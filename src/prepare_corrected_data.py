"""Build a leaf-group-isolated source cache and preserve official target holdout.

Inputs are pinned separately in provenance/datasets.json. Never mutates historical runs.
"""
from __future__ import annotations
import argparse, collections, hashlib, io, json, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from run_provenance import file_digest

PV_REV = '243d37baa0780d93a8305d64f382bc1d41dffe92'
PD_REV = 'ace645742b6af1b0df9aaefc2c9b0c988e141814'
MAPPING = {'apple scab leaf':0,'apple rust leaf':2,'apple leaf':3,'blueberry leaf':4,
           'cherry leaf':6,'corn gray leaf spot':7,'corn rust leaf':8,
           'corn leaf blight':9,'grape leaf':14,'grape leaf black rot':11}

class UnionFind:
    def __init__(self,n): self.p=list(range(n))
    def find(self,i):
        while self.p[i]!=i:
            self.p[i]=self.p[self.p[i]];i=self.p[i]
        return i
    def join(self,a,b): self.p[self.find(a)]=self.find(b)

def dhash(im):
    a=np.asarray(im.convert('L').resize((9,8)),dtype=np.int16)
    return sum(int(v)<<i for i,v in enumerate((a[:,1:]>a[:,:-1]).ravel()))

def records(path, leafmap=None):
    rows=[]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
        for r in batch.to_pylist():
            raw=r['image']['bytes'];im=Image.open(io.BytesIO(raw)).convert('RGB')
            name=r['image'].get('path') or str(len(rows))
            key=os.path.splitext(name.split('___')[-1])[0].lower()
            groups=leafmap.get(key,[]) if leafmap else []
            rows.append({'file':str(path),'row':len(rows),'name':name,
                         'label':r.get('label'),'objects':r.get('objects'),
                         'sha256':hashlib.sha256(raw).hexdigest(),
                         'pixel_sha256':hashlib.sha256(np.asarray(im).tobytes()).hexdigest(),
                         'dhash':f'{dhash(im):016x}', 'leaf_groups':groups})
    return rows

def component_ids(rows):
    uf=UnionFind(len(rows));seen={}
    for i,r in enumerate(rows):
        keys=['raw:'+r['sha256'],'pixel:'+r['pixel_sha256'],'name:'+r['name'].lower()]
        keys+=['leaf:'+g for g in r['leaf_groups']]
        for k in keys:
            if k in seen:uf.join(i,seen[k])
            else:seen[k]=i
    return [uf.find(i) for i in range(len(rows))]

def group_split(labels, groups, fraction=0.1, seed=1234):
    rng=np.random.RandomState(seed);train=[];val=[]
    labels=np.asarray(labels)
    for lab in sorted(set(labels.tolist())):
        unique=sorted({groups[i] for i in np.where(labels==lab)[0]})
        if len(unique)<2:raise ValueError(f'Class {lab} has fewer than two independent groups')
        rng.shuffle(unique);n=max(1,min(len(unique)-1,int(round(len(unique)*fraction))))
        chosen=set(unique[:n])
        for i in np.where(labels==lab)[0]:
            (val if groups[i] in chosen else train).append(int(i))
    if {groups[i] for i in train}&{groups[i] for i in val}:
        raise ValueError('Group crosses labels and internal partitions')
    return sorted(train),sorted(val)

def decode_selected(rows, size):
    requested=collections.defaultdict(dict)
    for out_idx,r in enumerate(rows):requested[r['file']][r['row']]=out_idx
    result=np.empty((len(rows),size,size,3),dtype=np.uint8)
    for file,indices in requested.items():
        offset=0
        for batch in pq.ParquetFile(file).iter_batches(batch_size=64,columns=['image']):
            for cell in batch.column(0).to_pylist():
                if offset in indices:
                    im=Image.open(io.BytesIO(cell['bytes'])).convert('RGB')
                    result[indices[offset]]=np.asarray(im.resize((size,size),Image.Resampling.BILINEAR))
                offset+=1
    return result

def save_json(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')

def prepare_source(root, cap=320, size=160):
    leafmap=json.loads((root/'provenance/sources/leaf-map.json').read_text())
    tr=records(root/'data/pv_train.parquet',leafmap);te=records(root/'data/pv_test.parquet',leafmap)
    all_rows=tr+te;ids=component_ids(all_rows);test_groups=set(ids[len(tr):])
    # Known leaves are grouped; missing leaf IDs use image identity components.
    # Their biological independence remains unverified and is reported explicitly.
    eligible=[i for i,r in enumerate(tr) if ids[i] not in test_groups]
    hashes=np.array([int(r['dhash'],16) for r in all_rows],dtype=np.uint64)
    candidates=[];near_train=set()
    for start in range(0,len(eligible),128):
        ix=np.array(eligible[start:start+128]);distance=np.bitwise_count(hashes[ix,None]^hashes[None,len(tr):])
        for a,b in np.argwhere(distance<=4):
            near_train.add(int(ix[a]))
            candidates.append({'train_row':int(ix[a]),'test_row':int(b),'hamming':int(distance[a,b])})
    eligible=[i for i in eligible if i not in near_train]
    rng=np.random.RandomState(0);rng.shuffle(eligible);counts=collections.Counter();selected=[]
    for i in eligible:
        lab=tr[i]['label']
        if counts[lab]<cap:selected.append(i);counts[lab]+=1
    selected=sorted(selected);train_rows=[tr[i] for i in selected]
    labels=[r['label'] for r in train_rows];groups=[ids[i] for i in selected]
    if set(labels)!=set(range(15)):raise ValueError('Source filtering removed an entire class')
    ti,vi=group_split(labels,groups)
    source_meta=json.loads(pq.ParquetFile(root/'data/pv_train.parquet').schema_arrow.metadata[b'huggingface'])
    names=source_meta['info']['features']['label']['names']
    if isinstance(names,dict):names=[names[str(i)] for i in range(len(names))]
    species_names=sorted({n.split('___')[0].split('__')[0] for n in names})
    species={s:i for i,s in enumerate(species_names)}
    out=root/'cache_corrected'
    if out.exists() and any(out.iterdir()):raise ValueError('Corrected cache already contains artifacts')
    out.mkdir(exist_ok=True)
    for split,rows in [('train',train_rows),('test',te)]:
        np.save(out/f'{split}_x.npy',decode_selected(rows,size))
        np.save(out/f'{split}_y.npy',np.array([r['label'] for r in rows],dtype=np.int64))
        np.save(out/f'{split}_s.npy',np.array([species[names[r['label']].split('___')[0].split('__')[0]] for r in rows],dtype=np.int64))
    save_json(out/'meta.json',{'class_names':names,'species_names':species_names,
        'n_classes':len(names),'n_species':len(species_names),'img_size':size,
        'train_size':len(train_rows),'test_size':len(te),'train_per_class':dict(counts),
        'protocol':'known leaf IDs + image-identity fallback; exclude test-overlap components and dHash<=4 candidates; component-disjoint validation'})
    save_json(out/'split_indices.json',{'train_indices':ti,'validation_indices':vi,'split_seed':1234,
        'val_frac':0.1,'group_ids':groups,'method':'per-class shuffled leaf components'})
    audit={'source_train_rows':len(tr),'source_test_rows':len(te),
        'mapped_train_rows':sum(bool(r['leaf_groups']) for r in tr),
        'mapped_test_rows':sum(bool(r['leaf_groups']) for r in te),
        'excluded_train_test_component_overlap':sum(ids[i] in test_groups for i in range(len(tr))),
        'eligible_train_rows':len(eligible),'excluded_near_duplicate_train_rows':len(near_train),'retained_train_rows_without_leaf_ids':sum(not r['leaf_groups'] for r in train_rows),'retained_train_cache_rows':len(selected),
        'optimizer_train_rows':len(ti),'validation_rows':len(vi),
        'retained_train_per_class':dict(counts),'source_test_per_class':dict(collections.Counter(r['label'] for r in te)),
        'known_group_or_exact_overlap_after_filter':0,
        'near_duplicate_candidates_hamming_le4':len(candidates),
        'near_duplicate_review':'all screening candidates conservatively excluded from source training; similarity does not prove same leaf',
        'test_rows_without_leaf_ids':sum(not r['leaf_groups'] for r in te)}
    save_json(root/'provenance/source_audit.json',audit)
    save_json(root/'provenance/near_duplicate_candidates.json',candidates)
    save_json(root/'provenance/source_rows.json',{'train':tr,'test':te,'retained_train_original_rows':selected})
    print(json.dumps(audit,indent=2),flush=True)

def prepare_target(root,size=144):
    rows=[];official=[];exclusions=collections.Counter()
    files=[('train','pd_train-00000.parquet'),('train','pd_train-00001.parquet'),('test','pd_test-00000.parquet')]
    for split,file in files:
        rr=records(root/'data'/file)
        for r in rr:
            cats=[c for c in (r['objects'] or {}).get('category',[]) if c]
            if not cats:exclusions['no_category']+=1;continue
            top,count=collections.Counter(cats).most_common(1)[0]
            # Strict majority: tied categories do not define an unambiguous image label.
            if count<=len(cats)/2 and len(set(cats))>1:exclusions['ambiguous_multilabel']+=1;continue
            key=' '.join(top.replace('_',' ').lower().split())
            if key not in MAPPING:exclusions['unmapped_category']+=1;continue
            r['label']=MAPPING[key];rows.append(r);official.append(split)
    source=json.loads((root/'provenance/source_rows.json').read_text())
    sr=source['train']+source['test'];raw={r['sha256'] for r in sr};pixels={r['pixel_sha256'] for r in sr}
    hashes=np.array([int(r['dhash'],16) for r in sr],dtype=np.uint64)
    keep=[]
    for i,r in enumerate(rows):
        if r['sha256'] in raw or r['pixel_sha256'] in pixels:
            exclusions['source_exact_overlap']+=1
        elif np.bitwise_count(hashes^np.uint64(int(r['dhash'],16))).min()<=4:
            exclusions['source_similarity_candidate']+=1
        else:keep.append(i)
    rows=[rows[i] for i in keep];official=[official[i] for i in keep]
    ids=component_ids(rows)
    labels_by_component=collections.defaultdict(set)
    for r,g in zip(rows,ids):labels_by_component[g].add(r['label'])
    conflict={g for g,v in labels_by_component.items() if len(v)>1}
    keep=[i for i,g in enumerate(ids) if g not in conflict]
    exclusions['conflicting_label_component_rows']=len(rows)-len(keep)
    rows=[rows[i] for i in keep];official=[official[i] for i in keep]
    ids=component_ids(rows);test_groups={ids[i] for i,s in enumerate(official) if s=='test'}
    exclusions['target_train_test_identity_overlap']=sum(s=='train' and ids[i] in test_groups for i,s in enumerate(official))
    train=[i for i,s in enumerate(official) if s=='train' and ids[i] not in test_groups]
    test=[i for i,s in enumerate(official) if s=='test']
    target_hashes=np.array([int(r['dhash'],16) for r in rows],dtype=np.uint64)
    similarity_train={i for i in train if np.bitwise_count(target_hashes[test]^target_hashes[i]).min()<=4}
    exclusions['target_train_test_similarity_candidate']=len(similarity_train)
    train=[i for i in train if i not in similarity_train]
    ti,vi=group_split([rows[i]['label'] for i in train],[ids[i] for i in train],fraction=0.2)
    out=root/'cache_pd_corrected';out.mkdir(exist_ok=False)
    np.save(out/'x.npy',decode_selected(rows,size));np.save(out/'y.npy',np.array([r['label'] for r in rows],dtype=np.int64))
    save_json(out/'split_indices.json',{'train_indices':[train[i] for i in ti],
        'validation_indices':[train[i] for i in vi],'test_indices':test,'group_ids':ids,
        'split_seed':1234,'protocol':'official test remains held out; exact duplicates grouped'})
    info={'n':len(rows),'per_class':dict(collections.Counter(r['label'] for r in rows)),
        'official_train_retained':len(train),'official_test':len(test),'exclusions':dict(exclusions),
        'target_train_rows':len(ti),'target_validation_rows':len(vi),
        'test_per_class':dict(collections.Counter(rows[i]['label'] for i in test)),
        'source':'agyaatcoder/PlantDoc','revision':PD_REV,'mapping':MAPPING,
        'label_policy':'strict majority of object-category counts; mapped categories only',
        'caveat':'official test membership preserved after overlap/conflict filtering; source similarity candidates and target-train/test candidates screened at dHash<=4; biological independence not certified'}
    save_json(out/'meta.json',info);save_json(root/'provenance/target_audit.json',info)
    save_json(root/'provenance/target_rows.json',rows)
    print(json.dumps(info,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--target',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if a.target:prepare_target(root)
    else:prepare_source(root)
