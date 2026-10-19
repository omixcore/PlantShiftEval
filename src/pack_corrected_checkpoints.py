"""Archive verified source/adaptation checkpoints separately from source control."""
import argparse,json,zipfile
from datetime import datetime,timezone
from pathlib import Path
from run_corrected_study import matrix,valid_completed
from run_provenance import file_digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--group',choices=['scratch','mtdnet_pt','resnet18'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    jobs=[j for j in matrix() if (j['model'] not in ('mtdnet_pt','resnet18') if a.group=='scratch' else j['model']==a.group)]
    expected=55 if a.group=='scratch' else 5
    if len(jobs)!=expected:raise ValueError('Unexpected archive configuration count')
    records=[];files={}
    for j in jobs:
        run=a.source_root/'runs_corrected'/j['run_id'];evaldir=a.source_root/'report/corrected'/j['run_id']
        if not valid_completed(run):raise ValueError('Incomplete or changed checkpoint: '+j['run_id'])
        e=json.loads((evaldir/'evaluation.json').read_text())
        if file_digest(run/'model.pt')!=e['checkpoint_sha256']:raise ValueError('Checkpoint evaluation mismatch')
        for name,sha in e['outputs_sha256'].items():
            if file_digest(evaldir/name)!=sha:raise ValueError('Changed evaluation artifact: '+name)
        for folder in (run,evaldir):
            for f in folder.iterdir():
                if f.is_file():files[str(f.relative_to(a.source_root))]=f
        log=a.source_root/'logs'/f"{j['run_id']}.log"
        if log.exists():files[str(log.relative_to(a.source_root))]=log
        records.append(j)
    for path in ['DATASET_NOTICE.md','provenance/datasets.json','provenance/source_audit.json','provenance/target_audit.json','evidence/corrected/class_names.json']:
        f=a.repo/path
        if f.exists():files[path]=f
    manifest={'group':a.group,'runs':records,'created_utc':datetime.now(timezone.utc).isoformat(),
        'workflow_run_id':39265931897,'training_commit':'bd7cd1d94ff2591fa58dac5eb69f0e66225b8f72',
        'file_sha256':{name:file_digest(f) for name,f in sorted(files.items())},
        'scope':'Source checkpoints, original logits, metrics, manifests, logs and domain-evaluation outputs; pretrained archives include supervised target fine-tuning weights. AdaBN statistics are reproducible from the source checkpoint and frozen target-training images.'}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for name,f in sorted(files.items()):z.write(f,name)
        z.writestr('CHECKPOINT_ARCHIVE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'path':str(a.out),'runs':len(jobs),'bytes':a.out.stat().st_size,'sha256':file_digest(a.out)}),flush=True)

if __name__=='__main__':main()
