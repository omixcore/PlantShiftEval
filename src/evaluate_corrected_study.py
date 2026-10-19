"""Evaluate frozen corrected checkpoints; target official test never selects models.

Writes per-run raw outputs before aggregation. Historical reports are untouched.
"""
from __future__ import annotations
import argparse, copy, json, os, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import load_cache, Split
from models import build_model
from train import LeafDS,evaluate,ece_score
from robustness import run_condition
from run_provenance import file_digest
from run_corrected_study import valid_completed
ROOT=Path(__file__).resolve().parents[1]
GRID=[('noise',1),('noise',2),('blur',1),('blur',2),('brightness',2),('contrast',2),('occlusion',1),('occlusion',2)]

def loader(x,y,species=None,size=112,train=False,seed=0):
    if species is None:species=np.zeros(len(y),dtype=np.int64)
    return DataLoader(LeafDS(x,y,species,train,size),batch_size=64,shuffle=train,
                      generator=torch.Generator().manual_seed(seed),num_workers=0)

def scores(logits,labels):
    from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score
    p=torch.softmax(torch.as_tensor(logits),1).numpy();pred=p.argmax(1)
    return {'n':len(labels),'accuracy':float(accuracy_score(labels,pred)),
            'macro_f1_15':float(f1_score(labels,pred,labels=list(range(15)),average='macro',zero_division=0)),
            'balanced_accuracy_supported':float(balanced_accuracy_score(labels,pred)),
            'ece':ece_score(p,labels)}

def temperature(z,y):
    from scipy.optimize import minimize_scalar
    z=np.asarray(z,dtype=np.float64)
    def nll(logt):
        v=z/np.exp(logt);v-=v.max(1,keepdims=True)
        return float(np.mean(np.log(np.exp(v).sum(1))-v[np.arange(len(y)),y]))
    fit=minimize_scalar(nll,bounds=(-3,3),method='bounded')
    return float(np.exp(fit.x))

def save_predictions(path,res):
    np.savez_compressed(path,logits=res['logits'],labels=res['labels'])

def adapt_bn(model,train_loader):
    model.eval()
    for module in model.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
            module.train();module.reset_running_stats();module.momentum=None
    with torch.no_grad():
        for xb,_,_ in train_loader:model(xb)
    model.eval();return model

def finetune(model,x,y,train_ids,val_ids,out,seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    tr=loader(x[train_ids],y[train_ids],train=True,seed=seed)
    va=loader(x[val_ids],y[val_ids]);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.02)
    history=[];best=-1;state=None;best_epoch=None
    for epoch in range(1,7):
        model.train()
        for xb,yb,_ in tr:
            pred=model(xb)['disease'];loss=torch.nn.functional.cross_entropy(pred,yb)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        res=evaluate(model,va,'cpu');history.append({'epoch':epoch,'val_macro_f1':res['macro_f1']})
        if res['macro_f1']>best:
            best=res['macro_f1'];state=copy.deepcopy(model.state_dict());best_epoch=epoch
    model.load_state_dict(state);torch.save(state,out/'target_finetuned.pt')
    return model,{'epochs':6,'lr':1e-4,'best_epoch':best_epoch,'history':history,
                  'selection':'target training-internal validation macro-F1; no official target test access'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='*');p.add_argument('--threads',type=int,default=4);p.add_argument('--adapt',action='store_true');a=p.parse_args()
    torch.set_num_threads(a.threads)
    meta,tr,te=load_cache(str(ROOT/'cache_corrected'))
    source_split=json.loads((ROOT/'cache_corrected/split_indices.json').read_text());v=source_split['validation_indices']
    target=ROOT/'cache_pd_corrected';x=np.load(target/'x.npy');y=np.load(target/'y.npy')
    split=json.loads((target/'split_indices.json').read_text());test_ids=split['test_indices']
    outroot=ROOT/'report/corrected';outroot.mkdir(parents=True,exist_ok=True)
    runs=[ROOT/'runs_corrected'/r for r in a.runs] if a.runs else sorted((ROOT/'runs_corrected').glob('*'))
    for run in runs:
        if not valid_completed(run):print('SKIP incomplete',run.name,flush=True);continue
        mj=json.loads((run/'metrics.json').read_text());out=outroot/run.name;out.mkdir(exist_ok=True)
        resultpath=out/'evaluation.json'
        if resultpath.exists():
            previous=json.loads(resultpath.read_text())
            if previous.get('checkpoint_sha256')!=file_digest(run/'model.pt'):raise ValueError('Checkpoint changed after target evaluation')
            if not a.adapt or 'adaptation' in previous or mj['model'] not in ('mtdnet_pt','resnet18'):
                print('VERIFIED existing evaluation',run.name,flush=True);continue
            raise ValueError('Do not silently reevaluate the target test; use the recorded evaluation and an explicit adaptation continuation')
        print('EVALUATE',run.name,flush=True)
        torch.manual_seed(mj['seed']);model=build_model(mj['model'],meta['n_classes'],meta['n_species'])
        model.load_state_dict(torch.load(run/'model.pt',map_location='cpu',weights_only=True))
        val=evaluate(model,loader(tr.images[v],tr.labels[v],tr.species[v]),'cpu')
        save_predictions(out/'source_validation.npz',val);t=temperature(val['logits'],val['labels'])
        targettest=loader(x[test_ids],y[test_ids]);zero=evaluate(model,targettest,'cpu')
        save_predictions(out/'target_zero_shot.npz',zero)
        result={'run_id':run.name,'model':mj['model'],'seed':mj['seed'],
                'checkpoint_sha256':file_digest(run/'model.pt'),'source_manifest_sha256':file_digest(run/'run_manifest.json'),
                'target_split_sha256':file_digest(target/'split_indices.json'),
                'zero_shot':scores(zero['logits'],zero['labels']),
                'source_validation_temperature':t,
                'zero_shot_source_calibrated':scores(zero['logits']/t,zero['labels']),
                'target_test_rows':test_ids,'protocol':'official target test only; no target-test tuning'}
        source_test=loader(te.images,te.labels,te.species)
        result['corruptions']={f'{kind}_{sev}':run_condition(model,source_test,kind,sev,seed=0) for kind,sev in GRID}
        if a.adapt and mj['model'] in ('mtdnet_pt','resnet18'):
            train_ids=split['train_indices'];val_ids=split['validation_indices']
            bn=adapt_bn(copy.deepcopy(model),loader(x[train_ids],y[train_ids]))
            bres=evaluate(bn,targettest,'cpu');save_predictions(out/'target_adabn.npz',bres)
            ft,details=finetune(copy.deepcopy(model),x,y,train_ids,val_ids,out,mj['seed'])
            fres=evaluate(ft,targettest,'cpu');save_predictions(out/'target_supervised_ft.npz',fres)
            result['adaptation']={'adabn':scores(bres['logits'],bres['labels']),
                'adabn_access':'target training images only; labels not supplied to model; BN running-statistics update',
                'supervised_ft':scores(fres['logits'],fres['labels']), 'finetuning_protocol':details,
                'target_train_rows':train_ids,'target_validation_rows':val_ids}
        result['outputs_sha256']={f.name:file_digest(f) for f in out.iterdir() if f.is_file() and f.name!='evaluation.json'}
        resultpath.write_text(json.dumps(result,indent=2)+'\n');print('COMPLETED evaluation',run.name,flush=True)

if __name__=='__main__':main()
