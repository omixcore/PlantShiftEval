"""Conditional target-ensemble uncertainty; distinct from across-seed variation."""
import argparse,json,math
from pathlib import Path
import numpy as np
from run_corrected_study import matrix

def wilson(k,n):
    if n<=0:raise ValueError('Positive number of test images required')
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,center-half),min(1,center+half)]

def summarize(root):
    groups={}
    for j in matrix():
        key=(j['model'],j['epochs'],j['lr']);groups.setdefault(key,[]).append(j)
    records=[]
    for (model,epochs,lr),jobs in groups.items():
        methods=['zero_shot']+(['adabn','supervised_ft'] if model in ('mtdnet_pt','resnet18') else [])
        for method in methods:
            files=[root/'report/corrected'/j['run_id']/f'target_{method}.npz' for j in jobs]
            if not all(f.exists() for f in files):continue
            probabilities=[];labels=None
            for file in files:
                b=np.load(file);z=b['logits'];y=b['labels']
                if labels is None:labels=y
                elif not np.array_equal(labels,y):raise ValueError('Target prediction order/labels differ')
                z=z-z.max(1,keepdims=True);p=np.exp(z);probabilities.append(p/p.sum(1,keepdims=True))
            pred=np.mean(probabilities,axis=0).argmax(1);n=len(labels);k=int((pred==labels).sum())
            per_class=[]
            for lab in range(15):
                support=int((labels==lab).sum());correct=int(((labels==lab)&(pred==lab)).sum())
                per_class.append({'class_index':lab,'support':support,'correct':correct,
                                  'recall':correct/support if support else None})
            records.append({'model':model,'epochs':epochs,'lr':lr,'method':method,
                'ensemble_seeds':len(jobs),'n_images':n,'correct':k,'accuracy':k/n,
                'wilson_ci95':wilson(k,n),'per_class':per_class,
                'uncertainty_unit':'target images conditional on a fixed five-seed probability ensemble',
                'caveat':'Approximate binomial image interval; unknown acquisition dependencies and population selection are not captured. Ensemble accuracy differs from mean per-seed accuracy.'})
    out=root/'report/corrected_summary';out.mkdir(parents=True,exist_ok=True)
    (out/'target_ensemble_uncertainty.json').write_text(json.dumps(records,indent=2)+'\n')
    return records

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    print('Target ensemble summaries:',len(summarize(a.root)))
