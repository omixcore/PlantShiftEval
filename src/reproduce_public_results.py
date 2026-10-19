"""Recompute reported metrics from public prediction arrays without model weights."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from run_corrected_study import matrix
from run_provenance import file_digest

def metrics(z,y):
    z=np.asarray(z);y=np.asarray(y);pred=z.argmax(1)
    confusion=np.bincount(y*15+pred,minlength=225).reshape(15,15)
    tp=np.diag(confusion).astype(float);den=confusion.sum(0)+confusion.sum(1)
    f1=np.divide(2*tp,den,out=np.zeros(15),where=den>0).mean()
    q=z-z.max(1,keepdims=True);p=np.exp(q);p=p/p.sum(1,keepdims=True)
    conf=p.max(1);correct=pred==y;ece=0
    for lo,hi in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
        mask=(conf>lo)&(conf<=hi)
        if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
    return {'accuracy':float(correct.mean()),'macro_f1':float(f1),'ece':float(ece)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    folder=a.root/'evidence/corrected';index=json.loads((folder/'index.json').read_text());byid={r['run_id']:r for r in index}
    missing=[j['run_id'] for j in matrix() if j['run_id'] not in byid]
    if missing:raise SystemExit('Public prediction study incomplete: '+str(len(missing))+' missing runs')
    rows=[];differences=[]
    def check(run,name,actual,expected,tol=1e-10):
        if not math.isclose(actual,expected,rel_tol=0,abs_tol=tol):differences.append({'run_id':run,'metric':name,'actual':actual,'expected':expected})
    for job in matrix():
        run=job['run_id'];path=folder/run;bypath=path/'predictions.npz'
        if file_digest(bypath)!=byid[run]['predictions_sha256']:raise ValueError('Changed public predictions: '+run)
        recorded=json.loads((path/'metrics.json').read_text());e=json.loads((path/'evaluation.json').read_text())
        with np.load(bypath,allow_pickle=False) as b:
            source=metrics(b['source_logits'],b['source_labels']);target=metrics(b['target_zero_shot_logits'],b['target_zero_shot_labels'])
            calibrated=metrics(b['target_zero_shot_logits']/e['source_validation_temperature'],b['target_zero_shot_labels'])
            check(run,'source_macro_f1',source['macro_f1'],recorded['test_macro_f1'])
            check(run,'source_accuracy',source['accuracy'],recorded['test_acc'])
            check(run,'source_ece',source['ece'],recorded['ece'],2e-6)
            check(run,'target_accuracy',target['accuracy'],e['zero_shot']['accuracy'])
            check(run,'target_ece',target['ece'],e['zero_shot']['ece'],2e-6)
            check(run,'target_calibrated_ece',calibrated['ece'],e['zero_shot_source_calibrated']['ece'],2e-6)
            row=dict(job,source_macro_f1=source['macro_f1'],source_accuracy=source['accuracy'],source_ece=source['ece'],target_accuracy=target['accuracy'],target_ece=target['ece'],target_calibrated_ece=calibrated['ece'],corruption_accuracy=float(np.mean(list(e['corruptions'].values()))))
            for method in ('adabn','supervised_ft'):
                if 'adaptation' in e:
                    key='target_'+method;score=metrics(b[key+'_logits'],b[key+'_labels']);row[method+'_accuracy']=score['accuracy']
                    check(run,method+'_accuracy',score['accuracy'],e['adaptation'][method]['accuracy'])
                else:row[method+'_accuracy']=None
            rows.append(row)
    out=a.root/'report/corrected_summary';out.mkdir(parents=True,exist_ok=True)
    with (out/'public_recomputed_metrics.csv').open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result={'runs_checked':len(rows),'metric_disagreements':differences,'ece_absolute_tolerance':2e-6,'accuracy_f1_absolute_tolerance':1e-10,'status':'passed' if not differences else 'failed','scope':'Statistical metric regeneration from public arrays; corruption aggregation uses archived per-condition records. Model retraining/inference is a separate reproduction route.'}
    (out/'public_reproduction.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if differences:raise SystemExit(1)

if __name__=='__main__':main()
