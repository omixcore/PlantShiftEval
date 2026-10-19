"""Execute the prespecified 65-run study; resume only hash-verified completed runs."""
from __future__ import annotations
import argparse, concurrent.futures, json, os, subprocess, sys
from pathlib import Path
from run_provenance import file_digest
ROOT=Path(__file__).resolve().parents[1]

def matrix():
    jobs=[]
    for seed in range(5):
        # Core comparisons first, then additional recipe cells and ablations.
        for model in ('mtdnet','smallcnn','mtdnet_pt','resnet18'):
            jobs.append({'model':model,'epochs':6,'lr':'3e-4' if model in ('mtdnet_pt','resnet18') else '1e-3','seed':seed})
    for seed in range(5):
        for model in ('mtdnet','smallcnn'):
            for epochs,lr in [(6,'2e-3'),(12,'1e-3'),(12,'2e-3')]:
                jobs.append({'model':model,'epochs':epochs,'lr':lr,'seed':seed})
        for model in ('mtd_concat','mtd_single','mtd_appearance'):
            jobs.append({'model':model,'epochs':6,'lr':'1e-3','seed':seed})
    for j in jobs:j['run_id']=f"{j['model']}_e{j['epochs']}_lr{j['lr']}_s{j['seed']}"
    return jobs

def valid_completed(out):
    p=out/'run_manifest.json'
    if not p.exists():return False
    try:
        r=json.loads(p.read_text())
        return r['status']=='completed' and all(file_digest(out/k)==v for k,v in r['outputs'].items()) and set(r['outputs'])=={'model.pt','metrics.json','logits.pt'}
    except (OSError,KeyError,ValueError):return False

def run(job,threads):
    out=ROOT/'runs_corrected'/job['run_id']; log=ROOT/'logs'/f"{job['run_id']}.log"
    if valid_completed(out):
        recorded=json.loads((out/'metrics.json').read_text())
        if any(recorded[k]!=job[k] for k in ('model','epochs','seed')) or recorded['lr']!=float(job['lr']):
            return {'run_id':job['run_id'],'status':'blocked_configuration_mismatch'}
        return {'run_id':job['run_id'],'status':'verified_existing'}
    if out.exists() and any(out.iterdir()):return {'run_id':job['run_id'],'status':'blocked_partial_output'}
    cmd=[sys.executable,str(ROOT/'src/train.py'),'--model',job['model'],'--cache','cache_corrected',
         '--out',str(out.relative_to(ROOT)),'--epochs',str(job['epochs']),'--lr',job['lr'],
         '--seed',str(job['seed']),'--size','112','--threads',str(threads),'--val-frac','0.1']
    env=dict(os.environ,OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS='1')
    print('START',job['run_id'],flush=True)
    with log.open('w') as handle:
        code=subprocess.call(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
    result={'run_id':job['run_id'],'returncode':code,'status':'completed' if code==0 and valid_completed(out) else 'failed'}
    print(json.dumps(result),flush=True);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--workers',type=int,default=2);p.add_argument('--threads',type=int,default=4);p.add_argument('--core-only',action='store_true');a=p.parse_args()
    jobs=matrix();(ROOT/'experiments').mkdir(exist_ok=True)
    (ROOT/'experiments/corrected_matrix.json').write_text(json.dumps(jobs,indent=2)+'\n')
    if a.core_only:jobs=jobs[:20]
    if not a.execute:print(f'{len(jobs)} planned runs; pass --execute to train.');return
    (ROOT/'logs').mkdir(exist_ok=True)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        futures=[pool.submit(run,j,a.threads) for j in jobs]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            (ROOT/'experiments/execution_status.json').write_text(json.dumps(results,indent=2)+'\n')
    if any(r['status'] not in ('completed','verified_existing') for r in results):raise SystemExit(1)

if __name__=='__main__':main()
