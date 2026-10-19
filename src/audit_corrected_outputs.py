"""Verify complete evidence across runs before final reporting."""
from pathlib import Path
import argparse,json
from run_corrected_study import matrix,valid_completed
from run_provenance import file_digest

def audit(root):
    failures=[];cache_signatures=set();target_signatures=set();code_signatures=set();commits=set();count=0
    for job in matrix():
        run=root/'runs_corrected'/job['run_id'];ep=root/'report/corrected'/job['run_id']/'evaluation.json'
        if not valid_completed(run):failures.append(job['run_id']+': training outputs missing or changed');continue
        m=json.loads((run/'metrics.json').read_text());manifest=json.loads((run/'run_manifest.json').read_text())
        if any(m[k]!=job[k] for k in ('model','epochs','seed')) or m['lr']!=float(job['lr']):failures.append(job['run_id']+': configuration mismatch')
        if not m.get('val_split') or m.get('selection_metric')!='validation_macro_f1' or m.get('test_evaluations')!=1:failures.append(job['run_id']+': source selection/evaluation protocol mismatch')
        commits.add(manifest.get('git_commit'))
        code_signatures.add(tuple(sorted(manifest['source_sha256'].items())))
        signature=tuple(sorted((Path(i['path']).name,i['sha256']) for i in manifest['inputs']))
        cache_signatures.add(signature)
        split=manifest['split']
        if set(split['train_indices'])&set(split['validation_indices']):failures.append(job['run_id']+': validation overlap')
        if not ep.exists():failures.append(job['run_id']+': evaluation missing');continue
        e=json.loads(ep.read_text())
        if e['checkpoint_sha256']!=file_digest(run/'model.pt') or e['source_manifest_sha256']!=file_digest(run/'run_manifest.json'):failures.append(job['run_id']+': evaluation provenance mismatch')
        for name,sha in e['outputs_sha256'].items():
            if file_digest(ep.parent/name)!=sha:failures.append(job['run_id']+': changed evaluation output '+name)
        if len(e['corruptions'])!=8:failures.append(job['run_id']+': incomplete corruption grid')
        target_signatures.add((e['target_split_sha256'],tuple(e['target_test_rows'])))
        if job['model'] in ('mtdnet_pt','resnet18'):
            ad=e.get('adaptation',{})
            if any(k not in ad for k in ('adabn','supervised_ft','finetuning_protocol')):failures.append(job['run_id']+': adaptation missing')
            else:
                tr=set(ad['target_train_rows']);va=set(ad['target_validation_rows']);te=set(e['target_test_rows'])
                if tr&va or tr&te or va&te:failures.append(job['run_id']+': target partition overlap')
        count+=1
    if len(cache_signatures)!=1:failures.append('Source-cache fingerprints differ across runs')
    if len(target_signatures)!=1:failures.append('Target-test assignments differ across runs')
    if len(code_signatures)!=1 or len(commits)!=1:failures.append('Training source revisions/fingerprints differ across runs')
    report={'planned':65,'audited_training_and_evaluation_pairs':count,'source_input_fingerprint_sets':len(cache_signatures),'target_split_fingerprint_sets':len(target_signatures),'training_source_fingerprint_sets':len(code_signatures),'training_commits':sorted(commits,key=str),'failures':failures,'status':'passed' if not failures and count==65 else 'incomplete_or_failed'}
    out=root/'report/corrected_summary';out.mkdir(parents=True,exist_ok=True)
    (out/'evidence_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();report=audit(a.root);print(json.dumps(report,indent=2));raise SystemExit(0 if report['status']=='passed' else 1)
