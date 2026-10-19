"""Regenerate corrected tables/figures and results text; fail if evidence is incomplete."""
from __future__ import annotations
import csv,json,math,statistics,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t,ttest_rel
from run_corrected_study import matrix,valid_completed
ROOT=Path(__file__).resolve().parents[1]

def paired_effect(a,b):
    seeds=sorted(set(a)&set(b));d=np.array([a[s]-b[s] for s in seeds],dtype=float)
    if len(seeds)<2:return {'n':len(seeds),'status':'insufficient_seeds'}
    mean=float(d.mean());sd=float(d.std(ddof=1));half=float(student_t.ppf(.975,len(d)-1)*sd/math.sqrt(len(d)))
    p=float(ttest_rel([a[s] for s in seeds],[b[s] for s in seeds]).pvalue)
    if not math.isfinite(p):p=1.0 if mean==0 else 0.0
    return {'n':len(seeds),'seeds':seeds,'mean_difference':mean,'sample_sd_difference':sd,
            'ci95_student_t':[mean-half,mean+half],'p_unadjusted':p,
            'uncertainty_unit':'paired training seeds, conditional on fixed data split',
            'assumption':'Student-t interval/test assumes approximately normal seed differences; n=5 offers limited power'}

def holm(records):
    ordered=sorted([r for r in records if 'p_unadjusted' in r],key=lambda r:r['p_unadjusted']);floor=0
    for k,r in enumerate(ordered):
        floor=max(floor,min(1,(len(ordered)-k)*r['p_unadjusted']));r['p_holm']=floor

def main():
    out=ROOT/'report/corrected_summary';out.mkdir(parents=True,exist_ok=True)
    rows=[];missing=[];details={}
    for job in matrix():
        run=ROOT/'runs_corrected'/job['run_id'];ev=ROOT/'report/corrected'/job['run_id']/'evaluation.json'
        if not valid_completed(run) or not ev.exists():missing.append(job['run_id']);continue
        metric=json.loads((run/'metrics.json').read_text());e=json.loads(ev.read_text());details[job['run_id']]=e
        row=dict(job,params=metric['params'],source_macro_f1=metric['test_macro_f1'],source_accuracy=metric['test_acc'],
            source_ece=metric['ece'],target_accuracy=e['zero_shot']['accuracy'],target_ece=e['zero_shot']['ece'],
            target_calibrated_ece=e['zero_shot_source_calibrated']['ece'],corruption_accuracy=statistics.mean(e['corruptions'].values()))
        for method in ('adabn','supervised_ft'):
            row[method+'_accuracy']=e.get('adaptation',{}).get(method,{}).get('accuracy')
        rows.append(row)
    status={'planned':65,'completed_training_and_evaluation':len(rows),'missing':missing,'status':'complete' if not missing else 'partial'}
    (out/'completeness.json').write_text(json.dumps(status,indent=2)+'\n')
    (out/'per_run.json').write_text(json.dumps(rows,indent=2)+'\n')
    if rows:
        with (out/'per_run.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    grouped=defaultdict(list)
    for r in rows:grouped[(r['model'],r['epochs'],r['lr'])].append(r)
    aggregates=[]
    measures=('source_macro_f1','source_accuracy','source_ece','target_accuracy','target_ece','target_calibrated_ece','corruption_accuracy','adabn_accuracy','supervised_ft_accuracy')
    for (model,epochs,lr),rr in sorted(grouped.items()):
        record={'model':model,'epochs':epochs,'lr':lr,'n_seeds':len(rr),'params':rr[0]['params']}
        for measure in measures:
            values=[r[measure] for r in rr if r[measure] is not None]
            record[measure]={'mean':statistics.mean(values),'sd':statistics.stdev(values) if len(values)>1 else None,'n':len(values)} if values else None
        aggregates.append(record)
    (out/'aggregates.json').write_text(json.dumps(aggregates,indent=2)+'\n')
    def values(model,epochs=6,lr='1e-3',measure='source_macro_f1'):
        return {r['seed']:r[measure] for r in grouped[(model,epochs,lr)] if r[measure] is not None}
    comparisons=[]
    for name,left,right,lr in [('primary_mtd_vs_cnn','mtdnet','smallcnn','1e-3'),
        ('gate','mtdnet','mtd_concat','1e-3'),('auxiliary_task','mtdnet','mtd_single','1e-3'),
        ('texture_branch_with_capacity_change','mtdnet','mtd_appearance','1e-3'),
        ('pretrained_increment','mtdnet_pt','resnet18','3e-4')]:
        comparisons.append(dict(name=name,**paired_effect(values(left,lr=lr),values(right,lr=lr))))
    holm(comparisons)
    # Difference-of-differences on fixed seed pairs: interaction of epochs and LR.
    recipe=[]
    for model in ('mtdnet','smallcnn'):
        a=values(model,12,'2e-3');b=values(model,6,'2e-3');c=values(model,12,'1e-3');d=values(model,6,'1e-3')
        seeds=set(a)&set(b)&set(c)&set(d)
        recipe.append(dict(model=model,contrast='(12-6 epochs at LR2e-3) minus (12-6 epochs at LR1e-3)',
            **paired_effect({s:a[s]-b[s] for s in seeds},{s:c[s]-d[s] for s in seeds})))
    (out/'paired_comparisons.json').write_text(json.dumps({'family':comparisons,'factorial_interactions':recipe},indent=2)+'\n')
    text=['# Corrected experimental results','',f"Evidence status: **{status['status']}** ({len(rows)}/65 training-and-evaluation pairs).",
          '','This document is generated only from corrected run artifacts. It does not replace historical results with hypothetical values. The source data were filtered for known leaf overlap and similarity candidates; incomplete leaf IDs remain a limitation. The external evaluation uses the official mapped PlantDoc test partition, with target training/validation separate.','',
          '| Model | Epochs | Peak LR | Seeds | Source macro-F1 (mean ± sample SD) | Target accuracy |',
          '|---|---:|---:|---:|---:|---:|']
    for r in aggregates:
        m=r['source_macro_f1'];sd='NA' if m['sd'] is None else f"{m['sd']:.4f}"
        text.append(f"| {r['model']} | {r['epochs']} | {r['lr']} | {r['n_seeds']} | {m['mean']:.4f} ± {sd} | {r['target_accuracy']['mean']:.4f} |")
    text+=['','## Interpretation boundaries','',
        'Use the paired intervals and individual seed points to assess the magnitude and stability of effects. A nonsignificant contrast does not establish equivalence. Texture-branch removal changes capacity and cannot establish a unique texture mechanism. Source-test scores, synthetic corruption and independent target-domain performance answer different questions.',
        '', 'AdaBN accesses only target-training images without labels. Supervised fine-tuning uses target-training labels and target-validation checkpoint selection. Both are reported separately from zero-shot transfer. Source-fitted calibration uses only the source validation labels. The small official target holdout limits precision and breadth of field-generalization claims.',
        '', 'Historical metrics cannot be compared directly to this cleaned source protocol because both split validity and retained training composition changed. No claim of universal superiority, semantic disentanglement, deployment readiness or practical equivalence is generated automatically.']
    (ROOT/'manuscript').mkdir(exist_ok=True);(ROOT/'manuscript/corrected_results.md').write_text('\n'.join(text)+'\n')
    if rows and not missing:figures(rows,grouped,out)
    print(json.dumps(status),flush=True)
    if missing:raise SystemExit(1)

def figures(rows,grouped,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.tight_layout()
        for ext in ('svg','pdf','png'):fig.savefig(out/f'{name}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    def plot_groups(ax,configs,measure):
        for i,(model,ep,lr) in enumerate(configs):
            vv=[r[measure] for r in grouped[(model,ep,lr)] if r[measure] is not None]
            if not vv:continue
            ax.scatter(i+np.linspace(-.08,.08,len(vv)),vv,color='#3284b7',s=25,zorder=3)
            ax.errorbar(i,np.mean(vv),yerr=np.std(vv,ddof=1) if len(vv)>1 else 0,fmt='_',color='#163a50',capsize=4)
        labels={'mtdnet':'MTD-Net','smallcnn':'Plain CNN','mtdnet_pt':'MTD-P','resnet18':'ResNet-18','mtd_concat':'No gate','mtd_single':'No auxiliary task','mtd_appearance':'Appearance only'}
        ax.set_xticks(range(len(configs)),[labels[m] for m,_,_ in configs],rotation=20,ha='right')
    core=[('mtdnet',6,'1e-3'),('smallcnn',6,'1e-3'),('mtdnet_pt',6,'3e-4'),('resnet18',6,'3e-4')]
    figure_rows={
        'Figure1_source_comparisons':[r for r in rows if (r['model'],r['epochs'],r['lr']) in core],
        'Figure2_training_factorial':[r for r in rows if r['model'] in ('mtdnet','smallcnn')],
        'Figure3_component_ablations':[r for r in rows if r['model'] in ('mtd_concat','mtd_single','mtd_appearance') or (r['model'],r['epochs'],r['lr'])==('mtdnet',6,'1e-3')],
        'Figure4_domain_limits':[r for r in rows if (r['model'],r['epochs'],r['lr']) in core],
        'Figure5_target_access':[r for r in rows if r['model'] in ('mtdnet_pt','resnet18')]}
    for name,records in figure_rows.items():
        with (out/(name+'_data.csv')).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(records)
    f,ax=plt.subplots(figsize=(6,3.4));plot_groups(ax,core,'source_macro_f1');ax.set_ylabel('Source test macro-F1');save(f,'Figure1_source_comparisons')
    f,axes=plt.subplots(1,2,figsize=(7,3.4))
    for ax,model in zip(axes,('mtdnet','smallcnn')):
        for lr,color in [('1e-3','#3284b7'),('2e-3','#e08b49')]:
            means=[]
            for ep in (6,12):
                vv=[r['source_macro_f1'] for r in grouped[(model,ep,lr)]];means.append(np.mean(vv) if vv else np.nan)
                if vv:ax.scatter(ep+np.linspace(-.12,.12,len(vv)),vv,color=color,s=15,alpha=.65)
            ax.plot([6,12],means,'o-',label='LR '+lr,color=color)
        ax.set(title={'mtdnet':'MTD-Net','smallcnn':'Plain CNN'}[model],xlabel='Epochs',ylabel='Source test macro-F1',xticks=[6,12]);ax.legend(frameon=False)
    save(f,'Figure2_training_factorial')
    f,ax=plt.subplots(figsize=(6,3.4));plot_groups(ax,[('mtdnet',6,'1e-3'),('mtd_concat',6,'1e-3'),('mtd_single',6,'1e-3'),('mtd_appearance',6,'1e-3')],'source_macro_f1');ax.set_ylabel('Source test macro-F1');save(f,'Figure3_component_ablations')
    f,axes=plt.subplots(1,2,figsize=(8,3.4))
    plot_groups(axes[0],core,'corruption_accuracy');axes[0].set_ylabel('Mean synthetic-corruption accuracy')
    plot_groups(axes[1],core,'target_accuracy');axes[1].set_ylabel('Official target-test accuracy');save(f,'Figure4_domain_limits')
    f,axes=plt.subplots(1,2,figsize=(8,3.4))
    for ax,model in zip(axes,('mtdnet_pt','resnet18')):
        for i,measure in enumerate(('target_accuracy','adabn_accuracy','supervised_ft_accuracy')):
            vv=[r[measure] for r in grouped[(model,6,'3e-4')] if r[measure] is not None]
            if vv:ax.scatter(i+np.linspace(-.06,.06,len(vv)),vv,color='#3284b7');ax.plot(i,np.mean(vv),'_',color='#163a50',ms=15)
        ax.set(title={'mtdnet_pt':'MTD-P','resnet18':'ResNet-18'}[model],xticks=[0,1,2],xticklabels=['Zero-shot','AdaBN','Supervised FT'],ylabel='Official target-test accuracy')
    save(f,'Figure5_target_access')

if __name__=='__main__':main()
