"""Bootstrap target metrics and paired comparisons from pinned public predictions."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from run_provenance import file_digest

CORE = {"MTD-Net": ("mtdnet","1e-3"), "Plain CNN": ("smallcnn","1e-3"),
        "MTD-P": ("mtdnet_pt","3e-4"), "ResNet-18": ("resnet18","3e-4")}
N_CLASSES = 15

def softmax(z):
    z=np.asarray(z,dtype=np.float64); z=z-z.max(axis=1,keepdims=True)
    e=np.exp(z); return e/e.sum(axis=1,keepdims=True)

def class_metrics(y,pred):
    y=np.asarray(y,dtype=np.int64); pred=np.asarray(pred,dtype=np.int64)
    if len(y)==0 or y.shape!=pred.shape or np.any(y<0) or np.any(y>=N_CLASSES) or np.any(pred<0) or np.any(pred>=N_CLASSES):
        raise ValueError("Invalid labels/predictions")
    cm=np.bincount(y*N_CLASSES+pred,minlength=N_CLASSES*N_CLASSES).reshape(N_CLASSES,N_CLASSES)
    tp=np.diag(cm).astype(float); sup=cm.sum(1).astype(int); pc=cm.sum(0).astype(int)
    pr=np.divide(tp,pc,out=np.zeros(N_CLASSES),where=pc>0)
    re=np.divide(tp,sup,out=np.zeros(N_CLASSES),where=sup>0)
    f1=np.divide(2*pr*re,pr+re,out=np.zeros(N_CLASSES),where=(pr+re)>0)
    present=np.flatnonzero(sup>0)
    rows=[]
    for k in range(N_CLASSES):
        rows.append({"class_index":k,"support":int(sup[k]),"predicted_count":int(pc[k]),"correct":int(tp[k]),
                     "precision":float(pr[k]) if pc[k] else None,"recall":float(re[k]) if sup[k] else None,
                     "f1":float(f1[k]) if sup[k] else None})
    return {"n":int(len(y)),"correct":int(np.sum(y==pred)),"accuracy":float(np.mean(y==pred)),
            "macro_f1_15":float(f1.mean()),"macro_f1_supported":float(f1[present].mean()),
            "balanced_accuracy_supported":float(re[present].mean()),"n_supported_classes":int(len(present)),
            "per_class":rows}

def probability_metrics(y,p):
    p=np.asarray(p,dtype=np.float64)
    if p.ndim!=2 or p.shape!=(len(y),N_CLASSES) or not np.isfinite(p).all(): raise ValueError("Invalid probabilities")
    return class_metrics(y,p.argmax(1))

def bootstrap_indices(y,reps=5000,seed=20261020,stratified=True):
    y=np.asarray(y); rng=np.random.default_rng(seed)
    if stratified:
        strata=[np.flatnonzero(y==lab) for lab in sorted(np.unique(y).tolist())]
        if any(len(ix)==0 for ix in strata): raise ValueError("Empty class stratum")
        return [np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in strata]) for _ in range(reps)]
    return [rng.integers(0,len(y),size=len(y)) for _ in range(reps)]

def ci(values):
    return [float(np.quantile(np.asarray(values,dtype=float),.025)),float(np.quantile(np.asarray(values,dtype=float),.975))]

def bootstrap_model(y,p,stratified,ordinary):
    point=probability_metrics(y,p); acc=[]; macro=[]; cls={i:{"precision":[],"recall":[],"f1":[]} for i in range(N_CLASSES)}
    for ix in ordinary: acc.append(float(np.mean(p[ix].argmax(1)==y[ix])))
    for ix in stratified:
        m=probability_metrics(y[ix],p[ix]); macro.append(m["macro_f1_supported"])
        for item in m["per_class"]:
            for k in ("precision","recall","f1"):
                if item[k] is not None: cls[item["class_index"]][k].append(item[k])
    per=[]
    for item in point["per_class"]:
        row=dict(item); k=row["class_index"]
        for metric in ("precision","recall","f1"):
            row[metric+"_bootstrap_ci95"]=ci(cls[k][metric]) if cls[k][metric] else None
        per.append(row)
    return {"point":point,"accuracy_image_bootstrap_ci95":ci(acc),
            "macro_f1_supported_stratified_bootstrap_ci95":ci(macro),"per_class":per,
            "interval_method":{"replicates":len(stratified),"seed":20261020,
            "accuracy":"ordinary image bootstrap","macro_f1_and_classwise":"class-stratified image bootstrap"}}

def mcnemar_exact(a,b):
    a=np.asarray(a,dtype=bool); b=np.asarray(b,dtype=bool)
    win=int(np.sum(a&~b)); loss=int(np.sum(~a&b)); n=win+loss
    p=1.0 if n==0 else min(1.0,2*sum(math.comb(n,k) for k in range(min(win,loss)+1)/(2**n)))
    return {"dinov2_only_correct":win,"baseline_only_correct":loss,"discordant_images":n,"mcnemar_exact_p":float(p)}

def holm(ps):
    order=np.argsort(ps); out=[1.0]*len(ps); running=0.0
    for rank,idx in enumerate(order):
        running=max(running,(len(ps)-rank)*float(ps[idx])); out[int(idx)]=min(1.0,running)
    return out

def load_core(root):
    folder=root/"evidence/corrected"; index=json.loads((folder/"index.json").read_text()); byid={r["run_id"]:r for r in index}
    labels=None; models={}
    for name,(model,lr) in CORE.items():
        ps=[]; seed_metrics=[]
        for seed in range(5):
            rid=f"{model}_e6_lr{lr}_s{seed}"
            if rid not in byid: raise FileNotFoundError("Missing prediction index entry: "+rid)
            path=folder/rid/"predictions.npz"
            if file_digest(path)!=byid[rid]["predictions_sha256"]: raise ValueError("Prediction checksum mismatch: "+rid)
            with np.load(path,allow_pickle=False) as z:
                y=z["target_zero_shot_labels"].astype(np.int64); p=softmax(z["target_zero_shot_logits"])
            if labels is None: labels=y
            elif not np.array_equal(labels,y): raise ValueError("Target order/labels differ: "+rid)
            ps.append(p); seed_metrics.append(probability_metrics(y,p))
        models[name]={"probabilities":np.mean(ps,axis=0),"seed_probabilities":ps,"seed_metrics":seed_metrics}
    return labels,models

def analyze(root,reps=5000,seed=20261020):
    y,models=load_core(root)
    if len(y)!=92: raise ValueError(f"Expected 92 mapped target-test images, found {len(y)}")
    strat=bootstrap_indices(y,reps,seed,True); ordinary=bootstrap_indices(y,reps,seed+1,False)
    output={}
    for name,rec in models.items():
        seed_stats={}
        for metric in ("accuracy","macro_f1_15","macro_f1_supported","balanced_accuracy_supported"):
            vals=[r[metric] for r in rec["seed_metrics"]]
            seed_stats[metric]={"mean":float(np.mean(vals)),"sample_sd":float(np.std(vals,ddof=1)),"per_seed":vals}
        output[name]={"model":CORE[name][0],"n_seeds":5,"mean_per_seed_metrics":seed_stats,
                      "five_seed_probability_ensemble":bootstrap_model(y,rec["probabilities"],strat,ordinary)}
    comparisons=[]; dino_path=root/"report/additional_baselines/dinov2_predictions.npz"
    if dino_path.exists():
        with np.load(dino_path,allow_pickle=False) as z:
            dy=z["target_labels"].astype(np.int64); dp=z["target_probabilities"].astype(float)
        if not np.array_equal(y,dy): raise ValueError("DINOv2 and baseline target labels/order differ")
        dino_pred=dp.argmax(1); ps=[]
        for name,rec in models.items():
            bp=rec["probabilities"]; bpred=bp.argmax(1); diffa=[]; difff=[]
            dm=class_metrics(y,dino_pred); bm=class_metrics(y,bpred)
            for ix in strat:
                a=class_metrics(y[ix],dino_pred[ix]); b=class_metrics(y[ix],bpred[ix])
                diffa.append(a["accuracy"]-b["accuracy"]); difff.append(a["macro_f1_supported"]-b["macro_f1_supported"])
            mc=mcnemar_exact(y==dino_pred,y==bpred); ps.append(mc["mcnemar_exact_p"])
            comparisons.append({"baseline":name,"dinov2_metrics":dm,"baseline_metrics":bm,
                "accuracy_difference":dm["accuracy"]-bm["accuracy"],"accuracy_difference_stratified_ci95":ci(diffa),
                "macro_f1_supported_difference":dm["macro_f1_supported"]-bm["macro_f1_supported"],
                "macro_f1_supported_difference_stratified_ci95":ci(difff),**mc})
        for row,p in zip(comparisons,holm(ps)): row["mcnemar_holm_adjusted_p"]=p
        output["DINOv2-S/14 frozen linear probe"]={"model":"dinov2_vits14","n_seeds":1,
            "training":"deterministic logistic-regression linear probe; C selected on source validation only",
            "five_seed_probability_ensemble":bootstrap_model(y,dp,strat,ordinary)}
    perclass=[]
    for name,rec in output.items():
        ens=rec["five_seed_probability_ensemble"]
        for row in ens["per_class"]: perclass.append({"model":name,**row})
    result={"schema_version":1,"dataset":"pinned PlantDoc official test after overlap and label mapping",
        "n_images":len(y),"n_supported_classes":len(np.unique(y)),
        "class_support":{str(int(k)):int(np.sum(y==k)) for k in np.unique(y)},
        "bootstrap_replicates":reps,"bootstrap_seed":seed,"models":output,"paired_dinov2_comparisons":comparisons,
        "interpretation":["Training-seed SD and image-bootstrap intervals quantify different sources of variability.",
        "Accuracy intervals use ordinary image bootstrap; macro-F1 and classwise intervals preserve observed class support.",
        "Intervals condition on 92 selected images; unknown parent-image/acquisition dependence and dataset-selection uncertainty are not captured.",
        "DINOv2 uses frozen features; its linear-probe regularization is selected on source validation, without target-label tuning.",
        "Comparisons use identical target rows and five-seed probability ensembles, not published scores from different label spaces or splits."]}
    out=root/"report/corrected_summary"; out.mkdir(parents=True,exist_ok=True)
    (out/"target_bootstrap.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    with (out/"target_per_class_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["model","class_index","support","predicted_count","correct","precision","precision_bootstrap_ci95","recall","recall_bootstrap_ci95","f1","f1_bootstrap_ci95"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(perclass)
    md=["# Target-domain statistical audit","",f"- Images: {len(y)}; represented classes: {len(np.unique(y))}",
        f"- Bootstrap replicates: {reps}; seed: {seed}","","| Model | Accuracy | 95% image-bootstrap CI | Supported-class macro-F1 | 95% stratified-bootstrap CI |",
        "|---|---:|---:|---:|---:|"]
    for name,rec in output.items():
        e=rec["five_seed_probability_ensemble"]; m=e["point"]; a,b=e["accuracy_image_bootstrap_ci95"]; c,d=e["macro_f1_supported_stratified_bootstrap_ci95"]
        md.append(f"| {name} | {m['accuracy']:.4f} | {a:.4f}–{b:.4f} | {m['macro_f1_supported']:.4f} | {c:.4f}–{d:.4f} |")
    md += ["","","## Paired DINOv2 comparisons","","| Baseline | Accuracy difference | 95% paired CI | McNemar exact p | Holm-adjusted p |","|---|---:|---:|---:|---:|"]
    for row in comparisons:
        a,b=row["accuracy_difference_stratified_ci95"]
        md.append(f"| {row['baseline']} | {row['accuracy_difference']:+.4f} | {a:+.4f}–{b:+.4f} | {row['mcnemar_exact_p']:.4g} | {row['mcnemar_holm_adjusted_p']:.4g} |")
    md += ["","","Per-class support, precision, recall, F1 and class-stratified intervals are in target_per_class_metrics.csv.",
           "Intervals are conditional on the selected holdout and do not establish acquisition independence.",""]
    (out/"target_bootstrap.md").write_text("\n".join(md),encoding="utf-8")
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--bootstrap",type=int,default=5000);p.add_argument("--seed",type=int,default=20261020);a=p.parse_args()
    r=analyze(a.root,a.bootstrap,a.seed)
    print(json.dumps({"n_images":r["n_images"],"models":list(r["models"]),"paired_comparisons":len(r["paired_dinov2_comparisons"])},indent=2))
if __name__=="__main__": main()
