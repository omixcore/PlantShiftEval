"""Frozen DINOv2-S/14 linear-probe baseline on the pinned PlantVillage/PlantDoc split.

The backbone is loaded from a pinned official DINOv2 code revision. The only
supervised fitting and hyperparameter selection use source training/validation.
The target official test is opened once for final evaluation and paired comparison.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from run_provenance import file_digest
from target_bootstrap import class_metrics, probability_metrics

ROOT=Path(__file__).resolve().parents[1]
DINO_REPO="7764ea0f912e53c92e82eb78a2a1631e92725fc8"
IMAGE_SIZE=224
MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)

def load_model():
    model=torch.hub.load(f"facebookresearch/dinov2:{DINO_REPO}","dinov2_vits14",pretrained=True,trust_repo=True)
    model.eval()
    return model

def weights_sha256(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        t=tensor.detach().cpu().contiguous()
        h.update(name.encode("utf-8"));h.update(str(t.dtype).encode("ascii"));h.update(str(tuple(t.shape)).encode("ascii"))
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def extract(model,images,device,batch_size,threads):
    torch.set_num_threads(threads)
    out=[]
    mean=MEAN.to(device);std=STD.to(device)
    with torch.inference_mode():
        for start in range(0,len(images),batch_size):
            arr=np.asarray(images[start:start+batch_size])
            x=torch.from_numpy(arr).permute(0,3,1,2).contiguous().to(device=device,dtype=torch.float32).div_(255.)
            x=F.interpolate(x,size=(IMAGE_SIZE,IMAGE_SIZE),mode="bicubic",align_corners=False)
            x=(x-mean)/std
            feat=model.forward_features(x)["x_norm_clstoken"]
            out.append(feat.detach().cpu().numpy().astype(np.float32))
            if (start//batch_size)%20==0:
                print(f"features {start+len(arr)}/{len(images)}",flush=True)
    return normalize(np.concatenate(out,axis=0),norm="l2").astype(np.float32)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=ROOT)
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--batch-size",type=int,default=16)
    ap.add_argument("--threads",type=int,default=2)
    ap.add_argument("--c-values",default="0.001,0.01,0.1,1,10,100")
    args=ap.parse_args()
    root=args.root
    src=root/"cache_corrected"; tgt=root/"cache_pd_corrected"
    required=[src/"train_x.npy",src/"train_y.npy",src/"test_x.npy",src/"test_y.npy",
              src/"split_indices.json",tgt/"x.npy",tgt/"y.npy",tgt/"split_indices.json"]
    missing=[str(p) for p in required if not p.exists()]
    if missing: raise FileNotFoundError("Pinned caches/splits missing: "+", ".join(missing))
    train_x=np.load(src/"train_x.npy",mmap_mode="r"); train_y=np.load(src/"train_y.npy")
    source_test_x=np.load(src/"test_x.npy",mmap_mode="r"); source_test_y=np.load(src/"test_y.npy")
    split=json.loads((src/"split_indices.json").read_text(encoding="utf-8"))
    train_idx=np.asarray(split["train_indices"],dtype=np.int64)
    val_idx=np.asarray(split["validation_indices"],dtype=np.int64)
    if set(train_idx)&set(val_idx) or set(train_idx)|set(val_idx)!=set(range(len(train_y))):
        raise ValueError("Source split is not a disjoint full partition")
    if set(np.unique(train_y[train_idx]))!=set(range(15)):
        raise ValueError("Linear-probe training partition must contain all 15 source classes")
    target_x=np.load(tgt/"x.npy",mmap_mode="r"); target_y=np.load(tgt/"y.npy")
    tsplit=json.loads((tgt/"split_indices.json").read_text(encoding="utf-8"))
    target_idx=np.asarray(tsplit["test_indices"],dtype=np.int64)
    target_test_y=target_y[target_idx]
    if len(target_idx)!=92: raise ValueError(f"Expected pinned 92-image target holdout, found {len(target_idx)}")
    device=torch.device(args.device)
    model=load_model().to(device)
    weight_hash=weights_sha256(model)
    # Source train/validation share the same cached pixels and frozen split used by the study.
    ftrain=extract(model,train_x,device,args.batch_size,args.threads)
    ftest=extract(model,source_test_x,device,args.batch_size,args.threads)
    ftarget=extract(model,target_x[target_idx],device,args.batch_size,args.threads)
    ftrain=np.asarray(ftrain); ftest=np.asarray(ftest); ftarget=np.asarray(ftarget)
    candidates=[]
    best=None
    for c in [float(v) for v in args.c_values.split(",")]:
        clf=LogisticRegression(C=c,class_weight="balanced",solver="lbfgs",max_iter=3000,tol=1e-5,random_state=0)
        clf.fit(ftrain[train_idx],train_y[train_idx])
        val_probs=clf.predict_proba(ftrain[val_idx])
        # sklearn class order must remain the 15 source labels in numeric order.
        if not np.array_equal(clf.classes_,np.arange(15)): raise ValueError("Unexpected linear-probe class ordering")
        val=probability_metrics(train_y[val_idx],val_probs)
        rec={"C":c,"source_validation_macro_f1_15":val["macro_f1_15"],
             "source_validation_macro_f1_supported":val["macro_f1_supported"],
             "source_validation_accuracy":val["accuracy"]}
        candidates.append(rec)
        if best is None or rec["source_validation_macro_f1_15"]>best["source_validation_macro_f1_15"]:
            best=rec
    # Refit the chosen linear classifier on all permitted source training rows, including validation.
    clf=LogisticRegression(C=best["C"],class_weight="balanced",solver="lbfgs",max_iter=3000,tol=1e-5,random_state=0)
    clf.fit(ftrain,train_y)
    source_probs=clf.predict_proba(ftest)
    target_probs=clf.predict_proba(ftarget)
    source_metrics=probability_metrics(source_test_y,source_probs)
    target_metrics=probability_metrics(target_test_y,target_probs)
    cache_hashes={str(p.relative_to(root)):file_digest(p) for p in required}
    try:
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    except Exception: commit=None
    try:
        hub_path=Path(torch.hub.get_dir())/f"facebookresearch_dinov2_{DINO_REPO}"
        code_commit=subprocess.check_output(["git","-C",str(hub_path),"rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: code_commit=DINO_REPO
    out=root/"report"/"additional_baselines";out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"dinov2_predictions.npz",target_probabilities=target_probs.astype(np.float32),
                        target_labels=target_test_y.astype(np.int64),source_probabilities=source_probs.astype(np.float32),
                        source_labels=source_test_y.astype(np.int64))
    result={
      "status":"completed","model":"DINOv2-S/14 (dinov2_vits14) frozen backbone + source-trained logistic-regression linear probe",
      "official_code_repository":"https://github.com/facebookresearch/dinov2",
      "official_code_commit":code_commit,"requested_code_ref":DINO_REPO,"weights_state_sha256":weight_hash,
      "feature":"forward_features()['x_norm_clstoken'], L2-normalized",
      "input":{"cached_source_size":list(train_x.shape[1:3]),"cached_target_size":list(target_x.shape[1:3]),
               "resize":[IMAGE_SIZE,IMAGE_SIZE],"interpolation":"bicubic","normalization_mean":[0.485,0.456,0.406],
               "normalization_std":[0.229,0.224,0.225],"batch_size":args.batch_size,"device":str(device)},
      "classifier":{"method":"sklearn LogisticRegression","solver":"lbfgs","class_weight":"balanced","max_iter":3000,
                    "tolerance":1e-5,"candidate_C_values":[float(v) for v in args.c_values.split(",")],
                    "selection_metric":"source-validation macro-F1 over 15 source classes",
                    "selected_C":best["C"],"candidate_validation_results":candidates,
                    "refit":"all 3255 permitted source-cache rows (training + source validation); no target labels used"},
      "source_test":{"n":len(source_test_y),**source_metrics},
      "target_test":{"n":len(target_test_y),**target_metrics,"supported_class_ids":[int(x) for x in np.unique(target_test_y)],
                     "target_image_indices":target_idx.tolist()},
      "target_predictions_file":"report/additional_baselines/dinov2_predictions.npz",
      "input_sha256":cache_hashes,"study_git_commit":commit,
      "environment":{"python":__import__("platform").python_version(),"torch":torch.__version__,
                     "numpy":np.__version__,"scikit_learn":__import__("sklearn").__version__},
      "protocol_caveats":["Target official test was used only for final evaluation, not model or C selection.",
          "This evaluates the 10-class mapped subset with a 15-class source classifier; predictions outside target-supported classes count as target errors.",
          "Source cache images were stored at 160x160 and target cache images at 144x144 before resizing to 224x224; upsampling does not restore lost detail.",
          "The DINOv2 feature extractor is deterministic; uncertainty over target images is reported separately by target_bootstrap.py."]
    }
    (out/"dinov2_s14.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"completed","selected_C":best["C"],"source_macro_f1":source_metrics["macro_f1_15"],
      "target_accuracy":target_metrics["accuracy"],"target_macro_f1_supported":target_metrics["macro_f1_supported"],
      "weights_sha256":weight_hash,"output":str(out/"dinov2_s14.json")},indent=2),flush=True)

if __name__=="__main__": main()
