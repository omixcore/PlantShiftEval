"""Can the PlantVillage -&gt; PlantDoc collapse be repaired?

`crossdomain.py` measures the damage; this script asks what it takes to undo it.
Four interventions are compared, ordered by the amount of target supervision
they consume:

  source        no target data at all (the zero-shot number, recomputed here so
                every method shares one code path)
  bn            unlabeled target images only: re-estimate every BatchNorm
                running statistic on the target training split (AdaBN-style
                [Li et al. 2016; Carlucci et al. 2019]); no gradient steps,
                no labels
  probe-k       k labelled images per class, backbone frozen, logistic
                regression on the penultimate features
  finetune      the whole labelled target training split, 3 epochs

Everything is evaluated on the target test split, which is never used for
fitting.  Reported: 15-way accuracy, macro-F1 over the ten classes that occur,
crop accuracy implied by the predicted class, and ECE.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crossdomain import CATEGORY_MAP, CROP_OF, PD_FILES, norm_cat  # noqa: E402
from data import Split, load_cache  # noqa: E402
from models import build_model  # noqa: E402
from train import LeafDS, ece_score  # noqa: E402


# --------------------------------------------------------------------------- #
# target cache that keeps PlantDoc's own train/test split
# --------------------------------------------------------------------------- #
def build_split_cache(out: str = "cache_pd_split", size: int = 144) -&gt; dict:
    os.makedirs(out, exist_ok=True)
    buckets = {"train": ([], []), "test": ([], [])}
    for p in PD_FILES:
        if not os.path.exists(p):
            continue
        split = "test" if "test" in os.path.basename(p) else "train"
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=64, columns=["image", "objects"]):
            for rec in b.to_pylist():
                obj = rec.get("objects") or {}
                cats = [c for c in (obj.get("category") or []) if c]
                if not cats:
                    continue
                top, cnt = collections.Counter(cats).most_common(1)[0]
                if cnt &lt; 0.5 * len(cats):
                    continue
                key = norm_cat(top)
                if key not in CATEGORY_MAP:
                    continue
                cell = rec["image"]
                raw = cell.get("bytes") if isinstance(cell, dict) else cell
                if not raw:
                    continue
                im = Image.open(io.BytesIO(raw)).convert("RGB").resize((size, size),
                                                                      Image.BILINEAR)
                buckets[split][0].append(np.asarray(im, dtype=np.uint8))
                buckets[split][1].append(CATEGORY_MAP[key])
    info = {}
    for split, (xs, ys) in buckets.items():
        x = np.stack(xs).astype(np.uint8)
        y = np.asarray(ys, dtype=np.int64)
        np.save(os.path.join(out, f"{split}_x.npy"), x)
        np.save(os.path.join(out, f"{split}_y.npy"), y)
        info[split] = {"n": int(len(y)),
                       "per_class": {int(k): int(v) for k, v in
                                     zip(*np.unique(y, return_counts=True))}}
    info["img_size"] = size
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


def load_split_cache(d: str):
    with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
        info = json.load(f)
    out = {}
    for split in ("train", "test"):
        out[split] = (np.load(os.path.join(d, f"{split}_x.npy")),
                      np.load(os.path.join(d, f"{split}_y.npy")))
    return info, out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def strip_head(model: nn.Module) -&gt; int:
    """Replace the final classifier with Identity; return feature width."""
    if hasattr(model, "disease_head"):           # MTD-Net family
        dim = model.disease_head[-1].in_features
        model.disease_head[-1] = nn.Identity()
        return dim
    if hasattr(model, "head"):                   # SmallCNN
        dim = model.head[-1].in_features
        model.head[-1] = nn.Identity()
        return dim
    inner = model.net                           # _OutWrap baselines
    if hasattr(inner, "fc"):
        dim = inner.fc.in_features
        inner.fc = nn.Identity()
        return dim
    for i in range(len(inner.classifier) - 1, -1, -1):
        if isinstance(inner.classifier[i], nn.Linear):
            dim = inner.classifier[i].in_features
            inner.classifier[i] = nn.Identity()
            return dim
    raise TypeError("cannot locate classifier head")


@torch.no_grad()
def features(model, x, size=112) -&gt; np.ndarray:
    ds = LeafDS(x, np.zeros(len(x), dtype=np.int64), np.zeros(len(x), dtype=np.int64),
                False, size)
    ld = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    model.eval()
    out = []
    for xb, _, _ in ld:
        z = model(xb)["disease"]
        out.append((z if z.dim() == 2 else z.flatten(1)).float())
    return torch.cat(out).numpy()


@torch.no_grad()
def logits_of(model, x, size=112) -&gt; np.ndarray:
    ds = LeafDS(x, np.zeros(len(x), dtype=np.int64), np.zeros(len(x), dtype=np.int64),
                False, size)
    ld = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    model.eval()
    out = []
    for xb, _, _ in ld:
        out.append(model(xb)["disease"].float())
    return torch.cat(out).numpy()


def softmax(z: np.ndarray) -&gt; np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def bn_adapt(model, x, size=112, bs=16) -&gt; None:
    """AdaBN: recompute BN statistics on unlabeled target images."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None       # cumulative moving average
    was_training = model.training
    model.train()
    ds = LeafDS(x, np.zeros(len(x), dtype=np.int64), np.zeros(len(x), dtype=np.int64),
                False, size)
    ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    with torch.no_grad():
        for xb, _, _ in ld:
            model(xb)
    for m in bns:
        m.momentum = 0.1
    if not was_training:
        model.eval()


def metrics(logits_or_probs: np.ndarray, y: np.ndarray, present) -&gt; dict:
    probs = logits_or_probs if logits_or_probs.min() &gt;= 0 and \
        np.allclose(logits_or_probs.sum(1), 1, atol=1e-3) else softmax(logits_or_probs)
    pred = probs.argmax(1)
    p10 = probs[:, present]
    pred10 = np.asarray(present)[p10.argmax(1)]
    crop_ok = np.array([CROP_OF[int(t)] == CROP_OF[int(t_)] for t, t_ in zip(pred, y)])
    y_crop = np.array([CROP_OF[int(v)] for v in y])
    return {
        "acc15": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred10, average="macro", zero_division=0,
                                   labels=list(present))),
        "crop_acc": float(crop_ok.mean()),
        "crop_macro": float(np.mean([crop_ok[y_crop == c].mean()
                                     for c in np.unique(y_crop)])),
        "conf": float(probs.max(1).mean()),
        "ece": ece_score(probs, y),
    }


def sample_k(y: np.ndarray, k: int, seed: int) -&gt; np.ndarray:
    rs = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y):
        pool = np.where(y == c)[0]
        idx.extend(rs.choice(pool, size=min(k, len(pool)), replace=False).tolist())
    return np.array(sorted(idx))


def finetune(model, x, y, size=112, epochs=3, lr=1e-4, bs=32, seed=0) -&gt; nn.Module:
    torch.manual_seed(seed)
    ds = LeafDS(x, y, np.zeros(len(y), dtype=np.int64), True, size)
    ld = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=False)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    n = max(1, len(ld) * epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=n,
                                                pct_start=0.3)
    model.train()
    for ep in range(epochs):
        for xb, yb, _ in ld:
            out = model(xb)
            loss = crit(out["disease"], yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--cache-pd", default="cache_pd_split")
    ap.add_argument("--out", default="report/adaptation.json")
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 25])
    ap.add_argument("--probe-seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    meta, _, _ = load_cache(args.cache)
    if args.rebuild or not os.path.exists(os.path.join(args.cache_pd, "meta.json")):
        info = build_split_cache(args.cache_pd)
        print("built", json.dumps(info, ensure_ascii=False)[:300], flush=True)
    info, splits = load_split_cache(args.cache_pd)
    # PlantDoc's own test shard holds only 96 mapped images, far too few to
    # separate the interventions, so both shards are pooled and evaluated with
    # stratified k-fold cross-validation.
    X = np.concatenate([splits["train"][0], splits["test"][0]])
    Y = np.concatenate([splits["train"][1], splits["test"][1]])
    present = sorted({int(v) for v in Y})
    print(f"target pooled: n={len(Y)} classes={present}", flush=True)

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)
    folds = list(skf.split(X, Y))

    results = {}
    for r in args.runs:
        with open(os.path.join(r, "metrics.json"), encoding="utf-8") as f:
            mj = json.load(f)
        name = os.path.basename(r)
        acc: dict[str, list[dict]] = {}

        def add(method, m):
            acc.setdefault(method, []).append(m)

        xtr_all, ytr_all = X, Y
        for fi, (tr_idx, te_idx) in enumerate(folds):
            xtr, ytr = X[tr_idx], Y[tr_idx]
            xte, yte = X[te_idx], Y[te_idx]

            model = build_model(mj["model"], meta["n_classes"], meta["n_species"])
            model.load_state_dict(torch.load(os.path.join(r, "model.pt"), map_location="cpu"))
            model.eval()
            add("source", metrics(logits_of(model, xte, args.size), yte, present))

            bn_adapt(model, xtr, args.size)
            add("bn", metrics(logits_of(model, xte, args.size), yte, present))

            fmodel = build_model(mj["model"], meta["n_classes"], meta["n_species"])
            fmodel.load_state_dict(torch.load(os.path.join(r, "model.pt"), map_location="cpu"))
            fmodel.eval()
            dim = strip_head(fmodel)
            ftr = features(fmodel, xtr, args.size)
            fte = features(fmodel, xte, args.size)
            for k in args.ks:
                for s in range(args.probe_seeds):
                    idx = sample_k(ytr, k, s)
                    clf = LogisticRegression(max_iter=3000, C=1.0)
                    clf.fit(ftr[idx], ytr[idx])
                    full = np.zeros((len(xte), meta["n_classes"]))
                    full[:, clf.classes_] = clf.predict_proba(fte)
                    add(f"probe_{k}", metrics(full, yte, present))

            ft = build_model(mj["model"], meta["n_classes"], meta["n_species"])
            ft.load_state_dict(torch.load(os.path.join(r, "model.pt"), map_location="cpu"))
            lr = 3e-4 if mj["model"] in ("mtdnet", "mtd_concat", "mtd_single", "smallcnn") else 1e-4
            ft = finetune(ft, xtr, ytr, args.size, epochs=3, lr=lr, seed=fi)
            add("finetune", metrics(logits_of(ft, xte, args.size), yte, present))
            print(f"  [{name}] fold {fi + 1}/{len(folds)} done", flush=True)

        methods = {}
        for method, rows in acc.items():
            entry = {}
            for key in ("acc15", "macro_f1", "crop_acc", "crop_macro", "conf", "ece"):
                vals = [row[key] for row in rows]
                entry[key] = float(np.mean(vals))
                entry[key + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) &gt; 1 else 0.0
            entry["n"] = len(rows)
            entry["feature_dim"] = 0
            if method.startswith("probe"):
                entry["feature_dim"] = int(dim)
                entry["n_shots_per_class"] = int(method.split("_")[1])
            methods[method] = entry
        results[name] = {"model": mj["model"], "params": mj["params"],
                         "methods": methods}
        print(f"==&gt; {name:22s} " + " | ".join(
            f"{m}:{v['macro_f1']:.3f}" for m, v in methods.items()), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"target_info": info, "results": results,
                   "protocol": f"{args.folds}-fold stratified CV over pooled "
                               f"{len(Y)} PlantDoc images"}, f,
                  ensure_ascii=False, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()