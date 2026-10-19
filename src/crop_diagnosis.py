"""Why does crop identity break on field images, and what repairs it?

The zero-shot result in the paper reports two numbers: 15-way accuracy 0.246 and
crop accuracy 0.614 (macro 0.465).  Those two numbers are not independent -- the
15-way prediction *is* a crop prediction plus a lesion prediction -- so this
script decomposes the error into its crop and lesion parts, and then tests which
repairs work without any target labels.

Diagnosis
---------
For every image we have the disease posterior p15 (15 PlantVillage classes) and
the species posterior q5 (5 crops, from the auxiliary head).  The crop implied by
p15 is CROP_OF[argmax p15]; the crop head gives an independent estimate.  We
report, per true crop:
    n, crop_recall (from p15), lesion_recall | crop correct, contribution
which separates "cannot tell apple from grape" from "cannot tell rust from
blight once it knows it is maize".

Repairs tested (all label-free unless stated)
---------------------------------------------
1. hierarchical decode      p(d) = q5[crop(d)] * p15(d) / sum_{d' in crop(d)} p15(d')
2. prior correction         q' propto q5 / pi_src   (undo the PlantVillage crop prior)
3. BBSE prior estimation    estimate the target crop prior from predictions alone,
                            then q' propto q5 * pi_hat / pi_src
4. k-shot crop probe        k in {1,5,25} labels per crop on frozen features,
                            5-fold CV -- labels the diagnosis, not a free lunch
5. resolution               same weights at 112 / 144 / 224 px (crop cues are
                            high-frequency; the model was trained at 112)

Output: report/crop_diagnosis.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crossdomain import CROP_OF, load_pd  # noqa: E402
from data import load_cache  # noqa: E402
from models import build_model  # noqa: E402
from train import LeafDS  # noqa: E402

CROPS = ["apple", "blueberry", "cherry", "maize", "grape"]
GROUPS = {c: [k for k, v in CROP_OF.items() if v == c] for c in range(5)}


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def infer(model, meta, x, n_classes, size=144, bs=128):
    """Return disease probs, species probs and the appearance descriptor."""
    y0 = np.zeros(len(x), dtype=np.int64)
    ds = LeafDS(x, y0, y0, False, size)
    ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    cache = []

    def hook(mod, inp, out):
        cache.append(inp[0].detach().float().cpu().numpy())

    h = model.species_head.register_forward_hook(hook)
    model.eval()
    p, q = [], []
    for xb, _, _ in ld:
        o = model(xb)
        p.append(torch.softmax(o["disease"].float(), 1).numpy())
        q.append(torch.softmax(o["species"].float(), 1).numpy())
    h.remove()
    return np.concatenate(p), np.concatenate(q), np.concatenate(cache)


def run_name(path):
    return os.path.basename(os.path.normpath(path))


def extract(runs, out_npz, size=144, threads=16):
    torch.set_num_threads(threads)
    meta, tr, te = load_cache("cache")
    _, x, y = load_pd("cache_pd")
    store = {}
    for d in runs:
        name = run_name(d)
        mj = json.load(open(os.path.join(d, "metrics.json"), encoding="utf-8"))
        model = build_model(mj["model"], meta["n_classes"], meta["n_species"])
        model.load_state_dict(torch.load(os.path.join(d, "model.pt"),
                                         map_location="cpu", weights_only=True))
        p, q, f = infer(model, meta, x, meta["n_classes"], size=size)
        store[f"pd_p_{name}"] = p
        store[f"pd_q_{name}"] = q
        store[f"pd_f_{name}"] = f
        # source-domain crop confusion comes from one seed, 15-way argmax
        ps, qs, fs = infer(model, meta, te.images, meta["n_classes"], size=size)
        store[f"src_p_{name}"] = ps
        store[f"src_q_{name}"] = qs
        store[f"src_f_{name}"] = fs
        store[f"src_y_{name}"] = te.labels
        yc = CROP_OF_arr(y)
        print(f"==> {name:18s} pd crop_acc(aux head)={(q.argmax(1) == yc).mean():.3f} "
              f"crop_acc(15-way)={(CROP_OF_arr(p.argmax(1)) == yc).mean():.3f}")
    np.savez_compressed(out_npz, y=y, **store)
    print("saved", out_npz)


def CROP_OF_arr(idx):
    return np.array([CROP_OF[int(i)] for i in idx])


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def crop_metrics(p15, y):
    yc = CROP_OF_arr(y)
    pc = CROP_OF_arr(p15.argmax(1))
    acc = float((p15.argmax(1) == y).mean())
    crop_acc = float((pc == yc).mean())
    crop_macro = float(np.mean([(pc[yc == c] == c).mean() for c in np.unique(yc)]))
    return acc, crop_acc, crop_macro


def hierarchical(p15, q5, prior_w=None):
    """p(d) = q5[crop(d)] * p15(d)/sum_{crop} p15, optionally re-weighting q5."""
    q = q5.copy()
    if prior_w is not None:
        q = q * prior_w[None, :]
        q = q / np.clip(q.sum(1, keepdims=True), 1e-12, None)
    out = np.zeros_like(p15)
    for c, ks in GROUPS.items():
        denom = np.clip(p15[:, ks].sum(1, keepdims=True), 1e-12, None)
        out[:, ks] = q[:, [c]] * p15[:, ks] / denom
    return out / np.clip(out.sum(1, keepdims=True), 1e-12, None)


def oracle_crop_acc(p15, y):
    """Accuracy if the crop were always known: argmax inside the true crop."""
    ok = 0
    for i, yi in enumerate(y):
        ks = GROUPS[CROP_OF[int(yi)]]
        ok += int(ks[int(p15[i, ks].argmax())] == yi)
    return ok / len(y)


def decompose(p15, y):
    """Per-crop split of the error into crop identity vs lesion identity."""
    yc = CROP_OF_arr(y)
    pc = CROP_OF_arr(p15.argmax(1))
    rows = {}
    for c, name in enumerate(CROPS):
        m = yc == c
        if not m.any():
            continue
        crop_rec = float((pc[m] == c).mean())
        right = m & (pc == c) & (p15.argmax(1) == y)
        lesion_given = float(right.sum() / max(m.sum(), 1))
        lesion_given_crop = float(right.sum() / max((m & (pc == c)).sum(), 1))
        rows[name] = {
            "n": int(m.sum()), "crop_recall": crop_rec,
            "acc15": float((p15.argmax(1)[m] == y[m]).mean()),
            "lesion_recall": lesion_given_crop,
            "contribution_to_correct": float(right.sum() / len(y)),
        }
    return rows


def top2_crop(q5, y):
    yc = CROP_OF_arr(y)
    order = np.argsort(-q5, axis=1)[:, :2]
    return float(np.mean([yc[i] in order[i] for i in range(len(yc))]))


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def bbse_prior(q5, src_cm):
    """Black-box shift estimation: solve C^T pi = mu for the target crop prior."""
    mu = np.bincount(q5.argmax(1), minlength=5) / len(q5)
    C = src_cm  # rows true, cols predicted
    pi, *_ = np.linalg.lstsq(C.T, mu, rcond=None)
    pi = np.clip(pi, 1e-3, None)
    return pi / pi.sum()


def source_cm(src_p, src_y):
    cm = np.zeros((5, 5))
    pc = CROP_OF_arr(src_p.argmax(1))
    yc = CROP_OF_arr(src_y)
    for t, p in zip(yc, pc):
        cm[int(t), int(p)] += 1
    return cm / cm.sum(1, keepdims=True)


def probe_scores(f_tr, y_tr, f_te, y_te, k, seed=0):
    """k labels per crop, logistic regression on frozen features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    yc_tr, yc_te = CROP_OF_arr(y_tr), CROP_OF_arr(y_te)
    keep = []
    for c in range(5):
        idx = np.where(yc_tr == c)[0]
        if len(idx) == 0:
            continue
        keep.extend(rng.choice(idx, size=min(k, len(idx)), replace=False).tolist())
    keep = np.array(keep)
    sc = StandardScaler().fit(f_tr[keep])
    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(sc.transform(f_tr[keep]), yc_tr[keep])
    pred = clf.predict(sc.transform(f_te))
    return pred, float((pred == yc_te).mean()), clf.predict_proba(sc.transform(f_te))


# --------------------------------------------------------------------------- #
def analyse(npz, models, folds=5):
    d = np.load(npz)
    y = d["y"]
    yc = CROP_OF_arr(y)
    out = {"models": {}, "crops": CROPS}
    from sklearn.model_selection import StratifiedKFold

    for name in models:
        seeds = sorted({k.split("_")[-1] for k in d.files if k.startswith(f"pd_p_{name}_")})
        if not seeds:
            continue
        p = np.mean([d[f"pd_p_{name}_{s}"] for s in seeds], 0)
        q = np.mean([d[f"pd_q_{name}_{s}"] for s in seeds], 0)
        f = np.mean([d[f"pd_f_{name}_{s}"] for s in seeds], 0)
        src_p = d[f"src_p_{name}_{seeds[0]}"]
        src_y = d[f"src_y_{name}_{seeds[0]}"]

        acc, crop_acc, crop_macro = crop_metrics(p, y)
        row = {"n_seeds": len(seeds), "acc15": acc, "crop_acc": crop_acc,
               "crop_macro": crop_macro,
               "top2_crop": top2_crop(q, y),
               "oracle_crop_acc": oracle_crop_acc(p, y),
               "crop_conf_when_wrong": float(q.max(1)[CROP_OF_arr(p.argmax(1)) != yc].mean()),
               "decompose": decompose(p, y),
               "hier_raw": float((hierarchical(p, q).argmax(1) == y).mean())}
        # prior correction with the known source prior
        src_prior = np.array([sum((src_y == k).sum() for k in ks) for ks in GROUPS.values()],
                             dtype=np.float64) if src_y is not None else None
        if src_prior is not None:
            src_prior = src_prior / src_prior.sum()
            row["source_crop_prior"] = src_prior.tolist()
            row["hier_uniform_prior"] = float(
                (hierarchical(p, q, 1.0 / np.clip(src_prior, 1e-6, None)).argmax(1) == y).mean())
            cm = source_cm(src_p, src_y)
            row["source_cm"] = cm.tolist()
            pi_hat = bbse_prior(q, cm)
            row["bbse_prior"] = pi_hat.tolist()
            row["true_target_prior"] = (np.bincount(yc, minlength=5) / len(yc)).tolist()
            row["hier_bbse"] = float(
                (hierarchical(p, q, pi_hat / np.clip(src_prior, 1e-6, None)).argmax(1) == y).mean())
            row["crop_acc_bbse"] = float((CROP_OF_arr(hierarchical(p, q, pi_hat / np.clip(
                src_prior, 1e-6, None)).argmax(1)) == yc).mean())
        # k-shot crop probe (5-fold)
        row["probe"] = {}
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        for k in (1, 5, 25):
            a, a15 = [], []
            pred_all = np.zeros(len(y), dtype=int)
            for tr_idx, te_idx in skf.split(f, yc):
                pred, s, proba = probe_scores(f[tr_idx], y[tr_idx], f[te_idx], y[te_idx], k)
                a.append(s)
                pred_all[te_idx] = pred
                qfull = np.zeros((len(te_idx), 5))
                for j, c in enumerate(np.unique(yc[tr_idx])):
                    qfull[:, int(c)] = proba[:, j]
                ph = hierarchical(p[te_idx], qfull)
                a15.append(float((ph.argmax(1) == y[te_idx]).mean()))
            row["probe"][str(k)] = {
                "crop_acc": float(np.mean(a)), "crop_sd": float(np.std(a, ddof=1)),
                "acc15": float(np.mean(a15)),
                "per_crop": {CROPS[c]: float((pred_all[yc == c] == c).mean())
                             for c in range(5)}}
        row["crop_ece"] = ece(q.max(1), (q.argmax(1) == yc).astype(float))
        out["models"][name] = row
        print(f"==> {name:16s} acc15={acc:.3f} crop={crop_acc:.3f} "
              f"oracle_crop={row['oracle_crop_acc']:.3f} hier={row['hier_raw']:.3f} "
              f"uniform={row.get('hier_uniform_prior', float('nan')):.3f} "
              f"bbse={row.get('hier_bbse', float('nan')):.3f} "
              f"probe25={row['probe']['25']['crop_acc']:.3f}/{row['probe']['25']['acc15']:.3f}")
    return out


def res_sweep(runs, sizes=(112, 144, 192, 224), threads=16):
    """Same weights, different input size: are crop cues lost to downsampling?"""
    torch.set_num_threads(threads)
    meta, _, _ = load_cache("cache")
    _, x, y = load_pd("cache_pd")
    yc = CROP_OF_arr(y)
    out = {}
    for d in runs:
        name = run_name(d)
        mj = json.load(open(os.path.join(d, "metrics.json"), encoding="utf-8"))
        model = build_model(mj["model"], meta["n_classes"], meta["n_species"])
        model.load_state_dict(torch.load(os.path.join(d, "model.pt"),
                                         map_location="cpu", weights_only=True))
        out[name] = {}
        for s in sizes:
            p, q, _ = infer(model, meta, x, meta["n_classes"], size=s)
            acc = float((p.argmax(1) == y).mean())
            crop = float((CROP_OF_arr(p.argmax(1)) == yc).mean())
            crop_macro = float(np.mean([(CROP_OF_arr(p.argmax(1))[yc == c] == c).mean()
                                        for c in range(5)]))
            out[name][str(s)] = {"acc15": acc, "crop_acc": crop,
                                 "crop_macro": crop_macro,
                                 "oracle_crop_acc": oracle_crop_acc(p, y)}
            print(f"{name:16s} {s:3d}px crop={crop:.3f} macro={crop_macro:.3f} "
                  f"acc15={acc:.3f} oracle={out[name][str(s)]['oracle_crop_acc']:.3f}")
    with open("report/crop_resolution.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote report/crop_resolution.json")


def adabn_sweep(runs, size=144, threads=16):
    """Re-estimate BN statistics on the unlabeled target images (AdaBN),
    then measure what happens to *crop* accuracy specifically."""
    torch.set_num_threads(threads)
    meta, _, _ = load_cache("cache")
    _, x, y = load_pd("cache_pd")
    yc = CROP_OF_arr(y)
    out = {}
    for d in runs:
        name = run_name(d)
        mj = json.load(open(os.path.join(d, "metrics.json"), encoding="utf-8"))
        model = build_model(mj["model"], meta["n_classes"], meta["n_species"])
        model.load_state_dict(torch.load(os.path.join(d, "model.pt"),
                                         map_location="cpu", weights_only=True))
        p0, q0, _ = infer(model, meta, x, meta["n_classes"], size=size)
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.reset_running_stats()
                m.momentum = None
        y0 = np.zeros(len(x), dtype=np.int64)
        ds = LeafDS(x, y0, y0, False, size)
        ld = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        model.train()
        with torch.no_grad():
            for xb, _, _ in ld:
                model(xb)
        model.eval()
        p1, q1, _ = infer(model, meta, x, meta["n_classes"], size=size)
        row = {}
        for tag, p, q in (("source_stats", p0, q0), ("adabn", p1, q1)):
            row[tag] = {
                "crop_acc": float((CROP_OF_arr(p.argmax(1)) == yc).mean()),
                "crop_macro": float(np.mean([(CROP_OF_arr(p.argmax(1))[yc == c] == c).mean()
                                             for c in range(5)])),
                "acc15": float((p.argmax(1) == y).mean()),
                "oracle_crop_acc": oracle_crop_acc(p, y),
                "head_crop_acc": float((q.argmax(1) == yc).mean()),
            }
        out[name] = row
        print(f"{name:16s} crop {row['source_stats']['crop_acc']:.3f} -> "
              f"{row['adabn']['crop_acc']:.3f} | acc15 {row['source_stats']['acc15']:.3f} -> "
              f"{row['adabn']['acc15']:.3f} | aux head {row['source_stats']['head_crop_acc']:.3f} -> "
              f"{row['adabn']['head_crop_acc']:.3f}")
    with open("report/crop_adabn.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote report/crop_adabn.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[])
    ap.add_argument("--npz", default="report/crop_diagnosis.npz")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--res-sweep", action="store_true")
    ap.add_argument("--adabn", action="store_true")
    ap.add_argument("--size", type=int, default=144)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="report/crop_diagnosis.json")
    a = ap.parse_args()

    if a.adabn:
        adabn_sweep(a.runs, size=a.size, threads=a.threads)
        return
    if a.res_sweep:
        res_sweep(a.runs, threads=a.threads)
        return
    if a.extract:
        extract(a.runs, a.npz, size=a.size, threads=a.threads)
        return
    models = a.models or ["mtdnet_pt", "mtdnet"]
    res = analyse(a.npz, models)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()