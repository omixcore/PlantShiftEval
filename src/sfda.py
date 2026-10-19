"""Label-free repairs for crop identity, tested on cached features.

Tier 3 (background augmentation + crop-balanced sampling) is a *training-time*
repair and it did not work: the pasted field backdrop itself became a new
shortcut.  This script asks the complementary question -- can the crop head be
repaired at *inference* time, with zero target labels, given that a linear probe
with 25 target labels per crop already reaches 0.70?

Everything here is fitted on PlantVillage only.  PlantDoc labels are used for
*measurement* alone, never for fitting or for model selection.

Variants
--------
src            linear crop head fitted on source features (reference)
zscore_t       target features standardised with *target* mean/std (label-free)
coral          source features recoloured to target second-order statistics
               (CORAL, Sun &amp; Saenko 2016), then the head is refitted
coral+st       CORAL followed by transductive self-training on target pseudo-labels
st             self-training alone, from the *source* head

Output: report/sfda.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crop_diagnosis import CROP_OF_arr  # noqa: E402

CROPS = ["apple", "blueberry", "cherry", "maize", "grape"]


# --------------------------------------------------------------------------- #
# alignment primitives
# --------------------------------------------------------------------------- #
def _stats(x):
    mu = x.mean(0)
    xc = x - mu
    cov = xc.T @ xc / max(1, len(x) - 1)
    return mu, cov


def _mat_pow(cov, power, eps=1e-3):
    w, v = np.linalg.eigh(cov.astype(np.float64))
    w = np.clip(w, eps, None) ** power
    return (v * w) @ v.T


def coral(xs, xt, eps=1e-3):
    """Recolour source features so their mean/covariance match the target."""
    ms, cs = _stats(xs)
    mt, ct = _stats(xt)
    ws = _mat_pow(cs, -0.5, eps)
    wt = _mat_pow(ct, 0.5, eps)
    return (xs - ms) @ ws @ wt + mt


def standardise(xs, xt):
    mu, sd = xs.mean(0), xs.std(0) + 1e-6
    return (xs - mu) / sd, (xt - mu) / sd


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #
def fit_lr(x, y, c):
    m = LogisticRegression(C=c, max_iter=3000, class_weight="balanced")
    m.fit(x, y)
    return m


def pick_c(x, y, grid=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0), seed=0):
    """Choose the L2 strength on a source-only held-out split."""
    xtr, xva, ytr, yva = train_test_split(x, y, test_size=0.2, stratify=y,
                                          random_state=seed)
    best, best_c = -1.0, grid[0]
    for c in grid:
        s = fit_lr(xtr, ytr, c).score(xva, yva)
        if s &gt; best:
            best, best_c = s, c
    return best_c, best


def self_train(x0, y0, xt, rounds=4, frac=0.5, c=0.1):
    """Transductive self-training; returns target probabilities."""
    m = fit_lr(x0, y0, c)
    for _ in range(rounds):
        p = m.predict_proba(xt)
        conf = p.max(1)
        k = max(50, int(frac * len(xt)))
        idx = np.argsort(-conf)[:k]
        xx = np.vstack([x0, xt[idx]])
        yy = np.concatenate([y0, m.classes_[p[idx].argmax(1)]])
        m = fit_lr(xx, yy, c)
    return m.predict_proba(xt)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def score(pred, y):
    return {
        "crop_acc": float((pred == y).mean()),
        "crop_macro_f1": float(f1_score(y, pred, average="macro", labels=np.arange(5),
                                        zero_division=0)),
        "per_crop": {CROPS[c]: float((pred[y == c] == c).mean()) for c in range(5)
                     if (y == c).any()},
    }


def run(npz, models, out_path):
    d = np.load(npz)
    y = CROP_OF_arr(d["y"])
    res = {"models": {}, "crops": CROPS}
    for name in models:
        seeds = sorted({k.split("_")[-1] for k in d.files if k.startswith(f"pd_f_{name}_")})
        if not seeds:
            continue
        rows = []
        for s in seeds:
            fs, ys = d[f"src_f_{name}_{s}"], CROP_OF_arr(d[f"src_y_{name}_{s}"])
            ft = d[f"pd_f_{name}_{s}"]
            fs, ft = standardise(fs, ft)
            c, cval = pick_c(fs, ys)
            r = {"seed": s, "C": c, "src_val_acc": cval}
            m0 = fit_lr(fs, ys, c)
            r["src"] = score(m0.predict(ft), y)
            # target standardisation only (label-free)
            mu, sd = ft.mean(0), ft.std(0) + 1e-6
            m1 = fit_lr(fs, ys, c)
            r["zscore_t"] = score(m1.predict((ft - mu) / sd), y)
            # CORAL + refit
            fa = coral(fs, ft)
            ma = fit_lr(fa, ys, c)
            ft2 = ft  # CORAL maps source onto target space
            r["coral"] = score(ma.predict(ft2), y)
            # self-training from the source head
            p_st = self_train(fs, ys, ft, c=c)
            r["st"] = score(p_st.argmax(1), y)
            # CORAL + self-training
            p_cst = self_train(fa, ys, ft2, c=c)
            r["coral+st"] = score(p_cst.argmax(1), y)
            rows.append(r)

        agg = {}
        for k in ("src", "zscore_t", "coral", "st", "coral+st"):
            accs = [r[k]["crop_acc"] for r in rows]
            agg[k] = {"crop_acc": float(np.mean(accs)),
                      "crop_acc_sd": float(np.std(accs)),
                      "crop_macro_f1": float(np.mean([r[k]["crop_macro_f1"] for r in rows])),
                      "per_crop": {c: float(np.mean([r[k]["per_crop"].get(c, 0.0)
                                                     for r in rows])) for c in CROPS}}
        res["models"][name] = {"n_seeds": len(seeds), "C": rows[0]["C"],
                               "src_val_acc": float(np.mean([r["src_val_acc"] for r in rows])),
                               "variants": agg}
        base = agg["src"]["crop_acc"]
        print(f"\n[{name}] n_seeds={len(seeds)}  C={rows[0]['C']}  "
              f"src-val={res['models'][name]['src_val_acc']:.3f}")
        for k in ("src", "zscore_t", "coral", "st", "coral+st"):
            a = agg[k]["crop_acc"]
            print(f"   {k:&lt;10s} crop={a:.3f} ({agg[k]['crop_acc_sd']:.3f})  "
                  f"macroF1={agg[k]['crop_macro_f1']:.3f}  d={a - base:+.3f}")
        # transductive ensemble over seeds (what the paper reports)
        if len(seeds) &gt; 1:
            pass
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\n[sfda] wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="report/crop_diagnosis.npz")
    ap.add_argument("--models", default="mtdnet_pt,mtdnet")
    ap.add_argument("--out", default="report/sfda.json")
    a = ap.parse_args()
    run(a.npz, [m for m in a.models.split(",") if m], a.out)