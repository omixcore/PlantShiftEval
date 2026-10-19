"""Target-domain crop confusion, baseline vs background-augmented training.

Reads the two feature dumps produced by `crop_diagnosis.py --extract` and writes
the row-normalised 5x5 crop confusion matrices plus the mass that lands on a
semantically impossible cell (apple/cherry/blueberry -&gt; maize or grape, which
differ in leaf shape, not just colour).

Output: report/bg_confusion.json
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "src" if __import__("os").path.isdir("src") else ".")
from crop_diagnosis import CROP_OF_arr  # noqa: E402

CROPS = ["apple", "blueberry", "cherry", "maize", "grape"]
# cells a botanist would call impossible from leaf morphology alone
IMPOSSIBLE = {(0, 3), (0, 4), (2, 3), (2, 4), (1, 3)}


def cm_of(npz, name):
    d = np.load(npz)
    seeds = sorted({k[len(f"pd_q_{name}_"):] for k in d.files
                    if k.startswith(f"pd_q_{name}_")})
    q = np.mean([d[f"pd_q_{name}_{s}"] for s in seeds], 0)
    y = CROP_OF_arr(d["y"])
    p = q.argmax(1)
    cm = np.zeros((5, 5))
    for t, k in zip(y, p):
        cm[t, k] += 1
    cm = cm / cm.sum(1, keepdims=True)
    mass = float(sum(cm[i, j] for i, j in IMPOSSIBLE))
    return {"seeds": len(seeds), "cm": cm, "impossible_mass": mass,
            "acc": float((p == y).mean()),
            "pred_prior": np.bincount(p, minlength=5).tolist()}


if __name__ == "__main__":
    import os
    out = {
        "crops": CROPS,
        "baseline": cm_of("report/crop_feats_pt.npz", "mtdnet_pt"),
        "bg_aug": cm_of("report/crop_feats_bg.npz", "bg_mtdnet_pt"),
    }
    if os.path.exists("report/crop_feats_bggray.npz"):
        out["bg_gray"] = cm_of("report/crop_feats_bggray.npz", "bggray_mtdnet_pt")
    for k in ("baseline", "bg_aug", "bg_gray"):
        if k in out:
            out[k]["cm"] = out[k]["cm"].tolist()
    with open("report/bg_confusion.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    for k in ("baseline", "bg_aug", "bg_gray"):
        if k not in out:
            continue
        r = out[k]
        print(f"{k:9s} crop_acc={r['acc']:.3f}  impossible={r['impossible_mass']:.3f}")
        print("          pred prior", np.round(np.array(r["pred_prior"]) / 956, 3))
    print("wrote report/bg_confusion.json")