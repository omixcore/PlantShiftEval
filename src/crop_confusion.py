"""Compute the crop-level confusion matrix for Figure 5B.

Loads the 3-seed ensemble of MTD-P, predicts the mapped PlantDoc subset once,
and stores a 5x5 row-normalised crop confusion matrix as an .npy file that
figures_pub.fig5_cross picks up.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crossdomain import CROP_OF, load_pd  # noqa: E402
from data import load_cache  # noqa: E402
from models import build_model  # noqa: E402
from train import LeafDS  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

RUNS = "runs"
MODEL = "mtdnet_pt"
CROPS = ["apple", "blueberry", "cherry", "maize", "grape"]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


@torch.no_grad()
def infer(model, x, size=112):
    ds = LeafDS(x, np.zeros(len(x), dtype=np.int64), np.zeros(len(x), dtype=np.int64),
                False, size)
    ld = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    model.eval()
    out = []
    for xb, _, _ in ld:
        out.append(model(xb)["disease"].float())
    return torch.cat(out).numpy()


def main():
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) &gt; 1 else 8)
    meta, _, _ = load_cache("cache")
    _, x, y = load_pd("cache_pd")
    probs = []
    for s in (0, 1, 2):
        p = os.path.join(RUNS, f"{MODEL}_s{s}", "metrics.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p, encoding="utf-8"))
        m = build_model(j["model"], meta["n_classes"], meta["n_species"])
        m.load_state_dict(torch.load(os.path.join(RUNS, f"{MODEL}_s{s}", "model.pt"),
                                     map_location="cpu", weights_only=True))
        probs.append(softmax(infer(m, x)))
    probs = np.mean(probs, 0)
    pred = probs.argmax(1)
    k = len(CROPS)
    cm = np.zeros((k, k))
    for t, p in zip(y, pred):
        cm[CROP_OF[int(t)], CROP_OF[int(p)]] += 1
    cm = cm / cm.sum(1, keepdims=True)
    out = os.path.join("report", "figures", "pub")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "_crop_cm.npy"), cm)
    print("crop confusion (rows=true):")
    for name, row in zip(CROPS, cm):
        print(f"{name:10s} " + " ".join(f"{v:.2f}" for v in row))


if __name__ == "__main__":
    main()