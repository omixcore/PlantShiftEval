"""Corruption-robustness benchmark.

PlantVillage is a laboratory-style dataset: uniform background, controlled
illumination.  A model that only works there is not useful in a field.  We
therefore stress-test every trained model with synthetic degradations that
mimic real acquisition problems (sensor noise, defocus, over-exposure, leaf
occlusion, haze) and report the accuracy drop.

Each corruption has 2 severity levels; the mean accuracy over all conditions is
reported as `robust_acc`, and `drop` = clean_acc - robust_acc.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_cache  # noqa: E402
from models import build_model  # noqa: E402
from train import LeafDS  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denorm(x):
    return (x * STD + MEAN).clamp(0, 1)


def norm(x):
    return (x - MEAN) / STD


def apply_corrupt(x01: torch.Tensor, kind: str, sev: int, rng: np.random.Generator):
    """x01: (B,3,H,W) in [0,1]."""
    if kind == "noise":
        s = 0.04 * sev
        return (x01 + torch.from_numpy(rng.normal(0, s, x01.shape).astype(np.float32))).clamp(0, 1)
    if kind == "blur":
        k = 2 * sev + 1
        w = torch.ones(3, 1, k, k) / (k * k)
        return F.conv2d(F.pad(x01, (k // 2,) * 4, mode="reflect"), w, groups=3)
    if kind == "brightness":
        return (x01 * (1.0 + 0.30 * sev) + 0.10 * sev).clamp(0, 1)
    if kind == "contrast":
        m = x01.mean(dim=(1, 2, 3), keepdim=True)
        return ((x01 - m) * (1.0 - 0.35 * sev) + m).clamp(0, 1)
    if kind == "occlusion":
        out = x01.clone()
        b, _, h, w = out.shape
        side = int(h * (0.18 + 0.10 * sev))
        for i in range(b):
            y = rng.integers(0, h - side)
            x = rng.integers(0, w - side)
            out[i, :, y:y + side, x:x + side] = 0.0
        return out
    raise ValueError(kind)


@torch.no_grad()
def run_condition(model, loader, kind, sev, seed=0):
    model.eval()
    rng = np.random.default_rng(seed)
    correct = total = 0
    for xb, yb, _ in loader:
        x = denorm(xb)
        if kind != "clean":
            x = apply_corrupt(x, kind, sev, rng)
        out = model(norm(x))
        correct += int((out["disease"].argmax(1) == yb).sum())
        total += len(yb)
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--cache", default="cache")
    p.add_argument("--out", default="runs/robustness.json")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--size", type=int, default=112)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    meta, _, te = load_cache(args.cache)
    ds = LeafDS(te.images, te.labels, te.species, False, args.size)
    ld = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    grid = [("noise", 1), ("noise", 2), ("blur", 1), ("blur", 2),
            ("brightness", 2), ("contrast", 2), ("occlusion", 1), ("occlusion", 2)]

    results = {}
    for r in args.runs:
        name = os.path.basename(r)
        with open(os.path.join(r, "metrics.json"), encoding="utf-8") as f:
            mj = json.load(f)
        model = build_model(mj["model"], meta["n_classes"], meta["n_species"])
        model.load_state_dict(torch.load(os.path.join(r, "model.pt"), map_location="cpu"))
        clean = run_condition(model, ld, "clean", 0)
        cur = {"clean": clean}
        for kind, sev in grid:
            cur[f"{kind}_s{sev}"] = run_condition(model, ld, kind, sev)
            print(f"{name:22s} {kind:11s} s{sev} acc={cur[f'{kind}_s{sev}']:.3f}", flush=True)
        vals = [v for k, v in cur.items() if k != "clean"]
        cur["robust_acc"] = float(np.mean(vals))
        cur["drop"] = float(clean - cur["robust_acc"])
        results[name] = cur
        print(f"==> {name}: clean={clean:.3f} robust={cur['robust_acc']:.3f} "
              f"drop={cur['drop']:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()