"""Label-free post-hoc calibration for the cross-domain experiments.

Question
--------
Zero-shot transfer to PlantDoc leaves every model both wrong and confident
(ECE > 0.4).  Fine-tuning repairs the calibration, but fine-tuning needs target
labels.  Is any of the miscalibration repairable *without* a single target
label?

Protocol
--------
Temperature scaling (Guo et al., 2017) has exactly one parameter, fitted by
minimising negative log-likelihood on data whose labels we do have -- the
source test split.  The fitted temperature is then applied unchanged to the
target logits.  Because argmax(z / T) = argmax(z), accuracy cannot move: any
ECE reduction is pure calibration gain, which makes the comparison clean.

Three conditions are reported per seed:

  raw        no rescaling (the number quoted in crossdomain.json)
  source-T   T fitted on PlantVillage test logits      <- label-free
  oracle-T   T fitted on PlantDoc labels               <- upper bound only,
             included to show how much of the error is a pure scale problem

Output: report/calibration.json  (per seed + mean/sd per model)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crossdomain import infer, load_pd, softmax  # noqa: E402
from data import load_cache  # noqa: E402
from models import build_model  # noqa: E402

N_BINS = 15


def ece(prob: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    conf = prob.max(1)
    pred = prob.argmax(1)
    acc = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        total += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(total)


def nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    z = torch.from_numpy(logits / T)
    return float(F.cross_entropy(z, torch.from_numpy(y)))


def fit_temperature(logits: np.ndarray, y: np.ndarray,
                    lo: float = 0.05, hi: float = 30.0, n: int = 600) -> float:
    """Grid search in log space: one parameter, convex enough that this is exact
    to 4 significant digits and avoids pulling in an optimiser dependency."""
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), n))
    losses = [nll(logits, y, t) for t in grid]
    return float(grid[int(np.argmin(losses))])


def summarise(logits: np.ndarray, y: np.ndarray, T: float) -> dict:
    p = softmax(logits / T)
    pred = p.argmax(1)
    return {
        "acc": float((pred == y).mean()),
        "conf": float(p.max(1).mean()),
        "ece": ece(p, y),
        "nll": float(nll(logits, y, T)),
        "T": float(T),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--cache-pd", default="cache_pd")
    ap.add_argument("--out", default="report/calibration.json")
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    meta, _, _ = load_cache(args.cache)
    pd_info, x_pd, y_pd = load_pd(args.cache_pd)

    groups: dict[str, list[str]] = {}
    for r in args.runs:
        with open(os.path.join(r, "metrics.json"), encoding="utf-8") as f:
            groups.setdefault(json.load(f)["model"], []).append(r)

    out: dict[str, dict] = {}
    for model_name, dirs in sorted(groups.items()):
        per_seed = []
        for r in sorted(dirs):
            with open(os.path.join(r, "metrics.json"), encoding="utf-8") as f:
                mj = json.load(f)
            z_src = torch.load(os.path.join(r, "logits.pt"),
                               weights_only=False)["logits"].astype(np.float64)
            y_src = torch.load(os.path.join(r, "logits.pt"),
                               weights_only=False)["labels"]
            model = build_model(mj["model"], meta["n_classes"], meta["n_species"])
            model.load_state_dict(torch.load(os.path.join(r, "model.pt"),
                                             map_location="cpu"))
            z_tgt = infer(model, x_pd, args.size).astype(np.float64)

            T_src = fit_temperature(z_src, y_src)
            T_orc = fit_temperature(z_tgt, y_pd)
            per_seed.append({
                "run": os.path.basename(r),
                "raw": summarise(z_tgt, y_pd, 1.0),
                "source_T": summarise(z_tgt, y_pd, T_src),
                "oracle_T": summarise(z_tgt, y_pd, T_orc),
            })
            print(f"{os.path.basename(r):16s} T_src={T_src:6.3f} "
                  f"ECE {per_seed[-1]['raw']['ece']:.3f} -> "
                  f"{per_seed[-1]['source_T']['ece']:.3f} "
                  f"(oracle {per_seed[-1]['oracle_T']['ece']:.3f})", flush=True)

        agg = {}
        for cond in ("raw", "source_T", "oracle_T"):
            for k in ("acc", "conf", "ece", "nll"):
                v = np.array([s[cond][k] for s in per_seed], dtype=np.float64)
                agg[f"{cond}_{k}_mean"] = float(v.mean())
                agg[f"{cond}_{k}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out[model_name] = {
            "n_seeds": len(per_seed),
            "per_seed": per_seed,
            "mean": agg,
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"target": "PlantDoc mapped subset", "n": int(len(y_pd)),
                   "n_bins": N_BINS, "results": out}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()