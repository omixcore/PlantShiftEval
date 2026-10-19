"""Aggregate the multi-seed corruption benchmark.

runs/robustness_ms.json holds one record per *run* (model x seed).  The
manuscript needs two different things from it:

1. per-model mean +- sd across seeds, so a single lucky seed cannot carry the
   robustness claim;
2. per-seed paired deltas between the two contrasts that matter
   (MTD-Net vs the parameter-matched CNN, MTD-P vs ResNet-18), because the
   seeds are paired: same data order, same augmentation stream.

Output: report/robustness_summary.json
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

RUNS = "runs/robustness_ms.json"
OLD = "runs/robustness.json"
OUT = "report/robustness_summary.json"

CONDITIONS = ["clean", "noise_s1", "noise_s2", "blur_s1", "blur_s2",
              "brightness_s2", "contrast_s2", "occlusion_s1", "occlusion_s2"]


def model_of(run: str) -> str:
    return re.sub(r"_s\d+$", "", run)


def load() -> dict[str, dict[int, dict]]:
    if not os.path.exists(RUNS):
        raise FileNotFoundError(
            "Missing runs/robustness_ms.json: the repository contains only the aggregate "
            "report/robustness_summary.json, not the raw per-seed, per-condition scores "
            "needed to rebuild it. Archive those run-level results before rerunning this "
            "aggregation."
        )
    with open(RUNS, encoding="utf-8") as f:
        ms = json.load(f)
    out: dict[str, dict[int, dict]] = {}
    for run, rec in ms.items():
        s = int(run.rsplit("_s", 1)[1])
        out.setdefault(model_of(run), {})[s] = rec
    # single-seed models from the first benchmark pass (mobilenet, mtd_single)
    if os.path.exists(OLD):
        with open(OLD, encoding="utf-8") as f:
            old = json.load(f)
        for run, rec in old.items():
            m = model_of(run)
            if m in out:
                continue
            s = int(run.rsplit("_s", 1)[1])
            out.setdefault(m, {})[s] = rec
    return out


def mean_sd(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def main() -> None:
    data = load()
    summary = {"per_model": {}, "paired": {}, "n_seeds": {}}

    for m, by_seed in sorted(data.items()):
        seeds = sorted(by_seed)
        summary["n_seeds"][m] = len(seeds)
        entry = {"n_seeds": len(seeds), "per_condition": {}}
        for cond in CONDITIONS:
            vals = [by_seed[s][cond] for s in seeds if cond in by_seed[s]]
            if vals:
                mu, sd = mean_sd(vals)
                entry["per_condition"][cond] = {"mean": mu, "sd": sd}
        for key in ("clean", "robust_acc", "drop"):
            vals = [by_seed[s][key] for s in seeds if key in by_seed[s]]
            mu, sd = mean_sd(vals)
            entry[key] = {"mean": mu, "sd": sd}
        entry["per_seed"] = {str(s): {"clean": by_seed[s]["clean"],
                                      "robust_acc": by_seed[s]["robust_acc"],
                                      "drop": by_seed[s]["drop"]} for s in seeds}
        summary["per_model"][m] = entry

    pairs = [("mtdnet", "smallcnn"), ("mtdnet_pt", "resnet18"),
             ("mtdnet", "smallcnn"), ("mtdnet_pt", "mtdnet")]
    seen = set()
    for a, b in pairs:
        if (a, b) in seen or a not in data or b not in data:
            continue
        seen.add((a, b))
        common = sorted(set(data[a]) & set(data[b]))
        if not common:
            continue
        rec = {"n_pairs": len(common), "seeds": common}
        for cond in ("clean", "robust_acc", "mean_over_conditions"):
            if cond == "mean_over_conditions":
                da = [np.mean([data[a][s][c] for c in CONDITIONS[1:]]) for s in common]
                db = [np.mean([data[b][s][c] for c in CONDITIONS[1:]]) for s in common]
            else:
                da = [data[a][s][cond] for s in common]
                db = [data[b][s][cond] for s in common]
            d = np.asarray(da) - np.asarray(db)
            mu, sd = mean_sd([float(v) for v in d])
            rec[cond] = {
                "delta_mean_pt": mu * 100,
                "delta_sd_pt": sd * 100,
                "a_mean": float(np.mean(da)),
                "b_mean": float(np.mean(db)),
                "per_seed_pt": [float(v) * 100 for v in d],
                "all_positive": bool((d > 0).all()),
            }
        # clean-vs-corrupted amplification of the same contrast
        summary["paired"][f"{a}_minus_{b}"] = rec

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    for m, e in summary["per_model"].items():
        print(f"{m:14s} n={e['n_seeds']}  clean={e['clean']['mean']:.3f}"
              f"+-{e['clean']['sd']:.3f}  robust={e['robust_acc']['mean']:.3f}"
              f"+-{e['robust_acc']['sd']:.3f}  drop={e['drop']['mean']*100:.1f}pt")
    for k, r in summary["paired"].items():
        print(f"{k:24s} clean {r['clean']['delta_mean_pt']:+.1f} "
              f"({r['clean']['delta_sd_pt']:.1f}) | robust "
              f"{r['robust_acc']['delta_mean_pt']:+.1f} "
              f"({r['robust_acc']['delta_sd_pt']:.1f}) | "
              f"all seeds positive: {r['robust_acc']['all_positive']}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()