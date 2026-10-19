"""Seed replication + statistical comparison.

Two different questions are answered here, and they need different tests.

1. "Is the ranking stable across random seeds?"
   -> per-model mean/std of macro-F1 over the 3 seeds, plus a paired t-test
      across seeds (same seed = same data order / augmentation stream, so the
      seeds are paired).  n=3 gives very little power, so this is reported as
      descriptive statistics plus an exact-ish p-value, never as proof.

2. "Is the gap larger than test-set sampling noise?"
   -> each model is represented by the softmax average of its 3 seeds (a
     3-run ensemble, i.e. the same compute budget for every model), then
      (a) McNemar's exact test on the paired correctness vectors, and
      (b) a paired bootstrap over test images giving a 95% CI for the
         macro-F1 difference.
   Holm-Bonferroni correction is applied across the family of comparisons.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def macro_f1(y: np.ndarray, p: np.ndarray, n_class: int) -> float:
    conf = np.bincount(y * n_class + p, minlength=n_class * n_class).reshape(n_class, n_class)
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    used = (conf.sum(0) + conf.sum(1)) > 0
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    return float(f1[used].mean())


def load_runs(runs_root: str):
    """{model: {seed: {'logits': np.ndarray, 'labels': np.ndarray, 'metrics': dict}}}"""
    out: dict[str, dict[int, dict]] = {}
    for d in sorted(glob.glob(os.path.join(runs_root, "*_s*"))):
        mj_p = os.path.join(d, "metrics.json")
        lg_p = os.path.join(d, "logits.pt")
        if not (os.path.exists(mj_p) and os.path.exists(lg_p)):
            continue
        with open(mj_p, encoding="utf-8") as f:
            mj = json.load(f)
        blob = torch.load(lg_p, map_location="cpu", weights_only=False)
        out.setdefault(mj["model"], {})[int(mj["seed"])] = {
            "logits": blob["logits"].numpy() if torch.is_tensor(blob["logits"]) else np.asarray(blob["logits"]),
            "labels": blob["labels"].numpy() if torch.is_tensor(blob["labels"]) else np.asarray(blob["labels"]),
            "metrics": mj,
            "dir": d,
        }
    return out


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def holm(pvals: list[float]) -> list[float]:
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(idx):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="report/significance.json")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs = load_runs(args.runs)
    seeds_by_model = {m: sorted(v) for m, v in runs.items()}
    models = [m for m, s in seeds_by_model.items() if len(s) >= 3]
    models.sort(key=lambda m: -np.mean([runs[m][s]["metrics"]["test_macro_f1"] for s in seeds_by_model[m]]))
    print("replicated models:", {m: seeds_by_model[m] for m in models})

    # ---- per-model seed statistics ------------------------------------- #
    stats = {}
    for m in models:
        f1s = np.array([runs[m][s]["metrics"]["test_macro_f1"] for s in seeds_by_model[m]])
        accs = np.array([runs[m][s]["metrics"]["test_acc"] for s in seeds_by_model[m]])
        # 3-seed ensemble
        probs = np.mean([softmax(runs[m][s]["logits"]) for s in seeds_by_model[m]], 0)
        y = runs[m][seeds_by_model[m][0]]["labels"]
        p = probs.argmax(1)
        stats[m] = {
            "n_seeds": len(f1s),
            "macro_f1_mean": float(f1s.mean()), "macro_f1_std": float(f1s.std(ddof=1)),
            "macro_f1_seeds": [float(v) for v in f1s],
            "acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1)),
            "params": int(runs[m][seeds_by_model[m][0]]["metrics"]["params"]),
            "ens_macro_f1": macro_f1(y, p, int(y.max()) + 1),
            "ens_acc": float((p == y).mean()),
            "ens_correct": (p == y),
            "ens_probs": probs,
            "labels": y,
        }
        print(f"{m:20s} f1={f1s.mean():.4f}±{f1s.std(ddof=1):.4f} "
              f"(ens {stats[m]['ens_macro_f1']:.4f})")

    # ---- pairwise tests ------------------------------------------------- #
    from scipy.stats import binomtest, ttest_rel
    pairs = []
    for a, b in itertools.combinations(models, 2):
        ya, yb = stats[a]["labels"], stats[b]["labels"]
        assert (ya == yb).all()
        ca, cb = stats[a]["ens_correct"], stats[b]["ens_correct"]
        b01 = int(np.sum(~ca & cb))  # a wrong, b right
        b10 = int(np.sum(ca & ~cb))  # a right, b wrong
        mcn = binomtest(b01, b01 + b10, 0.5).pvalue if (b01 + b10) > 0 else 1.0

        fa = np.array(stats[a]["macro_f1_seeds"])
        fb = np.array(stats[b]["macro_f1_seeds"])
        t_p = float(ttest_rel(fa, fb).pvalue) if len(fa) > 1 else float("nan")

        # paired bootstrap over test images on the ensembled predictions
        rng = np.random.default_rng(args.seed)
        n = len(ya)
        k = int(ya.max()) + 1
        pa, pb = stats[a]["ens_probs"].argmax(1), stats[b]["ens_probs"].argmax(1)
        deltas = np.empty(args.boot)
        for i in range(args.boot):
            idx = rng.integers(0, n, n)
            deltas[i] = macro_f1(ya[idx], pa[idx], k) - macro_f1(yb[idx], pb[idx], k)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        pairs.append({
            "a": a, "b": b,
            "delta_f1_mean_seeds": float(fa.mean() - fb.mean()),
            "delta_f1_ens": float(stats[a]["ens_macro_f1"] - stats[b]["ens_macro_f1"]),
            "boot_ci_low": float(lo), "boot_ci_high": float(hi),
            "boot_p_two_sided": float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean())),
            "mcnemar_b_a_wrong_b_right": b01, "mcnemar_b_a_right_b_wrong": b10,
            "mcnemar_p": float(mcn),
            "ttest_seed_p": t_p,
        })

    adj_mcn = holm([p["mcnemar_p"] for p in pairs])
    adj_boot = holm([p["boot_p_two_sided"] for p in pairs])
    for p, pm, pb in zip(pairs, adj_mcn, adj_boot):
        p["mcnemar_p_holm"] = float(pm)
        p["boot_p_holm"] = float(pb)
        p["significant_holm"] = bool(pm < 0.05)
        print(f"{p['a']:>20s} vs {p['b']:<20s} dF1={p['delta_f1_ens']:+.4f} "
              f"CI[{p['boot_ci_low']:+.4f},{p['boot_ci_high']:+.4f}] "
              f"mcnemar p={p['mcnemar_p']:.2e} (holm {pm:.2e})")

    payload = {
        "n_test": int(len(stats[models[0]]["labels"])) if models else 0,
        "n_boot": args.boot,
        "models": {m: {k: v for k, v in s.items()
                       if k not in ("ens_correct", "ens_probs", "labels")}
                   for m, s in stats.items()},
        "pairs": pairs,
        "notes": ("paired t-test uses the 3 seeds as pairs; McNemar uses the 3-seed "
                  "ensemble's per-image correctness; bootstrap resamples test images "
                  "with the seed fixed at %d; Holm correction within this family." % args.seed),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    # keep ensembled predictions for downstream figures
    np.savez("runs/_ens_probs.npz",
             **{f"{m}__probs": s["ens_probs"] for m, s in stats.items()},
             **{f"{m}__label": s["labels"] for m, s in stats.items()})
    print("saved", args.out)


if __name__ == "__main__":
    main()