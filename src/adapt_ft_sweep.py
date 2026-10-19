"""Can the from-scratch network be *made* adaptable?

The manuscript reports a negative result: on PlantDoc, full fine-tuning of the
from-scratch MTD-Net (0.206 macro-F1) lands *below* a 25-shot linear probe
(0.277).  If that is a property of the features it is a finding; if it is a
property of the optimiser settings (3 epochs at 3e-4 on 688 images, i.e. ~64
gradient steps) it is a bug in the protocol.

This script holds the folds, the labels and the model fixed and varies only the
adaptation recipe, so the comparison isolates the recipe:

  base       lr 3e-4, 3 epochs, everything trainable   (the manuscript setting)
  lowlr8     lr 1e-4, 8 epochs, everything trainable
  bnfreeze   lr 1e-4, 8 epochs, BN statistics frozen
  shallow    lr 3e-4, 5 epochs, stem + appearance path frozen

Output: report/adapt_ft_sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptation import finetune, load_split_cache, logits_of, metrics  # noqa: E402
from data import load_cache  # noqa: E402
from models import build_model  # noqa: E402

CONFIGS = [
    # name,      lr,    epochs, freeze_mode
    ("base", 3e-4, 3, None),
    ("lowlr8", 1e-4, 8, None),
    ("bnfreeze", 1e-4, 8, "bn"),
    ("shallow", 3e-4, 5, "stem_app"),
]


def apply_freeze(model: nn.Module, mode: str | None) -> None:
    if mode == "stem_app":
        for p in model.stem.parameters():
            p.requires_grad = False
        for p in model.app.parameters():
            p.requires_grad = False


def set_bn_eval(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/mtdnet_s0")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--cache-pd", default="cache_pd_split")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="report/adapt_ft_sweep.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    meta, _, _ = load_cache(args.cache)
    _info, splits = load_split_cache(args.cache_pd)
    X = np.concatenate([splits["train"][0], splits["test"][0]])
    Y = np.concatenate([splits["train"][1], splits["test"][1]])
    present = sorted({int(v) for v in Y})

    from sklearn.model_selection import StratifiedKFold
    folds = list(StratifiedKFold(n_splits=args.folds, shuffle=True,
                                 random_state=0).split(X, Y))

    with open(os.path.join(args.run, "metrics.json"), encoding="utf-8") as f:
        mj = json.load(f)

    out: dict[str, list[dict]] = {c[0]: [] for c in CONFIGS}
    for fi, (tr_idx, te_idx) in enumerate(folds):
        xtr, ytr = X[tr_idx], Y[tr_idx]
        xte, yte = X[te_idx], Y[te_idx]
        for name, lr, epochs, mode in CONFIGS:
            model = build_model(mj["model"], meta["n_classes"], meta["n_species"])
            model.load_state_dict(torch.load(os.path.join(args.run, "model.pt"),
                                             map_location="cpu"))
            model.eval()
            apply_freeze(model, mode)
            if mode == "bn":
                # keep BN in eval mode through the whole fine-tune
                orig_train = model.train
                model.train = lambda *a, m=model, orig=orig_train: (
                    set_bn_eval(m), orig(*a))[1]
            ft = finetune(model, xtr, ytr, args.size, epochs=epochs, lr=lr, seed=fi)
            m = metrics(logits_of(ft, xte, args.size), yte, present)
            out[name].append(m)
            print(f"fold {fi} {name:9s} f1={m['macro_f1']:.3f} "
                  f"acc={m['acc15']:.3f}", flush=True)

    summary = {}
    for name, rows in out.items():
        f1 = np.array([r["macro_f1"] for r in rows])
        summary[name] = {"macro_f1_mean": float(f1.mean()),
                         "macro_f1_sd": float(f1.std(ddof=1)),
                         "acc_mean": float(np.mean([r["acc15"] for r in rows]))}
        print(f"==> {name:9s} {f1.mean():.3f} +- {f1.std(ddof=1):.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"run": args.run, "folds": args.folds,
                   "summary": summary, "per_fold": out}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()