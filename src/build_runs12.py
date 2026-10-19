"""Assemble the 12-epoch arm into a directory that significance.py can consume.

significance.load_runs() keys runs by the `model` field of metrics.json, so the
12-epoch runs (whose directory names carry an `opt_` prefix) cannot live next to
the 6-epoch baselines without colliding on (model, seed).  This script copies
only metrics.json and logits.pt -- a few hundred kilobytes per run -- into
runs12/, renaming each run to the canonical `<model>_s<seed>` form.
"""

from __future__ import annotations

import json
import os
import shutil

SRC = {
    "mtdnet": "runs/opt_mtdnet_long_s%d",
    "smallcnn": "runs/opt_smallcnn_long_s%d",
}
DST = "runs12"


def main() -> None:
    os.makedirs(DST, exist_ok=True)
    n = 0
    for model, tpl in SRC.items():
        for s in range(3):
            src = tpl % s
            if not os.path.exists(os.path.join(src, "logits.pt")):
                continue
            dst = os.path.join(DST, f"{model}_s{s}")
            os.makedirs(dst, exist_ok=True)
            for f in ("metrics.json", "logits.pt"):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
            with open(os.path.join(dst, "metrics.json"), encoding="utf-8") as fh:
                m = json.load(fh)
            print(f"{src} -> {dst}  f1={m['test_macro_f1']:.4f} epochs={m['epochs']}")
            n += 1
    print(f"copied {n} runs into {DST}")


if __name__ == "__main__":
    main()
