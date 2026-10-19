"""Collect the budget-optimisation experiments into one summary.

Question the manuscript's negative results hinge on: how much of the
"from-scratch networks are unstable and barely beat a plain CNN" story is a
property of the architecture, and how much is a property of a six-epoch
training budget?  This script compares, seed by seed, the 6-epoch baseline
against a 12-epoch / higher-LR schedule, for MTD-Net and the parameter-matched
plain CNN.

Output: report/optimization.json
"""

from __future__ import annotations

import glob
import json
import os

OUT = "report/optimization.json"
PAIRS = {
    "mtdnet": ("runs/mtdnet_s%d", "runs/opt_mtdnet_long_s%d"),
    "smallcnn": ("runs/smallcnn_s%d", "runs/opt_smallcnn_long_s%d"),
}


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main() -&gt; None:
    out = {}
    for model, (base_tpl, opt_tpl) in PAIRS.items():
        rows = []
        for s in range(3):
            bp, op = base_tpl % s, opt_tpl % s
            if not (os.path.exists(bp + "/metrics.json") and os.path.exists(op + "/metrics.json")):
                continue
            b, o = jload(bp + "/metrics.json"), jload(op + "/metrics.json")
            rows.append({
                "seed": s,
                "base_f1": b["test_macro_f1"], "base_ece": b["ece"],
                "opt_f1": o["test_macro_f1"], "opt_ece": o["ece"],
                "opt_epochs": o["epochs"], "opt_lr": o["lr"],
                "delta_f1_pt": (o["test_macro_f1"] - b["test_macro_f1"]) * 100,
            })
        if not rows:
            continue
        import numpy as np
        bf = [r["base_f1"] for r in rows]
        of = [r["opt_f1"] for r in rows]
        out[model] = {
            "n_seeds": len(rows),
            "base_mean": float(np.mean(bf)),
            "base_sd": float(np.std(bf, ddof=1)) if len(bf) &gt; 1 else 0.0,
            "opt_mean": float(np.mean(of)),
            "opt_sd": float(np.std(of, ddof=1)) if len(of) &gt; 1 else 0.0,
            "gain_mean_pt": float(np.mean([r["delta_f1_pt"] for r in rows])),
            "per_seed": rows,
            "opt_latency_ms": jload(rows[0] and opt_tpl % rows[0]["seed"] + "/metrics.json")["ms_per_image"],
        }
    # gate-variant single-seed screening, if present
    g = "runs/opt_gate_s0/metrics.json"
    if os.path.exists(g):
        m = jload(g)
        b = jload("runs/mtdnet_s0/metrics.json")
        b12p = "runs/opt_mtdnet_long_s0/metrics.json"
        b12 = jload(b12p) if os.path.exists(b12p) else None
        out["mtdnet_g_seed0"] = {
            "opt_f1": m["test_macro_f1"], "ece": m["ece"], "epochs": m["epochs"],
            "base_f1_6ep": b["test_macro_f1"],
            "base_f1_12ep": b12["test_macro_f1"] if b12 else None,
            "delta_vs_12ep_pt": ((m["test_macro_f1"] - b12["test_macro_f1"]) * 100
                                 if b12 else None),
            "delta_vs_6ep_pt": (m["test_macro_f1"] - b["test_macro_f1"]) * 100,
        }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for k, v in out.items():
        print(k, json.dumps(v, indent=1)[:300])
    print("wrote", OUT)


if __name__ == "__main__":
    main()