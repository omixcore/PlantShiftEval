"""Figure: where the zero-shot error comes from, and what repairs the crop half.

Panel A  per-crop crop-identity recall, zero-shot vs 25 labels/crop probe
Panel B  decomposition of 15-way accuracy: correct / lost to crop identity /
         lost to lesion identity, for MTD-P and for the from-scratch model
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "red": "#D55E00", "grey": "#666666", "black": "#222222"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

d = json.load(open("report/crop_diagnosis.json", encoding="utf-8"))
pt = d["models"]["mtdnet_pt"]
fr = d["models"]["mtdnet"]
crops = ["apple", "blueberry", "cherry", "maize", "grape"]
zs = [pt["decompose"][c]["crop_recall"] for c in crops]
pr = [pt["probe"]["25"]["per_crop"][c] for c in crops]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5),
                               gridspec_kw={"width_ratios": [1.15, 1]})

x = np.arange(len(crops))
ax1.bar(x - 0.19, zs, 0.36, color=C["grey"], label="zero-shot (no target labels)")
ax1.bar(x + 0.19, pr, 0.36, color=C["blue"], label="25 labels/crop linear probe")
ax1.axhline(0.2, color=C["red"], lw=0.8, ls=":")
ax1.text(-0.45, 0.208, "chance", color=C["red"], fontsize=6, ha="left")
ax1.set_xticks(x)
ax1.set_xticklabels(crops, fontsize=7)
ax1.set_ylabel("crop-identity recall")
ax1.set_ylim(0, 1.05)
ax1.legend(frameon=False, fontsize=6.2, loc="upper left")
ax1.set_title("A  Three of five crops collapse; labels restore them", fontsize=7.5,
              loc="left", color=C["black"])

rows = [("MTD-P\n(pre-trained)", pt), ("MTD-Net\n(from scratch)", fr)]
for i, (name, r) in enumerate(rows):
    ok = r["acc15"]
    crop_part = r["oracle_crop_acc"] - ok
    lesion_part = 1 - r["oracle_crop_acc"]
    ok, cp, lp = ok * 100, crop_part * 100, lesion_part * 100
    ax2.barh(i, ok, color=C["green"], height=0.5)
    ax2.barh(i, cp, left=ok, color=C["orange"], height=0.5)
    ax2.barh(i, lp, left=ok + cp, color=C["grey"], height=0.5)
    ax2.text(ok / 2, i, f"{ok:.1f}", va="center", ha="center", fontsize=6.5,
             color="white")
    if cp > 8:
        ax2.text(ok + cp / 2, i, f"+{cp:.1f}", va="center", ha="center",
                 fontsize=6.5, color="white")
    ax2.text(ok + cp + lp / 2, i, f"{lp:.1f}", va="center", ha="center",
             fontsize=6.5, color="white")
ax2.set_yticks(range(len(rows)))
ax2.set_yticklabels([n for n, _ in rows], fontsize=7)
ax2.set_xlim(0, 100)
ax2.set_xlabel("share of the 956 field images (%)")
ax2.set_title("B  Crop identity costs ~1/3, lesion identity ~2/3", fontsize=7.5,
              loc="left", color=C["black"])
handles = [plt.Rectangle((0, 0), 1, 1, color=C["green"]),
           plt.Rectangle((0, 0), 1, 1, color=C["orange"]),
           plt.Rectangle((0, 0), 1, 1, color=C["grey"])]
ax2.legend(handles, ["correct", "lost to crop identity", "lost to lesion identity"],
           frameon=False, fontsize=6.2, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, -0.42))

fig.tight_layout()
os.makedirs("report/figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"report/figures/crop_diagnosis.{ext}", dpi=350, bbox_inches="tight")
print("wrote report/figures/crop_diagnosis.png/.pdf")