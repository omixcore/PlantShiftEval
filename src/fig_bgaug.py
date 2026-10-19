"""Figure: does background augmentation repair crop identity?  No.

Panel A  per-crop crop-identity recall on PlantDoc, baseline vs augmented
Panel B  where the 956 predictions actually go, against the true crop prior:
         augmentation pushes even more mass onto maize, which is exactly the
         backdrop shortcut it was meant to remove
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "red": "#D55E00", "grey": "#666666", "black": "#222222",
     "purple": "#7B52AB"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

crops = ["apple", "blueberry", "cherry", "maize", "grape"]
cmp_ = json.load(open("report/bg_aug_compare.json", encoding="utf-8"))
cmj = json.load(open("report/bg_confusion.json", encoding="utf-8"))

base = [cmp_["per_crop_recall"][c]["baseline"] for c in crops]
aug = [cmp_["per_crop_recall"][c]["bg_aug"] for c in crops]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.55),
                               gridspec_kw={"width_ratios": [1.25, 1]})

x = np.arange(len(crops))
ax1.bar(x - 0.19, base, 0.36, color=C["grey"], label="MTD-P baseline")
ax1.bar(x + 0.19, aug, 0.36, color=C["orange"], label="+ background aug. + crop balance")
ax1.axhline(0.2, color=C["red"], lw=0.8, ls=":")
ax1.text(-0.47, 0.225, "chance", color=C["red"], fontsize=6, ha="left",
         bbox=dict(fc="white", ec="none", pad=0.6))
ax1.set_xticks(x)
ax1.set_xticklabels(crops, fontsize=7)
ax1.set_ylabel("crop-identity recall (PlantDoc, 0 labels)")
ax1.set_ylim(0, 1.12)
ax1.legend(frameon=False, fontsize=6.2, loc="upper left", ncol=1)
ax1.set_title("A  The repair hurts the crops it was aimed at", fontsize=7.5,
              loc="left", color=C["black"])

pred_b = np.array(cmj["baseline"]["pred_prior"]) / 956
pred_a = np.array(cmj["bg_aug"]["pred_prior"]) / 956
true_p = json.load(open("report/crop_diagnosis.json", encoding="utf-8"))[
    "models"]["mtdnet_pt"]["true_target_prior"]
ax2.bar(x - 0.19, pred_b, 0.36, color=C["grey"], label="baseline")
ax2.bar(x + 0.19, pred_a, 0.36, color=C["orange"], label="+ background aug.")
ax2.plot(x, true_p, "o", ms=4.2, color=C["blue"], label="true crop prior", zorder=5)
ax2.set_xticks(x)
ax2.set_xticklabels(crops, fontsize=7, rotation=20, ha="right")
ax2.set_ylabel("share of 956 predictions")
ax2.set_ylim(0, 0.80)
ax2.legend(frameon=False, fontsize=6.2, loc="upper left")
ax2.annotate("", xy=(3.19, 0.645), xytext=(3.19, 0.562),
             arrowprops=dict(arrowstyle="-|&gt;", color=C["red"], lw=0.9))
ax2.text(3.30, 0.60, "+8.3", color=C["red"], fontsize=6.2, va="center")
ax2.set_title("B  …and piles even more mass onto maize", fontsize=7.5,
              loc="left", color=C["black"])

fig.tight_layout()
os.makedirs("report/figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"report/figures/fig_bgaug.{ext}", dpi=350, bbox_inches="tight")
print("wrote report/figures/fig_bgaug.png/.pdf")