"""Publication-grade figures for the manuscript.

Every figure is written twice: a 600-dpi PNG for quick inspection and a PDF
(vector) for the production upload.  Sizes follow the journal's column widths
(80 mm single, 180 mm double).  Panel letters are uppercase and placed in the
top-left corner, as required by the author guidelines.

Run `python src/figures_pub.py --which 1 2 ...` to rebuild a subset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_cache  # noqa: E402
from models import build_model  # noqa: E402

MM = 1 / 25.4
W1, W2 = 80 * MM, 180 * MM          # single / double column width in inches
DPI = 600

# Okabe-Ito, colour-blind safe
C = {
    "blue": "#0072B2", "verm": "#D55E00", "green": "#009E73",
    "orange": "#E69F00", "purple": "#CC79A7", "sky": "#56B4E9",
    "grey": "#8C8C8C", "black": "#1A1A1A",
}
MODEL_STYLE = {
    "resnet18":          ("ResNet-18", C["blue"]),
    "mobilenet_v3_small": ("MobileNetV3-S", C["sky"]),
    "mtdnet_pt":         ("MTD-P", C["verm"]),
    "mtdnet":            ("MTD-Net", C["green"]),
    "mtd_concat":        ("MTD-Net w/o gate", C["purple"]),
    "mtd_single":        ("MTD-Net w/o auxiliary head", C["orange"]),
    "smallcnn":          ("Plain CNN (matched)", C["grey"]),
}
ORDER = ["mtdnet_pt", "resnet18", "mobilenet_v3_small", "mtdnet",
         "mtd_concat", "mtd_single", "smallcnn"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# --------------------------------------------------------------------------- #
def panel(ax, letter, dx=-0.14, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="left")


def save(fig, name, outdir):
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, name + ".png")
    pdf = os.path.join(outdir, name + ".pdf")
    fig.savefig(png, dpi=DPI)
    fig.savefig(pdf)
    plt.close(fig)
    return png


def load_json(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def short(name):
    s = name.split("___")[-1].replace("_", " ")
    return s[:22]


# --------------------------------------------------------------------------- #
# Figure 1 - architecture
# --------------------------------------------------------------------------- #
def fig1_arch(outdir):
    fig, ax = plt.subplots(figsize=(W2, 62 * MM))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, label, fc="#FFFFFF", ec=C["black"], fs=6.2, lw=0.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.15,rounding_size=0.9",
                                    fc=fc, ec=ec, lw=lw, zorder=3))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, zorder=4, linespacing=1.3)

    def arrow(x1, y1, x2, y2, lw=0.7, color=C["black"], ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|&gt;",
                                     mutation_scale=7, lw=lw, color=color,
                                     linestyle=ls, shrinkA=1.5, shrinkB=1.5,
                                     connectionstyle=f"arc3,rad={rad}", zorder=2))

    def seg(x1, y1, x2, y2, color=C["black"], lw=0.7, ls="-"):
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, zorder=2)

    # input + stem
    box(0.5, 16.5, 9.0, 13.0, "leaf image\n112 × 112\n× 3", fc="#F2F2F2")
    box(11.5, 18.5, 11.0, 9.0, "stem\n3×3 conv / s2\n→ 3×3 DSConv / s2\n32 ch, H/4", fc="#E8F0F7")
    arrow(9.5, 23.0, 11.5, 23.0)

    # trunk split
    seg(22.5, 23.0, 24.3, 23.0)
    seg(24.3, 23.0, 24.3, 35.1)
    seg(24.3, 23.0, 24.3, 15.1)
    arrow(24.3, 35.1, 26.0, 35.1)
    arrow(24.3, 15.1, 26.0, 15.1)

    # ---- appearance path (upper) ----
    ax.text(45, 42.6, "appearance path — crop identity, colour, global shape",
            fontsize=6.6, ha="center", color=C["blue"])
    ya = 32.0
    box(26.0, ya, 9.2, 6.2, "DS block\n32→64, s2", fc="#E8F0F7")
    box(36.1, ya, 9.2, 6.2, "DS block\n64→128, s2", fc="#E8F0F7")
    box(46.2, ya, 9.2, 6.2, "DS block\n128→256, s2", fc="#E8F0F7")
    box(56.3, ya, 7.6, 6.2, "GAP\n$g$ (256)", fc="#D6E6F2")
    for x1, x2 in [(35.2, 36.1), (45.3, 46.2), (55.4, 56.3)]:
        arrow(x1, ya + 3.1, x2, ya + 3.1)
    box(67.5, 39.2, 11.5, 4.8, "crop head, 5-way", fc="#FDF0E8", ec=C["verm"])
    arrow(63.9, 36.4, 67.5, 40.4, rad=0.18, color=C["verm"])

    # ---- texture path (lower) ----
    ax.text(45, 6.4, "texture path — lesion edge and spot granularity, retained at H/4",
            fontsize=6.6, ha="center", color=C["green"])
    yt = 12.0
    box(26.0, yt, 9.6, 6.2, "DS block\n32→96, d1", fc="#E6F4EE")
    box(36.5, yt, 9.6, 6.2, "DS block\n96→96, d1", fc="#E6F4EE")
    box(47.0, yt, 9.6, 6.2, "DS block\n96→96, d2", fc="#E6F4EE")
    box(57.5, yt, 9.6, 6.2, "DS block\n96→96, d2", fc="#E6F4EE")
    for x1, x2 in [(35.6, 36.5), (46.1, 47.0), (56.6, 57.5)]:
        arrow(x1, yt + 3.1, x2, yt + 3.1)

    # ---- gate ----
    box(40.5, 21.5, 14.0, 5.6, "cross-scale gate\nchannel + spatial prior, from $g$",
        fc="#FFF7E0", ec=C["orange"])
    arrow(52.0, yt + 6.2, 49.0, 21.5, rad=-0.08, color=C["green"])
    arrow(60.1, 33.0, 52.6, 24.2, rad=0.10, color=C["blue"], ls="--")

    box(70.5, 12.6, 12.4, 6.4, "gated texture\nGAP → $d$ (96)", fc="#D9EFE5")
    arrow(67.1, 15.1, 70.5, 15.8)

    # ---- fusion + head ----
    box(85.0, 14.4, 6.4, 15.0, "[ $d$ ; $g$ ]\n304-d", fc="#F2F2F2")
    arrow(82.9, 15.8, 85.0, 18.4)
    arrow(61.6, 33.6, 87.0, 27.5, rad=-0.22, color=C["blue"], ls="--")
    box(93.2, 17.0, 6.6, 9.6, "FC\n256\n↓\n15-way", fc="#FDF0E8", ec=C["verm"])
    arrow(91.4, 21.8, 93.2, 21.8)

    ax.text(50, 1.0, "trained from scratch in the small-data regime; the pre-trained variant "
                     "replaces the stem and appearance path with the corresponding ResNet-18 "
                     "stages, keeping the wiring unchanged",
            fontsize=6.0, ha="center", style="italic", color=C["grey"])
    return save(fig, "Figure_1", outdir)


# --------------------------------------------------------------------------- #
# Figure 2 - training dynamics and seed spread
# --------------------------------------------------------------------------- #
def fig2_training(outdir, runs="runs"):
    picks = ["resnet18", "mtdnet_pt", "mtdnet", "smallcnn"]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 58 * MM),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for m in picks:
        curves = []
        for s in (0, 1, 2):
            p = os.path.join(runs, f"{m}_s{s}", "metrics.json")
            if not os.path.exists(p):
                continue
            j = load_json(p)
            curves.append([h["val_macro_f1"] for h in j["history"]])
        if not curves:
            continue
        cur = np.array(curves)
        ep = np.arange(1, cur.shape[1] + 1)
        lab, col = MODEL_STYLE[m]
        ax.plot(ep, cur.mean(0), color=col, lw=1.1, marker="o", ms=2.4, label=lab)
        ax.fill_between(ep, cur.min(0), cur.max(0), color=col, alpha=0.16, lw=0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0.45, 1.0)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.legend(frameon=False, loc="lower right", handlelength=1.4)
    ax.set_title("training trajectory (mean and range of 3 seeds)", fontsize=7.2)
    panel(ax, "A")

    ax = axes[1]
    sig = load_json("report/significance.json", {}) or {}
    models = sig.get("models", {})
    xs = np.arange(len(ORDER))
    for i, m in enumerate(ORDER):
        v = models.get(m)
        if not v:
            continue
        lab, col = MODEL_STYLE[m]
        vals = v.get("macro_f1_seeds") or [v["macro_f1_mean"]]
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.13, 0.13, len(vals)),
                   vals, s=13, color=col, alpha=0.55, lw=0, zorder=3)
        ax.scatter([i], [v["macro_f1_mean"]], s=26, color="white", ec=col,
                   lw=1.0, zorder=4)
        ax.plot([i - 0.26, i + 0.26], [v["ens_macro_f1"]] * 2, color=col, lw=1.4,
                zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_STYLE[m][0].replace(" ", "\n", 1) for m in ORDER],
                       rotation=38, ha="right", fontsize=6.2)
    ax.set_ylabel("test macro-F1")
    ax.set_ylim(0.78, 1.0)
    ax.set_title("individual seeds (dots), mean (open), ensemble (bar)", fontsize=7.2)
    panel(ax, "B")
    return save(fig, "Figure_2", outdir)


# --------------------------------------------------------------------------- #
# ensemble logits helpers
# --------------------------------------------------------------------------- #
def ensemble(runs="runs", model="mtdnet_pt", seeds=(0, 1, 2)):
    acc, y = [], None
    for s in seeds:
        p = os.path.join(runs, f"{model}_s{s}", "logits.pt")
        if not os.path.exists(p):
            continue
        b = torch.load(p, map_location="cpu", weights_only=False)
        z = b["logits"]
        acc.append(torch.softmax(torch.from_numpy(np.asarray(z)), 1).numpy())
        y = np.asarray(b["labels"])
    if not acc:
        return None, None
    return np.mean(acc, 0), y


# --------------------------------------------------------------------------- #
# Figure 3 - per-class behaviour
# --------------------------------------------------------------------------- #
def fig3_classes(outdir):
    meta = load_json("cache/meta.json")
    names = [short(n) for n in meta["class_names"]]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 74 * MM),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    picks = ["resnet18", "mtdnet", "mtdnet_pt"]
    per = {}
    for m in picks:
        probs, y = ensemble(model=m)
        if probs is None:
            continue
        per[m] = f1_score(y, probs.argmax(1), average=None, labels=range(len(names)),
                          zero_division=0)
    if len(per) == 3:
        x = np.arange(len(names))
        w = 0.27
        for k, m in enumerate(picks):
            lab, col = MODEL_STYLE[m]
            ax.bar(x + (k - 1) * w, per[m], width=w, color=col, label=lab,
                   lw=0, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=55, ha="right", fontsize=5.9)
        ax.set_ylim(0.4, 1.02)
        ax.set_ylabel("F1 (per class)")
        ax.legend(frameon=False, ncol=1, loc="lower left", handlelength=1.2)
        ax.axhline(1.0, color=C["grey"], lw=0.4, ls=":")
        gap = per["mtdnet_pt"] - per["mtdnet"]
        worst = np.argsort(gap)[:3]
        for i in worst:
            if gap[i] &lt; 0.06:
                ax.annotate("", xy=(i, 1.0), xytext=(i, 1.0),
                            arrowprops=None)
        ax.set_title("per-class F1, 3-seed ensemble", fontsize=7.2)
        panel(ax, "A", dx=-0.10)

    ax = axes[1]
    probs, y = ensemble(model="mtdnet_pt")
    if probs is not None:
        cm = confusion_matrix(y, probs.argmax(1), labels=range(len(names)),
                              normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=4.6)
        ax.set_yticklabels(names, fontsize=4.6)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        cb.set_label("row-normalised", fontsize=6)
        ax.set_title("MTD-P confusion", fontsize=7.2)
        panel(ax, "B", dx=-0.30)
    return save(fig, "Figure_3", outdir)


# --------------------------------------------------------------------------- #
# Figure 4 - forest plot
# --------------------------------------------------------------------------- #
def fig4_forest(outdir):
    sig = load_json("report/significance.json", {})
    pairs = [p for p in sig.get("pairs", []) if p.get("delta_f1_ens") is not None]
    if not pairs:
        return None
    focus = [("mtdnet_pt", "resnet18"), ("resnet18", "mtdnet"),
             ("mtdnet", "smallcnn"), ("mtdnet", "mtd_concat"),
             ("mtdnet", "mtd_single"), ("mtdnet_pt", "mtdnet"),
             ("mobilenet_v3_small", "smallcnn"), ("mtd_concat", "smallcnn")]
    sel = []
    for a, b in focus:
        for p in pairs:
            if {p["a"], p["b"]} == {a, b}:
                sel.append(p)
                break
    sel = sel or pairs
    fig, ax = plt.subplots(figsize=(W1 * 1.45, 52 * MM))
    ys = np.arange(len(sel))[::-1]
    for y, p in zip(ys, sel):
        d = p["delta_f1_ens"] * 100
        lo, hi = p.get("boot_ci_low", d) * 100, p.get("boot_ci_high", d) * 100
        sigq = p.get("significant_holm")
        col = C["verm"] if sigq else C["grey"]
        ax.plot([lo, hi], [y, y], color=col, lw=1.3, solid_capstyle="round")
        ax.scatter([d], [y], s=26, color=col, zorder=3, lw=0)
        ax.text(hi + 0.35, y, f"{d:+.1f}", va="center", fontsize=6.2, color=col)
    ax.axvline(0, color=C["black"], lw=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{MODEL_STYLE[p['a']][0]}  vs  {MODEL_STYLE[p['b']][0]}"
                        for p in sel], fontsize=6.4)
    ax.set_xlabel("difference in macro-F1 (percentage points), 3-seed ensemble")
    ax.set_xlim(min(p.get("boot_ci_low", 0) for p in sel) * 100 - 1,
                max(p.get("boot_ci_high", 0) for p in sel) * 100 + 3)
    for lab, c, txt in [("significant after Holm", C["verm"], ""),
                        ("not significant", C["grey"], "")]:
        ax.scatter([], [], color=c, s=22, label=lab)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.4)
    ax.set_ylim(-0.8, len(sel) - 0.2)
    return save(fig, "Figure_4", outdir)


# --------------------------------------------------------------------------- #
# Figure 5 - cross-dataset
# --------------------------------------------------------------------------- #
def fig5_cross(outdir):
    cd = load_json("report/crossdomain.json", {})
    res = cd.get("results", {})
    if not res:
        return None
    order = [m for m in ORDER if m in res]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 66 * MM),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    keys = [("acc15", "15-way accuracy"), ("acc10", "re-normalised (10 classes)"),
            ("within_crop", "within-crop"), ("crop_acc", "crop identification")]
    x = np.arange(len(order))
    w = 0.19
    shades = [C["blue"], C["sky"], C["green"], C["orange"]]
    for k, ((key, lab), col) in enumerate(zip(keys, shades)):
        vals = [res[m][key] for m in order]
        ax.bar(x + (k - 1.5) * w, vals, width=w, color=col, label=lab, lw=0, zorder=3)
    ax.axhline(1 / 15, color=C["grey"], lw=0.5, ls=":")
    ax.text(len(order) - 0.45, 1 / 15 + 0.012, "chance (15-way)", fontsize=5.6,
            color=C["grey"], ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_STYLE[m][0].replace(" ", "\n", 1) for m in order],
                       rotation=35, ha="right", fontsize=6.2)
    ax.set_ylabel("fraction correct")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, ncol=2, loc="upper right", handlelength=1.1,
              columnspacing=0.9)
    ax.set_title("PlantVillage → PlantDoc, zero-shot", fontsize=7.2)
    panel(ax, "A", dx=-0.12)

    ax = axes[1]
    cm_p = os.path.join("report", "figures", "pub", "_crop_cm.npy")
    if os.path.exists(cm_p):
        cm = np.load(cm_p)
        im = ax.imshow(cm, cmap="Oranges", vmin=0, vmax=1)
        crops = ["apple", "blueberry", "cherry", "maize", "grape"]
        ax.set_xticks(range(len(crops)))
        ax.set_yticks(range(len(crops)))
        ax.set_xticklabels(crops, rotation=40, ha="right", fontsize=6)
        ax.set_yticklabels(crops, fontsize=6)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] &gt; 0.02:
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                            fontsize=5.2, color="white" if cm[i, j] &gt; 0.55 else C["black"])
        ax.set_xlabel("predicted crop")
        ax.set_ylabel("true crop")
        ax.set_title("where the errors go (MTD-P)", fontsize=7.2)
        panel(ax, "B", dx=-0.28)
    return save(fig, "Figure_5", outdir)


# --------------------------------------------------------------------------- #
# Figure 6 - adaptation
# --------------------------------------------------------------------------- #
def fig6_adapt(outdir):
    ad = load_json("report/adaptation.json", {})
    res = ad.get("results", {})
    if not res:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(W2, 64 * MM),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    slots = ["source", "bn", "probe_1", "probe_5", "probe_10", "probe_25", "finetune"]
    xlab = ["0\n(source)", "0\n(BN only)", "1", "5", "10", "25", "all\n(≈86)"]
    picks = ["mtdnet_pt", "resnet18", "mtdnet", "smallcnn"]
    for m in picks:
        key = f"{m}_s0"
        if key not in res:
            continue
        meth = res[key]["methods"]
        vals = [meth.get(s, {}).get("macro_f1", np.nan) for s in slots]
        errs = [meth.get(s, {}).get("macro_f1_sd", 0) for s in slots]
        lab, col = MODEL_STYLE[m]
        ax.errorbar(np.arange(len(slots)), vals, yerr=errs, color=col, lw=1.1,
                    marker="o", ms=3.0, capsize=1.6, elinewidth=0.6, label=lab)
    ax.set_xticks(range(len(slots)))
    ax.set_xticklabels(xlab, fontsize=6.2)
    ax.set_xlabel("labelled PlantDoc images per class")
    ax.set_ylabel("macro-F1 on held-out PlantDoc folds")
    ax.set_ylim(0, 0.85)
    ax.legend(frameon=False, loc="upper left", handlelength=1.3)
    ax.set_title("what it takes to close the gap", fontsize=7.2)
    panel(ax, "A")

    ax = axes[1]
    keys = ["acc15", "macro_f1", "crop_acc"]
    labs = ["15-way acc.", "macro-F1", "crop acc."]
    x = np.arange(len(picks))
    w = 0.26
    for k, (key, lab) in enumerate(zip(keys, labs)):
        vals = []
        for m in picks:
            ent = res.get(f"{m}_s0", {}).get("methods", {}).get("finetune", {})
            vals.append(ent.get(key, np.nan))
        ax.bar(x + (k - 1) * w, vals, width=w, color=[C["blue"], C["green"], C["orange"]][k],
               label=lab, lw=0, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_STYLE[m][0].replace(" ", "\n", 1) for m in picks],
                       rotation=30, ha="right", fontsize=6.2)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score after adaptation")
    ax.legend(frameon=False, loc="upper left", handlelength=1.1)
    ax.set_title("after fine-tuning on all target labels", fontsize=7.2)
    panel(ax, "B", dx=-0.18)
    return save(fig, "Figure_6", outdir)


# --------------------------------------------------------------------------- #
# Figure 7 - corruption grid
# --------------------------------------------------------------------------- #
def fig7_robust(outdir):
    # multi-seed benchmark if present, else the original single-seed pass
    ms = load_json("runs/robustness_ms.json", {})
    rob = load_json("runs/robustness.json", {})
    if not rob and not ms:
        return None
    conds = ["clean", "noise_s1", "noise_s2", "blur_s1", "blur_s2",
             "brightness_s2", "contrast_s2", "occlusion_s1", "occlusion_s2"]
    labels = ["clean", "noise\nσ=0.04", "noise\nσ=0.08", "blur\n3×3", "blur\n5×5",
              "bright\n+60%", "contrast\n−70%", "occl.\n18%", "occl.\n28%"]

    cells: dict[str, dict[str, list[float]]] = {}
    for run, rec in {**rob, **ms}.items():
        model, seed = run.rsplit("_s", 1)
        if not seed.isdigit():
            continue
        for c in conds:
            if c in rec:
                cells.setdefault(model, {}).setdefault(c, []).append(rec[c])
    order = [m for m in ORDER if m in cells]
    if not order:
        return None
    M = np.array([[float(np.mean(cells[m][c])) if cells[m].get(c) else np.nan
                   for c in conds] for m in order])
    SD = np.array([[float(np.std(cells[m][c], ddof=1)) if len(cells[m].get(c, [])) &gt; 1
                    else 0.0 for c in conds] for m in order])

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(W2, 64 * MM),
        gridspec_kw={"width_ratios": [2.35, 1.0], "wspace": 0.42})
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(labels, fontsize=5.9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([MODEL_STYLE[m][0] for m in order], fontsize=6.4)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                txt = f"{M[i, j]:.2f}" + (f"\n±{SD[i, j]:.2f}" if SD[i, j] &gt; 0.0005 else "")
                ax.text(j, i, txt, ha="center", va="center", fontsize=4.9,
                        color="white" if M[i, j] &lt; 0.5 else C["black"])
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.012)
    cb.set_label("mean accuracy", fontsize=6.4)
    cb.ax.tick_params(labelsize=5.8)
    ax.set_xlabel("synthetic degradation (severity)", fontsize=7)
    panel(ax, "A", dx=-0.26, dy=1.04)

    # panel B: clean vs corrupted, mean +- sd over seeds
    y = np.arange(len(order))[::-1]
    clean_mu, clean_sd, rob_mu, rob_sd = [], [], [], []
    for m in order:
        cl = cells[m]["clean"]
        clean_mu.append(float(np.mean(cl)))
        clean_sd.append(float(np.std(cl, ddof=1)) if len(cl) &gt; 1 else 0.0)
        n_seed = max(len(v) for v in cells[m].values())
        per_seed = [np.mean([cells[m][c][k] for c in conds[1:]
                             if c in cells[m] and k &lt; len(cells[m][c])])
                    for k in range(n_seed)]
        rob_mu.append(float(np.mean(per_seed)))
        rob_sd.append(float(np.std(per_seed, ddof=1)) if n_seed &gt; 1 else 0.0)
    bx.barh(y + 0.19, clean_mu, height=0.34, color=[MODEL_STYLE[m][1] for m in order],
            alpha=0.35, edgecolor="none")
    bx.barh(y - 0.19, rob_mu, height=0.34, color=[MODEL_STYLE[m][1] for m in order],
            edgecolor="none")
    bx.errorbar(clean_mu, y + 0.19, xerr=clean_sd, fmt="none", ecolor=C["black"],
                elinewidth=0.7, capsize=1.6)
    bx.errorbar(rob_mu, y - 0.19, xerr=rob_sd, fmt="none", ecolor=C["black"],
                elinewidth=0.7, capsize=1.6)
    bx.set_yticks(y)
    bx.set_yticklabels([])
    bx.set_xlim(0, 1.02)
    bx.set_xlabel("accuracy", fontsize=7)
    bx.axvline(0.5, color=C["grey"], lw=0.4, ls=":")
    for yy, v in zip(y - 0.19, rob_mu):
        bx.text(min(v + 0.02, 0.92), yy, f"{v:.2f}", va="center", fontsize=5.6)
    bx.text(0.02, y[-1] + 0.62, "clean", fontsize=6, color=C["grey"])
    bx.text(0.02, y[-1] + 0.18, "corrupted (mean of 8)", fontsize=6, color=C["black"])
    panel(bx, "B", dx=-0.30, dy=1.04)
    return save(fig, "Figure_7", outdir)


# --------------------------------------------------------------------------- #
# Figure 8 - Grad-CAM
# --------------------------------------------------------------------------- #
def fig8_cam(outdir, size=112):
    from analyze import cam_layer, gradcam
    meta, _, te = load_cache("cache")
    picks = ["resnet18", "mtdnet_pt", "mtdnet"]
    mods, layers = [], []
    for m in picks:
        p = os.path.join("runs", f"{m}_s0", "metrics.json")
        if not os.path.exists(p):
            continue
        j = load_json(p)
        mod = build_model(j["model"], meta["n_classes"], meta["n_species"])
        mod.load_state_dict(torch.load(os.path.join("runs", f"{m}_s0", "model.pt"),
                                       map_location="cpu", weights_only=True))
        mod.eval()
        mods.append(mod)
        layers.append(cam_layer(mod, j["model"]))
    if not mods:
        return None
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    # four visually distinct lesions (one per crop where possible)
    want = [0, 9, 11, 4]
    idx = []
    for c in want:
        w = np.where(te.labels == c)[0]
        if len(w):
            idx.append(int(w[0]))
    fig, axes = plt.subplots(1 + len(mods), len(idx),
                             figsize=(W2 * 0.92, (1 + len(mods)) * 38 * MM))
    axes = np.atleast_2d(axes)
    for j, i in enumerate(idx):
        img = torch.from_numpy(te.images[i]).permute(2, 0, 1).float() / 255.0
        img = torch.nn.functional.interpolate(img[None], size=(size, size),
                                              mode="bilinear", align_corners=False)
        base = img[0].permute(1, 2, 0).numpy()
        axes[0, j].imshow(base)
        axes[0, j].set_title(short(meta["class_names"][te.labels[i]]), fontsize=6.0)
        x = (img - mean) / std
        for r, (mod, layer) in enumerate(zip(mods, layers), start=1):
            if layer is None:
                continue
            cam, _ = gradcam(mod, x, layer)
            axes[r, j].imshow(base)
            axes[r, j].imshow(cam, cmap="magma", alpha=0.45)
    for r in range(axes.shape[0]):
        axes[r, 0].set_ylabel("input" if r == 0 else MODEL_STYLE[picks[r - 1]][0],
                              fontsize=6.2)
    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    return save(fig, "Figure_8", outdir)


# --------------------------------------------------------------------------- #
# Graphical abstract
# --------------------------------------------------------------------------- #
def graphical_abstract(outdir):
    fig, ax = plt.subplots(figsize=(W2, 52 * MM))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    def card(x, y, w, h, title, body, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8",
                                    fc="white", ec=col, lw=0.9))
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=7.2,
                fontweight="bold", color=col)
        ax.text(x + w / 2, y + h - 4.4, body, ha="center", va="top", fontsize=6.3,
                linespacing=1.35)

    card(1, 3, 30, 24, "Question",
         "Does decoupling texture from\nappearance, or ImageNet\npre-training, drive accuracy\nin leaf-disease recognition?", C["blue"])
    card(35, 3, 30, 24, "Design",
         "7 models, 3 seeds, 17 + 4 runs\nMcNemar + bootstrap + Holm\n2 datasets, 5-fold target CV\ncalibration and corruption", C["green"])
    card(69, 3, 30, 24, "Answer",
         "Structure helps only without\npre-training (+4.2 pt); the gain\nvanishes on a pre-trained trunk\nand does not survive field shift", C["verm"])
    for x in (31, 65):
        ax.add_patch(FancyArrowPatch((x + 0.4, 15), (x + 3.6, 15), arrowstyle="-|&gt;",
                                     mutation_scale=9, lw=0.9, color=C["grey"]))
    return save(fig, "Graphical_Abstract", outdir)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["all"])
    ap.add_argument("--outdir", default="report/figures/pub")
    a = ap.parse_args()
    want = set(a.which)
    todo = {
        "1": fig1_arch, "2": fig2_training, "3": fig3_classes, "4": fig4_forest,
        "5": fig5_cross, "6": fig6_adapt, "7": fig7_robust, "8": fig8_cam,
        "ga": graphical_abstract,
    }
    for k, fn in todo.items():
        if "all" in want or k in want:
            try:
                p = fn(a.outdir)
                print("ok", k, p)
            except Exception as e:                        # keep going
                print("FAIL", k, type(e).__name__, e)


if __name__ == "__main__":
    main()