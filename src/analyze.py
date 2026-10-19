"""Post-hoc analysis: summary table + figures.

Figures
-------
fig_training   validation macro-F1 per epoch (mean of seeds, band = min-max)
fig_confusion  normalised confusion matrix of MTD-Net (best seed)
fig_per_class  per-class F1, ResNet18 vs MTD-Net
fig_robust     clean vs corrupted accuracy
fig_cam        Grad-CAM comparison
fig_seeds      per-seed dots + mean/SD  (needs significance.json)
fig_forest     paired-bootstrap CI of the macro-F1 gap (needs significance.json)
fig_cross      PlantDoc zero-shot views (needs crossdomain.json)
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_cache  # noqa: E402
from models import build_model  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 160, "savefig.bbox": "tight"})
PALETTE = ["#888780", "#378ADD", "#1D9E75", "#D85A30", "#7F77DD", "#BA7517", "#C43D8E"]
LABEL = {"mtdnet": "MTD-Net (scratch)", "mtdnet_pt": "MTD-P (pretrained)",
         "mtd_concat": "MTD, no gate", "mtd_single": "MTD, no aux head",
         "resnet18": "ResNet18", "mobilenet_v3_small": "MobileNetV3-S",
         "smallcnn": "plain CNN (same budget)"}


def nice(name: str) -&gt; str:
    return LABEL.get(name, name)


def load_runs(run_dirs):
    out = []
    for r in sorted(run_dirs):
        if not os.path.exists(os.path.join(r, "metrics.json")):
            continue
        with open(os.path.join(r, "metrics.json"), encoding="utf-8") as f:
            m = json.load(f)
        m["dir"] = r
        m["name"] = m["model"]
        out.append(m)
    return out


def fig_training(grouped, path):
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for i, (model, runs) in enumerate(grouped.items()):
        n = max(len(r["history"]) for r in runs)
        mat = np.full((len(runs), n), np.nan)
        for j, r in enumerate(runs):
            for h in r["history"]:
                mat[j, h["epoch"] - 1] = h["val_macro_f1"]
        mu = np.nanmean(mat, 0)
        ep = np.arange(1, n + 1)
        c = PALETTE[i % len(PALETTE)]
        ax.plot(ep, mu, marker="o", ms=2.6, lw=1.4, color=c, label=nice(model))
        if len(runs) &gt; 1:
            ax.fill_between(ep, np.nanmin(mat, 0), np.nanmax(mat, 0), color=c, alpha=0.13, lw=0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val macro-F1")
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(path)
    plt.close(fig)


def fig_confusion(rep, meta, path, which="mtdnet"):
    m = rep.get(which) or next(iter(rep.values()))
    cm = np.array(m["confusion"], dtype=float)
    cm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    labels = [c.replace("___", " | ").replace("__", " | ")[:26] for c in meta["class_names"]]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=5.6)
    ax.set_yticklabels(labels, fontsize=5.6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.colorbar(im, fraction=0.036, pad=0.02, label="fraction")
    fig.savefig(path)
    plt.close(fig)


def fig_robust(rob, path, keep=None):
    names = [n for n in rob if (keep is None or n in keep)]
    clean = [rob[n]["clean"] for n in names]
    robust = [rob[n]["robust_acc"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.bar(x - 0.19, clean, 0.36, color="#B5D4F4", label="clean")
    ax.bar(x + 0.19, robust, 0.36, color="#185FA5", label="corrupted (mean)")
    for i, (c, r) in enumerate(zip(clean, robust)):
        ax.text(i - 0.19, c + 0.012, f"{c:.2f}", ha="center", fontsize=6.2)
        ax.text(i + 0.19, r + 0.012, f"{r:.2f}", ha="center", fontsize=6.2, color="#0C447C")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_s0", "") for n in names], rotation=18, ha="right", fontsize=7.2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.grid(axis="y", alpha=0.18, lw=0.6)
    ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(path)
    plt.close(fig)


def fig_per_class(rep, meta, path, a="resnet18", b="mtdnet"):
    ra = rep[a]
    rb = rep[b]
    cm_a = np.array(ra["confusion"], dtype=float)
    cm_b = np.array(rb["confusion"], dtype=float)
    f1a = 2 * np.diag(cm_a) / np.clip(cm_a.sum(0) + cm_a.sum(1), 1, None)
    f1b = 2 * np.diag(cm_b) / np.clip(cm_b.sum(0) + cm_b.sum(1), 1, None)
    labels = [c.replace("___", "|").replace("__", "|") for c in meta["class_names"]]
    order = np.argsort(f1b)
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.barh(y + 0.19, f1a[order], 0.36, color="#B5D4F4", label=nice(a))
    ax.barh(y - 0.19, f1b[order], 0.36, color="#0F6E56", label=nice(b))
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in order], fontsize=5.8)
    ax.set_xlabel("class F1")
    ax.set_xlim(0, 1.02)
    ax.grid(axis="x", alpha=0.18, lw=0.6)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def gradcam(model, x, layer, cls=None):
    feats, grads = {}, {}
    def fh(_, i, o): feats["v"] = o
    def bh(_, gi, go): grads["v"] = go[0]
    h1 = layer.register_forward_hook(fh)
    h2 = layer.register_full_backward_hook(bh)
    model.zero_grad()
    out = model(x)
    logit = out["disease"]
    c = int(logit.argmax(1)) if cls is None else cls
    logit[0, c].backward()
    h1.remove(); h2.remove()
    w = grads["v"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * feats["v"]).sum(1))[0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = torch.nn.functional.interpolate(cam[None, None], size=x.shape[-2:],
                                          mode="bilinear", align_corners=False)[0, 0]
    return cam.detach().numpy(), c


def cam_layer(model, name):
    if name == "resnet18":
        return model.net.layer4
    if name == "mobilenet_v3_small":
        return model.net.features[-1]
    if name in ("mtdnet", "mtdnet_pt", "mtd_concat", "mtd_single"):
        return model.det[-1]
    return None


def fig_cam(rep, meta, te, path, size=112):
    torch.set_num_threads(8)
    picks = [p for p in ("resnet18", "mtdnet", "mtdnet_pt") if p in rep]
    models, layers = [], []
    for k in picks:
        m = rep[k]
        mod = build_model(m["model"], meta["n_classes"], meta["n_species"])
        mod.load_state_dict(torch.load(os.path.join(m["dir"], "model.pt"), map_location="cpu"))
        mod.eval()
        models.append(mod)
        layers.append(cam_layer(mod, m["model"]))

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    idx = []
    for c in range(meta["n_classes"]):
        w = np.where(te.labels == c)[0]
        if len(w):
            idx.append(int(w[0]))
    idx = idx[:8]

    fig, axes = plt.subplots(1 + len(models), len(idx),
                             figsize=(1.05 * len(idx), 1.05 * (1 + len(models))))
    for j, i in enumerate(idx):
        img = torch.from_numpy(te.images[i]).permute(2, 0, 1).float() / 255.0
        img = torch.nn.functional.interpolate(img[None], size=(size, size),
                                              mode="bilinear", align_corners=False)
        x = (img - mean) / std
        axes[0, j].imshow(img[0].permute(1, 2, 0).numpy())
        axes[0, j].set_title(meta["class_names"][te.labels[i]].split("___")[-1][:14], fontsize=4.6)
        for row, (mod, layer) in enumerate(zip(models, layers), start=1):
            if layer is None:
                continue
            cam, _ = gradcam(mod, x, layer)
            axes[row, j].imshow(img[0].permute(1, 2, 0).numpy())
            axes[row, j].imshow(cam, cmap="magma", alpha=0.45)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    for row, k in enumerate(picks, start=1):
        axes[row, 0].set_ylabel(k, fontsize=6)
    fig.savefig(path)
    plt.close(fig)


def fig_seeds(sig, path):
    models = sig["models"]
    order = sorted(models, key=lambda m: models[m]["macro_f1_mean"])
    fig, ax = plt.subplots(figsize=(5.4, 0.42 * len(order) + 1.0))
    for i, m in enumerate(order):
        s = models[m]
        vals = s["macro_f1_seeds"]
        ax.scatter(vals, [i] * len(vals), s=26, color="#378ADD", zorder=3,
                   edgecolor="white", linewidth=0.6)
        ax.errorbar(s["macro_f1_mean"], i, xerr=s["macro_f1_std"], fmt="o",
                    color="#0C447C", ms=0, capsize=4, lw=1.4, zorder=4)
        ax.text(s["macro_f1_mean"], i + 0.26, f"{s['macro_f1_mean']:.3f}±{s['macro_f1_std']:.3f}",
                ha="center", fontsize=6.4, color="#0C447C")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([nice(m) for m in order], fontsize=7.4)
    ax.set_xlabel("macro-F1 on the held-out PlantVillage test split")
    ax.grid(axis="x", alpha=0.18, lw=0.6)
    ax.set_ylim(-0.6, len(order) - 0.4)
    fig.savefig(path)
    plt.close(fig)


def fig_forest(sig, path, keep=None):
    pairs = [p for p in sig["pairs"]
             if keep is None or {p["a"], p["b"]} &lt;= keep]
    if not pairs:
        return
    pairs.sort(key=lambda p: p["delta_f1_ens"])
    fig, ax = plt.subplots(figsize=(5.6, 0.5 * len(pairs) + 1.1))
    for i, p in enumerate(pairs):
        lo, hi = p["boot_ci_low"], p["boot_ci_high"]
        sig_flag = (lo &gt; 0) or (hi &lt; 0)
        c = "#1D9E75" if sig_flag else "#B0483A"
        ax.plot([lo, hi], [i, i], lw=1.8, color=c, solid_capstyle="round")
        ax.plot(p["delta_f1_ens"], i, "o", ms=5.5, color=c, zorder=3)
        ax.text(hi + 0.002, i, f"{p['delta_f1_ens']*100:+.1f}", va="center", fontsize=6.4, color=c)
    ax.axvline(0, color="#333", lw=0.9, ls="--")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([f"{nice(p['a'])} − {nice(p['b'])}" for p in pairs], fontsize=7)
    ax.set_xlabel("Δ macro-F1 (3-seed ensemble), 95% paired-bootstrap CI")
    ax.grid(axis="x", alpha=0.18, lw=0.6)
    fig.savefig(path)
    plt.close(fig)


def fig_cross(cd, path, order=None):
    res = cd["results"]
    names = order or sorted(res, key=lambda n: -res[n]["acc15"])
    acc15 = [res[n]["acc15"] for n in names]
    within = [res[n]["within_crop"] for n in names]
    crop = [res[n]["crop_macro"] for n in names]
    chance_w = res[names[0]]["within_chance"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.bar(x - 0.27, acc15, 0.25, color="#B5D4F4", label="15-way zero-shot")
    ax.bar(x, within, 0.25, color="#378ADD", label="within-crop (crop known)")
    ax.bar(x + 0.27, crop, 0.25, color="#0F6E56", label="crop accuracy")
    ax.axhline(chance_w, color="#993c1d", lw=0.9, ls=":")
    ax.text(len(names) - 0.5, chance_w + 0.01, "within-crop chance", fontsize=6,
            color="#993c1d", ha="right")
    ax.axhline(0.2, color="#7F77DD", lw=0.9, ls=":")
    ax.text(len(names) - 0.5, 0.21, "crop chance (5 crops)", fontsize=6, color="#7F77DD", ha="right")
    for i, v in enumerate(acc15):
        ax.text(i - 0.27, v + 0.012, f"{v:.2f}", ha="center", fontsize=5.8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_s0", "") for n in names], rotation=18, ha="right", fontsize=7)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("accuracy on PlantDoc")
    ax.grid(axis="y", alpha=0.18, lw=0.6)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--figdir", default="report/figures")
    ap.add_argument("--robust", default="runs/robustness.json")
    ap.add_argument("--robust-summary", default="report/robustness_summary.json")
    ap.add_argument("--sig", default="report/significance.json")
    ap.add_argument("--cross", default="report/crossdomain.json")
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--no-cam", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.figdir, exist_ok=True)
    meta, _, te = load_cache(args.cache)
    runs = load_runs(args.runs)
    grouped = collections.OrderedDict()
    for key in ("resnet18", "mobilenet_v3_small", "mtdnet_pt", "mtdnet",
                "mtd_concat", "mtd_single", "smallcnn"):
        sub = [r for r in runs if r["name"] == key]
        if sub:
            grouped[key] = sorted(sub, key=lambda r: r["seed"])
    for r in runs:  # anything not in the canonical order
        if r["name"] not in grouped:
            grouped[r["name"]] = [r]

    rep = {k: max(v, key=lambda r: r["test_macro_f1"]) for k, v in grouped.items()}

    fig_training(grouped, os.path.join(args.figdir, "fig_training.png"))
    fig_confusion(rep, meta, os.path.join(args.figdir, "fig_confusion.png"))
    if "resnet18" in rep and "mtdnet" in rep:
        fig_per_class(rep, meta, os.path.join(args.figdir, "fig_per_class.png"))
    rob = {}
    if os.path.exists(args.robust):
        with open(args.robust, encoding="utf-8") as f:
            rob = json.load(f)
    rbs = {}
    if os.path.exists(args.robust_summary):
        with open(args.robust_summary, encoding="utf-8") as f:
            rbs = json.load(f)
    rob_plot = {}
    for model, rec in rbs.get("per_model", {}).items():
        rob_plot[model] = {
            "clean": rec.get("clean", {}).get("mean", 0.0),
            "robust_acc": rec.get("robust_acc", {}).get("mean", 0.0),
        }
    if not rob_plot:
        rob_plot = rob
    fig_robust(rob_plot, os.path.join(args.figdir, "fig_robust.png"),
               keep=set(grouped) if rbs.get("per_model") else {f"{m}_s0" for m in grouped})
    if not args.no_cam:
        try:
            fig_cam(rep, meta, te, os.path.join(args.figdir, "fig_cam.png"), args.size)
        except Exception as e:
            print("CAM failed:", repr(e))

    if os.path.exists(args.sig):
        with open(args.sig, encoding="utf-8") as f:
            sig = json.load(f)
        fig_seeds(sig, os.path.join(args.figdir, "fig_seeds.png"))
        fig_forest(sig, os.path.join(args.figdir, "fig_forest.png"),
                   keep=set(sig["models"].keys()))
    if os.path.exists(args.cross):
        with open(args.cross, encoding="utf-8") as f:
            cd = json.load(f)
        ordered = [n for n in ("mtdnet_pt", "resnet18", "mobilenet_v3_small",
                               "mtdnet", "mtd_concat", "mtd_single", "smallcnn")
                   if n in cd["results"]]
        fig_cross(cd, os.path.join(args.figdir, "fig_cross.png"), ordered)

    rows = []
    for model, sub in grouped.items():
        f1s = np.array([r["test_macro_f1"] for r in sub])
        accs = np.array([r["test_acc"] for r in sub])
        eces = np.array([r["ece"] for r in sub])
        sp = [r["species_acc"] for r in sub if r["species_acc"] is not None]
        rr = rbs.get("per_model", {}).get(model, {})
        legacy_rr = rob.get(f"{model}_s0") or rob.get(os.path.basename(rep[model]["dir"]), {})
        clean_value = rr.get("clean", {}).get("mean", legacy_rr.get("clean", 0))
        robust_value = rr.get("robust_acc", {}).get("mean", legacy_rr.get("robust_acc", 0))
        rows.append({
            "model": model, "params(M)": round(sub[0]["params"] / 1e6, 2),
            "n_seeds": len(sub),
            "test_acc": round(float(accs.mean()), 4),
            "macro_f1": round(float(f1s.mean()), 4),
            "macro_f1_sd": round(float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0, 4),
            "ece": round(float(eces.mean()), 4),
            "species_acc": round(float(np.mean(sp)), 4) if sp else "",
            "ms/img": round(float(np.mean([r["ms_per_image"] for r in sub])), 2),
            "robustness_n_seeds": int(rr.get("n_seeds", 1 if legacy_rr else 0)),
            "clean": round(float(clean_value), 4),
            "robust_acc": round(float(robust_value), 4),
        })
    # clean-room latency (measured separately, single process) overrides the
    # latency recorded during training, which was polluted by parallel jobs
    lat_p = os.path.join("runs", "latency.json")
    if os.path.exists(lat_p):
        with open(lat_p, encoding="utf-8") as f:
            lat = json.load(f)
        for r in rows:
            if f"{r['model']}_s0" in lat:
                r["ms/img"] = lat[f"{r['model']}_s0"]
    rows.sort(key=lambda x: -x["macro_f1"])
    with open("report/summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open("report/summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    hdr = "| " + " | ".join(rows[0].keys()) + " |"
    sep = "|" + "---|" * len(rows[0])
    body = "\n".join("| " + " | ".join(str(v) for v in r.values()) + " |" for r in rows)
    with open("report/summary.md", "w", encoding="utf-8") as f:
        f.write(f"{hdr}\n{sep}\n{body}\n")
    print(hdr); print(body)


if __name__ == "__main__":
    main()