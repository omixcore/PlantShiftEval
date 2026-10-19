"""Training / evaluation entry point."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_cache  # noqa: E402
from models import build_model, count_params  # noqa: E402


class BgSwapField:
    """Paste the leaf onto a field background (copy-paste augmentation).

    Why: on PlantVillage the background is a uniform studio backdrop, so the
    crop head can learn "grey backdrop + leaf fills frame" instead of leaf
    morphology.  That shortcut is what collapses on PlantDoc, where apple leaves
    get predicted as maize.  Replacing the backdrop with an actual field scene
    (and jittering scale/position) removes the shortcut at training time.

    Procedure
    ---------
    1. estimate the backdrop colour from the 6-pixel border (PlantVillage
       backgrounds are near-uniform, so this is reliable),
    2. take the largest connected component of pixels far from that colour,
       fill holes -&gt; leaf mask,
    3. composite the leaf, scaled 0.7-1.0x and placed at a random offset, onto
       a blurred unlabeled PlantDoc frame.

    Blurring the field frame keeps its colour/illumination statistics while
    suppressing foreign leaf edges, so the pasted leaf stays the only sharp
    object.  If the mask is implausible (too small or the whole frame) the
    sample is returned unchanged.
    """

    def __init__(self, bank: torch.Tensor, p: float = 0.7, thr: float = 42.0):
        self.bank = bank          # uint8, (N,3,H,W), already blurred
        self.p = p
        self.thr = thr

    def _mask(self, arr: np.ndarray) -&gt; np.ndarray:
        b = np.concatenate([arr[:6].reshape(-1, 3), arr[-6:].reshape(-1, 3),
                            arr[:, :6].reshape(-1, 3), arr[:, -6:].reshape(-1, 3)])
        bg = np.median(b.astype(np.float32), 0)
        d = np.linalg.norm(arr.astype(np.float32) - bg, axis=2)
        m = ndimage.binary_closing(d &gt; self.thr, np.ones((5, 5)))
        m = ndimage.binary_fill_holes(m)
        lab, n = ndimage.label(m)
        if n &gt; 1:
            sizes = ndimage.sum(m, lab, range(1, n + 1))
            m = lab == (1 + int(np.argmax(sizes)))
        return m

    def __call__(self, img: torch.Tensor) -&gt; torch.Tensor:
        if random.random() &gt; self.p:
            return img
        arr = img.permute(1, 2, 0).numpy()
        m = self._mask(arr)
        frac = float(m.mean())
        if frac &lt; 0.12 or frac &gt; 0.97:
            return img
        bg = self.bank[random.randrange(len(self.bank))].float()
        H, W = bg.shape[1], bg.shape[2]
        leaf = img.float()
        msk = torch.from_numpy(m).float().unsqueeze(0)
        s = random.uniform(0.7, 1.0)
        if s &lt; 0.999:
            nh, nw = max(8, int(round(H * s))), max(8, int(round(W * s)))
            leaf = F.interpolate(leaf.unsqueeze(0), size=(nh, nw), mode="bilinear",
                                 align_corners=False)[0]
            msk = F.interpolate(msk.unsqueeze(0), size=(nh, nw), mode="bilinear",
                                align_corners=False)[0]
        else:
            nh, nw = H, W
        canvas = bg.clone()
        oy = random.randint(0, max(0, H - nh))
        ox = random.randint(0, max(0, W - nw))
        ch, cw = min(nh, H - oy), min(nw, W - ox)
        mm = msk[:, :ch, :cw]
        canvas[:, oy:oy + ch, ox:ox + cw] = (leaf[:, :ch, :cw] * mm
                                             + canvas[:, oy:oy + ch, ox:ox + cw] * (1 - mm))
        return canvas.clamp(0, 255).to(torch.uint8)


class LeafDS(Dataset):
    def __init__(self, x, y, s, train: bool, size: int = 128, bg=None):
        self.x, self.y, self.s = x, y, s
        common = [T.ToDtype(torch.float32, scale=True),
                  T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        if train:
            # background swap goes first: the mask needs the original backdrop,
            # and the crop afterwards sees a real field scene
            self.tf = T.Compose(([bg] if bg is not None else []) + [
                T.RandomResizedCrop(size, scale=(0.55, 1.0), ratio=(0.8, 1.25)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(20),
                T.ColorJitter(0.25, 0.25, 0.25, 0.05),
                *common,
            ])
        else:
            self.tf = T.Compose([T.Resize(size + 16), T.CenterCrop(size), *common])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = torch.from_numpy(self.x[i]).permute(2, 0, 1).contiguous()
        aug = self.tf(img)
        return aug, int(self.y[i]), int(self.s[i])


def ece_score(probs, labels, bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == labels).astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf &gt; lo) &amp; (conf &lt;= hi)
        if m.sum() &gt; 0:
            ece += (m.mean()) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    logits, ys, aux_ok, aux_n = [], [], 0, 0
    t0 = time.time()
    n = 0
    for xb, yb, sb in loader:
        xb = xb.to(device)
        out = model(xb)
        logits.append(out["disease"].float().cpu())
        ys.append(yb)
        if out.get("species") is not None:
            aux_ok += int((out["species"].argmax(1).cpu() == sb).sum())
            aux_n += len(sb)
        n += len(xb)
    dt = time.time() - t0
    logits = torch.cat(logits).numpy()
    ys = torch.cat(ys).numpy()
    probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
    pred = probs.argmax(1)
    return {
        "acc": float(accuracy_score(ys, pred)),
        "macro_f1": float(f1_score(ys, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(ys, pred, average="weighted", zero_division=0)),
        "ece": ece_score(probs, ys),
        "species_acc": (aux_ok / aux_n) if aux_n else None,
        "ms_per_image": 1000.0 * dt / max(n, 1),
        "logits": logits,
        "labels": ys,
        "confusion": confusion_matrix(ys, pred).tolist(),
    }


def build_bg_bank(args, sigma: float = 2.5) -&gt; torch.Tensor:
    if getattr(args, "bg_gray", False):
        return _gray_bank(args, sigma)
    """Blurred, *unlabeled* PlantDoc frames used as field backdrops.

    Only the pixels are used, never the labels: this is transductive
    augmentation, equivalent to collecting a pile of background photographs.
    """
    return _color_bank(args, sigma)


def _gray_bank(args, sigma):
    """Grayscale backdrops: removes the vegetation-colour shortcut that the
    blurred field frames otherwise reintroduce (see report/bg_aug_compare)."""
    x = np.load(os.path.join(args.bg_cache, "x.npy"))
    out = np.empty_like(x)
    for i in range(len(x)):
        g = ndimage.gaussian_filter(x[i].astype(np.float32), (sigma, sigma, 0))
        lum = g @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        out[i] = np.repeat(lum[:, :, None], 3, axis=2).clip(0, 255).astype(np.uint8)
    bank = torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()
    print(f"[bg] GRAY bank {tuple(bank.shape)}", flush=True)
    return bank


def _color_bank(args, sigma):
    x = np.load(p)
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = ndimage.gaussian_filter(x[i].astype(np.float32),
                                         (sigma, sigma, 0)).clip(0, 255).astype(np.uint8)
    bank = torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()
    print(f"[bg] bank {tuple(bank.shape)} from {p} (sigma={sigma})", flush=True)
    return bank


def train_one(args, meta, tr, te, va=None):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)

    n_dis, n_sp = meta["n_classes"], meta["n_species"]
    model = build_model(args.model, n_dis, n_sp).to("cpu")
    n_par = count_params(model)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    bg = BgSwapField(build_bg_bank(args, args.bg_sigma), p=args.bg_p) if args.bg_aug else None
    tr_ds = LeafDS(tr.images, tr.labels, tr.species, True, args.size, bg=bg)
    te_ds = LeafDS(te.images, te.labels, te.species, False, args.size)
    if args.crop_balance:
        cnt = np.bincount(tr.species, minlength=n_sp).astype(np.float64)
        w = 1.0 / np.clip(cnt, 1, None)[tr.species]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(tr.species), replacement=True)
        tr_ld = DataLoader(tr_ds, batch_size=args.bs, sampler=sampler, num_workers=0,
                           drop_last=True)
        print(f"[balance] crop counts {cnt.astype(int).tolist()} -&gt; per-crop "
              f"weights {np.round(1.0 / cnt, 5).tolist()}", flush=True)
    else:
        tr_ld = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=0,
                           drop_last=True)
    te_ld = DataLoader(te_ds, batch_size=128, shuffle=False, num_workers=0)
    # When a validation split is available, epoch selection uses it and the test
    # split is touched exactly once, after training.
    if va is not None:
        va_ld = DataLoader(LeafDS(va.images, va.labels, va.species, False, args.size),
                           batch_size=128, shuffle=False, num_workers=0)
    else:
        va_ld = te_ld

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, len(tr_ld) * args.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.25)

    hist = []
    best = {"macro_f1": -1.0}
    for ep in range(1, args.epochs + 1):
        model.train()
        run, seen = 0.0, 0
        t0 = time.time()
        for xb, yb, sb in tr_ld:
            out = model(xb)
            loss = crit(out["disease"], yb)
            if out.get("species") is not None and args.aux_w &gt; 0:
                loss = loss + args.aux_w * crit(out["species"], sb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            run += float(loss.detach()) * len(xb)
            seen += len(xb)
        res = evaluate(model, va_ld, "cpu")
        rec = {"epoch": ep, "train_loss": run / max(seen, 1),
               "val_acc": res["acc"], "val_macro_f1": res["macro_f1"],
               "sec": round(time.time() - t0, 1)}
        hist.append(rec)
        print(f"[{args.model} s{args.seed}] ep{ep:02d} loss={rec['train_loss']:.3f} "
              f"acc={res['acc']:.4f} f1={res['macro_f1']:.4f} ({rec['sec']}s)",
              flush=True)
        if res["macro_f1"] &gt; best["macro_f1"]:
            best = {k: v for k, v in res.items()}
            best["epoch"] = ep
            torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))

    if va is not None:
        sel_epoch = best.get("epoch")
        model.load_state_dict(torch.load(os.path.join(args.out, "model.pt"), map_location="cpu"))
        test_res = evaluate(model, te_ld, "cpu")
        best = {k: v for k, v in test_res.items()}
        best["epoch"] = sel_epoch
        print(f"[{args.model} s{args.seed}] test acc={best['acc']:.4f} "
              f"f1={best['macro_f1']:.4f} (selected ep{sel_epoch})", flush=True)
    out = {
        "model": args.model, "seed": args.seed, "params": n_par,
        "val_split": va is not None,
        "epochs": args.epochs, "lr": args.lr, "batch": args.bs,
        "bg_aug": bool(args.bg_aug), "bg_p": args.bg_p,
        "crop_balance": bool(args.crop_balance),
        "test_acc": best["acc"], "test_macro_f1": best["macro_f1"],
        "test_weighted_f1": best["weighted_f1"], "ece": best["ece"],
        "species_acc": best["species_acc"], "ms_per_image": best["ms_per_image"],
        "best_epoch": best.get("epoch"), "history": hist,
        "confusion": best["confusion"],
    }
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    torch.save({"logits": best["logits"], "labels": best["labels"]},
               os.path.join(args.out, "logits.pt"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--cache", default="cache")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.02)
    p.add_argument("--aux-w", type=float, default=0.3)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="cap training set (debug)")
    p.add_argument("--val-frac", type=float, default=0.0,
                   help="hold out this fraction of the training split for epoch selection")
    p.add_argument("--bg-aug", action="store_true",
                   help="paste leaves onto unlabeled field backgrounds (copy-paste aug)")
    p.add_argument("--bg-cache", default="cache_pd", help="source of background frames")
    p.add_argument("--bg-p", type=float, default=0.7, help="probability of background swap")
    p.add_argument("--bg-sigma", type=float, default=2.5, help="blur applied to backgrounds")
    p.add_argument("--bg-gray", action="store_true",
                   help="grayscale the background frames (kill the colour shortcut)")
    p.add_argument("--crop-balance", action="store_true",
                   help="sample crops with equal probability")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta, tr, te = load_cache(args.cache)
    if args.limit:
        idx = np.random.RandomState(0).permutation(len(tr.labels))[:args.limit]
        tr.images, tr.labels, tr.species = tr.images[idx], tr.labels[idx], tr.species[idx]

    va = None
    if args.val_frac &gt; 0:
        from data import Split
        rs = np.random.RandomState(1234)          # fixed, independent of --seed
        perm = rs.permutation(len(tr.labels))
        n_va = int(round(args.val_frac * len(perm)))
        va_idx, tr_idx = perm[:n_va], perm[n_va:]
        va = Split(tr.images[va_idx], tr.labels[va_idx], tr.species[va_idx])
        tr = Split(tr.images[tr_idx], tr.labels[tr_idx], tr.species[tr_idx])
        print(f"[split] train={len(tr.labels)} val={len(va.labels)} test={len(te.labels)}", flush=True)
    res = train_one(args, meta, tr, te, va)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("history", "confusion")}, ensure_ascii=False))


if __name__ == "__main__":
    main()