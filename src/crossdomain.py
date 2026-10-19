"""Zero-shot cross-domain evaluation on PlantDoc.

PlantVillage is lab-style: one leaf, uniform background, studio light.
PlantDoc is field-style: cluttered background, several leaves per frame,
varying exposure, phone cameras.  A model trained *only* on PlantVillage and
tested *directly* on PlantDoc measures how much of its accuracy came from the
easy acquisition conditions rather than from the lesion itself.

Protocol
--------
1. PlantDoc category names are mapped onto the 15 PlantVillage classes; only
   images whose PlantDoc label has a counterpart in the training taxonomy are
   kept (the rest belong to crops we never trained on).
2. Three views are reported, because they isolate different failure modes:

   acc15      full 15-way accuracy (includes classes absent from PlantDoc)
   acc10      renormalised over the 10 classes that actually occur in the
              mapped PlantDoc subset -- removes the "can never be right" penalty
   within     decision restricted to the true crop's classes -&gt; measures lesion
              recognition given that the crop is known
   crop_acc   accuracy of the *crop* implied by the predicted class -&gt; measures
              whether the model still knows what plant it is looking at

If a model's acc15 collapses but crop_acc stays high, the failure is in lesion
recognition, not in crop recognition; if crop_acc collapses too, the model was
riding on background/appearance cues.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_cache  # noqa: E402
from models import build_model  # noqa: E402
from train import LeafDS  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

PD_FILES = [
    "data/pd_train-00000.parquet",
    "data/pd_train-00001.parquet",
    "data/pd_test-00000.parquet",
]

# PlantDoc category (normalised) -&gt; PlantVillage class index
CATEGORY_MAP = {
    "apple scab leaf": 0,
    "apple rust leaf": 2,
    "apple leaf": 3,
    "blueberry leaf": 4,
    "cherry leaf": 6,
    "corn gray leaf spot": 7,
    "corn rust leaf": 8,
    "corn leaf blight": 9,
    "grape leaf": 14,
    "grape leaf black rot": 11,
}
CROP_OF = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 2,
           7: 3, 8: 3, 9: 3, 10: 3, 11: 4, 12: 4, 13: 4, 14: 4}


def norm_cat(s: str) -&gt; str:
    return " ".join(s.replace("_", " ").lower().split())


def softmax(z: np.ndarray) -&gt; np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def build_pd_cache(out: str = "cache_pd", size: int = 144) -&gt; dict:
    os.makedirs(out, exist_ok=True)
    xs, ys, kept_cat = [], [], []
    unmapped = collections.Counter()
    multi = 0
    for p in PD_FILES:
        if not os.path.exists(p):
            continue
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=64, columns=["image", "objects"]):
            for rec in b.to_pylist():
                obj = rec.get("objects") or {}
                cats = [c for c in (obj.get("category") or []) if c]
                if not cats:
                    continue
                top, cnt = collections.Counter(cats).most_common(1)[0]
                if cnt &lt; 0.5 * len(cats):
                    multi += 1
                    continue
                key = norm_cat(top)
                if key not in CATEGORY_MAP:
                    unmapped[top] += 1
                    continue
                cell = rec["image"]
                raw = cell.get("bytes") if isinstance(cell, dict) else cell
                if not raw:
                    continue
                im = Image.open(io.BytesIO(raw)).convert("RGB").resize((size, size),
                                                                      Image.BILINEAR)
                xs.append(np.asarray(im, dtype=np.uint8))
                ys.append(CATEGORY_MAP[key])
                kept_cat.append(top)
    if not xs:
        raise RuntimeError("no PlantDoc image mapped")
    x = np.stack(xs)
    y = np.asarray(ys, dtype=np.int64)
    np.save(os.path.join(out, "x.npy"), x)
    np.save(os.path.join(out, "y.npy"), y)
    info = {
        "n": int(len(y)),
        "per_class": {int(k): int(v) for k, v in
                      zip(*np.unique(y, return_counts=True))},
        "raw_categories": {k: int(v) for k, v in
                           collections.Counter(kept_cat).most_common()},
        "unmapped_categories": {k: int(v) for k, v in unmapped.most_common()},
        "skipped_multilabel": int(multi),
        "img_size": size,
    }
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


def load_pd(cache_pd: str):
    with open(os.path.join(cache_pd, "meta.json"), encoding="utf-8") as f:
        info = json.load(f)
    return info, np.load(os.path.join(cache_pd, "x.npy")), np.load(os.path.join(cache_pd, "y.npy"))


@torch.no_grad()
def infer(model, x, size=112):
    ds = LeafDS(x, np.zeros(len(x), dtype=np.int64), np.zeros(len(x), dtype=np.int64),
                False, size)
    ld = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    out = []
    model.eval()
    for xb, _, _ in ld:
        out.append(model(xb)["disease"].float())
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--cache", default="cache")
    p.add_argument("--cache-pd", default="cache_pd")
    p.add_argument("--out", default="report/crossdomain.json")
    p.add_argument("--size", type=int, default=112)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    meta, _, _ = load_cache(args.cache)
    if args.rebuild or not os.path.exists(os.path.join(args.cache_pd, "meta.json")):
        info = build_pd_cache(args.cache_pd)
        print("built cache:", json.dumps({k: info[k] for k in ("n", "per_class")},
                                         ensure_ascii=False))
    pd_info, x, y = load_pd(args.cache_pd)
    present = sorted(set(int(v) for v in y))
    print(f"PlantDoc mapped subset: {len(y)} images, classes {present}")

    # group runs by model, then average the softmax over seeds -&gt; one entry per
    # model, computed with the same "3 runs" budget as the significance study
    groups: dict[str, list[str]] = {}
    for r in args.runs:
        with open(os.path.join(r, "metrics.json"), encoding="utf-8") as f:
            mj = json.load(f)
        groups.setdefault(mj["model"], []).append(r)

    results = {}
    for model_name, dirs in groups.items():
        per_seed = []
        mj = None
        for r in sorted(dirs):
            with open(os.path.join(r, "metrics.json"), encoding="utf-8") as f:
                mj = json.load(f)
            model = build_model(mj["model"], meta["n_classes"], meta["n_species"])
            model.load_state_dict(torch.load(os.path.join(r, "model.pt"), map_location="cpu"))
            per_seed.append(softmax(infer(model, x, args.size)))
        probs = np.mean(per_seed, 0)
        name = model_name
        # per-seed spread of the headline number (single-run wobble)
        seed_acc15 = [float((p.argmax(1) == y).mean()) for p in per_seed]

        pred15 = probs.argmax(1)
        acc15 = float((pred15 == y).mean())
        # renormalised over the classes that actually occur
        p10 = probs[:, present]
        pred10 = np.asarray(present)[p10.argmax(1)]
        acc10 = float((pred10 == y).mean())
        # within-crop, but only for crops that have &gt;1 class in this subset
        # (otherwise the metric is trivially 1.0 and inflates the mean)
        ok, n_w, chance = 0, 0, 0.0
        for i, yi in enumerate(y):
            cand = [c for c in present if CROP_OF.get(c) == CROP_OF[yi]]
            if len(cand) &lt; 2:
                continue
            n_w += 1
            chance += 1.0 / len(cand)
            ok += int(cand[int(probs[i, cand].argmax())] == yi)
        within = float(ok / n_w) if n_w else 0.0
        within_chance = float(chance / n_w) if n_w else 0.0
        # crop accuracy implied by the full prediction
        crop_correct = np.array([CROP_OF[int(t)] == CROP_OF[int(t_)]
                                 for t, t_ in zip(pred15, y)])
        crop_acc = float(crop_correct.mean())
        y_crop = np.array([CROP_OF[int(v)] for v in y])
        crop_macro = float(np.mean([crop_correct[y_crop == c].mean()
                                    for c in np.unique(y_crop)]))
        per_class = {}
        for c in present:
            m = y == c
            cand = [k for k in present if CROP_OF[k] == CROP_OF[c]]
            per_class[str(c)] = {
                "n": int(m.sum()),
                "recall15": float((pred15[m] == c).mean()),
                "recall_within": float(np.mean([
                    c == max(cand, key=lambda k: probs[i, k]) for i in np.where(m)[0]])),
            }
        results[name] = {
            "model": mj["model"], "n_seeds": len(dirs), "params": mj["params"],
            "acc15": acc15, "acc10": acc10, "within_crop": within,
            "within_n": int(n_w), "within_chance": within_chance,
            "crop_acc": crop_acc, "crop_macro": crop_macro,
            "acc15_seeds": seed_acc15,
            "acc15_seed_sd": float(np.std(seed_acc15, ddof=1)) if len(seed_acc15) &gt; 1 else 0.0,
            "per_class": per_class,
            "mean_conf": float(probs.max(1).mean()),
        }
        print(f"==&gt; {name:22s} n={len(dirs)} acc15={acc15:.3f} acc10={acc10:.3f} "
              f"within={within:.3f}(ch={within_chance:.2f},n={n_w}) "
              f"crop={crop_macro:.3f} conf={results[name]['mean_conf']:.3f}",
              flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"pd_info": pd_info, "results": results}, f, ensure_ascii=False, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()