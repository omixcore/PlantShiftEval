"""Data loading utilities for PlantVillage parquet dumps.

The parquet files used here come from the HuggingFace dataset
`GVJahnavi/Plant_village_subset` (a PlantVillage distribution).  Each row holds
an encoded image plus an integer class label; class names live in the parquet
schema metadata under the `huggingface` key.

We convert once into compact uint8 numpy caches so that subsequent training
runs never pay the JPEG-decoding cost again.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


@dataclass
class Split:
    images: np.ndarray  # (N, H, W, 3) uint8
    labels: np.ndarray  # (N,) int64
    species: np.ndarray  # (N,) int64


def _class_names(pf: pq.ParquetFile) -&gt; List[str]:
    meta = pf.schema_arrow.metadata or {}
    raw = meta.get(b"huggingface")
    if raw:
        try:
            info = json.loads(raw.decode("utf-8"))
            names = info["info"]["features"]["label"]["names"]
            if isinstance(names, list) and names:
                return [str(n) for n in names]
        except Exception:
            pass
    return []


def _image_column(columns: List[str]) -&gt; str:
    for c in columns:
        if c in ("image", "img", "pixel_values"):
            return c
    if "bytes" in columns:
        return None  # struct column: handled separately
    return columns[0]


def _decode(cell) -&gt; Image.Image:
    if isinstance(cell, dict):
        cell = cell.get("bytes") or cell.get("path")
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell)).convert("RGB")
    if isinstance(cell, str) and os.path.exists(cell):
        return Image.open(cell).convert("RGB")
    raise TypeError(f"unsupported image cell type: {type(cell)}")


def read_split(path: str, size: int, per_class_cap: int | None = None,
               class_names: List[str] | None = None) -&gt; Split:
    pf = pq.ParquetFile(path)
    names = class_names or _class_names(pf)
    top = pf.schema_arrow.names
    col = "image" if "image" in top else top[0]

    imgs, labs = [], []
    counts: dict[int, int] = {}
    for batch in pf.iter_batches(batch_size=128, columns=[col, "label"]):
        df = batch.to_pandas()
        for cell, lab in zip(df[col].values, df["label"].values):
            lab = int(lab)
            if per_class_cap is not None and counts.get(lab, 0) &gt;= per_class_cap:
                continue
            im = _decode(cell)
            if im.size != (size, size):
                im = im.resize((size, size), Image.BILINEAR)
            imgs.append(np.asarray(im, dtype=np.uint8))
            labs.append(lab)
            counts[lab] = counts.get(lab, 0) + 1
    if not imgs:
        raise RuntimeError(f"no images decoded from {path}")

    labels = np.asarray(labs, dtype=np.int64)
    names = names or [str(i) for i in range(int(labels.max()) + 1)]
    species_names = [n.split("___")[0].split("__")[0].strip() for n in names]
    uniq = {s: i for i, s in enumerate(sorted(set(species_names)))}
    species = np.asarray([uniq[species_names[l]] for l in labels], dtype=np.int64)
    return Split(np.stack(imgs), labels, species)


def build_cache(out_dir: str, size: int = 160, per_class_cap: int = 320,
                seed: int = 0) -&gt; dict:
    os.makedirs(out_dir, exist_ok=True)
    root = os.path.dirname(out_dir.rstrip("/\\"))
    train_p = os.path.join(root, "data", "pv_train.parquet")
    test_p = os.path.join(root, "data", "pv_test.parquet")

    names = _class_names(pq.ParquetFile(train_p))
    tr = read_split(train_p, size, per_class_cap, names)
    te = read_split(test_p, size, None, names)

    np.save(os.path.join(out_dir, "train_x.npy"), tr.images)
    np.save(os.path.join(out_dir, "train_y.npy"), tr.labels)
    np.save(os.path.join(out_dir, "train_s.npy"), tr.species)
    np.save(os.path.join(out_dir, "test_x.npy"), te.images)
    np.save(os.path.join(out_dir, "test_y.npy"), te.labels)
    np.save(os.path.join(out_dir, "test_s.npy"), te.species)

    species_names = sorted({n.split("___")[0].split("__")[0].strip() for n in names})
    meta = {
        "class_names": names,
        "species_names": species_names,
        "n_classes": len(names),
        "n_species": len(species_names),
        "train_size": int(len(tr.labels)),
        "test_size": int(len(te.labels)),
        "img_size": size,
        "train_per_class": {names[k]: int(v) for k, v in
                            zip(*np.unique(tr.labels, return_counts=True))},
        "test_per_class": {names[k]: int(v) for k, v in
                           zip(*np.unique(te.labels, return_counts=True))},
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def load_cache(cache_dir: str) -&gt; Tuple[dict, Split, Split]:
    with open(os.path.join(cache_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    tr = Split(np.load(os.path.join(cache_dir, "train_x.npy")),
               np.load(os.path.join(cache_dir, "train_y.npy")),
               np.load(os.path.join(cache_dir, "train_s.npy")))
    te = Split(np.load(os.path.join(cache_dir, "test_x.npy")),
               np.load(os.path.join(cache_dir, "test_y.npy")),
               np.load(os.path.join(cache_dir, "test_s.npy")))
    return meta, tr, te


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m = build_cache(os.path.join(here, "cache"))
    print(json.dumps({k: m[k] for k in ("n_classes", "n_species", "train_size", "test_size")},
                     indent=2))
    print("classes:", m["class_names"][:6], "...")
    print("species:", m["species_names"])