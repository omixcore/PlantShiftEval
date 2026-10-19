"""Clean-room latency benchmark (single process, no competing jobs)."""

from __future__ import annotations

import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_cache  # noqa: E402
from models import build_model  # noqa: E402

torch.set_num_threads(16)
meta, _, _ = load_cache("cache")
out = {}
for d in sorted(glob.glob("runs/*_s0")):
    mp = os.path.join(d, "metrics.json")
    if not os.path.exists(mp):
        continue
    with open(mp, encoding="utf-8") as f:
        m = json.load(f)
    model = build_model(m["model"], meta["n_classes"], meta["n_species"])
    model.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location="cpu"))
    model.eval()
    x = torch.randn(1, 3, 112, 112)
    with torch.inference_mode():
        for _ in range(5):
            model(x)
        t = time.time()
        for _ in range(40):
            model(x)
        ms = (time.time() - t) / 40 * 1000
    out[os.path.basename(d)] = round(ms, 2)
    print(f"{os.path.basename(d):24s} {ms:6.2f} ms/img")

with open("runs/latency.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)