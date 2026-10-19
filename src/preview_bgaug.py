"""Visual check for the background-swap augmentation (run before a long retrain)."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import BgSwapField, build_bg_bank  # noqa: E402


class A:
    bg_cache = "cache_pd"
    bg_sigma = 2.5


def main():
    torch.manual_seed(0)
    bank = build_bg_bank(A)
    aug = BgSwapField(bank, p=1.0)
    x = np.load("cache/train_x.npy", mmap_mode="r")
    y = np.load("cache/train_y.npy")
    idx = np.random.RandomState(3).choice(len(y), 8, replace=False)
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.3))
    for j, i in enumerate(idx):
        img = torch.from_numpy(np.array(x[i])).permute(2, 0, 1).contiguous()
        out = aug(img.clone())
        axes[0, j].imshow(img.permute(1, 2, 0).numpy())
        axes[1, j].imshow(out.permute(1, 2, 0).numpy())
        for r in (0, 1):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
        axes[1, j].set_title(f"y={int(y[i])}", fontsize=7)
    axes[0, 0].set_ylabel("PlantVillage", fontsize=8)
    axes[1, 0].set_ylabel("swapped", fontsize=8)
    fig.tight_layout()
    fig.savefig("report/figures/bgaug_preview.png", dpi=150, bbox_inches="tight")
    print("wrote report/figures/bgaug_preview.png")


if __name__ == "__main__":
    main()