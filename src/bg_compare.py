"""Compare the background-augmented MTD-P runs against the baseline on PlantDoc.

Reads report/crop_diagnosis.json (baseline) and report/crop_diagnosis_bg.json
(the --bg-aug --crop-balance retrain), plus per-run metrics.json for the
source-domain numbers, and writes report/bg_aug_compare.json.
"""

from __future__ import annotations

import json

import numpy as np

BASE = json.load(open("report/crop_diagnosis.json", encoding="utf-8"))["models"]["mtdnet_pt"]
NEW = json.load(open("report/crop_diagnosis_bg.json", encoding="utf-8"))["models"]["bg_mtdnet_pt"]


def src_stats(prefix):
    f1, acc = [], []
    for s in range(3):
        m = json.load(open(f"runs/{prefix}_s{s}/metrics.json", encoding="utf-8"))
        f1.append(m["test_macro_f1"])
        acc.append(m["test_acc"])
    return float(np.mean(f1)), float(np.std(f1, ddof=1)), float(np.mean(acc))


def main():
    f1b, sdb, accb = src_stats("mtdnet_pt")
    f1n, sdn, accn = src_stats("bg_mtdnet_pt")
    crops = ["apple", "blueberry", "cherry", "maize", "grape"]
    out = {
        "source_test_macro_f1": {"baseline": [f1b, sdb], "bg_aug": [f1n, sdn]},
        "source_test_acc": {"baseline": accb, "bg_aug": accn},
        "plantdoc": {},
        "per_crop_recall": {},
    }
    for k in ("acc15", "crop_acc", "crop_macro", "top2_crop", "oracle_crop_acc"):
        out["plantdoc"][k] = {"baseline": BASE[k], "bg_aug": NEW[k],
                              "delta": NEW[k] - BASE[k]}
    for c in crops:
        out["per_crop_recall"][c] = {
            "baseline": BASE["decompose"][c]["crop_recall"],
            "bg_aug": NEW["decompose"][c]["crop_recall"]}
    out["probe25"] = {"baseline": BASE["probe"]["25"]["crop_acc"],
                      "bg_aug": NEW["probe"]["25"]["crop_acc"]}
    with open("report/bg_aug_compare.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"source test macro-F1 : {f1b:.4f} +- {sdb:.4f}  -&gt;  {f1n:.4f} +- {sdn:.4f}")
    print(f"source test acc      : {accb:.4f}  -&gt;  {accn:.4f}")
    print(f"{'metric':22s}{'baseline':&gt;10s}{'bg_aug':&gt;10s}{'delta':&gt;9s}")
    for k, v in out["plantdoc"].items():
        print(f"{k:22s}{v['baseline']:10.3f}{v['bg_aug']:10.3f}{v['delta']:+9.3f}")
    print("per-crop crop-identity recall:")
    for c in crops:
        b, n = out["per_crop_recall"][c]["baseline"], out["per_crop_recall"][c]["bg_aug"]
        print(f"  {c:10s}{b:8.3f}{n:8.3f}{n - b:+8.3f}")
    print("wrote report/bg_aug_compare.json")


if __name__ == "__main__":
    main()