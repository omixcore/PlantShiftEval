# Reproducibility guide

This repository includes Python source, run-level metric JSONs, and aggregate numeric reports. It does not include raw images, tensor caches, or trained weights. The summary was re-aggregated from recorded run metrics and the existing multi-seed corruption summary; training was **not rerun**.

## Environment

Python 3.10 or newer is required.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencies are currently unpinned because the source snapshot contained no environment lock. Pin Python, PyTorch, torchvision, hardware/CUDA, and package versions before submission.

## Required local data

```text
data/pv_train.parquet
data/pv_test.parquet
data/pd_train-00000.parquet
data/pd_train-00001.parquet
data/pd_test-00000.parquet
```

The PlantVillage loader references Hugging Face dataset `GVJahnavi/Plant_village_subset`. The exact PlantDoc Parquet revision, license, and hashes are not recorded; see [DATA_PROVENANCE.md](DATA_PROVENANCE.md).

## Rebuild cache and run a scratch comparison

The default cap in `src/data.py` is aligned to 320 training images per class. After placing the Parquet files in `data/`:

```bash
python src/data.py
for seed in 0 1 2; do
  python src/train.py --model mtdnet --cache cache --out "runs/mtdnet_s${seed}" --epochs 6 --size 112 --lr 1e-3 --seed "${seed}"
  python src/train.py --model smallcnn --cache cache --out "runs/smallcnn_s${seed}" --epochs 6 --size 112 --lr 1e-3 --seed "${seed}"
done
```

For pretrained comparisons use `--model mtdnet_pt` and `--model resnet18` with peak LR `3e-4`. Keep the data cache, seed, command, and output path recorded.

## Main scripts

- `src/significance.py`: paired seed tests, ensemble tests, bootstrap intervals, Holm correction.
- `src/robustness.py` / `src/robustness_stats.py`: synthetic corruption and aggregation.
- `src/crossdomain.py`: zero-shot PlantDoc transfer.
- `src/calibration.py`: calibration comparisons.
- `src/adaptation.py` / `src/adapt_ft_sweep.py`: exploratory target adaptation.
- `src/analyze.py`: summaries and diagnostic figures; uses the multi-seed robustness aggregate.
- `src/make_report.py`: HTML report from aggregate JSON.
- `src/make_manuscript.py`: manuscript generation; requires local cache metadata and run outputs.

CI validates Python syntax, JSON syntax, and summary consistency. It does not download datasets, retrain models, or verify external data licensing.
