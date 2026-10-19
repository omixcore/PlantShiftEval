# Reproducibility guide

## Current status

This repository includes Python source, run-level classification metric JSONs, and aggregate numeric reports. It does not include raw images, tensor caches, trained weights, all per-run prediction logits, complete training logs, or raw per-seed per-condition corruption records. Training was **not rerun** during this mirror.

**Important validity caveat:** the historical metric files do not record whether an internal validation split was used. The earlier training entry point defaulted to `--val-frac=0`, and when validation was omitted, the test split was used for per-epoch checkpoint selection. Therefore, the recorded metrics cannot be certified as independent held-out estimates. Treat them as provisional and regenerate the headline comparisons before making definitive claims.

The current `src/train.py` defaults to `--val-frac 0.1`, rejects zero, and refuses to run checkpoint selection without a training-internal validation split. This protects future runs; it does not retroactively fix the historical metrics.

For the corrected control matrix and reporting decisions, see [CORRECTED_EXPERIMENT_PLAN.md](CORRECTED_EXPERIMENT_PLAN.md). Each new run saves `run_manifest.json` before training and marks it completed only after hashing the checkpoint, metrics and logits. It records full arguments, original-cache training/validation indices, input and source hashes, Git revision/dirty status, Python/package versions and target-background access. It is a local audit record, not upstream dataset verification. Console logs and an environment lock must still be archived separately. Output directories must be new or empty to prevent overwriting previous runs. Background augmentation uses target images and must be reported as transductive, even without target labels.

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

Dependencies are currently unpinned. Before submission, record exact Python, PyTorch, torchvision, package, and hardware/CUDA versions.

## Required local data

```text
data/pv_train.parquet
data/pv_test.parquet
data/pd_train-00000.parquet
data/pd_train-00001.parquet
data/pd_test-00000.parquet
```

The PlantVillage loader references Hugging Face dataset `GVJahnavi/Plant_village_subset`. The exact PlantDoc Parquet revision, license, and hashes are not recorded; see [DATA_PROVENANCE.md](DATA_PROVENANCE.md). Obtain the files under their applicable terms and record checksums before running experiments.

## Rebuild cache and rerun a scratch comparison

The default cap in `src/data.py` is aligned to 320 training images per class. After placing the Parquet files in `data/`:

```bash
python src/data.py
for seed in 0 1 2; do
  python src/train.py --model mtdnet --cache cache --out "runs_corrected/mtdnet_s${seed}" --epochs 6 --size 112 --lr 1e-3 --val-frac 0.1 --seed "${seed}"
  python src/train.py --model smallcnn --cache cache --out "runs_corrected/smallcnn_s${seed}" --epochs 6 --size 112 --lr 1e-3 --val-frac 0.1 --seed "${seed}"
done
```

Use new output directories such as `runs_corrected/`; do not overwrite historical metrics. For pretrained comparisons, use `--model mtdnet_pt` and `--model resnet18` with peak LR `3e-4`, and pass `--val-frac 0.1` for every run. Select checkpoints only on the training-internal validation split and evaluate the test set once after training.

The 12-epoch comparison must likewise be regenerated under the corrected split. Its current recorded contrast also changes peak LR from 1e-3 to 2e-3, so it is a combined training-recipe comparison, not an isolated epoch-budget experiment.

## Analysis scripts and missing inputs

- `src/significance.py`: seed-level and ensemble tests; requires per-run `logits.pt` files that are not included in this mirror.
- `src/robustness.py`: creates corruption measurements from local data and trained models.
- `src/robustness_stats.py`: aggregates `runs/robustness_ms.json`; that raw multi-seed, per-condition file is not included. Only the precomputed `report/robustness_summary.json` is tracked, so the complete aggregate cannot currently be rebuilt from raw run records.
- `src/crossdomain.py`: zero-shot PlantDoc transfer.
- `src/calibration.py`: calibration comparisons.
- `src/adaptation.py` / `src/adapt_ft_sweep.py`: exploratory target adaptation.
- `src/analyze.py`: summaries and diagnostic figures.
- `src/make_report.py`: HTML report from aggregate JSON.
- `src/make_manuscript.py`: manuscript package generator; requires local `cache/meta.json`, `cache_pd/meta.json`, figures, and run artifacts not included in the mirror.

## What CI validates

GitHub Actions compiles Python source, validates JSON syntax, and checks consistency between tracked run metrics and summary tables. It does not download datasets, train models, regenerate logits, verify data licenses, or certify that the historical results were obtained with an untouched test set.

## Execute the complete corrected study

Use a CPU PyTorch installation and the pinned scientific dependencies listed in `requirements-study.txt`. From the repository root:

```bash
python src/fetch_study_data.py
python src/prepare_corrected_data.py
python src/prepare_corrected_data.py --target
python src/run_corrected_study.py --execute --workers 2 --threads 4
python src/evaluate_corrected_study.py --adapt --threads 4
python src/summarize_corrected_study.py
```

The training runner generates the 65-cell matrix, preserves console logs and resumes only completed runs whose output hashes verify. Partial outputs require explicit inspection; they are never silently treated as completed. Data preparation refuses to overwrite populated caches. The final summary reports missing cells and exits unsuccessfully while any are missing. The GitHub Actions study workflow performs the same preparation once, then distributes fixed experiment cells and archives checkpoints, logits, logs and evaluation records. Generated partial Markdown is explicitly labeled partial. Final plots and numerical manuscript claims must be reviewed after completion.
