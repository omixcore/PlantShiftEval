# Reproducibility guide

## Completed study and public statistical reproduction

All 65 primary runs completed with five seeds per configuration in [workflow 39265931897](https://github.com/omixcore/PlantShiftEval/actions/runs/39265931897), training revision `bd7cd1d94ff2591fa58dac5eb69f0e66225b8f72`. Every run has recorded Python/package versions, source and input hashes, checkpoint/output hashes and frozen split indices. See `report/corrected_summary/evidence_audit.json` and `provenance/corrected_workflow.json`.

Install NumPy to regenerate source/target classification and calibration metrics directly from the 65 public prediction arrays:

```bash
python src/reproduce_public_results.py
```

The checked publication archive reproduces without disagreements. Accuracy and macro-F1 tolerance is 1e-10; ECE tolerance is 2e-6 for floating-point softmax differences. Corruption averages use the archived eight per-condition accuracy records, not new inference. This route does not require torch or weights. Small platform/thread differences during fresh training are possible; a separate CPU-repeat diagnostic is in `provenance/local_cpu_repeat.json`.

## Full training and inference

Use Python 3.10+ and `requirements-study.txt`. Install its torch/torchvision CPU versions from the official CPU wheel index. The exact realized environment is also preserved per run. The workflow uses two CPU threads per job; use `--threads 2` for the closest match. Data URLs, revisions, hashes and licenses are in `provenance/datasets.json` and `DATASET_NOTICE.md`; raw image data are not committed.

## Execute the complete corrected study

Use a CPU PyTorch installation and the pinned scientific dependencies listed in `requirements-study.txt`. From the repository root:

```bash
python src/fetch_study_data.py
python src/prepare_corrected_data.py
python src/prepare_corrected_data.py --target
python src/run_corrected_study.py --execute --workers 2 --threads 2
python src/evaluate_corrected_study.py --adapt --threads 2
python src/audit_corrected_outputs.py
python src/summarize_corrected_study.py
python src/target_uncertainty.py
python src/export_public_predictions.py --source-root .
python src/reproduce_public_results.py
python src/assemble_corrected_manuscript.py
```

The training runner generates the 65-cell matrix, preserves console logs and resumes only completed runs whose output hashes verify. Partial outputs require explicit inspection; they are never silently treated as completed. Data preparation refuses to overwrite populated caches. The final summary reports missing cells and exits unsuccessfully while any are missing. The GitHub Actions study workflow performs the same preparation once, then distributes fixed experiment cells and archives checkpoints, logits, logs and evaluation records. Generated partial Markdown is explicitly labeled partial. All 65 observed runs are complete; final figures have been inspected. The same completeness guards apply to new replications.

## Checkpoints and final document

The workflow experiment artifacts contain original checkpoints, logits, console logs and evaluation outputs; pretrained artifacts also contain target fine-tuned weights. Separate verified archives group the 55 scratch runs, five MTD-P runs and five ResNet-18 runs. `provenance/checkpoint_archives.json` records archive hashes and sizes. Keep a durable archive copy because GitHub Actions retention is finite. AdaBN statistics can be reconstructed from source weights and the frozen target-training images. Public prediction arrays remain committed in Git for weight-free statistical reproduction.

To export the manuscript after complete results exist:

```bash
python src/export_corrected_docx.py --out PlantShiftEval_corrected_manuscript.docx
```

The DOCX figures use the generated PNGs; editable SVG and PDF figures plus plotting CSVs are in `report/corrected_summary`. Document rendering requires LibreOffice and the document verification tools, separately from numerical reproduction.

## Historical scripts

The old `runs/`, `runs12/`, older summary files and pooled target adaptation scripts are preserved as provisional historical records. They are not inputs to corrected reporting. The old entry point could select on test when validation was absent. The corrected study uses frozen internal source validation and independent official target-test membership; its different training population prevents a direct estimate of the impact of that old bug alone.
