# PlantShiftEval

**Testing texture-representation hypotheses in plant disease recognition: training-strategy effects and cross-domain generalization limits**

PlantShiftEval is the evaluation project; **MTD-Net** is the candidate model under study. The proposed paper is a methodological investigation of whether texture/appearance decoupling produces a stable, transferable benefit—not a claim that a new network is universally superior.

This repository mirrors Python scripts, recorded run-level metrics, and numeric result summaries from the [source MTD-Net repository on Gitee](https://gitee.com/omixcore/MTD-Net) (snapshot accessed 2026-10-19). Raw images, tensor caches, trained weights, per-run prediction logits, complete training logs, generated figure binaries, and the original DOCX package are not included. Training was **not rerun** during this mirror.

## Critical validity note: historical results are provisional

A post hoc code audit found that the earlier training entry point defaulted to `--val-frac=0`. When no validation split was supplied, the training loop used the test split for per-epoch checkpoint selection. The tracked historical `metrics.json` files do not record the validation-split setting, so we cannot certify that every recorded score is an independent held-out estimate.

The current `src/train.py` has been corrected: it defaults to a 10% validation split drawn from the training partition and rejects a zero validation fraction. **This code correction does not retroactively validate the historical metrics.** All headline configurations must be rerun with the corrected protocol before definitive performance claims or submission.

## Recorded results (historical; not yet revalidated)

| Model / protocol | PlantVillage macro-F1 (mean ± sample SD) | Training seeds | Corruption accuracy (mean ± SD) | Corruption seeds |
|---|---:|---:|---:|---:|
| MTD-Net, scratch, 6 epochs, LR 1e-3 | 0.8632 ± 0.0467 | 3 | 0.5885 ± 0.0587 | 3 |
| Parameter-matched plain CNN, scratch, 6 epochs, LR 1e-3 | 0.8323 ± 0.0079 | 3 | 0.5085 ± 0.0296 | 3 |
| MTD-Net without cross-scale gate | 0.8717 ± 0.0242 | 3 | 0.6288 | 1 |
| MTD-P, ImageNet-pretrained ResNet-18 trunk | 0.9862 ± 0.0012 | 3 | 0.8967 ± 0.0093 | 3 |
| ResNet-18, ImageNet-pretrained | 0.9861 ± 0.0021 | 3 | 0.8864 ± 0.0064 | 3 |

These are the tracked summaries, **not corrected rerun results**. Their p-values and confidence intervals may also be affected if the test split influenced checkpoint selection.

### Cautious interpretation of the recorded metrics

- At six epochs, MTD-Net's mean macro-F1 is 3.09 points above the plain CNN, but its across-seed SD is 4.67 points; the paired seed-level comparison is inconclusive (p = 0.396, n = 3).
- The no-gate ablation is numerically similar and slightly higher in mean macro-F1. The current ablations do not establish an independent benefit from the gate or auxiliary species head.
- MTD-P and ResNet-18 are effectively tied on the recorded in-domain macro-F1; the added architecture should not be described as a general improvement over a pretrained baseline.
- The 12-epoch comparison changes both epoch count and peak learning rate (6 epochs / LR 1e-3 versus 12 epochs / LR 2e-3). It demonstrates recipe sensitivity, not an isolated epoch-count effect.
- The multi-seed corruption aggregate suggests a positive MTD-Net–plain-CNN difference, but it has only three seeds and cannot establish that the texture pathway itself causes robustness.

## Cross-domain limits

The mapped PlantDoc subset contains 956 images across 10 mapped categories. Recorded zero-shot 15-way accuracy is 0.2458 for MTD-P, 0.2385 for ResNet-18, 0.1036 for scratch MTD-Net, and 0.0973 for the plain CNN. This indicates a substantial source-to-target gap under the current mapping, not complete deployment validation.

Source-fitted temperature scaling worsens target-domain ECE in the recorded analysis. Target-label-fitted oracle calibration is a diagnostic upper bound only. The current adaptation script pools mapped PlantDoc train and test shards for stratified five-fold cross-validation; those results are exploratory, not an independent held-out target test.

## Contents and remaining gaps

The [corrected experiment plan](docs/CORRECTED_EXPERIMENT_PLAN.md) translates the three research questions into matched controls, a five-seed epoch-by-learning-rate factorial study, reporting rules, and an explicit evidence-status table. It is a prospective plan, not a report of completed training. New training runs preserve original-cache validation indices, complete arguments, input/source/output SHA-256 fingerprints and environment metadata in `run_manifest.json`; existing output directories cannot be overwritten.

- `src/`: training, evaluation, reporting, and manuscript-generation scripts.
- `runs/*/metrics.json`, `runs12/*/metrics.json`: run-level classification metrics; no weights or logits.
- `runs/latency.json`, `runs/robustness.json`: recorded latency and single-seed corruption inputs.
- `report/`: aggregate numeric results and statistical outputs.
- `docs/`: research positioning, limitations, provenance, and reproduction steps.
- `tests/` and GitHub Actions: syntax, JSON, and summary-consistency checks.

Full statistical regeneration still requires per-run prediction logits and raw multi-seed, per-condition corruption outputs; the latter are not present as `runs/robustness_ms.json`. End-to-end reproduction also requires the exact PlantVillage and PlantDoc source revisions, licenses, checksums, and a locked software environment. See [DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md) and [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Proposed paper title

**Testing Texture-Representation Hypotheses in Plant Disease Recognition: Training-Strategy Effects and Cross-Domain Generalization Limits**

The paper should not be treated as submission-ready until the historical runs have been regenerated with an untouched test set and the data/software provenance gaps are resolved.
