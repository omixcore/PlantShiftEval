# PlantShiftEval

**Testing texture-representation hypotheses in plant disease recognition: training-strategy effects and cross-domain generalization limits**

PlantShiftEval is the evaluation project; **MTD-Net** is the candidate model under study. The paper is positioned as a methodological investigation of whether texture/appearance decoupling provides a stable, transferable benefit—not as a claim that a new network is universally superior.

This repository mirrors the Python scripts, recorded run-level metrics, and key numeric result summaries from the [source MTD-Net repository on Gitee](https://gitee.com/omixcore/MTD-Net) (snapshot accessed 2026-10-19). Raw images, tensor caches, trained weights, generated figure binaries, and the original DOCX package are not included. The metrics are historical; training was **not rerun** during this mirror.

## Recorded results

| Model / protocol | PlantVillage macro-F1 (mean ± sample SD) | Training seeds | Corruption accuracy (mean ± SD) | Corruption seeds |
|---|---:|---:|---:|---:|
| MTD-Net, scratch, 6 epochs, LR 1e-3 | 0.8632 ± 0.0467 | 3 | 0.5885 ± 0.0587 | 3 |
| Parameter-matched plain CNN, scratch, 6 epochs, LR 1e-3 | 0.8323 ± 0.0079 | 3 | 0.5085 ± 0.0296 | 3 |
| MTD-Net without cross-scale gate | 0.8717 ± 0.0242 | 3 | 0.6288 | 1 |
| MTD-P, ImageNet-pretrained ResNet-18 trunk | 0.9862 ± 0.0012 | 3 | 0.8967 ± 0.0093 | 3 |
| ResNet-18, ImageNet-pretrained | 0.9861 ± 0.0021 | 3 | 0.8864 ± 0.0064 | 3 |

### Interpretation, not a victory claim

- At six epochs, MTD-Net's mean macro-F1 is 3.09 points above the plain CNN, but its across-seed SD is 4.67 points. The paired seed-level test does not establish a stable gain (p = 0.396). The positive image-level ensemble bootstrap is conditional on this fixed test set and does not replace more training seeds.
- The no-gate ablation is numerically similar and slightly higher in mean macro-F1. The current ablations do **not** establish an independent benefit from the gate or auxiliary species head.
- MTD-P and ResNet-18 are effectively tied on in-domain macro-F1; the added architecture should not be described as a general improvement over a pretrained baseline.
- The 12-epoch comparison changes both epoch count and peak learning rate (6 epochs / LR 1e-3 versus 12 epochs / LR 2e-3). It demonstrates training-recipe sensitivity, not an isolated epoch-count effect.

## Cross-domain limits

The mapped PlantDoc subset contains 956 images across 10 mapped categories. Zero-shot 15-way accuracy is 0.2458 for MTD-P, 0.2385 for ResNet-18, 0.1036 for scratch MTD-Net, and 0.0973 for the plain CNN. Even the best recorded model transfers poorly under this mapping protocol. Unmapped categories are excluded, so this is not a complete deployment evaluation.

Source-fitted temperature scaling worsens target-domain ECE for MTD-Net, MTD-P, and ResNet-18. Target-label-fitted oracle calibration is only a diagnostic upper bound. The current adaptation script pools mapped PlantDoc train and test shards for stratified five-fold cross-validation; those results are exploratory, not an independent held-out target test.

## Contents and remaining gaps

- `src/`: training, evaluation, reporting, and manuscript-generation scripts.
- `runs/*/metrics.json`, `runs12/*/metrics.json`: run-level metrics, no weights.
- `report/`: aggregate numeric results and statistical outputs.
- `docs/`: research positioning, limitations, provenance, and reproduction steps.
- `tests/` and GitHub Actions: syntax, JSON, and summary-consistency checks.

End-to-end reproduction still requires the exact PlantDoc Parquet source/revision, license and checksums, and a locked software environment. See [DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md) and [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Proposed paper title

**Testing Texture-Representation Hypotheses in Plant Disease Recognition: Training-Strategy Effects and Cross-Domain Generalization Limits**
