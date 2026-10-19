# Recorded results and limitations

**Status: provisional.** The tables below reproduce the tracked historical metrics; they are not corrected rerun results. The historical `metrics.json` files do not record whether a training-internal validation split was used. The earlier training entry point defaulted to `--val-frac=0` and used the test split for per-epoch checkpoint selection when validation was omitted. We therefore cannot certify that these are unbiased held-out estimates. Current code fixes the future training path, but the experiments must be rerun before definitive claims.

## Recorded PlantVillage metrics

| Model | Mean macro-F1 | Across-seed sample SD | Seeds |
|---|---:|---:|---:|
| MTD-Net (scratch) | 0.8632 | 0.0467 | 3 |
| Plain CNN (parameter-matched scratch) | 0.8323 | 0.0079 | 3 |
| MTD-Net without gate | 0.8717 | 0.0242 | 3 |
| MTD-Net without auxiliary species head | 0.8601 | 0.0559 | 3 |
| MTD-P (pretrained ResNet-18 trunk) | 0.9862 | 0.0012 | 3 |
| ResNet-18 | 0.9861 | 0.0021 | 3 |

The recorded MTD-Net minus plain-CNN mean difference is **+3.09 macro-F1 points**. The paired seed-level test is inconclusive (p = 0.396; n = 3), while the image-level ensemble bootstrap interval is positive (Δ macro-F1 = +4.21 points; 95% CI +3.07 to +5.39; McNemar p = 1.21 × 10⁻¹¹). The image-level analysis is conditional on the recorded test set and ensemble; because test-set checkpoint-selection leakage cannot be ruled out, these inferential statistics are provisional too.

The no-gate variant has a slightly higher recorded mean macro-F1, but its seed-level contrast is not significant (p = 0.589) and the ensemble difference is near zero with a confidence interval crossing zero. The auxiliary-head comparison is inconclusive (p = 0.644). MTD-P and ResNet-18 have recorded macro-F1 means of 0.9862 and 0.9861, respectively; the current evidence does not establish a reliable difference.

## Training-recipe sensitivity

The recorded contrast is 6 epochs at peak LR 1e-3 versus 12 epochs at peak LR 2e-3.

| Model | 6-epoch mean macro-F1 (SD) | 12-epoch / higher-LR mean macro-F1 (SD) |
|---|---:|---:|
| MTD-Net | 0.8632 (0.0467) | 0.9628 (0.0056) |
| Plain CNN | 0.8323 (0.0079) | 0.9428 (0.0090) |

Both scratch models improve and become less variable under the second recipe. Because epoch count and learning rate both change, the comparison does not isolate a pure training-budget effect. These historical scores also require rerunning with the corrected validation protocol.

## Synthetic corruption

These are recorded mean accuracies across eight synthetic corruptions, not PlantDoc accuracies.

| Model | Clean accuracy (mean ± SD) | Corruption accuracy (mean ± SD) | Corruption seeds |
|---|---:|---:|---:|
| MTD-Net | 0.8812 ± 0.0354 | 0.5885 ± 0.0587 | 3 |
| Plain CNN | 0.8499 ± 0.0050 | 0.5085 ± 0.0296 | 3 |
| MTD-P | 0.9877 ± 0.0014 | 0.8967 ± 0.0093 | 3 |
| ResNet-18 | 0.9871 ± 0.0017 | 0.8864 ± 0.0064 | 3 |

The recorded MTD-Net–plain-CNN mean corruption-accuracy difference is +8.00 points, with paired-seed SD 6.85 points and only three seeds. This is a descriptive robustness signal, not proof that the texture branch alone causes the effect. The raw multi-seed per-condition input used to build the aggregate is missing from the repository, so the complete robustness aggregation is not currently reproducible from run-level inputs.

## Zero-shot PlantDoc transfer

The mapping retains 956 images across 10 categories. Unmapped categories are excluded, and 15-way accuracy penalizes classes absent from PlantDoc.

| Model | PlantDoc 15-way accuracy |
|---|---:|
| MTD-P | 0.2458 |
| ResNet-18 | 0.2385 |
| MTD-Net (scratch) | 0.1036 |
| Plain CNN | 0.0973 |

The best recorded model remains at about 24.6% 15-way accuracy, indicating a substantial source-to-target gap under this mapping protocol, not complete deployment validation. These scores should also be regenerated after the test-selection issue is resolved.

## Calibration and adaptation

On mapped PlantDoc, recorded mean raw ECE is 0.492 for scratch MTD-Net, 0.386 for MTD-P, and 0.326 for ResNet-18. Source-fitted temperature scaling worsens these to 0.602, 0.534, and 0.499. Target-label-fitted oracle scaling is diagnostic only.

The adaptation analysis pools mapped PlantDoc train and test shards and uses stratified five-fold CV. It is exploratory, not an independent held-out target-test evaluation.

## Remaining limitations and required next steps

- Rerun all headline models with validation data split only from the training partition, and evaluate the untouched test set once.
- Increase seed count beyond three before claiming small architectural effects.
- Cross epoch count and learning rate to separate their effects.
- Gate and auxiliary-head effects are not isolated convincingly.
- Synthetic corruption does not replace independent field-domain validation.
- PlantDoc mapping and dominant-object filtering narrow the target problem.
- Exact dataset revisions, licenses, checksums, package lock, trained weights, logits, full training logs, and raw per-seed corruption inputs are not included.
- The full uncapped data-scaling arm is not tracked.
