# Results and limitations

## PlantVillage held-out evaluation

| Model | Mean macro-F1 | Across-seed sample SD | Seeds |
|---|---:|---:|---:|
| MTD-Net (scratch) | 0.8632 | 0.0467 | 3 |
| Plain CNN (parameter-matched scratch) | 0.8323 | 0.0079 | 3 |
| MTD-Net without gate | 0.8717 | 0.0242 | 3 |
| MTD-Net without auxiliary species head | 0.8601 | 0.0559 | 3 |
| MTD-P (pretrained ResNet-18 trunk) | 0.9862 | 0.0012 | 3 |
| ResNet-18 | 0.9861 | 0.0021 | 3 |

The scratch MTD-Net minus plain-CNN mean difference is **+3.09 macro-F1 points**. The paired seed-level test is not significant (p = 0.396; n = 3), while the three-seed ensemble's image-level bootstrap interval is positive (Δ macro-F1 = +4.21 points; 95% CI +3.07 to +5.39; McNemar p = 1.21 × 10⁻¹¹). The image-level test conditions on a fixed test set and ensemble; seed-level variability remains high. This does not justify a broad claim of stable architecture superiority.

The no-gate variant has a slightly higher mean macro-F1, but its seed-level contrast is not significant (p = 0.589) and the ensemble difference is near zero with a confidence interval crossing zero. The auxiliary-head comparison is also inconclusive (p = 0.644). MTD-P and ResNet-18 have macro-F1 means of 0.9862 and 0.9861, respectively, with no reliable difference.

## Training-recipe sensitivity

The recorded contrast is 6 epochs at peak LR 1e-3 versus 12 epochs at peak LR 2e-3.

| Model | 6-epoch mean macro-F1 (SD) | 12-epoch / higher-LR mean macro-F1 (SD) |
|---|---:|---:|
| MTD-Net | 0.8632 (0.0467) | 0.9628 (0.0056) |
| Plain CNN | 0.8323 (0.0079) | 0.9428 (0.0090) |

Both scratch models improve substantially and become less variable. Because epoch count and learning rate both change, this does not isolate a pure training-budget effect.

## Synthetic corruption

These are mean accuracies across eight synthetic corruptions, not PlantDoc accuracies.

| Model | Clean accuracy (mean ± SD) | Corruption accuracy (mean ± SD) | Corruption seeds |
|---|---:|---:|---:|
| MTD-Net | 0.8812 ± 0.0354 | 0.5885 ± 0.0587 | 3 |
| Plain CNN | 0.8499 ± 0.0050 | 0.5085 ± 0.0296 | 3 |
| MTD-P | 0.9877 ± 0.0014 | 0.8967 ± 0.0093 | 3 |
| ResNet-18 | 0.9871 ± 0.0017 | 0.8864 ± 0.0064 | 3 |

The MTD-Net–plain-CNN mean corruption-accuracy difference is +8.00 points, with paired-seed SD 6.85 points and only three seeds. This is a robustness signal worth retesting, not proof that the texture branch alone causes the effect.

## Zero-shot PlantDoc transfer

The mapping retains 956 images across 10 categories. Unmapped categories are excluded, and 15-way accuracy penalizes classes absent from PlantDoc.

| Model | PlantDoc 15-way accuracy |
|---|---:|
| MTD-P | 0.2458 |
| ResNet-18 | 0.2385 |
| MTD-Net (scratch) | 0.1036 |
| Plain CNN (scratch) | 0.0973 |

The best model remains at about 24.6% 15-way accuracy: a substantial source-to-field gap under this mapping protocol, not complete deployment validation.

## Calibration and adaptation

On mapped PlantDoc, mean raw ECE is 0.492 for scratch MTD-Net, 0.386 for MTD-P, and 0.326 for ResNet-18. Source-fitted temperature scaling worsens these to 0.602, 0.534, and 0.499. Target-label-fitted oracle scaling is diagnostic only.

The adaptation analysis pools mapped PlantDoc train and test shards and uses stratified five-fold CV. It is not an independent held-out target-test evaluation.

## Remaining limits

- Three seeds provide weak power for small architectural differences.
- Epoch count and learning rate are confounded in the longer-training contrast.
- Gate and auxiliary-head effects are not isolated convincingly.
- Synthetic corruption does not replace independent field-domain validation.
- PlantDoc mapping and dominant-object filtering narrow the target problem.
- The exact PlantDoc Parquet source, license, and checksums are not recorded.
- Dependencies are not locked; raw images, caches, and weights are not included.
