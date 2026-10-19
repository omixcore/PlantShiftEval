# Corrected results and limitations

Evidence: complete 65-run study; all configurations have five training seeds.

| Configuration | Source macro-F1 mean ± SD | Corruption accuracy | Target zero-shot accuracy |
|---|---:|---:|---:|
| MTD-Net | 0.6026 ± 0.0245 | 0.4496 | 0.1478 |
| Plain CNN | 0.5903 ± 0.0250 | 0.3784 | 0.1283 |
| MTD-P | 0.8745 ± 0.0292 | 0.7781 | 0.3370 |
| ResNet-18 | 0.8742 ± 0.0151 | 0.7437 | 0.3109 |

The primary six-epoch scratch contrast is +1.23 macro-F1 percentage points for MTD-Net versus the parameter-matched CNN, with a paired-seed 95% Student-t interval of −3.53 to +5.98 points (five seeds; Holm-adjusted p = 1). This does not establish a stable advantage or practical equivalence. MTD-P versus pretrained ResNet-18 differs by +0.03 points, with a seed interval of −4.41 to +4.47 points. Gate and auxiliary-task benefits are not established by the declared contrasts.

Training configuration has a substantial descriptive effect: MTD-Net means range from 0.6026 at six epochs / LR 1e-3 to 0.7930 at twelve epochs / LR 2e-3; corresponding CNN means range from 0.5903 to 0.7404. The full factorial reports each factor and its interaction rather than attributing the joint change to epochs alone. The four scratch recipe contrasts and their raw per-seed values are retained without selecting a single favorable cell as the primary test.

Source macro-F1 near 0.874 for the pretrained pair coexists with only 0.3370 / 0.3109 target zero-shot accuracy. These are different metrics and datasets, not directly subtractable performance estimates. Synthetic corruption, source-fitted calibration, image-only AdaBN and supervised target fine-tuning are separately reported. The study does not establish semantic texture disentanglement, universal superiority or field reliability.

## Interpretation limits

The cleaned source split contains 2,922 optimizer-training images, 333 validation images and 2,916 official test images. Filtering removes 5,820 training rows in known leaf/identity components overlapping test and 12 further similarity candidates before per-class capping. Class support is uneven; one retained source-cache class has only 16 images. Biological leaf IDs remain missing for 1,604 retained source-training-cache images and 857 source-test images. Image-identity fallback and similarity screening cannot certify full biological or acquisition independence.

The mapped target pool is 946 images after filtering, with 683 target-training, 170 validation and 92 official test images; one duplicate train row is unused. Results cover the selected mapped subset, not the whole PlantDoc population. Five seeds quantify training variability conditional on these splits. The supplementary ensemble Wilson image intervals do not capture acquisition dependence or dataset-selection uncertainty. Removing a branch changes parameter count and does not isolate a texture mechanism. Historical and corrected scores use different training populations and cannot isolate the impact of the old checkpoint-selection problem.

Historical scores are archived separately in [HISTORICAL_RESULTS_AND_LIMITATIONS.md](HISTORICAL_RESULTS_AND_LIMITATIONS.md). See [all configurations](../manuscript/corrected_results.md) and [paired contrasts](../report/corrected_summary/paired_comparisons.json).
