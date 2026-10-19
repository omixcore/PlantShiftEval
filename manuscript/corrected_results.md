# Corrected experimental results

Evidence status: **complete** (65/65 training-and-evaluation pairs).

This document is generated only from corrected run artifacts. It does not replace historical results with hypothetical values. The source data were filtered for known leaf overlap and similarity candidates; incomplete leaf IDs remain a limitation. The external evaluation uses the official mapped PlantDoc test partition, with target training/validation separate.

| Model | Epochs | Peak LR | Seeds | Source macro-F1 (mean ± sample SD) | Target accuracy |
|---|---:|---:|---:|---:|---:|
| mtd_appearance | 6 | 1e-3 | 5 | 0.5932 ± 0.0201 | 0.1109 |
| mtd_concat | 6 | 1e-3 | 5 | 0.6104 ± 0.0277 | 0.1413 |
| mtd_single | 6 | 1e-3 | 5 | 0.6186 ± 0.0268 | 0.1326 |
| mtdnet | 6 | 1e-3 | 5 | 0.6026 ± 0.0245 | 0.1478 |
| mtdnet | 6 | 2e-3 | 5 | 0.6708 ± 0.0102 | 0.1500 |
| mtdnet | 12 | 1e-3 | 5 | 0.7503 ± 0.0168 | 0.1696 |
| mtdnet | 12 | 2e-3 | 5 | 0.7930 ± 0.0149 | 0.1783 |
| mtdnet_pt | 6 | 3e-4 | 5 | 0.8745 ± 0.0292 | 0.3370 |
| resnet18 | 6 | 3e-4 | 5 | 0.8742 ± 0.0151 | 0.3109 |
| smallcnn | 6 | 1e-3 | 5 | 0.5903 ± 0.0250 | 0.1283 |
| smallcnn | 6 | 2e-3 | 5 | 0.6519 ± 0.0150 | 0.1196 |
| smallcnn | 12 | 1e-3 | 5 | 0.6913 ± 0.0116 | 0.1174 |
| smallcnn | 12 | 2e-3 | 5 | 0.7404 ± 0.0140 | 0.1391 |

## Interpretation boundaries

Use the paired intervals and individual seed points to assess the magnitude and stability of effects. A nonsignificant contrast does not establish equivalence. Texture-branch removal changes capacity and cannot establish a unique texture mechanism. Source-test scores, synthetic corruption and independent target-domain performance answer different questions.

AdaBN accesses only target-training images without labels. Supervised fine-tuning uses target-training labels and target-validation checkpoint selection. Both are reported separately from zero-shot transfer. Source-fitted calibration uses only the source validation labels. The small official target holdout limits precision and breadth of field-generalization claims.

Historical metrics cannot be compared directly to this cleaned source protocol because both split validity and retained training composition changed. No claim of universal superiority, semantic disentanglement, deployment readiness or practical equivalence is generated automatically.
