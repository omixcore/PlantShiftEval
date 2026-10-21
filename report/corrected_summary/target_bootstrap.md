# Target-domain statistical audit

- Images: 92; represented classes: 10
- Bootstrap replicates: 5000; seed: 20261020

| Model | Accuracy | 95% image-bootstrap CI | Supported-class macro-F1 | 95% stratified-bootstrap CI |
|---|---:|---:|---:|---:|
| MTD-Net | 0.1630 | 0.0870–0.2391 | 0.1671 | 0.0960–0.2334 |
| Plain CNN | 0.1196 | 0.0543–0.1957 | 0.1077 | 0.0565–0.1567 |
| MTD-P | 0.3370 | 0.2391–0.4348 | 0.3249 | 0.2340–0.4051 |
| ResNet-18 | 0.3261 | 0.2283–0.4239 | 0.3027 | 0.2184–0.3815 |
| DINOv2-S/14 frozen linear probe | 0.3696 | 0.2717–0.4674 | 0.3628 | 0.2838–0.4345 |


## Paired DINOv2 comparisons

| Baseline | Accuracy difference | 95% paired CI | McNemar exact p | Holm-adjusted p |
|---|---:|---:|---:|---:|
| MTD-Net | +0.2065 | +0.1196–+0.2935 | 0.0003107 | 0.0009322 |
| Plain CNN | +0.2500 | +0.1739–+0.3370 | 1.55e-06 | 6.199e-06 |
| MTD-P | +0.0326 | -0.0652–+0.1304 | 0.7011 | 1 |
| ResNet-18 | +0.0435 | -0.0543–+0.1304 | 0.5413 | 1 |


Per-class support, precision, recall, F1 and class-stratified intervals are in target_per_class_metrics.csv.
Intervals are conditional on the selected holdout and do not establish acquisition independence.
