# Target-domain statistical audit

- Images: 92; represented classes: 10
- Bootstrap replicates: 5000; seed: 20261020

| Model | Accuracy | 95% image-bootstrap CI | Supported-class macro-F1 | 95% stratified-bootstrap CI |
|---|---:|---:|---:|---:|
| MTD-Net | 0.1630 | 0.0870–0.2391 | 0.1671 | 0.0960–0.2334 |
| Plain CNN | 0.1196 | 0.0543–0.1957 | 0.1077 | 0.0565–0.1567 |
| MTD-P | 0.3370 | 0.2391–0.4348 | 0.3249 | 0.2340–0.4051 |
| ResNet-18 | 0.3261 | 0.2283–0.4239 | 0.3027 | 0.2184–0.3815 |


## Paired DINOv2 comparisons

| Baseline | Accuracy difference | 95% paired CI | McNemar exact p | Holm-adjusted p |
|---|---:|---:|---:|---:|


Per-class support, precision, recall, F1 and class-stratified intervals are in target_per_class_metrics.csv.
Intervals are conditional on the selected holdout and do not establish acquisition independence.
