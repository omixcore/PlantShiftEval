# PlantShiftEval bibliography audit

**Audit date:** 20 October 2026  
**Manuscript:** `manuscript/PlantShiftEval_corrected_manuscript.md`  
**Reference count:** 50; all entries are cited in the manuscript.

## Screening approach

The list was expanded around the manuscript's actual claims: plant-disease image recognition and datasets; texture and dataset bias; laboratory-to-field transfer; domain adaptation and domain generalization; self-supervised and foundation-model representations; corruption robustness, calibration and uncertainty; and the optimization methods used in the corrected experiments. Recent plant-specific work from 2024–2026 is included alongside foundational sources that define the methods and evaluation concepts.

Each row links to a DOI resolver, official proceedings/preprint record, or the official PlantWild project page. The bibliography distinguishes peer-reviewed articles from arXiv preprints and conference proceedings rather than presenting them as equivalent publication types. The “why retained” column describes the reference's role in this manuscript; it is not a claim that the cited paper validates MTD-Net's results.

## Reference-by-reference relevance and source record

| No. | Short title | Source record | Topic | Why retained |
|---:|---|---|---|---|
| 1 | PlantVillage CNN baseline | [Source record](https://doi.org/10.3389/fpls.2016.01419) | Plant-disease benchmarks | Core supervised plant-disease benchmark; establishes in-domain task. |
| 2 | PlantVillage image repository | [Source record](https://arxiv.org/abs/1511.08060) | Dataset provenance | Dataset provenance and access context. |
| 3 | CNN disease diagnosis | [Source record](https://doi.org/10.1016/j.compag.2018.01.009) | Plant-disease baseline | Early benchmark evidence across multiple disease categories. |
| 4 | Transfer-learning comparison | [Source record](https://doi.org/10.1016/j.compag.2018.03.032) | Transfer learning | Supports the role of initialization/model choice. |
| 5 | PlantDoc field dataset | [Source record](https://doi.org/10.1145/3371158.3371196) | Field dataset | Directly relevant external, non-laboratory evaluation source. |
| 6 | Mobile capture / field imagery | [Source record](https://doi.org/10.1016/j.compag.2018.04.002) | Field capture | Supports acquisition variability in field disease recognition. |
| 7 | Dataset size and variety | [Source record](https://doi.org/10.1016/j.compag.2018.08.013) | Dataset diversity | Direct evidence that dataset composition affects model performance. |
| 8 | Cassava field disease recognition | [Source record](https://doi.org/10.3389/fpls.2017.01852) | Crop-specific recognition | Crop-specific image-based recognition beyond a single benchmark. |
| 9 | Tomato disease and pest detection | [Source record](https://doi.org/10.3390/s17092022) | Plant-disease baseline | Plant-disease CNN baseline in agricultural imagery. |
| 10 | Rice disease recognition | [Source record](https://doi.org/10.1016/j.neucom.2017.06.023) | Crop-specific recognition | Cross-crop evidence for CNN-based disease classification. |
| 11 | Plant-disease image-analysis review | [Source record](https://doi.org/10.1016/j.biosystemseng.2016.01.017) | Review | Background complexity, symptom ambiguity and capture conditions. |
| 12 | Plant-disease CNN review | [Source record](https://doi.org/10.3389/fpls.2019.00941) | Review | Context for architecture-focused disease-recognition studies. |
| 13 | Texture versus shape bias | [Source record](https://arxiv.org/abs/1811.12231) | Texture representation | Motivates testing texture claims rather than inferring them from architecture names. |
| 14 | Dataset bias | [Source record](https://doi.org/10.1109/CVPR.2011.5995347) | Dataset bias | General evidence that dataset-specific cues can limit transfer. |
| 15 | Unfamiliar-environment recognition | [Source record](https://doi.org/10.1007/978-3-030-01270-0_28) | Domain shift | General benchmark evidence for environment-level distribution shift. |
| 16 | Plant disease unsupervised adaptation | [Source record](https://doi.org/10.34133/plantphenomics.0038) | Plant-specific adaptation | Direct plant-specific laboratory-to-field adaptation study. |
| 17 | Cross-domain few-shot learning | [Source record](https://doi.org/10.3389/fpls.2024.1434222) | Few-shot transfer | Plant-specific transfer under limited target labels. |
| 18 | Transformer plus Wasserstein adaptation | [Source record](https://doi.org/10.1016/j.compag.2024.109574) | Domain adaptation | Recent plant-specific hybrid/domain-alignment approach. |
| 19 | PlantWild benchmark | [Source record](https://tqwei05.github.io/PlantWild/) | Field dataset | Open in-the-wild multimodal disease-recognition dataset. |
| 20 | PlantSeg dataset | [Source record](https://arxiv.org/abs/2409.04038) | Field dataset | Field disease imagery with lesion segmentation annotations. |
| 21 | Plant-disease domain generalization benchmark | [Source record](https://doi.org/10.1016/j.engappai.2025.111168) | Domain generalization | Recent public cross-domain benchmark and baseline comparison. |
| 22 | Low-data few-shot plant disease recognition | [Source record](https://doi.org/10.1016/j.compag.2024.108812) | Few-shot transfer | Supports label-scarcity and transfer context. |
| 23 | Self-supervised unseen disease recognition | [Source record](https://doi.org/10.1016/j.neucom.2024.128608) | Self-supervised learning | Directly relevant to learned representations and unseen classes. |
| 24 | Unified self-supervised plant-disease learning | [Source record](https://doi.org/10.3390/electronics14173410) | Self-supervised learning | Recent unlabeled-representation study on lab and field imagery. |
| 25 | Attentive contrastive plant-disease learning | [Source record](https://doi.org/10.1016/j.rineng.2025.103922) | Contrastive learning | Recent contrastive-pretraining study. |
| 26 | Foundation models in agriculture | [Source record](https://doi.org/10.1016/j.compag.2025.110900) | Foundation models | DINOv2, adaptation and distillation in agricultural imagery. |
| 27 | PlantCLR | [Source record](https://doi.org/10.1038/s41598-026-45684-x) | Self-supervised learning | 2026 contrastive self-supervised plant-disease study. |
| 28 | Frozen foundation features | [Source record](https://doi.org/10.1016/j.atech.2026.102531) | Foundation models | 2026 laboratory-to-field feature-transfer study. |
| 29 | Cross-domain reliability gap | [Source record](https://doi.org/10.3389/fpls.2026.1826962) | Reliability and calibration | 2026 plant-specific study of accuracy, confidence and adaptation under shift. |
| 30 | Multi-foundation-model fusion | [Source record](https://doi.org/10.3389/fpls.2026.1860665) | Foundation-model comparison | 2026 in-the-wild plant-disease representation comparison. |
| 31 | ResNet | [Source record](https://doi.org/10.1109/CVPR.2016.90) | CNN architecture | Standard residual CNN baseline family. |
| 32 | Vision Transformer | [Source record](https://arxiv.org/abs/2010.11929) | Transformer architecture | Transformer-based visual representation background. |
| 33 | DINO | [Source record](https://doi.org/10.1109/ICCV48922.2021.00946) | Self-supervised learning | Self-supervised vision-transformer representation learning. |
| 34 | DINOv2 | [Source record](https://arxiv.org/abs/2304.07193) | Foundation models | Modern frozen foundation representation learning. |
| 35 | SimCLR | [Source record](https://proceedings.mlr.press/v119/chen20j.html) | Contrastive learning | Contrastive self-supervised pretraining objective. |
| 36 | BYOL | [Source record](https://arxiv.org/abs/2006.07733) | Self-supervised learning | Self-supervised representation learning objective. |
| 37 | MoCo | [Source record](https://doi.org/10.1109/CVPR42600.2020.00975) | Contrastive learning | Momentum-contrast representation learning. |
| 38 | DomainBed | [Source record](https://arxiv.org/abs/2007.01434) | Domain generalization | Protocol-sensitive domain-generalization benchmark. |
| 39 | WILDS | [Source record](https://proceedings.mlr.press/v139/koh21a.html) | Distribution-shift benchmark | Benchmark suite for naturally occurring distribution shifts. |
| 40 | Domain-generalization survey | [Source record](https://doi.org/10.1109/TPAMI.2022.3195549) | Domain generalization | Framework for separating domain generalization from adaptation. |
| 41 | DANN | [Source record](https://jmlr.org/papers/v17/15-239.html) | Domain adaptation | Representative adversarial domain-alignment method. |
| 42 | Deep CORAL | [Source record](https://doi.org/10.1007/978-3-319-49409-8_35) | Domain adaptation | Representative feature-statistics alignment method. |
| 43 | AdaBN | [Source record](https://arxiv.org/abs/1603.04779) | Image-only adaptation | Directly supports the image-only target-statistics adaptation condition. |
| 44 | Common corruptions benchmark | [Source record](https://arxiv.org/abs/1903.12261) | Corruption robustness | Supports synthetic corruption stress testing. |
| 45 | Covariate-shift adaptation | [Source record](https://proceedings.neurips.cc/paper_files/paper/2020/hash/85690f81aadc1749175c187784afc9ee-Abstract.html) | Covariate-shift adaptation | Shows how unlabeled test-like data changes corruption robustness. |
| 46 | Neural-network calibration | [Source record](https://proceedings.mlr.press/v70/guo17a.html) | Calibration | Supports expected calibration error and temperature-scaling context. |
| 47 | Uncertainty under dataset shift | [Source record](https://arxiv.org/abs/1906.02530) | Uncertainty under shift | Supports interpreting confidence under target shift. |
| 48 | Deep ensembles | [Source record](https://arxiv.org/abs/1612.01474) | Ensemble uncertainty | Supports ensemble uncertainty context. |
| 49 | AdamW | [Source record](https://arxiv.org/abs/1711.05101) | Optimization | Optimizer used in the corrected study. |
| 50 | Super-convergence / large learning rates | [Source record](https://arxiv.org/abs/1708.07120) | Training schedule | Supports learning-rate schedule context. |

## Editorial checks applied

- Removed the previous nine-reference ceiling by integrating 50 references into the introduction, related-work discussion and methods context.
- Prioritized directly relevant plant-disease datasets and laboratory-to-field studies, including PlantDoc, PlantWild and PlantSeg.
- Added 2024–2026 research on domain generalization, self-supervised learning, frozen foundation-model features, cross-domain reliability and multi-backbone representations.
- Retained foundational computer-vision sources only where they explain a concept used in the argument or protocol.
- Kept all numerical claims tied to the corrected result tables; literature results are used as context, not as substitute evidence for this study.
- Preserved the manuscript's direct positioning: the current evidence supports a reproducible evaluation of texture-oriented design choices, with training recipe and initialization effects reported alongside the architecture contrast.
