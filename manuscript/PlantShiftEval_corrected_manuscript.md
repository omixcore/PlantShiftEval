# Testing Texture-Representation Hypotheses in Plant Disease Recognition: Training-Strategy Effects and Cross-Domain Generalization Limits

**Evidence status: complete corrected numerical study; author-reviewed submission materials pending.**

## Abstract

Texture-oriented plant disease classifiers are often evaluated on controlled images, but within-domain accuracy alone cannot identify the contribution of texture-oriented components or predict field transfer. We evaluated MTD-Net in a corrected, traceable protocol comprising 65 fixed configurations, five seeds per configuration, training-internal checkpoint selection and a held-out PlantDoc test set. On the cleaned 15-class source test, six-epoch MTD-Net and a parameter-matched CNN achieved mean macro-F1 values of 0.6026 and 0.5903; the paired difference was +1.23 percentage points (seed-level 95% interval, −3.53 to +5.98). Across the sampled recipes, MTD-Net macro-F1 ranged from 0.6026 to 0.7930, showing that training duration and peak learning rate materially changed source performance. Pretrained MTD-P and ResNet-18 achieved 0.8745 and 0.8742 source macro-F1, but their mean zero-shot accuracies on 92 mapped target-test images were 0.3370 and 0.3109. Image-only adaptation raised these values to 0.3609 and 0.3391; supervised target fine-tuning raised them to 0.6109 and 0.6370. The evidence points to a small, uncertain scratch-architecture contrast, substantial training-recipe and initialization effects, and a persistent controlled-to-field gap. The contribution is a seed-replicated test of architecture and optimization effects under a fixed protocol, not a claim of state-of-the-art field transfer.

## Introduction

Image-based plant disease recognition has advanced rapidly through convolutional neural networks trained on labeled leaf photographs. PlantVillage helped establish large-scale supervised benchmarks [1,2], and subsequent studies reported strong performance across crops and disease categories [3,4]. Field-oriented datasets and comparative analyses have since shown that performance depends on image diversity, capture conditions, symptom presentation and background complexity [5–12]. A model evaluated only on controlled images may learn cues that are predictive within that collection but unreliable when the imaging context changes.

This issue is especially relevant to texture-oriented architectures. Disease symptoms often include local patterns such as lesions, speckling, discoloration and tissue damage, but an image classifier can also exploit leaf shape, color, surrounding vegetation, soil or acquisition artifacts. General computer-vision research demonstrates that CNN predictions can favor texture over shape and that dataset-specific visual regularities can impair transfer across environments [13–15]. Plant-disease methods have also explicitly encoded texture: a texture–color dual-branch network was designed to combine maize-lesion texture with RGB features in complex backgrounds [10], while intensity-invariant texture analysis for citrus Huanglongbing showed both the diagnostic value of local texture and its sensitivity to illumination [51]. These studies establish texture as a plausible cue, not that a generic high-resolution branch isolates lesion texture or that its contribution remains stable across domains.

Plant-disease research has increasingly addressed the laboratory-to-field gap through domain adaptation and cross-domain learning. Unsupervised adaptation has been evaluated across PlantDoc and other disease datasets [16], cross-domain few-shot protocols have tested transfer with limited target labels [17], and hybrid convolution–Transformer models have combined feature extraction with distribution alignment [18]. PlantWild provides an in-the-wild multimodal recognition benchmark [19], while PlantSeg is a peer-reviewed dataset with disease-area segmentation masks from natural imagery [20]. These resources make it possible to distinguish performance on curated laboratory images from performance under more heterogeneous visual conditions.

Recent plant-specific work includes public domain-generalization benchmarks for unseen field domains [21], few-shot recognition under limited labels [22], and self-supervised recognition of unseen crop–disease pairs [23]. Self-supervised methods have also been evaluated on laboratory and in-field images [24,25], while agricultural foundation-model studies compare DINOv2, adaptation and distillation [26]. PlantCLR studies contrastive pretraining followed by supervised fine-tuning across PlantVillage and cassava images [27]. Particularly close to the present task, Nguyen et al. report frozen DINOv2 features evaluated on PlantDoc [28]; Xiang et al. directly examine PlantVillage-to-PlantDoc transfer, calibration, selective prediction, target-aware adaptation and frozen backbones under parent-image-aware partitions [29]; and El Karch et al. compare frozen vision backbones and fusion on PlantWild and PlantDoc [30]. These studies mean that the existence of a laboratory-to-field gap, the usefulness of frozen foundation features, and the limits of simple adaptation are not new claims in themselves.

Here, MTD-Net is treated as a candidate texture-oriented configuration, not as a presumed improvement. We ask three questions: (i) does MTD-Net improve source-domain macro-F1 over a parameter-matched CNN under a fixed split and paired seeds; (ii) how much do epoch count and peak learning rate change the ranking; and (iii) do component contrasts or a pretrained MTD-Net configuration add measurable value relative to a conventional pretrained ResNet-18? We then compare synthetic corruption, zero-shot PlantDoc transfer, image-only adaptation and supervised target fine-tuning as distinct evaluation conditions. The specific contribution is the seed-paired analysis of architecture and training-recipe effects across 65 fixed configurations, with the source protocol and the 92-image, 10-class mapped target holdout made explicit.

## Related work

### Texture cues and architecture attribution

Texture has a long history in plant-disease image analysis, from intensity-invariant descriptors for citrus symptoms [51] to learned texture–color pathways for maize lesions [10]. In broader visual recognition, controlled texture–shape conflicts show that CNN cue preferences depend on training and representation choices [13]. Dataset-bias and unfamiliar-environment studies explain why a cue that is predictive in one collection may not transfer to another [14,15]. These lines of work motivate a texture-focused hypothesis, but they do not establish that a network branch named or designed for high-resolution processing encodes texture exclusively. In this study, the architecture comparisons and ablations measure configuration-level effects; branch removal also changes capacity, so the results do not isolate a unique texture mechanism.

ResNet provides a standard residual CNN backbone [31], while Vision Transformers offer a token-based alternative to convolutional feature hierarchies [32]. Self-supervised approaches including DINO, DINOv2, SimCLR, BYOL and MoCo learn visual representations through different pretraining objectives [33–37]. Plant-specific work on self-supervised learning and frozen foundation features extends this comparison to crop imagery [23–30]. These methods provide relevant alternatives to a new scratch-trained architecture, particularly when the goal is to determine whether custom components add value beyond a strong pretrained representation.

### Domain generalization and adaptation

DomainBed emphasizes that domain-generalization results depend on datasets, model selection and comparable training protocols [38], while WILDS provides benchmarks for naturally occurring distribution shifts [39]. A broad survey organizes domain generalization around representation learning, data augmentation, alignment and related strategies [40]. Domain-adversarial training and Deep CORAL are representative feature-alignment methods [41,42], while adaptive batch normalization (AdaBN) uses target-image statistics without target labels [43]. The plant-specific 2026 reliability study is particularly relevant because it evaluates calibration, selective prediction, adaptation and frozen representations in a controlled-to-field setting [29]. Accordingly, zero-shot transfer, image-only adaptation and supervised target fine-tuning are treated as distinct information conditions rather than as interchangeable scores.

### Robustness, calibration and training strategy

Synthetic corruption benchmarks offer controlled stress tests [44], and covariate-shift adaptation shows that access to unlabeled test-like images can alter corruption robustness [45]. Calibration methods quantify whether confidence tracks empirical correctness [46]; evaluations under dataset shift and deep ensembles further show why accuracy alone is insufficient to describe uncertainty [47,48]. Optimization choices also affect comparisons: AdamW decouples weight decay from the gradient update [49], while large-learning-rate schedules can change convergence behavior [50]. These references motivate the study design, but the magnitude of the epoch and learning-rate effects reported here comes from the corrected experiments, not from transferring numerical findings from other datasets.

## Materials and methods

### Study status and separation from historical experiments

An audit of the historical training entry point found a path in which the source test split could select checkpoints when internal validation was omitted. Historical run files do not preserve enough split metadata to rule out that path. Those scores remain provisional and are not used as corrected estimates. The present protocol also changes the retained training composition through overlap filtering, so numerical differences between historical and corrected tables cannot isolate the effect of fixing checkpoint selection.

The corrected study contains 65 fixed source-training configurations: 40 cells crossing two scratch models, two epoch counts, two peak learning rates and five seeds; 10 cells comparing two pretrained configurations over five seeds; and 15 component-ablation cells. The configuration list is stored in `experiments/corrected_matrix.json`. Seeds are paired by identifier across configurations. This pairing does not imply identical stochastic augmentation or optimization trajectories between architectures.

### Dataset identity, attribution and image lineage

The recovered source files are the 15-class `GVJahnavi/Plant_village_subset` at revision `243d37baa0780d93a8305d64f382bc1d41dffe92`, containing 11,322 training and 2,916 test images before the present filtering. The compatible target mirror is `agyaatcoder/PlantDoc` at revision `ace645742b6af1b0df9aaefc2c9b0c988e141814`, containing 2,342 official training and 236 official test images. Each Parquet file is identified by its upstream SHA-256 checksum. These recovered revisions define the corrected study; they are not asserted to be identical to undocumented historical local files.

All 14,238 source-image byte strings match original PlantVillage Git blobs at repository revision `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`, including copies with changed filenames. The subset card omits a license. The canonical `mohanty/PlantVillage` card at revision `9e97599868962bd0079b8db4b7f1efa9185fa1e7` declares CC BY-SA 3.0; the image-lineage check connects the recovered image content to that original collection. The target mirror and its cited upstream repository declare CC BY 4.0. Attribution and image-processing changes are recorded in `DATASET_NOTICE.md`.

### Source overlap audit and internal validation

We construct identity components using known original leaf-group identifiers, encoded-image hashes, decoded-pixel hashes and image filenames. Known group relationships are joined transitively. All source-training rows in components overlapping the official source test are excluded. A 64-bit difference hash computed on a 9 × 8 grayscale thumbnail provides a conservative additional similarity screen: source-training rows within Hamming distance four of any source-test image are excluded. This threshold is a processing rule, not proof that all flagged images depict the same leaf or that unflagged images are independent.

Filtering removes 5,820 original source-training rows through known leaf/identity overlap and another 12 similarity-candidate rows. Eligible rows are shuffled with seed zero and capped at 320 per class. The resulting cache contains 3,255 rows, including 1,604 without an original leaf-group identifier. Unknown leaves use image-identity components. Internal validation is assigned by shuffling components separately within each class with split seed 1234; approximately 10% of components are held out. The realized split contains 2,922 optimizer-training and 333 validation images. These row assignments are frozen and reused for every corrected model seed.

All 15 classes remain, but filtering produces substantial imbalance: the smallest class retains only 16 cache images. The official source test retains 2,916 images, of which 857 lack a known leaf identifier. The audit establishes separation for available identity information and screening rules; it does not certify biological independence for all unknown leaves or acquisition sessions. The primary result is conditional on this filtered, capped protocol.

### Candidate representations and controls

MTD-Net uses a shared convolutional stem, a downsampled appearance pathway, a higher-resolution pathway with depthwise-separable and dilated blocks, an appearance-conditioned channel/spatial gate, an auxiliary crop-species head and a disease classifier combining both descriptors. These are architectural designations. They do not establish that the higher-resolution branch contains only lesion texture or that appearance and disease information are disentangled.

The scratch comparison uses the repository's plain CNN, with its parameter count reported alongside MTD-Net. Gate removal replaces gating with concatenation. The single-task contrast disables the auxiliary species output and loss while retaining the base wiring. Appearance-only ablation removes the higher-resolution branch and gate and changes the classifier input dimension; it also changes capacity, which limits mechanistic attribution. MTD-P reuses ImageNet-pretrained ResNet-18 features and adds the candidate branch and heads. Its comparison with pretrained ResNet-18 estimates a conditional configuration difference, not an isolated effect of pretraining. Both pretrained implementations use the torchvision ResNet-18 default weight release.

### Optimization and checkpoint selection

Images are cached at 160 × 160 source resolution and transformed to 112 × 112 model inputs. Training uses random resized cropping, horizontal and vertical flips, rotation and color jitter, followed by ImageNet normalization. Evaluation uses resizing and center cropping without stochastic augmentation. AdamW [49] uses weight decay 0.02, a OneCycle learning-rate schedule [50], batch size 64, disease cross-entropy with label smoothing 0.05, and gradient clipping at norm five. The auxiliary loss weight is 0.3 when that output is enabled. Source background swapping and crop-balanced sampling are disabled in the corrected core study.

The scratch factorial crosses 6/12 epochs with peak learning rates 1e-3/2e-3. Component ablations use six epochs and peak learning rate 1e-3. Both pretrained configurations use six epochs and peak learning rate 3e-4. Model seeds are 0–4; Python, NumPy and PyTorch RNGs are initialized, and the data-order generator is independently seeded. The maximum validation macro-F1 selects the checkpoint, with the first checkpoint retained for tied maxima. The source test is evaluated once after training. A runtime test instruments evaluation calls to verify this separation on a small test fixture; it does not substitute for the real-data experiments.

### Corruption, transfer and target-label access

Synthetic evaluation applies eight fixed conditions to the source test: two noise severities, two blur severities, increased brightness, reduced contrast and two occlusion severities. Conditions and the common corruption RNG seed are fixed in `src/robustness.py`. Their mean accuracy is a synthetic stress-test outcome; this is not an ImageNet-C reproduction or a substitute for field-domain evaluation [44].

PlantDoc object-category names are mapped explicitly to ten supported source classes. Images without a mapped dominant category are excluded; tied categories do not define an unambiguous image label. Additional screening excludes two source-similarity candidates and eight rows in identity components with conflicting labels. One target-training duplicate overlapping official target test is unused. The resulting cached pool contains 946 rows, with 683 adaptation-training images, 170 target-validation images and 92 official target-test images. Target-training images within the stated similarity threshold of official target test are excluded; the recovered data contained none after the other filters. Official test membership is preserved after these documented exclusions rather than replaced by pooled cross-validation.

Zero-shot evaluation reads no target images during source training. Image-only adaptation updates batch-normalization running statistics using only the target-training images, without labels or gradient updates; statistics are reset and accumulated in one pass of minibatches of 64. Supervised adaptation starts from the same source checkpoint, fine-tunes disease loss with AdamW at learning rate 1e-4 and weight decay 0.02 for six epochs, and selects the checkpoint on target-validation macro-F1. AdaBN and supervised adaptation are applied to both pretrained configurations for all five seeds. Official target-test labels never choose adaptation epochs or hyperparameters. These three access conditions are reported separately.

### Outcomes, uncertainty and traceability

The primary source outcome is macro-F1. Accuracy, expected calibration error with 15 confidence bins, per-class behavior, synthetic-corruption accuracy and target accuracy are secondary outcomes. Source-fitted temperature scaling minimizes negative log-likelihood on source-validation predictions only [46]; the fitted temperature is then applied to target-test logits without target-label fitting. This evaluates transfer of a source-fitted calibrator, not an oracle target calibrator.

We report all seed points, means and sample standard deviations. Paired seed differences receive Student-t intervals and paired tests, conditional on the frozen dataset; their approximate-normality assumption and five-seed sample size limit precision. Holm adjustment is applied across five architectural/configuration contrasts. A difference-of-differences summarizes the epoch-by-learning-rate interaction for each scratch model. Nonsignificance is not interpreted as equivalence, and seed variability is not treated as target-population sampling uncertainty. The small, selected target holdout limits generalization beyond these images. As an additional descriptive diagnostic, probability-averaged five-seed target ensembles receive Wilson image-level intervals under an independent-image binomial assumption. These intervals are distinct from seed intervals and do not capture unknown acquisition dependencies or population selection.

Every completed run preserves parsed arguments, frozen source indices, input/source fingerprints, package and Python versions, checkpoint, logits and metrics. Output hashes must verify before a run can be resumed as complete. Per-run target predictions and corruption records are saved separately. Aggregation fails while any of the 65 training-and-evaluation pairs is missing; unexecuted cells never become final numerical evidence. Historical tables remain separate.


## Results

| Model | Epochs | Peak LR | Seeds | Source macro-F1 (mean ± sample SD) | Target accuracy |
|---|---:|---:|---:|---:|---:|
| Appearance only | 6 | 1e-3 | 5 | 0.5932 ± 0.0201 | 0.1109 |
| No gate | 6 | 1e-3 | 5 | 0.6104 ± 0.0277 | 0.1413 |
| No auxiliary task | 6 | 1e-3 | 5 | 0.6186 ± 0.0268 | 0.1326 |
| MTD-Net | 6 | 1e-3 | 5 | 0.6026 ± 0.0245 | 0.1478 |
| MTD-Net | 6 | 2e-3 | 5 | 0.6708 ± 0.0102 | 0.1500 |
| MTD-Net | 12 | 1e-3 | 5 | 0.7503 ± 0.0168 | 0.1696 |
| MTD-Net | 12 | 2e-3 | 5 | 0.7930 ± 0.0149 | 0.1783 |
| MTD-P | 6 | 3e-4 | 5 | 0.8745 ± 0.0292 | 0.3370 |
| ResNet-18 | 6 | 3e-4 | 5 | 0.8742 ± 0.0151 | 0.3109 |
| Plain CNN | 6 | 1e-3 | 5 | 0.5903 ± 0.0250 | 0.1283 |
| Plain CNN | 6 | 2e-3 | 5 | 0.6519 ± 0.0150 | 0.1196 |
| Plain CNN | 12 | 1e-3 | 5 | 0.6913 ± 0.0116 | 0.1174 |
| Plain CNN | 12 | 2e-3 | 5 | 0.7404 ± 0.0140 | 0.1391 |

All cells summarize five training seeds. Source macro-F1 and target accuracy are different outcomes; uncertainty is reported separately below.

### Model capacity

| Configuration | Trainable parameters |
|---|---:|
| MTD-Net | 394,042 |
| Plain CNN | 389,936 |
| No gate | 360,116 |
| No auxiliary task | 394,042 |
| Appearance only | 255,412 |
| MTD-P | 11,493,178 |
| ResNet-18 | 11,184,207 |

### Paired configuration contrasts

- MTD-Net minus plain CNN: Δ macro-F1 +1.23 points; seed-level 95% interval -3.53 to +5.98; Holm-adjusted p = 1; n = 5 paired seeds.
- MTD-Net minus no gate: Δ macro-F1 -0.78 points; seed-level 95% interval -2.50 to +0.94; Holm-adjusted p = 1; n = 5 paired seeds.
- MTD-Net minus no auxiliary task: Δ macro-F1 -1.60 points; seed-level 95% interval -3.71 to +0.51; Holm-adjusted p = 0.5129; n = 5 paired seeds.
- MTD-Net minus appearance only: Δ macro-F1 +0.94 points; seed-level 95% interval -1.61 to +3.49; Holm-adjusted p = 1; n = 5 paired seeds.
- MTD-P minus ResNet-18: Δ macro-F1 +0.03 points; seed-level 95% interval -4.41 to +4.47; Holm-adjusted p = 1; n = 5 paired seeds.

Intervals describe training-seed variability conditional on the frozen split. They do not establish practical equivalence or uncertainty over future acquisition settings.
### Training strategy contrasts

For MTD-Net, mean source macro-F1 changed from 0.6026 to 0.7503 between six and twelve epochs at peak LR 1e-3, and from 0.6708 to 0.7930 at peak LR 2e-3. The epoch-by-LR difference-of-differences was -2.55 points (seed-level 95% interval -5.85 to +0.76). These contrasts apply to the declared OneCycle schedules and do not identify a general optimal recipe.

For the plain CNN, mean source macro-F1 changed from 0.5903 to 0.6913 between six and twelve epochs at peak LR 1e-3, and from 0.6519 to 0.7404 at peak LR 2e-3. The epoch-by-LR difference-of-differences was -1.24 points (seed-level 95% interval -3.92 to +1.44). These contrasts apply to the declared OneCycle schedules and do not identify a general optimal recipe.

### Corruption and independent target transfer

| Configuration | Source clean accuracy | Mean corruption accuracy | Official target accuracy |
|---|---:|---:|---:|
| MTD-Net | 0.6800 | 0.4496 | 0.1478 |
| Plain CNN | 0.6566 | 0.3784 | 0.1283 |
| MTD-P | 0.8945 | 0.7781 | 0.3370 |
| ResNet-18 | 0.8859 | 0.7437 | 0.3109 |

These are means over five training seeds. Corruption accuracy averages eight declared conditions; target accuracy uses the same 92 independently held-out mapped target images for each model. The two outcomes must not be substituted for one another.

### Confidence under target shift

| Configuration | Raw target ECE | Target ECE after source fitted temperature |
|---|---:|---:|
| MTD-Net | 0.3988 | 0.4859 |
| Plain CNN | 0.3852 | 0.4993 |
| MTD-P | 0.2626 | 0.3780 |
| ResNet-18 | 0.2477 | 0.3624 |

Temperature fitting used source-validation labels only. ECE changes describe this source-fitted calibrator on the selected target test; no target-label oracle was fitted.

### Target image and label access

| Configuration | Zero shot accuracy | Image only AdaBN accuracy | Supervised fine tuning accuracy |
|---|---:|---:|---:|
| MTD-P | 0.3370 | 0.3609 | 0.6109 |
| ResNet-18 | 0.3109 | 0.3391 | 0.6370 |

All methods use the same held-out target test and five source seeds. The adaptation columns have additional target-domain information and are not zero-shot results. Complete per-seed values and sample SDs accompany the tables in the machine-readable evidence package.

### Conditional target image uncertainty

The five-seed MTD-P probability ensemble correctly classified 31/92 target-test images (accuracy 0.3370; image-level Wilson interval 0.2486 to 0.4383). The five-seed ResNet-18 probability ensemble correctly classified 30/92 target-test images (accuracy 0.3261; image-level Wilson interval 0.2389 to 0.4272). These image-level intervals do not account for possible dependence among images from the same plant, leaf, or acquisition session. The target holdout contains 92 images across ten represented classes, with class support ranging from 3 to 12 images. Per-class precision, recall, F1 and class-stratified bootstrap intervals are provided in report/corrected_summary/target_per_class_metrics.csv.

## Discussion

The matched scratch comparison gives MTD-Net a +1.23-point mean macro-F1 difference over the plain CNN, with a seed-level 95% interval spanning −3.53 to +5.98 points. This is a small and uncertain difference under the tested six-epoch recipe. The gate and auxiliary-task contrasts likewise do not show a consistent benefit. The results therefore do not establish a general architecture advantage; they estimate the effect of the complete candidate configuration under a fixed, reproducible comparison.

Training strategy produced larger descriptive changes than the primary architecture contrast. MTD-Net macro-F1 increased from 0.6026 to 0.7503 when training was extended from six to twelve epochs at peak LR 1e-3, and from 0.6708 to 0.7930 at peak LR 2e-3. The plain CNN also improved under longer schedules. This pattern places optimization budget alongside architecture as a major determinant of the reported score. The factorial design makes the tested epoch and learning-rate effects visible instead of assigning their joint change to a single factor. These results support the tested schedules and split; they do not establish a universal optimal training recipe.

Pretraining changed the performance level more strongly than adding the candidate branch. MTD-P and pretrained ResNet-18 reached nearly identical source macro-F1 values (0.8745 and 0.8742), and the paired contrast was centered near zero. Their zero-shot PlantDoc accuracies were higher than those of the scratch models, yet remained 0.3370 and 0.3109. The comparison is consistent with a central role for initialization and representation quality, but it does not establish that the candidate branch is useless in all settings.

The closest 2026 literature makes the scope of this paper's contribution especially important. Xiang et al. [29] already examine PlantVillage-to-PlantDoc transfer, calibration, selective prediction, target-aware adaptation and frozen backbones with parent-image-aware partitions. Nguyen et al. [28] report that frozen DINOv2 features with a linear probe achieve 51.1 ± 0.7% on PlantDoc under a full 27-class evaluation, and El Karch et al. [30] report 80.23 ± 0.41% on PlantWild and 76.97 ± 1.48% on PlantDoc for a frozen multi-backbone fusion configuration. Those figures are not directly comparable to the present results: this study uses a 15-class source subset, a 10-class target mapping and only 92 mapped target-test images, while the cited studies use different label spaces, target partitions, models and evaluation protocols. Nevertheless, these papers establish that the laboratory-to-field gap itself, frozen foundation-model evaluation and multi-backbone fusion are already active research areas. The defensible distinction here is the paired analysis of MTD-Net's architecture and training-recipe effects across 65 fixed configurations—not a claim to have introduced field-transfer evaluation or to outperform modern foundation-model baselines.

Plant-disease texture literature also narrows the interpretation of the proposed branch. Texture–color dual-branch models have been explicitly developed for maize lesions, and classical texture descriptors have been used for citrus disease analysis [10,51]. These studies support the biological plausibility of texture as a useful signal. The present ablations, however, test a high-resolution branch and its gating within a complete network; they do not show that branch activations are exclusively texture-based, nor do they separate texture from color, shape, lesion size and background information. A stronger mechanism claim would require matched-capacity controls and direct feature-level interventions on texture while holding other image properties constant.

Synthetic corruption and field transfer measure different kinds of shift. The pretrained models retained mean accuracy of 0.7781 and 0.7437 across the eight source-corruption conditions, but zero-shot PlantDoc accuracy was 0.3370 and 0.3109. Corrupting source images does not reproduce the combined changes in background, scene composition, label mapping and symptom presentation present in field datasets. Field-transfer results should therefore remain a primary outcome rather than being inferred from corruption robustness [16–21,28–30].

Target-data access materially changed the result. AdaBN used unlabeled target-training images and raised MTD-P accuracy from 0.3370 to 0.3609; supervised fine-tuning used target labels and target-validation checkpoint selection, reaching 0.6109. ResNet-18 showed a similar pattern, reaching 0.3391 with AdaBN and 0.6370 with supervised fine-tuning. These figures describe different operating conditions, not a single zero-shot ranking. Source-fitted temperature scaling also failed to improve target ECE in the reported configurations, consistent with the known difficulty of transferring calibration under dataset shift [29,44–48].

The study's reusable contribution is an auditable record of the fixed source and target partitions, explicit label mapping and target access, five-seed results, paired contrasts, saved predictions and reproducible summary figures. The source cache is class-imbalanced and has incomplete leaf identifiers, while the mapped PlantDoc test contains 92 images. These facts define the population represented by the estimates. A next-stage comparison should use several target datasets, parent-image or acquisition-level grouping, stronger frozen foundation-model baselines, matched initialization strata and a common label-access protocol. Such a design would test whether the recipe-level effects observed here persist across field settings.


].

Target-data access materially changed the result. AdaBN used unlabeled target-training images and raised MTD-P accuracy from 0.3370 to 0.3609; supervised fine-tuning used target labels and target-validation checkpoint selection, reaching 0.6109. ResNet-18 showed a similar pattern, reaching 0.3391 with AdaBN and 0.6370 with supervised fine-tuning. These figures describe different operating conditions, not a single zero-shot ranking. Source-fitted temperature scaling also failed to improve target ECE in the reported configurations, consistent with the known difficulty of transferring calibration under dataset shift [44–48].

The study’s practical contribution is an auditable evaluation record: fixed source and target partitions, explicit label mapping and target access, five-seed results, paired contrasts, saved predictions, and reproducible summary figures. The data audit also identifies the conditions that define the current estimates: the source cache is class-imbalanced and has incomplete leaf identifiers, while the mapped PlantDoc test contains 92 images. These facts define the population represented by the results. A useful next comparison is a multi-target benchmark with matched pretrained initializations, acquisition-level grouping and the same zero-shot/adaptation split across architectures.

## Conclusion

This study tests MTD-Net as a texture-oriented design under controlled comparisons of architecture, training recipe and target-data access. Its six-epoch scratch result was only 1.23 macro-F1 points above the parameter-matched CNN, with substantial uncertainty across seeds. Training duration and learning rate produced larger changes, while MTD-P did not improve source macro-F1 over pretrained ResNet-18. Both pretrained models transferred poorly to the selected PlantDoc holdout despite strong source performance; adaptation improved accuracy when target images or labels were available. The central finding is methodological and empirical: architecture claims are best judged alongside optimization controls, pretrained baselines and an independent field-domain evaluation.

## Figure legends

**Figure 1. Corrected source-domain configuration comparisons.** Individual points show five training seeds; the central mark and error bars show the mean and sample standard deviation. Scratch models use six epochs and peak LR 1e-3; pretrained models use six epochs and peak LR 3e-4.

**Figure 2. Training-recipe factorial.** Source-test macro-F1 for MTD-Net and the plain CNN at 6/12 epochs and peak LR 1e-3/2e-3. Points show individual seeds and lines join cell means, allowing the tested epoch and learning-rate contrasts to be read separately.

**Figure 3. Component contrasts.** Source-test macro-F1 for MTD-Net, gate removal, disabled auxiliary task and the appearance-only ablation. Points show seeds and error bars show sample SD. Because the appearance-only ablation also reduces parameter count, it represents a combined architecture-and-capacity contrast.

**Figure 4. Corruption and domain-transfer performance.** Mean accuracy over eight fixed synthetic source corruptions is plotted alongside accuracy on the mapped official PlantDoc holdout. The two outcomes describe different evaluation conditions.

**Figure 5. Target-information conditions.** Accuracy on the same held-out target test after zero-shot transfer, target-training-image AdaBN or supervised target-training fine-tuning. Individual points represent five source training seeds. Fine-tuning checkpoints are selected using target validation; target-test labels are used only for final evaluation.

## References

1. Mohanty SP, Hughes DP, Salathé M. Using Deep Learning for Image-Based Plant Disease Detection. *Frontiers in Plant Science*. 2016;7:1419. https://doi.org/10.3389/fpls.2016.01419

2. Hughes DP, Salathé M. An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv*. 2015; arXiv:1511.08060. https://arxiv.org/abs/1511.08060

3. Ferentinos KP. Deep learning models for plant disease detection and diagnosis. *Computers and Electronics in Agriculture*. 2018;145:311–318. https://doi.org/10.1016/j.compag.2018.01.009

4. Too EC, Li Y, Njuki S, Liu Y. A comparative study of fine-tuning deep learning models for plant disease identification. *Computers and Electronics in Agriculture*. 2019;161:272–279. https://doi.org/10.1016/j.compag.2018.03.032

5. Singh D, Jain N, Jain P, Kayal P, Kumawat S, Batra N. PlantDoc: A Dataset for Visual Plant Disease Detection. In: *Proceedings of the 7th ACM IKDD CoDS and 25th COMAD*. 2020:249–253. https://doi.org/10.1145/3371158.3371196

6. Picon A, Alvarez-Gila A, Seitz M, Ortiz-Barredo A, Echazarra J, Johannes A. Deep convolutional neural networks for mobile capture device-based crop disease classification in the wild. *Computers and Electronics in Agriculture*. 2019;161:280–290. https://doi.org/10.1016/j.compag.2018.04.002

7. Barbedo JGA. Impact of dataset size and variety on the effectiveness of deep learning and transfer learning for plant disease classification. *Computers and Electronics in Agriculture*. 2018;153:46–53. https://doi.org/10.1016/j.compag.2018.08.013

8. Ramcharan A, Baranowski K, McCloskey P, Ahmed B, Legg J, Hughes DP. Deep learning for image-based cassava disease detection. *Frontiers in Plant Science*. 2017;8:1852. https://doi.org/10.3389/fpls.2017.01852

9. Fuentes A, Yoon S, Kim SC, Park DS. A robust deep-learning-based detector for real-time tomato plant diseases and pests recognition. *Sensors*. 2017;17(9):2022. https://doi.org/10.3390/s17092022

10. Wang H, Pan X, Zhu Y, Li S, Zhu R. Maize leaf disease recognition based on TC-MRSN model in sustainable agriculture. *Computers and Electronics in Agriculture*. 2024;221:108915. https://doi.org/10.1016/j.compag.2024.108915

11. Barbedo JGA. A review on the main challenges in automatic plant disease identification based on visible range images. *Biosystems Engineering*. 2016;144:52–60. https://doi.org/10.1016/j.biosystemseng.2016.01.017

12. Boulent J, Foucher S, Théau J, St-Charles PL. Convolutional neural networks for the automatic identification of plant diseases. *Frontiers in Plant Science*. 2019;10:941. https://doi.org/10.3389/fpls.2019.00941

13. Geirhos R, Rubisch P, Michaelis C, Bethge M, Wichmann FA, Brendel W. ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. *International Conference on Learning Representations*. 2019. https://arxiv.org/abs/1811.12231

14. Torralba A, Efros AA. Unbiased look at dataset bias. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2011:1521–1528. https://doi.org/10.1109/CVPR.2011.5995347

15. Beery S, Van Horn G, Perona P. Recognition in Terra Incognita. In: *European Conference on Computer Vision*. 2018:456–473. https://doi.org/10.1007/978-3-030-01270-0_28

16. Wu X, Fan X, Luo P, Das Choudhury S, Tjahjadi T, Hu C. From Laboratory to Field: Unsupervised Domain Adaptation for Plant Disease Recognition in the Wild. *Plant Phenomics*. 2023;5:0038. https://doi.org/10.34133/plantphenomics.0038

17. Yang S, Feng Q, Zhang J, Yang W, Zhou W, Yan W. From laboratory to field: cross-domain few-shot learning for crop disease identification in the field. *Frontiers in Plant Science*. 2024;15:1434222. https://doi.org/10.3389/fpls.2024.1434222

18. Tunio MH, Li JP, Zeng X, Ahmed A, Shah SA, Shaikh HU, Mallah GA, Yahya IA. Advancing plant disease classification: A robust and generalized approach with transformer-fused convolution and Wasserstein domain adaptation. *Computers and Electronics in Agriculture*. 2024;227:109574. https://doi.org/10.1016/j.compag.2024.109574

19. Wei T, Chen Z, Huang Z, Yu X. Benchmarking In-the-Wild Multimodal Disease Recognition and A Versatile Baseline. In: *Proceedings of the 32nd ACM International Conference on Multimedia (MM '24)*. 2024:1593–1601. https://doi.org/10.1145/3664647.3680599

20. Wei T, Chen Z, Yu X, Chapman S, Melloy P, Huang Z. A Large-Scale In-the-wild Dataset for Plant Disease Segmentation. *Scientific Data*. 2026;13:205. https://doi.org/10.1038/s41597-025-06513-4

21. Zhan K, Peng Y, Liao M, Wang Y. Domain generalization plant leaf disease recognition: Toward from laboratory to field. *Engineering Applications of Artificial Intelligence*. 2025;156:111168. https://doi.org/10.1016/j.engappai.2025.111168

22. Rezaei M, Diepeveen D, Laga H, Jones MG, Sohel F. Plant disease recognition in a low data scenario using few-shot learning. *Computers and Electronics in Agriculture*. 2024;219:108812. https://doi.org/10.1016/j.compag.2024.108812

23. Chai AYH, Lee SH, Tay FS, Bonnet P, Joly A. Beyond supervision: Harnessing self-supervised learning in unseen plant disease recognition. *Neurocomputing*. 2024;610:128608. https://doi.org/10.1016/j.neucom.2024.128608

24. Huan X, Chen B, Zhou H. A unified self-supervised framework for plant disease detection on laboratory and in-field images. *Electronics*. 2025;14(17):3410. https://doi.org/10.3390/electronics14173410

25. Yilma G, Dagne M, Ahmed MK, Bellam RB. Attentive self-supervised contrastive learning (ASCL) for plant disease classification. *Results in Engineering*. 2025;25:103922. https://doi.org/10.1016/j.rineng.2025.103922

26. Espejo-Garcia B, Güldenring R, Nalpantidis L, Fountas S. Foundation vision models in agriculture: DINOv2, LoRA and knowledge distillation for disease and weed identification. *Computers and Electronics in Agriculture*. 2025;239:110900. https://doi.org/10.1016/j.compag.2025.110900

27. Shah SSA, Saeed F, Raza MU, Rehman A, Shaheryar M, Kim I-M, Yun S, Kang JM. PlantCLR: contrastive self-supervised pretraining for generalizable plant disease detection. *Scientific Reports*. 2026;16:10550. https://doi.org/10.1038/s41598-026-45684-x

28. Nguyen TA, Nguyen DS, Dang QM, Phan BNL, Nguyen LH. From laboratory to field: Frozen foundation-model features toward robust plant disease recognition. *Smart Agricultural Technology*. 2026; article 102531. https://doi.org/10.1016/j.atech.2026.102531

29. Xiang K, Shi D, Zhu X. Quantifying the reliability gap in cross-domain plant disease classification: benchmarking the limited efficacy of standard mitigation techniques under controlled-to-field shift. *Frontiers in Plant Science*. 2026;17:1826962. https://doi.org/10.3389/fpls.2026.1826962

30. El Karch H, Natij Y, Benaly M, El Gouri R, Mezouari A. Backbone diversity beats text supervision: a systematic study of frozen multi-foundation model fusion for in-the-wild plant disease recognition. *Frontiers in Plant Science*. 2026;17:1860665. https://doi.org/10.3389/fpls.2026.1860665

31. He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2016:770–778. https://doi.org/10.1109/CVPR.2016.90

32. Dosovitskiy A, Beyer L, Kolesnikov A, et al. An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale. *International Conference on Learning Representations*. 2021. https://arxiv.org/abs/2010.11929

33. Caron M, Touvron H, Misra I, et al. Emerging Properties in Self-Supervised Vision Transformers. In: *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2021:9650–9660. https://doi.org/10.1109/ICCV48922.2021.00951

34. Oquab M, Darcet T, Moutakanni T, et al. DINOv2: Learning Robust Visual Features without Supervision. *Transactions on Machine Learning Research*. 2024. https://arxiv.org/abs/2304.07193

35. Chen T, Kornblith S, Norouzi M, Hinton G. A Simple Framework for Contrastive Learning of Visual Representations. In: *Proceedings of the 37th International Conference on Machine Learning*. 2020;119:1597–1607. https://proceedings.mlr.press/v119/chen20j.html

36. Grill JB, Strub F, Altché F, et al. Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning. *Advances in Neural Information Processing Systems*. 2020;33:21271–21284. https://proceedings.neurips.cc/paper_files/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html

37. He K, Fan H, Wu Y, Xie S, Girshick R. Momentum Contrast for Unsupervised Visual Representation Learning. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020:9729–9738. https://doi.org/10.1109/CVPR42600.2020.00975

38. Gulrajani I, Lopez-Paz D. In Search of Lost Domain Generalization. *International Conference on Learning Representations*. 2021. https://arxiv.org/abs/2007.01434

39. Koh PW, Sagawa S, Marklund H, et al. WILDS: A Benchmark of in-the-Wild Distribution Shifts. In: *Proceedings of the 38th International Conference on Machine Learning*. 2021;139:5637–5664. https://proceedings.mlr.press/v139/koh21a.html

40. Zhou K, Liu Z, Qiao Y, Xiang T, Loy CC. Domain Generalization: A Survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. 2023;45(4):4396–4415. https://doi.org/10.1109/TPAMI.2022.3195549

41. Ganin Y, Ustinova E, Ajakan H, et al. Domain-Adversarial Training of Neural Networks. *Journal of Machine Learning Research*. 2016;17(59):1–35. https://jmlr.org/papers/v17/15-239.html

42. Sun B, Saenko K. Deep CORAL: Correlation Alignment for Deep Domain Adaptation. In: *European Conference on Computer Vision Workshops*. 2016:443–450. https://doi.org/10.1007/978-3-319-49409-8_35

43. Li Y, Wang N, Shi J, Liu J, Hou X. Revisiting Batch Normalization for Practical Domain Adaptation. *arXiv*. 2016; arXiv:1603.04779. https://arxiv.org/abs/1603.04779

44. Hendrycks D, Dietterich T. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. *International Conference on Learning Representations*. 2019. https://arxiv.org/abs/1903.12261

45. Schneider S, Rusak E, Eck L, Bringmann O, Brendel W, Bethge M. Improving robustness against common corruptions by covariate shift adaptation. *Advances in Neural Information Processing Systems*. 2020;33:11539–11551. https://proceedings.neurips.cc/paper_files/paper/2020/hash/85690f81aadc1749175c187784afc9ee-Abstract.html

46. Guo C, Pleiss G, Sun Y, Weinberger KQ. On Calibration of Modern Neural Networks. In: *Proceedings of the 34th International Conference on Machine Learning*. 2017;70:1321–1330. https://proceedings.mlr.press/v70/guo17a.html

47. Ovadia Y, Fertig E, Ren J, et al. Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift. *Advances in Neural Information Processing Systems*. 2019;32:13991–14002. https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html

48. Lakshminarayanan B, Pritzel A, Blundell C. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *Advances in Neural Information Processing Systems*. 2017;30:6402–6413. https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html

49. Loshchilov I, Hutter F. Decoupled Weight Decay Regularization. *International Conference on Learning Representations*. 2019. https://arxiv.org/abs/1711.05101

50. Smith LN, Topin N. Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates. *arXiv*. 2017; arXiv:1708.07120. https://arxiv.org/abs/1708.07120

51. Gómez-Flores W, Garza-Saldaña JJ, Varela-Fuentes SE. Detection of Huanglongbing disease based on intensity-invariant texture analysis of images in the visible spectrum. *Computers and Electronics in Agriculture*. 2019;162:825–835. https://doi.org/10.1016/j.compag.2019.05.032
