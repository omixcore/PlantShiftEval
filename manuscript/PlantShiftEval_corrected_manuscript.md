# Testing Texture-Representation Hypotheses in Plant Disease Recognition: Training-Strategy Effects and Cross-Domain Generalization Limits

**Evidence status: complete corrected numerical study; author-reviewed submission materials pending.**

## Abstract

Texture-oriented architectures for plant disease recognition require evaluation that separates configuration effects from optimization and domain shift. We examined MTD-Net using a corrected, auditable protocol with known-leaf and image-overlap screening, training-internal validation, and 65 fixed training configurations over five seeds. On the cleaned 15-class source test, six-epoch MTD-Net and a parameter-matched CNN achieved mean macro-F1 values of 0.6026 and 0.5903, respectively. The paired difference was +1.23 percentage points (seed-level 95% interval -3.53 to +5.98). Pretrained MTD-P and ResNet-18 achieved 0.8745 and 0.8742. Epoch-by-learning-rate controls and component ablations characterize protocol dependence without establishing semantic disentanglement. Mean zero-shot accuracies on 92 official target-test images were 0.337 and 0.311 for the pretrained pair. Corruption, calibration, image-only adaptation and supervised adaptation are reported separately. Limited leaf metadata, uneven source support and a small target holdout constrain interpretation. The evidence supports configuration-specific conclusions rather than universal architectural superiority.

## Introduction

Image-based plant disease recognition raises a representation question as well as a classification problem: which visual signals support a decision, and which remain informative when acquisition conditions change? PlantVillage enabled systematic learning from labeled leaf photographs [1], whereas PlantDoc introduced a more heterogeneous collection of non-laboratory imagery [2]. Performance on one collection does not by itself establish performance on another. A model can exploit a useful association in its training domain without learning a feature that transfers to a different imaging context.

Texture is a plausible source of disease information because local patterns can encode spots, lesions and surface changes. Nevertheless, naming a network branch “texture” does not establish what it represents. Evidence from controlled texture–shape conflicts in general object recognition shows that training data can alter cue preferences [3]; that observation motivates tests in plant imagery but does not demonstrate the behavior of the present model. A high-resolution feature branch can also carry shape, color and background information. Consequently, improvements by a multibranch model must be separated from claims of semantic disentanglement or an identified biological-image mechanism.

Training strategy is another plausible explanation for performance differences. A short optimization schedule may disadvantage an otherwise adequate baseline, and changing both learning rate and training duration prevents attribution to either factor alone. Similarly, comparing a pretrained model with a different scratch architecture confounds initialization with architecture and capacity. The informative question is whether the candidate design adds a repeatable benefit under a matched protocol, including when a conventional pretrained ResNet-18 is available [4].

This study therefore treats MTD-Net as a candidate configuration rather than an assumed superior network. PlantShiftEval denotes the evaluation project. We ask whether the configuration improves source-domain performance relative to a parameter-matched CNN, whether the ranking depends on an epoch-by-learning-rate contrast, and whether component differences persist under a frozen validation protocol. We then distinguish synthetic-corruption performance from held-out target-domain accuracy and confidence calibration [5,6]. Finally, we distinguish zero-shot transfer from adaptation with target images and adaptation with target labels. AdaBN provides a simple image-only diagnostic based on target-domain normalization statistics [7]. The purpose is to characterize the scope and uncertainty of the evidence, including inconclusive and negative findings.

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

Images are cached at 160 × 160 source resolution and transformed to 112 × 112 model inputs. Training uses random resized cropping, horizontal and vertical flips, rotation and color jitter, followed by ImageNet normalization. Evaluation uses resizing and center cropping without stochastic augmentation. AdamW [8] uses weight decay 0.02, a OneCycle learning-rate schedule [9], batch size 64, disease cross-entropy with label smoothing 0.05, and gradient clipping at norm five. The auxiliary loss weight is 0.3 when that output is enabled. Source background swapping and crop-balanced sampling are disabled in the corrected core study.

The scratch factorial crosses 6/12 epochs with peak learning rates 1e-3/2e-3. Component ablations use six epochs and peak learning rate 1e-3. Both pretrained configurations use six epochs and peak learning rate 3e-4. Model seeds are 0–4; Python, NumPy and PyTorch RNGs are initialized, and the data-order generator is independently seeded. The maximum validation macro-F1 selects the checkpoint, with the first checkpoint retained for tied maxima. The source test is evaluated once after training. A runtime test instruments evaluation calls to verify this separation on a small test fixture; it does not substitute for the real-data experiments.

### Corruption, transfer and target-label access

Synthetic evaluation applies eight fixed conditions to the source test: two noise severities, two blur severities, increased brightness, reduced contrast and two occlusion severities. Conditions and the common corruption RNG seed are fixed in `src/robustness.py`. Their mean accuracy is a synthetic stress-test outcome; this is not an ImageNet-C reproduction or a substitute for field-domain evaluation [5].

PlantDoc object-category names are mapped explicitly to ten supported source classes. Images without a mapped dominant category are excluded; tied categories do not define an unambiguous image label. Additional screening excludes two source-similarity candidates and eight rows in identity components with conflicting labels. One target-training duplicate overlapping official target test is unused. The resulting cached pool contains 946 rows, with 683 adaptation-training images, 170 target-validation images and 92 official target-test images. Target-training images within the stated similarity threshold of official target test are excluded; the recovered data contained none after the other filters. Official test membership is preserved after these documented exclusions rather than replaced by pooled cross-validation.

Zero-shot evaluation reads no target images during source training. Image-only adaptation updates batch-normalization running statistics using only the target-training images, without labels or gradient updates; statistics are reset and accumulated in one pass of minibatches of 64. Supervised adaptation starts from the same source checkpoint, fine-tunes disease loss with AdamW at learning rate 1e-4 and weight decay 0.02 for six epochs, and selects the checkpoint on target-validation macro-F1. AdaBN and supervised adaptation are applied to both pretrained configurations for all five seeds. Official target-test labels never choose adaptation epochs or hyperparameters. These three access conditions are reported separately.

### Outcomes, uncertainty and traceability

The primary source outcome is macro-F1. Accuracy, expected calibration error with 15 confidence bins, per-class behavior, synthetic-corruption accuracy and target accuracy are secondary outcomes. Source-fitted temperature scaling minimizes negative log-likelihood on source-validation predictions only [6]; the fitted temperature is then applied to target-test logits without target-label fitting. This evaluates transfer of a source-fitted calibrator, not an oracle target calibrator.

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

The five-seed MTD-P probability ensemble correctly classified 31/92 target-test images (accuracy 0.3370; image-level Wilson interval 0.2486 to 0.4383). The five-seed ResNet-18 probability ensemble correctly classified 30/92 target-test images (accuracy 0.3261; image-level Wilson interval 0.2389 to 0.4272). These ensemble estimates are distinct from mean per-seed accuracy. The approximate intervals condition on the selected images and assume independent image outcomes; unknown acquisition dependencies and population selection remain outside their scope. Per-class supports and recalls are archived.

### Observed configuration effects

At the primary six-epoch budget, the gate and auxiliary-task removals have numerically higher mean macro-F1 than the full candidate, while branch removal has a lower mean. None of the five declared architecture contrasts establishes a benefit after Holm adjustment. This is limited evidence, not a demonstration of equivalence or proof that those components can never help. The higher-budget scratch cells improve both models descriptively; their effect sizes are larger than the primary six-epoch architecture difference.

Source-fitted temperature scaling increases mean target ECE for all four core configurations. For MTD-P it changes from 0.2626 to 0.3780; for ResNet-18, from 0.2477 to 0.3624. AdaBN yields modest descriptive accuracy increases, while supervised target fine-tuning reaches 0.6109 and 0.6370. The latter result depends on target labels and does not repair zero-shot transfer. No target-method superiority test is asserted from these descriptive means.

## Discussion

The central methodological result is the set of conditional contrasts rather than a declaration that one architecture is generally best. The source comparison evaluates a whole candidate configuration, including its optimization and auxiliary supervision. Component ablations narrow the interpretation, but removing a branch changes capacity, and neither the branch name nor a predictive gain establishes a uniquely texture-based mechanism. Direct feature interventions and additional matched controls would be required for that stronger claim.

The factorial controls address an important ambiguity in the historical study: increasing training duration while also increasing peak learning rate does not isolate an epoch effect. The corrected design reports both factors and their interaction under common split assignments and seed identifiers. These schedules still sample a limited hyperparameter space and are not a comprehensive tuning contest. Any favorable contrast is specific to these declared budgets and should not be generalized to all well-tuned baselines.

The pretrained comparison estimates whether the candidate additions help when both configurations have access to pretrained ResNet-18 features. It does not isolate pretraining itself because the scratch and pretrained architectures differ. Parameter counts and data access accompany scores so that an apparent benefit is not conflated with an identical-capacity or identical-information comparison.

Source performance, synthetic stress tests and target transfer answer different questions. Corruptions modify source images while largely preserving the source labeling and acquisition context. The target experiment changes image composition, collection conditions and taxonomy support simultaneously; it does not identify a single causal reason for errors. Temperature scaling fitted on source validation addresses source confidence and need not remain calibrated under target shift. Numerical changes in target ECE must therefore be read alongside target accuracy and the explicit fitting split.

AdaBN and supervised fine-tuning introduce different information. The former accesses target-training pixels without labels; the latter uses labels and a separate target-validation set. Their gains or losses are conditional on that access and must not be presented as zero-shot generalization. Using official target-test membership after documented filtering provides a clearer held-out estimate than the historical pooled cross-validation, but only 92 mapped test images remain. Seed replication does not remove this small-test-set limitation.

Data curation also changes the population being measured. Known source leaf overlap was substantial, and removing it reduces some class supports severely. Missing biological group IDs remain unresolved for part of both source partitions. Difference-hash screening can remove similar independent images and can miss duplicates with larger transformations. The corrected study is thus more explicit about available evidence, without certifying complete biological independence or a representative field population. Historical and corrected scores cannot quantify the impact of checkpoint-selection leakage alone because the training composition also changed.

The reusable contribution is the traceable evaluation record: fixed inputs, class mapping, explicit target access, validation-selected checkpoints, seed-level contrasts and reconstructable figures. This record supports independent scrutiny of the conditional effects and negative findings. Further acquisition-level and deployment validation would be needed to assess practical use.

## Conclusion

This study evaluates texture-oriented configuration choices under explicit optimization and domain-access controls. The reported estimates, intervals and failure boundaries define where the current evidence applies. Additional datasets, better biological grouping metadata and independently collected target images are needed before making broader claims about disease-specific texture mechanisms or practical field reliability.

## Figure legends

**Figure 1. Corrected source-domain configuration comparisons.** Individual points show five training seeds; the central mark and error bars show the mean and sample standard deviation. Scratch models use six epochs and peak LR 1e-3; pretrained models use six epochs and peak LR 3e-4. Comparisons are interpreted within the declared initialization stratum.

**Figure 2. Training-recipe factorial.** Source-test macro-F1 for MTD-Net and the plain CNN at 6/12 epochs and peak LR 1e-3/2e-3. Points show individual seeds and lines join cell means. The design separates the sampled factors; it does not establish an exhaustive optimum.

**Figure 3. Component contrasts.** Source-test macro-F1 for the candidate model, gate removal, disabled auxiliary task and appearance-only ablation. Points show seeds and error bars show sample SD. Appearance-only removal changes parameter count and cannot by itself identify a texture-specific mechanism.

**Figure 4. Corruption and domain-transfer limits.** Mean accuracy over eight fixed synthetic source corruptions and accuracy on the mapped official target holdout. These are distinct evaluations. Source and target images differ in more than one acquisition factor.

**Figure 5. Target information conditions.** Accuracy on the same held-out target test after zero-shot transfer, target-training-image AdaBN or supervised target-training fine-tuning. Individual points represent five source training seeds. Fine-tuning checkpoints are chosen using only target validation; target-test labels are used for final evaluation only.

## References

1. Mohanty SP, Hughes DP, Salathé M. Using Deep Learning for Image-Based Plant Disease Detection. *Frontiers in Plant Science*. 2016;7:1419. https://doi.org/10.3389/fpls.2016.01419
2. Singh D, Jain N, Jain P, Kayal P, Kumawat S, Batra N. PlantDoc: A Dataset for Visual Plant Disease Detection. *Proceedings of CoDS COMAD 2020*. 2020:249–253. https://doi.org/10.1145/3371158.3371196
3. Geirhos R, Rubisch P, Michaelis C, Bethge M, Wichmann FA, Brendel W. ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. *ICLR*. 2019. https://arxiv.org/abs/1811.12231
4. He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. *CVPR*. 2016:770–778. https://arxiv.org/abs/1512.03385
5. Hendrycks D, Dietterich T. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. *ICLR*. 2019. https://arxiv.org/abs/1903.12261
6. Guo C, Pleiss G, Sun Y, Weinberger KQ. On Calibration of Modern Neural Networks. *PMLR*. 2017;70:1321–1330. https://proceedings.mlr.press/v70/guo17a.html
7. Li Y, Wang N, Shi J, Liu J, Hou X. Revisiting Batch Normalization for Practical Domain Adaptation. arXiv:1603.04779. 2016. https://arxiv.org/abs/1603.04779

8. Loshchilov I, Hutter F. Decoupled Weight Decay Regularization. *ICLR*. 2019. https://arxiv.org/abs/1711.05101
9. Smith LN, Topin N. Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates. arXiv:1708.07120v3. 2018. https://arxiv.org/abs/1708.07120
