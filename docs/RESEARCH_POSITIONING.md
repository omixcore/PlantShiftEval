# Research positioning

**Testing Texture-Representation Hypotheses in Plant Disease Recognition: Training-Strategy Effects and Cross-Domain Generalization Limits**

MTD-Net is an experimental vehicle for an original methodological investigation. Its contribution is the controlled, reproducible examination of texture-oriented configurations and their failure boundaries. Architectural novelty or superiority is not assumed.

## What the completed evidence supports

The primary six-epoch scratch contrast is +1.23 macro-F1 percentage points for MTD-Net versus the parameter-matched CNN, with a paired-seed 95% Student-t interval of −3.53 to +5.98 points (five seeds; Holm-adjusted p = 1). This does not establish a stable advantage or practical equivalence. MTD-P versus pretrained ResNet-18 differs by +0.03 points, with a seed interval of −4.41 to +4.47 points. Gate and auxiliary-task benefits are not established by the declared contrasts.

Training configuration has a substantial descriptive effect: MTD-Net means range from 0.6026 at six epochs / LR 1e-3 to 0.7930 at twelve epochs / LR 2e-3; corresponding CNN means range from 0.5903 to 0.7404. The full factorial reports each factor and its interaction rather than attributing the joint change to epochs alone. The four scratch recipe contrasts and their raw per-seed values are retained without selecting a single favorable cell as the primary test.

Source macro-F1 near 0.874 for the pretrained pair coexists with only 0.3370 / 0.3109 target zero-shot accuracy. These are different metrics and datasets, not directly subtractable performance estimates. Synthetic corruption, source-fitted calibration, image-only AdaBN and supervised target fine-tuning are separately reported. The study does not establish semantic texture disentanglement, universal superiority or field reliability.

## Frozen foundation-model baseline

The frozen DINOv2-S/14 source-only linear probe achieved source macro-F1 0.9374 and target accuracy 0.3696 on the same 92-image mapped target holdout. The target accuracy interval is 0.2717–0.4674; supported-class macro-F1 is 0.3628 with class-stratified interval 0.2838–0.4345. Regularization was selected on source validation, not on target labels. The paired comparisons and McNemar/Holm results are in report/corrected_summary/target_bootstrap.json. This baseline narrows the novelty claim to controlled architecture/optimization contrasts; it does not turn published DINOv2 results from different label spaces into direct comparisons.

## Narrative and boundaries

Lead with the representation hypothesis, declare the alternative explanation of optimization and initialization, and report the full corrected design before numerical claims. Use the primary paired contrast, all factorial cells and component controls. Treat non-significance as limited evidence, not equivalence. The branch labels do not prove semantic factorization; branch removal changes capacity. Separate source classification, synthetic stress tests and target transfer. Target adaptation uses additional image or label information and cannot be relabeled zero-shot generalization.

The cleaned source split contains 2,922 optimizer-training images, 333 validation images and 2,916 official test images. Filtering removes 5,820 training rows in known leaf/identity components overlapping test and 12 further similarity candidates before per-class capping. Class support is uneven; one retained source-cache class has only 16 images. Biological leaf IDs remain missing for 1,604 retained source-training-cache images and 857 source-test images. Image-identity fallback and similarity screening cannot certify full biological or acquisition independence.

The mapped target pool is 946 images after filtering, with 683 target-training, 170 validation and 92 official test images; one duplicate train row is unused. Results cover the selected mapped subset, not the whole PlantDoc population. Five seeds quantify training variability conditional on these splits. The supplementary ensemble Wilson image intervals do not capture acquisition dependence or dataset-selection uncertainty. Removing a branch changes parameter count and does not isolate a texture mechanism. Historical and corrected scores use different training populations and cannot isolate the impact of the old checkpoint-selection problem.

## Publication preparation still owned by the authors

Complete verified author/affiliation details, contributions, funding and competing-interest declarations; review the final numerical narrative and venue requirements; preserve durable access to pinned data and checkpoint archives. Broader deployment claims would require further independently acquired data and biological grouping metadata. These are limitations and author responsibilities, not unfinished cells in the completed 65-run experiment.
