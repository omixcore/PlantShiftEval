# Research positioning

## Recommended title

**Testing Texture-Representation Hypotheses in Plant Disease Recognition: Training-Strategy Effects and Cross-Domain Generalization Limits**

MTD-Net is the candidate architecture used to test the texture/appearance-decoupling hypothesis. The methodological contribution is the evaluation of that hypothesis under repeated seeds, component ablations, a different training recipe, synthetic corruptions, and cross-domain transfer—not an assumed state-of-the-art network.

## What the current evidence supports

- Texture/appearance decoupling is plausible, but the measured scratch-model gain is sensitive to seed and training setup.
- The gate and auxiliary crop-species head have not shown a consistent independent benefit.
- The 12-epoch comparison also raises peak learning rate, so it demonstrates recipe sensitivity rather than a pure epoch-budget effect.
- Pretrained MTD-P is effectively tied with ResNet-18 on in-domain macro-F1.
- Synthetic-corruption robustness and PlantDoc transfer are distinct tests; transfer remains poor even for the strongest recorded pretrained model.

## Recommended narrative

1. Motivate the gap between controlled leaf images and field imagery.
2. State the texture-representation hypothesis and the competing possibility that pre-training and optimization dominate.
3. Describe matched scratch baselines, pretrained baselines, seeds, and ablations before the results.
4. Report means with seed variability; do not claim a gate or auxiliary-head benefit that is not established.
5. Treat the longer/higher-LR comparison as a combined recipe contrast.
6. Contrast synthetic corruption with zero-shot PlantDoc transfer.
7. Close with the evidence boundary and the controls needed next.

## Claims to avoid

- Universal superiority or state-of-the-art claims without a narrowly defined benchmark and protocol.
- Attributing corruption gains to the texture branch alone.
- Claiming that epochs alone caused the 12-epoch improvement.
- Treating synthetic corruption as field validation.
- Calling pooled PlantDoc five-fold adaptation an independent held-out test.
- Claiming that the gate or auxiliary head improves performance in general.

## Needed before submission

1. Cross epoch count (6/12) with learning rate (1e-3/2e-3) using the same seeds and data.
2. Increase seed count beyond three for small architectural effects.
3. Pin dataset revisions, licenses, checksums, class mapping, and split definitions.
4. Use an independent PlantDoc holdout for adaptation, or label pooled CV exploratory.
5. Freeze dependency versions and archive exact commands and environments.
