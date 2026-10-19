# Research positioning

## Recommended title

**Testing Texture-Representation Hypotheses in Plant Disease Recognition: Training-Strategy Effects and Cross-Domain Generalization Limits**

MTD-Net is the candidate architecture used to test the texture/appearance-decoupling hypothesis. The proposed methodological contribution is the evaluation of that hypothesis under repeated seeds, component ablations, training-recipe sensitivity, synthetic corruptions, and cross-domain transfer—not an assumed state-of-the-art result.

## Critical status before publication

The currently tracked scores are historical and provisional. The run-level metric JSONs do not record the validation-split setting, and the earlier training entry point defaulted to `--val-frac=0`, which could use the test split for per-epoch checkpoint selection. The current code now requires a training-internal validation split, but the historical runs have not been regenerated. Do not present the existing p-values or metrics as definitive unbiased held-out results until that rerun is complete.

## What the recorded evidence suggests (provisionally)

- Texture/appearance decoupling is plausible, but the recorded scratch-model gain is small relative to seed variation and is not established as a stable effect.
- The gate and auxiliary crop-species head have not shown a consistent independent benefit.
- The 12-epoch comparison also raises peak learning rate, so it demonstrates recipe sensitivity rather than a pure epoch-budget effect.
- Pretrained MTD-P is effectively tied with ResNet-18 on recorded in-domain macro-F1.
- Synthetic-corruption robustness and PlantDoc transfer are distinct tests; transfer remains poor even for the strongest recorded pretrained model.
- The pooled PlantDoc adaptation results are exploratory and do not establish deployment readiness.

## Recommended narrative after corrected reruns

1. Motivate the gap between controlled leaf images and field imagery.
2. State the texture-representation hypothesis and the competing possibility that pre-training and optimization dominate.
3. Describe matched scratch baselines, pretrained baselines, seeds, and ablations before the results.
4. Report means with seed variability; do not claim a gate or auxiliary-head benefit that is not established.
5. Treat the longer/higher-LR comparison as a combined recipe contrast until a factorial control is run.
6. Contrast synthetic corruption with zero-shot PlantDoc transfer.
7. Close with the evidence boundary and the controls needed next.

## Claims to avoid

- Universal superiority or state-of-the-art claims.
- Attributing corruption gains to the texture branch alone.
- Claiming that epochs alone caused the 12-epoch improvement.
- Treating synthetic corruption as field validation.
- Calling pooled PlantDoc five-fold adaptation an independent held-out test.
- Claiming that the gate or auxiliary head improves performance in general.
- Treating the current historical scores as final, unbiased test estimates.

## Needed before submission

1. Rerun the headline comparisons with a training-internal validation split and a single final test evaluation.
2. Increase seed count beyond three for small architectural effects.
3. Cross epoch count (6/12) with learning rate (1e-3/2e-3) using the same seeds and data.
4. Pin dataset revisions, licenses, checksums, class mapping, and split definitions.
5. Archive the per-run logits, raw corruption outputs, commands, environment, and package lock.
6. Use an independent PlantDoc holdout for adaptation, or keep pooled CV explicitly exploratory.
