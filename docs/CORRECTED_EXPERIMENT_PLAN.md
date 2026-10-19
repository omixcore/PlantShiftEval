# Corrected experiment plan and claim boundaries

Status: prospective protocol; no new training results are reported here. Historical scores remain provisional. PlantShiftEval is the study and evaluation repository; MTD-Net is one candidate representation model.

## Questions and controls

| Question | Required comparison | Factors held fixed | Permitted conclusion |
|---|---|---|---|
| Does the candidate texture representation help? | Scratch MTD-Net vs parameter-matched plain CNN; no-gate and no-auxiliary-head ablations | Data, validation rows, augmentation, resolution, optimizer, LR, epochs, seeds | Effect of each tested configuration under this protocol; pathway mechanism remains a hypothesis |
| How much does the training recipe matter? | Each scratch model at epochs {6,12} crossed with peak LR {1e-3,2e-3} | Same validation rows, seed pairs, batch size and augmentation | Separate epoch and LR contrasts plus their interaction; the historical diagonal comparison cannot do this |
| Does adding the candidate head help a pretrained baseline? | MTD-P vs pretrained ResNet-18 | Pretraining source, source-data split, LR, epochs and seeds | Incremental effect conditional on this pretrained setup; no isolated attribution to pretraining itself |
| Does synthetic robustness transfer? | Clean and each corruption condition, then mapped PlantDoc zero-shot | Same checkpoints; documented severity and mapping | Corruption robustness and target transfer are separate outcomes |
| Does target adaptation help on unseen images? | Unadapted baseline vs each adaptation method | Target train/validation/test assignments frozen before tuning | Held-out target performance only when test images are excluded from adaptation and tuning |

Use seeds 0–4 for the corrected core study, paired across models. Fix the internal split seed at 1234 independently of model seed. Five seeds are a minimum study design, not a guarantee of adequate power. Primary outcome is source-test macro-F1; report accuracy, ECE and per-class results as secondary outcomes. Declare the primary comparison and multiplicity policy before examining corrected results.

## Execution order

1. Identify upstream dataset revisions, licenses and checksums; verify class mapping. Audit exact and near duplicates across splits and group by leaf/plant/acquisition where metadata allow. An internal validation split alone does not establish biological sample independence.
2. Build a single source cache and archive its creation command and metadata. Freeze the split before comparing models. Do not pool the target test images into a background bank for a claimed inductive experiment.
3. Run the factorial scratch study and component ablations in new directories. Keep historical runs intact. A partial or failed run is not a completed experiment.
4. Run matched pretrained configurations with LR 3e-4 and the same epoch budget. Compare within the pretrained pair. A comparison to a scratch architecture with different capacity does not isolate pretraining.
5. Freeze checkpoints using source validation only, then evaluate source test once per run. Final-test reporting must not become another hyperparameter-selection loop.
6. Evaluate corruptions and zero-shot transfer from the frozen checkpoints. Fit calibration on validation data, never the reported test labels. Separate any target-access method as transductive/adapted.
7. Regenerate aggregate tables from run artifacts. Archive per-image logits, label order, per-seed corruption outputs, execution logs and environment lock. Explain missing/failed runs without silently dropping them.

The corrected training entry point saves `run_manifest.json`: full parsed arguments, original-cache train/validation indices (also under `--limit`), SHA-256 input and source fingerprints, Python/package/platform versions, Git revision/dirty status and output fingerprints. Python, NumPy and PyTorch RNGs receive the model seed. This improves traceability; it does not certify upstream licensing, duplicate-free splits, deterministic execution across platforms, or experimental validity. Preserve console logs separately. A `started` manifest without completed outputs is not a successful run.

## Example factorial commands

From the repository root, after provenance checks and cache preparation:

```bash
for seed in 0 1 2 3 4; do
  for model in mtdnet smallcnn; do
    for epochs in 6 12; do
      for lr in 1e-3 2e-3; do
        python src/train.py --model "$model" --cache cache \
          --out "runs_corrected/${model}_e${epochs}_lr${lr}_s${seed}" \
          --epochs "$epochs" --size 112 --lr "$lr" --val-frac 0.1 --seed "$seed"
      done
    done
  done
 done
```

These commands describe 40 runs; they do not indicate that any have been executed. Adapt aggregation scripts explicitly to the new directory layout rather than mixing corrected and historical scores. Use a new empty output directory for every run; the entry point refuses to overwrite existing artifacts.

## Analysis and manuscript decisions

Report each seed and the paired differences, their mean and sample SD, and uncertainty intervals with the resampling unit stated. Distinguish variability across training seeds from uncertainty conditional on one fixed test-image set. With few seeds, avoid treating a nonsignificant result as equivalence or absence of an effect. A claim of practical equivalence requires a justified prespecified margin and an appropriate interval analysis.

Organize the paper around (1) representation hypothesis and matched controls, (2) training-recipe sensitivity, (3) component effects and uncertainty, and (4) corruption versus target-domain transfer, calibration and adaptation limits. Discuss negative and inconclusive findings alongside positive findings. Figures should show seed-level points and clearly identify historical versus corrected results. Do not populate final figures or the abstract with unexecuted experiment cells.

## Evidence status at this revision

| Category | Items | Evidence boundary |
|---|---|---|
| Completed | Repository migration, research positioning, historical summary checks, internal-validation requirement, prospective run provenance | Code/document work; not a model-training replication |
| Partly completed | Historical result interpretation and manuscript generator | Numerical summaries exist; underlying estimates remain provisional |
| Not completed | Corrected multi-seed training, factorial controls, independent target holdout evaluation | No corrected run artifacts are present |
| Missing evidence | Exact upstream data revisions/licenses, historical logits/weights/full logs, raw multi-seed corruption inputs | Blocks end-to-end regeneration and definitive submission claims |
