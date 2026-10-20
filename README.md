# PlantShiftEval

**Testing texture-representation hypotheses in plant disease recognition: training-strategy effects and cross-domain generalization limits**

MTD-Net is the candidate architecture in an original methodological study of texture-oriented configuration choices, optimization and transfer. The corrected study is complete: **65 training-and-evaluation pairs**, with five seeds per configuration, fixed data partitions, training-internal validation and one final source-test evaluation per run. All 67 workflow jobs (preparation, 65 experiments and aggregation) succeeded in [the archived study run](https://github.com/omixcore/PlantShiftEval/actions/runs/39265931897). Training source revision: `bd7cd1d94ff2591fa58dac5eb69f0e66225b8f72`.

## Corrected evidence

| Configuration | Source macro-F1 mean ± SD | Corruption accuracy | Target zero-shot accuracy |
|---|---:|---:|---:|
| MTD-Net | 0.6026 ± 0.0245 | 0.4496 | 0.1478 |
| Plain CNN | 0.5903 ± 0.0250 | 0.3784 | 0.1283 |
| MTD-P | 0.8745 ± 0.0292 | 0.7781 | 0.3370 |
| ResNet-18 | 0.8742 ± 0.0151 | 0.7437 | 0.3109 |

All rows average five seeds. Scratch models use six epochs / peak LR 1e-3; pretrained models use six epochs / peak LR 3e-4. Corruption averages eight conditions. The target holdout has 92 mapped official test images. Full SDs, all factorial cells and adaptation conditions are in the [results](manuscript/corrected_results.md) and [machine-readable summaries](report/corrected_summary).

The primary six-epoch scratch contrast is +1.23 macro-F1 percentage points for MTD-Net versus the parameter-matched CNN, with a paired-seed 95% Student-t interval of −3.53 to +5.98 points (five seeds; Holm-adjusted p = 1). This does not establish a stable advantage or practical equivalence. MTD-P versus pretrained ResNet-18 differs by +0.03 points, with a seed interval of −4.41 to +4.47 points. Gate and auxiliary-task benefits are not established by the declared contrasts.

Training configuration has a substantial descriptive effect: MTD-Net means range from 0.6026 at six epochs / LR 1e-3 to 0.7930 at twelve epochs / LR 2e-3; corresponding CNN means range from 0.5903 to 0.7404. The full factorial reports each factor and its interaction rather than attributing the joint change to epochs alone. The four scratch recipe contrasts and their raw per-seed values are retained without selecting a single favorable cell as the primary test.

Source macro-F1 near 0.874 for the pretrained pair coexists with only 0.3370 / 0.3109 target zero-shot accuracy. These are different metrics and datasets, not directly subtractable performance estimates. Synthetic corruption, source-fitted calibration, image-only AdaBN and supervised target fine-tuning are separately reported. The study does not establish semantic texture disentanglement, universal superiority or field reliability.

## Paper and evidence

- [Complete corrected manuscript (50 references)](manuscript/PlantShiftEval_corrected_manuscript.md), [reference authenticity and relevance audit](manuscript/reference_audit.md), and [research positioning](docs/RESEARCH_POSITIONING.md).
- [Five figures in editable SVG, PDF and PNG, with plotting CSVs](report/corrected_summary).
- [All 65 raw prediction arrays and run records](evidence/corrected), with SHA-256 index; [metric reproduction check](report/corrected_summary/public_reproduction.json) passed without disagreements.
- [Full evidence audit](report/corrected_summary/evidence_audit.json), [workflow record](provenance/corrected_workflow.json) and [six-task completion record](docs/EXECUTION_STATUS.md).
- [Data lineage and split audit](docs/DATA_PROVENANCE.md), [dataset attribution](DATASET_NOTICE.md), [reproduction instructions](docs/REPRODUCIBILITY.md) and [locked study dependencies](requirements-study.txt).

Statistical metric regeneration needs NumPy and the public arrays, without weights:

```bash
python src/reproduce_public_results.py
```

Full training and inference reproduction additionally require the pinned data and checkpoints or retraining. Raw images and tensor caches are not committed. Checkpoints and logs are retained in the workflow experiment artifacts and separate checked archives; workflow artifacts have finite retention. Keep a durable copy for long-term inference reproduction.

## Scope and limitations

The cleaned source split contains 2,922 optimizer-training images, 333 validation images and 2,916 official test images. Filtering removes 5,820 training rows in known leaf/identity components overlapping test and 12 further similarity candidates before per-class capping. Class support is uneven; one retained source-cache class has only 16 images. Biological leaf IDs remain missing for 1,604 retained source-training-cache images and 857 source-test images. Image-identity fallback and similarity screening cannot certify full biological or acquisition independence.

The mapped target pool is 946 images after filtering, with 683 target-training, 170 validation and 92 official test images; one duplicate train row is unused. Results cover the selected mapped subset, not the whole PlantDoc population. Five seeds quantify training variability conditional on these splits. The supplementary ensemble Wilson image intervals do not capture acquisition dependence or dataset-selection uncertainty. Removing a branch changes parameter count and does not isolate a texture mechanism. Historical and corrected scores use different training populations and cannot isolate the impact of the old checkpoint-selection problem.

Historical `runs/`, `runs12/` and earlier reports remain provisional because the former training default could select checkpoints on test. They are preserved for audit, excluded from corrected conclusions, and described in [the historical limitations record](docs/HISTORICAL_RESULTS_AND_LIMITATIONS.md). The study code descends from [MTD-Net on Gitee](https://gitee.com/omixcore/MTD-Net). Author identities, declarations and target-journal formatting require author completion before submission.
