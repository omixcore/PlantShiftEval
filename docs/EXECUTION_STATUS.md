# Six-task execution status

This record distinguishes implementation from completed model experiments. It must be updated from run artifacts rather than inferred from CI success.

| Task | Verified work | Remaining evidence |
|---|---|---|
| Data provenance and splitting | Five Parquet inputs pinned and SHA-256 verified; all 14,238 PlantVillage image bytes matched to original Git blobs; source/target filtering and frozen splits implemented | Some biological leaf IDs are unavailable; the audit cannot certify biological independence for every image |
| Core comparisons | Five-seed paired matrix, corrected training and real runtime test implemented; local first runs started | Completed runs and held-out evaluations for every planned seed |
| Training strategy | Full epochs {6,12} × LR {1e-3,2e-3} for both scratch models defined in 40 cells | All factorial cells must finish before effect/interaction interpretation |
| Component ablations | No gate, no auxiliary loss, and no high-resolution texture branch implemented, each with five seeds | Corrected component contrasts; texture removal also changes capacity |
| Domain evaluation | Official target test preserved; zero-shot, AdaBN image-only access, supervised FT label access, and source-validation calibration implemented | Finished independent held-out predictions, per-seed corruption records and uncertainty assessment |
| Paper and figures | Corrected results/table/five-figure generator implemented; incomplete evidence causes a nonzero exit | Final numerical results and figures only after all 65 training/evaluation pairs complete |

Local source audit: 3,255 retained cache rows, including 2,922 optimizer-training and 333 validation rows; 2,916 official source test rows. The filter removes 5,820 source-training rows in known leaf/identity components overlapping test, and 12 further similarity candidates before the per-class cap. Class counts are uneven; one class retains only 16 source-training-cache images. These cleaned data are a new protocol and must not be treated as directly comparable to historical scores.

The initial mapped target pool contained 956 images. Additional source-similarity and conflicting-label-component filtering retained a pool of 946, with 92 official target-test images, 683 target-training images and 170 target-validation images; one duplicate target-training row is unused. The small target holdout limits generalization and precision. Official test images and labels never select source or adapted checkpoints.

Run `python src/summarize_corrected_study.py` to generate a machine-readable completeness record. Missing experiment cells produce a partial report and an exit code of 1. A completed training run is not itself a complete domain evaluation. GitHub Actions validation does not certify model superiority or submission readiness.
