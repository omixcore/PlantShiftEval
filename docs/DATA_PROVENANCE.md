# Data provenance and licensing checklist

## PlantVillage

`src/data.py` identifies the Hugging Face dataset `GVJahnavi/Plant_village_subset` and expects `data/pv_train.parquet` and `data/pv_test.parquet`. The exact revision, license, retrieval date, and SHA-256 hashes for the specific files used in the recorded experiments are not preserved in this repository snapshot. Record and verify these before release.

## PlantDoc

The cross-domain code expects `data/pd_train-00000.parquet`, `data/pd_train-00001.parquet`, and `data/pd_test-00000.parquet`. The source snapshot does not identify the exact upstream repository/revision, license, or file checksums for these Parquet files. This is a reproducibility gap; do not infer a source or license from the dataset name alone. Pin the exact source, verify its terms, and record checksums before submission.

## Mapping and exclusions

The current protocol maps PlantDoc categories to the 15 PlantVillage classes and retains 956 images across 10 mapped categories. Unmapped categories and some multi-object images are excluded. Results describe this mapped subset, not all PlantDoc images or all field deployments.

## Test-set isolation

The historical run metrics do not preserve validation-split metadata. The previous training entry point allowed the test split to be used for per-epoch checkpoint selection when `--val-frac=0`. The current code now requires a nonzero training-internal validation split, but the old scores have not been regenerated. Treat historical performance and significance estimates as provisional until that rerun is complete.

## Data handling

Raw data and caches are not committed. Obtain data under upstream terms and do not redistribute without checking the applicable license and permissions.

## Recovered, pinned inputs for the corrected study (2026-10-19)

`provenance/datasets.json` records the exact recovered Parquet revisions, local names, file sizes, upstream SHA-256 values and license sources. All five files have been downloaded and verified. The recovered PlantVillage revision is `243d37baa0780d93a8305d64f382bc1d41dffe92`; the compatible PlantDoc object-detection mirror is `agyaatcoder/PlantDoc` at `ace645742b6af1b0df9aaefc2c9b0c988e141814`. This recovery does not prove that undocumented historical local Parquet files were byte-identical.

All 14,238 PlantVillage image bytes match original Git blobs in `spMohanty/PlantVillage-Dataset` revision `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`. The canonical `mohanty/PlantVillage` card at `9e97599868962bd0079b8db4b7f1efa9185fa1e7` declares CC BY-SA 3.0; the subset card itself omits a license. PlantDoc's mirror card and upstream repository declare CC BY 4.0. See `DATASET_NOTICE.md` for attribution and processing changes.

The corrected source cache removes known leaf/identity overlap with official test and conservatively excludes dHash distance <=4 candidates. Missing leaf IDs use image-identity components and remain a biological-independence limitation. Internal validation is disjoint by these components. Target processing excludes source similarity candidates and conflicting-label identity components, preserves official target-test membership, and splits only target training into adaptation training/validation. Full row manifests are generated locally and archived with workflow data artifacts; images are not committed.
