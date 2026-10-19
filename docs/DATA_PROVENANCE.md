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
