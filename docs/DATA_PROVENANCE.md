# Data provenance and licensing checklist

## PlantVillage

`src/data.py` identifies the Hugging Face dataset `GVJahnavi/Plant_village_subset` and expects `data/pv_train.parquet` and `data/pv_test.parquet`. Record the exact revision, license, retrieval date, and SHA-256 hashes before release.

## PlantDoc

The cross-domain code expects `data/pd_train-00000.parquet`, `data/pd_train-00001.parquet`, and `data/pd_test-00000.parquet`. The source snapshot does not identify the exact upstream repository/revision, license, or file checksums for these files. This is a real reproducibility gap; do not infer a source. Pin the exact source, verify its terms, and record checksums before submission.

## Mapping and exclusions

The current protocol maps PlantDoc categories to the 15 PlantVillage classes and retains 956 images across 10 mapped categories. Unmapped categories and some multi-object images are excluded. Results describe this mapped subset, not all PlantDoc or all field deployments.

Raw data and caches are not committed. Obtain data under upstream terms and do not redistribute without checking the license.
