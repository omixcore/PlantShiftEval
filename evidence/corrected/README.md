# Corrected study evidence

This directory contains run-level configuration/metric records and compressed NumPy prediction arrays for the fixed 65-run study. `index.json` identifies every exported run and the SHA-256 of its prediction file. No image files or model weights are stored here.

Each run contains `metrics.json`, `run_manifest.json`, `evaluation.json` and `predictions.npz`. Arrays preserve source-test logits/labels, source-validation logits/labels, and official target-test logits/labels. Pretrained configurations also include target-test predictions after image-only AdaBN and supervised fine-tuning. Source class order is recorded in `class_names.json`. Load with `numpy.load(path, allow_pickle=False)`.

After the completed evidence is available, run:

```bash
python src/reproduce_public_results.py
```

This regenerates source/target metrics and corruption aggregates without model weights. It fails if any planned run is missing or a checked metric disagrees. Expected calibration error allows an absolute tolerance of 2e-6 for NumPy versus PyTorch softmax arithmetic; accuracy and macro-F1 use tolerance 1e-10. Inference and full training reproduction additionally require the archived checkpoints or a rerun with the pinned data and protocol. Original checkpoint/logit hashes are retained in the run manifests.

The 65 primary runs all originate from the same fixed study workflow and source revision. Local CPU repeat diagnostics are kept separately and are not extra primary-study seeds. Historical reports elsewhere in the repository remain provisional and are not mixed with these records.
