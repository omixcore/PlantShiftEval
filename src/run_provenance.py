"""Dependency-free audit records for corrected experiments (not a validity certificate)."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_record(train_indices, validation_indices, source_size):
    train = [int(i) for i in train_indices]
    validation = [int(i) for i in validation_indices]
    if not train or not validation:
        raise ValueError("Training and validation partitions must both be nonempty")
    combined = train + validation
    if len(set(combined)) != len(combined):
        raise ValueError("Training/validation indices overlap or contain duplicates")
    if any(i < 0 or i >= source_size for i in combined):
        raise ValueError("Split index outside source training partition")
    return {"source_training_rows": source_size, "train_indices": train,
            "validation_indices": validation, "split_seed": 1234,
            "index_space": "original cache train arrays, before --limit",
            "test_used_for_selection": False}


def write_manifest(args, split, argv):
    root = Path(__file__).resolve().parents[1]
    cache = Path(args.cache)
    paths = [cache / name for name in (
        "meta.json", "train_x.npy", "train_y.npy", "train_s.npy",
        "test_x.npy", "test_y.npy", "test_s.npy")]
    if args.bg_aug:
        paths.append(Path(args.bg_cache) / "x.npy")
    inputs = [{"path": str(p.resolve()), "bytes": p.stat().st_size,
               "sha256": file_digest(p)} for p in paths]
    versions = {}
    for name in ("numpy", "torch", "torchvision", "scipy", "scikit-learn", "pyarrow", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    manifest = {
        "schema_version": 1, "status": "started",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(argv), "arguments": vars(args), "git_commit": commit,
        "git_dirty": dirty,
        "source_sha256": {str(p.relative_to(root)): file_digest(p)
                          for p in sorted((root / "src").glob("*.py"))},
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": versions, "inputs": inputs, "split": split,
        "evaluation_protocol": "validation selects checkpoint; final source test evaluated once",
        "target_access": "unlabeled target backgrounds (transductive)" if args.bg_aug else "none during source training",
        "caveat": "Cache hashes identify local inputs; upstream provenance and image-level split independence require separate verification.",
    }
    path = Path(args.out) / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def complete_manifest(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    record["outputs"] = {name: file_digest(path.parent / name)
                         for name in ("model.pt", "metrics.json", "logits.pt")}
    record["status"] = "completed"
    record["completed_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
