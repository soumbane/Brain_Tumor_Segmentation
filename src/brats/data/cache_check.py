"""Streaming mechanical cache audit; never predicts, rewrites labels or changes splits.

Counts detect truncated crops, but cannot prove voxelwise identity to raw NIfTI.
The report binds every exact NPZ byte stream and encoding to the frozen split.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from brats.artifacts import canonical_hash, safe_path, sha256_file
from brats.config import REPO_ROOT, DataConfig
from brats.constants import COHORTS, SEQUENCES, QUANT_CLIP_SIGMA, QUANT_MIN_CODE, QUANT_MAX_CODE
from brats.data.splits import verify

ENCODING = {"schema": "brats-cache-v1", "channels": list(SEQUENCES), "dtype": "uint8",
            "background": 0, "min_code": QUANT_MIN_CODE, "max_code": QUANT_MAX_CODE,
            "clip_sigma": QUANT_CLIP_SIGMA, "labels": [0, 1, 2, 3]}


def split_path(cfg: DataConfig) -> Path:
    return Path(os.environ.get("BRATS_SPLITS_CSV", str(REPO_ROOT / cfg.splits["output"])))


def _table(path: Path, required: set[str]) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"case_id": str, "patient_id": str, "cohort": str})
    if not required.issubset(table.columns) or table.empty:
        raise ValueError(f"missing required columns or empty input: {path.name}")
    if table[list(required)].isna().any().any():
        raise ValueError(f"missing required values: {path.name}")
    if table.duplicated(["cohort", "case_id"]).any():
        raise ValueError(f"duplicate cohort/case key: {path.name}")
    for column in {"brain_vox", "vox_ncr", "vox_ed", "vox_et"} & required:
        values = pd.to_numeric(table[column], errors="raise")
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"invalid voxel counts: {path.name}")
    return table


def _check_case(path: Path, row: dict, stats: dict | None, divisor: int) -> None:
    with np.load(path, allow_pickle=False) as cache:
        required = {"img", "orig_shape", "crop_start", "crop_stop", "affine", "norm_stats"}
        if not required.issubset(cache.files):
            raise ValueError("missing required NPZ metadata")
        image = cache["img"]
        if image.dtype != np.uint8 or image.ndim != 4 or image.shape[0] != len(SEQUENCES):
            raise ValueError("image must be uint8 (4,D,H,W)")
        vectors = [cache[key] for key in ("orig_shape", "crop_start", "crop_stop")]
        if any(value.shape != (3,) or value.dtype.kind not in "iu" for value in vectors):
            raise ValueError("geometry must be integer length-three vectors")
        shape, start, stop = vectors
        if np.any(shape <= 0) or np.any(start < 0) or np.any(start >= shape) or np.any(stop <= start):
            raise ValueError("invalid crop geometry")
        extent = stop - start
        if tuple(extent) != image.shape[1:] or np.any(extent % divisor):
            raise ValueError("cache shape does not match divisor-aligned crop")
        if tuple(shape) != tuple(int(value) for value in str(row["shape"]).split("x")):
            raise ValueError("original shape differs from source manifest")
        # The preprocessor only pads an axis when the rounded extent exceeds it.
        padded = stop > shape
        if np.any(padded & ((start != 0) | (stop - shape >= divisor))):
            raise ValueError("illegal padding extent")
        affine = cache["affine"]
        if (affine.shape != (4, 4) or affine.dtype.kind != "f" or not np.isfinite(affine).all()
                or abs(np.linalg.det(affine[:3, :3])) < 1e-8
                or not np.allclose(affine[3], [0, 0, 0, 1])
                or not np.allclose(np.linalg.norm(affine[:3, :3], axis=0), 1, atol=1e-3)):
            raise ValueError("invalid affine or non-1mm spacing")
        norm = cache["norm_stats"]
        if norm.shape != (4, 2) or norm.dtype.kind != "f" or not np.isfinite(norm).all() or np.any(norm[:, 1] <= 0):
            raise ValueError("invalid normalization statistics")
        mask = image[0] != 0
        if any(not np.array_equal(image[channel] != 0, mask) for channel in range(1, 4)):
            raise ValueError("brain masks differ between channels")
        if int(mask.sum()) != int(row["brain_vox"]):
            raise ValueError("brain voxel count differs from source manifest (possible truncated crop)")
        bbox = [tuple(int(value) for value in axis.split(":")) for axis in str(row["bbox"]).split(",")]
        if len(bbox) != 3:
            raise ValueError("invalid source brain bounding box")
        for axis, (low, high) in enumerate(bbox):
            if not (0 <= low < high <= shape[axis] and start[axis] <= low and stop[axis] >= high):
                raise ValueError("crop does not contain source brain bounding box")
            present = np.flatnonzero(np.any(mask, axis=tuple(other for other in range(3) if other != axis)))
            if not present.size or (present[0] + start[axis], present[-1] + start[axis] + 1) != (low, high):
                raise ValueError("cached brain bounding box differs from source manifest")
        labeled = stats is not None
        if ("seg" in cache.files) != labeled:
            raise ValueError("labeled/unlabeled membership mismatch")
        seg = cache["seg"] if labeled else None
        if labeled:
            if seg.dtype != np.uint8 or seg.shape != image.shape[1:] or not np.isin(seg, [0, 1, 2, 3]).all():
                raise ValueError("invalid segmentation shape/dtype/label values")
            for label, name in ((1, "vox_ncr"), (2, "vox_ed"), (3, "vox_et")):
                if int((seg == label).sum()) != int(stats[name]):
                    raise ValueError(f"{name} count differs from label_stats (possible truncated crop)")
        for axis in range(3):
            if padded[axis]:
                slices = [slice(None)] * 3
                slices[axis] = slice(int(shape[axis] - start[axis]), None)
                if np.any(image[(slice(None), *slices)]) or (labeled and np.any(seg[tuple(slices)])):
                    raise ValueError("padding is not zero")


def audit_cache(cfg: DataConfig, *, expected_split_sha256: str, splits_csv: Path | None = None) -> dict:
    splits_csv = splits_csv or split_path(cfg)
    actual_hash = sha256_file(splits_csv)
    if not expected_split_sha256 or actual_hash != expected_split_sha256:
        raise ValueError("frozen split SHA256 mismatch")
    keys = {"case_id", "cohort"}
    manifest_hash, labels_hash = sha256_file(cfg.manifest_csv), sha256_file(cfg.label_stats_csv)
    splits = _table(splits_csv, keys | {"patient_id", "split"})
    manifest = _table(cfg.manifest_csv, keys | {"patient_id", "split_source", "shape", "brain_vox", "bbox"})
    labels = _table(cfg.label_stats_csv, keys | {"vox_ncr", "vox_ed", "vox_et"})
    issues = verify(splits)
    if issues:
        raise ValueError("; ".join(issues))
    if not set(splits["split"]).issubset({"train", "val", "test", "official_val"}):
        raise ValueError("unknown split")
    if not set(manifest["cohort"]).issubset(COHORTS) or not set(manifest["split_source"]).issubset({"train", "val"}):
        raise ValueError("unknown cohort or source split")
    def keyed(table):
        return {(row["cohort"], row["case_id"]): row for row in table.to_dict("records")}
    source_rows, split_rows, label_rows = keyed(manifest), keyed(splits), keyed(labels)
    if source_rows.keys() != split_rows.keys():
        raise ValueError("split and manifest case coverage differ")
    if label_rows.keys() != {key for key, row in source_rows.items() if row["split_source"] == "train"}:
        raise ValueError("label_stats coverage differs from labeled manifest")
    expected_paths = {f"{cohort}/{case_id}.npz" for cohort, case_id in source_rows}
    observed_paths = {path.relative_to(cfg.cache_root).as_posix() for path in cfg.cache_root.rglob("*.npz")}
    if expected_paths != observed_paths:
        raise ValueError("cache file coverage differs from manifest (missing or unexpected NPZ)")
    entries = []
    for key, row in sorted(source_rows.items()):
        split = split_rows[key]
        if row["patient_id"] != split["patient_id"] or ((row["split_source"] == "val") != (split["split"] == "official_val")):
            raise ValueError("patient identity or labeled split membership mismatch")
        relative = f"{key[0]}/{key[1]}.npz"
        path = safe_path(cfg.cache_root, relative)
        before = sha256_file(path)
        try:
            _check_case(path, row, label_rows.get(key), int(cfg.preprocess["size_divisor"]))
        except (ValueError, OSError) as exc:
            raise ValueError(f"cache integrity failure at {relative}: {exc}") from exc
        if sha256_file(path) != before:
            raise ValueError("cache changed during audit")
        entries.append({"path": relative, "sha256": before, "size": path.stat().st_size})
    if (sha256_file(splits_csv) != actual_hash or sha256_file(cfg.manifest_csv) != manifest_hash
            or sha256_file(cfg.label_stats_csv) != labels_hash):
        raise ValueError("manifest or split changed during audit")
    identity = {"encoding": ENCODING, "split_sha256": actual_hash,
                "manifest_sha256": manifest_hash,
                "label_stats_sha256": labels_hash, "files": entries}
    return {"schema_version": 1, "status": "passed", "dataset_identity": canonical_hash(identity),
            "case_count": len(entries), **identity,
            "limitation": "Mechanical hash/geometry/count checks; not raw voxelwise equivalence."}


def verify_audit_identity(cfg: DataConfig, report_path: str | Path, *, expected_split_sha256: str) -> str:
    """Check current exact bytes against a completed CPU audit before GPU training."""
    report = json.loads(Path(report_path).read_text())
    identity = {key: report[key] for key in ("encoding", "split_sha256", "manifest_sha256", "label_stats_sha256", "files")}
    if report.get("schema_version") != 1 or report.get("status") != "passed" or report.get("dataset_identity") != canonical_hash(identity) or report["encoding"] != ENCODING or report.get("case_count") != len(report["files"]):
        raise ValueError("invalid cache audit report")
    for path, expected in ((split_path(cfg), expected_split_sha256), (cfg.manifest_csv, report["manifest_sha256"]),
                           (cfg.label_stats_csv, report["label_stats_sha256"])):
        if sha256_file(path) != expected:
            raise ValueError("data manifest/split changed since cache audit")
    if report["split_sha256"] != expected_split_sha256:
        raise ValueError("audit uses a different frozen split")
    expected = {entry["path"] for entry in report["files"]}
    observed = {path.relative_to(cfg.cache_root).as_posix() for path in cfg.cache_root.rglob("*.npz")}
    if len(expected) != len(report["files"]) or expected != observed:
        raise ValueError("cache coverage changed since audit")
    for entry in report["files"]:
        path = safe_path(cfg.cache_root, entry["path"])
        if path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
            raise ValueError("NPZ content changed since cache audit")
    return report["dataset_identity"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-config")
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--output", required=True, help="local/Snowflake-only integrity report (case identifiers)")
    args = parser.parse_args(argv)
    cfg = DataConfig.load(args.data_config) if args.data_config else DataConfig.load()
    output = Path(args.output).resolve()
    if output.is_relative_to(cfg.cache_root.resolve()) or output.exists():
        raise ValueError("audit output must be new and outside the read-only cache")
    report = audit_cache(cfg, expected_split_sha256=args.expected_split_sha256)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "cases": report["case_count"], "dataset_identity": report["dataset_identity"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
