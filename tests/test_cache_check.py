"""Mechanical all-case auditing detects historical crop loss without prediction."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from brats.artifacts import sha256_file
from brats.config import DataConfig
from brats.data.cache_check import audit_cache, verify_audit_identity


@pytest.fixture
def cache(tmp_path):
    cfg = replace(DataConfig.load(), cache_root=tmp_path / "cache", manifest_dir=tmp_path / "manifest",
                  splits={"output": str(tmp_path / "splits.csv")}, preprocess={"size_divisor": 16})
    cfg.manifest_dir.mkdir()
    rows, splits, labels = [], [], []
    for cohort in ("GLI", "MEN", "PED"):
        (cfg.cache_root / cohort).mkdir(parents=True)
        for index, split in enumerate(("train", "val", "test", "official_val")):
            case = f"{cohort}-{index}"
            labeled = split != "official_val"
            image = np.zeros((4, 16, 16, 32), dtype=np.uint8)
            image[:, 2:7, 3:8, 4:17] = 128
            seg = np.zeros(image.shape[1:], dtype=np.uint8)
            seg[3:5, 4:6, 8:10] = 3
            arrays = {"img": image, "orig_shape": np.array([16, 16, 17], dtype=np.int32),
                      "crop_start": np.array([0, 0, 0], dtype=np.int32),
                      "crop_stop": np.array([16, 16, 32], dtype=np.int32),
                      "affine": np.eye(4, dtype=np.float32), "norm_stats": np.ones((4, 2), dtype=np.float32)}
            if labeled:
                arrays["seg"] = seg
                labels.append({"case_id": case, "cohort": cohort, "vox_ncr": 0, "vox_ed": 0, "vox_et": 8})
            np.savez_compressed(cfg.cache_root / cohort / f"{case}.npz", **arrays)
            rows.append({"case_id": case, "cohort": cohort, "patient_id": case, "split_source": "train" if labeled else "val",
                         "shape": "16x16x17", "brain_vox": 325, "bbox": "2:7,3:8,4:17"})
            splits.append({"case_id": case, "cohort": cohort, "patient_id": case, "split": split})
    pd.DataFrame(rows).to_csv(cfg.manifest_csv, index=False)
    pd.DataFrame(labels).to_csv(cfg.label_stats_csv, index=False)
    pd.DataFrame(splits).to_csv(cfg.splits["output"], index=False)
    return cfg


def _audit(cfg):
    return audit_cache(cfg, expected_split_sha256=sha256_file(cfg.splits["output"]))


def _mutate(cfg, callback):
    path = cfg.cache_root / "GLI" / "GLI-0.npz"
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    callback(arrays)
    np.savez_compressed(path, **arrays)


def test_full_audit_binds_exact_npz_and_split_bytes(cache, tmp_path):
    report = _audit(cache)
    assert report["case_count"] == 12 and report["status"] == "passed"
    output = tmp_path / "audit.json"
    output.write_text(json.dumps(report))
    identity = verify_audit_identity(cache, output, expected_split_sha256=sha256_file(cache.splits["output"]))
    assert identity == report["dataset_identity"]
    _mutate(cache, lambda arrays: arrays["img"].__setitem__((0, 2, 3, 4), 129))
    with pytest.raises(ValueError, match="NPZ content"):
        verify_audit_identity(cache, output, expected_split_sha256=sha256_file(cache.splits["output"]))


@pytest.mark.parametrize("mutate,reason", [
    (lambda arrays: arrays["img"].__setitem__((slice(None), 2, 3, 4), 0), "brain voxel count"),
    (lambda arrays: arrays["seg"].__setitem__((3, 4, 8), 0), "vox_et"),
    (lambda arrays: arrays["seg"].__setitem__((0, 0, 0), 4), "label values"),
    (lambda arrays: arrays["seg"].__setitem__((0, 0, 20), 1), "vox_ncr"),
    (lambda arrays: arrays["affine"].__setitem__((0, 0), 0), "affine"),
    (lambda arrays: arrays.__setitem__("img", arrays["img"].astype(np.float32)), "uint8"),
    (lambda arrays: arrays.__delitem__("norm_stats"), "missing required"),
])
def test_corruption_is_a_stop_not_a_repair(cache, mutate, reason):
    _mutate(cache, mutate)
    path = cache.cache_root / "GLI" / "GLI-0.npz"
    before = sha256_file(path)
    with pytest.raises(ValueError, match=reason):
        _audit(cache)
    assert sha256_file(path) == before


def test_frozen_split_hash_required(cache):
    with pytest.raises(ValueError, match="split SHA256"):
        audit_cache(cache, expected_split_sha256="wrong")


def test_patient_leakage_reuses_split_contract(cache):
    splits = pd.read_csv(cache.splits["output"])
    splits.loc[1, "patient_id"] = splits.loc[0, "patient_id"]
    splits.to_csv(cache.splits["output"], index=False)
    with pytest.raises(ValueError, match="leakage"):
        _audit(cache)


def test_unexpected_cache_file_and_duplicate_manifest_rejected(cache):
    path = cache.cache_root / "GLI" / "unexpected.npz"
    path.write_bytes(b"unknown")
    with pytest.raises(ValueError, match="coverage"):
        _audit(cache)
    path.unlink()
    manifest = pd.read_csv(cache.manifest_csv)
    pd.concat([manifest, manifest.iloc[:1]]).to_csv(cache.manifest_csv, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        _audit(cache)
