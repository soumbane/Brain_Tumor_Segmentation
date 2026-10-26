"""Mechanical all-case auditing detects historical crop loss without prediction."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brats.artifacts import sha256_file
from brats.config import DataConfig
from brats.data.cache_check import audit_cache, verify_audit_identity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rebuild_cache


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


@pytest.fixture
def raw_cache(tmp_path, monkeypatch):
    import nibabel as nib
    from brats.constants import SEQUENCES
    from brats.data.preprocess import preprocess_case

    monkeypatch.delenv('BRATS_SPLITS_CSV', raising=False)
    raw = tmp_path / 'raw'
    raw.mkdir()
    manifest = tmp_path / 'metadata'
    manifest.mkdir()
    cfg = replace(DataConfig.load(), cache_root=tmp_path / 'existing', manifest_dir=manifest,
                  splits={'output': str(tmp_path / 'split.csv')},
                  preprocess={'crop_margin': 1, 'size_divisor': 16})
    rows, splits, labels = [], [], []
    for cohort in ('GLI', 'MEN', 'PED'):
        for index, split in enumerate(('train', 'val', 'test', 'official_val')):
            case = f'{cohort}-{index}'
            directory = raw / cohort / case
            directory.mkdir(parents=True)
            values = np.zeros((16, 16, 17), np.float32)
            values[2:7, 3:8, 4:17] = 100
            row = {'case_id': case, 'cohort': cohort, 'patient_id': case,
                   'split_source': 'val' if split == 'official_val' else 'train',
                   'shape': '16x16x17', 'brain_vox': 325, 'bbox': '2:7,3:8,4:17'}
            for sequence in SEQUENCES:
                path = directory / f'{case}-{sequence}.nii.gz'
                nib.save(nib.Nifti1Image(values, np.eye(4)), path)
                row[f'path_{sequence}'] = str(path)
            row['path_seg'] = ''
            if split != 'official_val':
                seg = np.zeros(values.shape, np.uint8)
                seg[3:5, 4:6, 8:10] = 3
                path = directory / f'{case}-seg.nii.gz'
                nib.save(nib.Nifti1Image(seg, np.eye(4)), path)
                row['path_seg'] = str(path)
                labels.append({'case_id': case, 'cohort': cohort, 'vox_ncr': 0, 'vox_ed': 0, 'vox_et': 8})
            preprocess_case(row, str(cfg.cache_root), 1, 16)
            rows.append(row)
            splits.append({'case_id': case, 'cohort': cohort, 'patient_id': case, 'split': split})
    pd.DataFrame(rows).to_csv(cfg.manifest_csv, index=False)
    pd.DataFrame(labels).to_csv(cfg.label_stats_csv, index=False)
    pd.DataFrame(splits).to_csv(cfg.splits['output'], index=False)
    cfg.integrity_csv.write_text('fixture_integrity\npassed\n')
    return cfg, raw


def test_repair_reuses_only_verified_files_rebuilds_missing_and_corrupt(raw_cache, tmp_path):
    cfg, raw = raw_cache
    missing = cfg.cache_root / 'GLI/GLI-0.npz'
    missing.unlink()
    corrupt = cfg.cache_root / 'MEN/MEN-0.npz'
    with np.load(corrupt, allow_pickle=False) as cache:
        arrays = {key: cache[key] for key in cache.files}
    arrays['norm_stats'][0, 0] += 10
    np.savez_compressed(corrupt, **arrays)
    before = {str(path): sha256_file(path) for path in cfg.cache_root.rglob('*.npz')}
    output = tmp_path / 'repair'
    summary = rebuild_cache.rebuild(cfg, raw_root=raw, output=output, reuse_cache=cfg.cache_root,
                                    source_root=str(raw), workers=1,
                                    expected_split_sha256=sha256_file(cfg.splits['output']))
    assert summary['status'] == 'passed'
    assert summary['actions'] == {'reused_verified': 10, 'rebuilt': 2}
    assert before == {str(path): sha256_file(path) for path in cfg.cache_root.rglob('*.npz')}
    assert not missing.exists()
    assert verify_audit_identity(replace(cfg, cache_root=output / 'cache'), output / 'cache-audit.json',
                                 expected_split_sha256=sha256_file(cfg.splits['output'])) == summary['dataset_identity']


def test_repair_failure_never_writes_completion(raw_cache, tmp_path):
    import nibabel as nib
    cfg, raw = raw_cache
    manifest = pd.read_csv(cfg.manifest_csv)
    path = manifest.iloc[0]['path_seg']
    volume = nib.load(path)
    seg = np.asanyarray(volume.dataobj).copy()
    seg[3, 4, 8] = 4
    nib.save(nib.Nifti1Image(seg, volume.affine), path)
    output = tmp_path / 'failed-repair'
    with pytest.raises(ValueError, match='no completion'):
        rebuild_cache.rebuild(cfg, raw_root=raw, output=output, reuse_cache=cfg.cache_root,
                              source_root=str(raw), workers=1,
                              expected_split_sha256=sha256_file(cfg.splits['output']))
    assert not (output / 'cache-audit.json').exists()
    report = json.loads((output / 'repair-report.json').read_text())
    assert report['status'] == 'failed' and len(report['failures']) == 1


@pytest.mark.parametrize('target', ['existing', 'inside_cache', 'inside_raw', 'wrong_hash'])
def test_repair_refuses_unsafe_paths_and_wrong_split(raw_cache, tmp_path, target):
    cfg, raw = raw_cache
    output = {'existing': cfg.cache_root, 'inside_cache': cfg.cache_root / 'new',
              'inside_raw': raw / 'GLI/new', 'wrong_hash': tmp_path / 'wrong'}[target]
    digest = '0' * 64 if target == 'wrong_hash' else sha256_file(cfg.splits['output'])
    with pytest.raises(ValueError):
        rebuild_cache.rebuild(cfg, raw_root=raw, output=output, source_root=str(raw), workers=1,
                              expected_split_sha256=digest)
    if target != 'existing':
        assert not output.exists()


def test_repair_rejects_candidate_symlink_escape(raw_cache, tmp_path):
    from brats.artifacts import ArtifactError
    cfg, raw = raw_cache
    rows, labels = rebuild_cache.load_inputs(cfg, raw, str(raw))
    row = rows[0]
    candidate = cfg.cache_root / row['cohort'] / f"{row['case_id']}.npz"
    external = tmp_path / 'external.npz'
    external.write_bytes(candidate.read_bytes())
    candidate.unlink()
    candidate.symlink_to(external)
    before = sha256_file(external)
    with pytest.raises(ArtifactError, match='escapes'):
        rebuild_cache.repair_case(row, labels[row['cohort'], row['case_id']],
                                  str(tmp_path / 'repair'), str(cfg.cache_root), 1, 16)
    assert sha256_file(external) == before


def test_repair_does_not_mark_complete_if_cache_changes_after_source_check(raw_cache, tmp_path, monkeypatch):
    cfg, raw = raw_cache
    real_audit = rebuild_cache.audit_cache
    def changed_audit(*args, **kwargs):
        audit = real_audit(*args, **kwargs)
        audit['files'][0]['sha256'] = '0' * 64
        return audit
    monkeypatch.setattr(rebuild_cache, 'audit_cache', changed_audit)
    output = tmp_path / 'changed-repair'
    with pytest.raises(ValueError, match='bytes changed'):
        rebuild_cache.rebuild(cfg, raw_root=raw, output=output, reuse_cache=cfg.cache_root,
                              source_root=str(raw), workers=1,
                              expected_split_sha256=sha256_file(cfg.splits['output']))
    assert not (output / 'cache-audit.json').exists()
    assert json.loads((output / 'repair-report.json').read_text())['status'] == 'failed'
