"""Repair into a new local cache, validating reused files against original NIfTI.

Never overwrites old caches, edits labels/splits, uploads files, or starts training.
A completion audit is written last, only after every case passes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path, PureWindowsPath
import shutil
import zipfile

import pandas as pd

from brats.artifacts import safe_id, safe_path, sha256_file
from brats.config import DataConfig
from brats.constants import COHORTS, SEQUENCES
from brats.data.cache_check import _check_case, audit_cache, split_path
from brats.data.preprocess import preprocess_case, roundtrip_passed, verify_roundtrip
from brats.data.splits import verify


def source_paths(row: dict) -> dict[str, Path]:
    return {key: Path(row[key]) for key in [*(f'path_{name}' for name in SEQUENCES), 'path_seg']
            if isinstance(row.get(key), str) and row[key]}


def repair_case(row: dict, stats: dict | None, cache_root: str, reuse_root: str | None,
                margin: int, divisor: int) -> dict:
    """Copy only a source-verified candidate; otherwise rebuild into the new root."""
    sources = source_paths(row)
    source_hashes = {key: sha256_file(path) for key, path in sources.items()}
    relative = f"{row['cohort']}/{row['case_id']}.npz"
    destination = safe_path(cache_root, relative)
    if destination.exists():
        raise ValueError('refusing to replace a repair output')
    candidate = safe_path(reuse_root, relative) if reuse_root else None
    check = None
    reason = 'missing_candidate'
    candidate_hash = None
    if candidate is not None and candidate.is_file():
        candidate_hash = sha256_file(candidate)
        try:
            _check_case(candidate, row, stats, divisor)
            check = verify_roundtrip(candidate, row, margin=margin, divisor=divisor)
            if not roundtrip_passed(check):
                raise ValueError('candidate failed original-source round-trip')
        except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
            reason = f'{type(exc).__name__}: {exc}'
            check = None
        if sha256_file(candidate) != candidate_hash:
            raise ValueError('reuse candidate changed during verification')
    if check is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with candidate.open('rb') as source, destination.open('xb') as target:
            shutil.copyfileobj(source, target)
        if sha256_file(destination) != candidate_hash or sha256_file(candidate) != candidate_hash:
            raise ValueError('reuse candidate changed during copy')
        action = 'reused_verified'
        reason = ''
    else:
        preprocess_case(row, cache_root, margin, divisor)
        _check_case(destination, row, stats, divisor)
        check = verify_roundtrip(destination, row, margin=margin, divisor=divisor)
        if not roundtrip_passed(check):
            raise ValueError(f"rebuilt case failed original-source verification: {check}")
        action = 'rebuilt'
    if any(sha256_file(path) != source_hashes[key] for key, path in sources.items()):
        raise ValueError('original NIfTI changed during repair/verification')
    return {'case_id': row['case_id'], 'cohort': row['cohort'], 'path': relative,
            'action': action, 'rebuild_reason': reason, 'size': destination.stat().st_size,
            'sha256': sha256_file(destination), 'reuse_sha256': candidate_hash,
            'source_sha256': source_hashes, 'roundtrip': check}


def load_inputs(cfg: DataConfig, raw_root: Path, source_root: str) -> tuple[list[dict], dict]:
    """Validate frozen metadata before resolving original paths within raw_root."""
    records = pd.read_csv(cfg.manifest_csv).to_dict('records')
    labels = pd.read_csv(cfg.label_stats_csv).to_dict('records')
    splits = pd.read_csv(split_path(cfg))
    issues = verify(splits)
    if issues:
        raise ValueError('; '.join(issues))
    keys = [(row['cohort'], row['case_id']) for row in records]
    split_keys = list(zip(splits.cohort, splits.case_id, strict=True))
    label_keys = [(row['cohort'], row['case_id']) for row in labels]
    if (not records or len(set(keys)) != len(keys) or len(set(split_keys)) != len(split_keys)
            or set(keys) != set(split_keys) or len(set(label_keys)) != len(label_keys)
            or set(label_keys) != {key for key, row in zip(keys, records, strict=True) if row['split_source'] == 'train'}):
        raise ValueError('manifest, split, or label coverage differs')
    by_split = {(row.cohort, row.case_id): row for row in splits.itertuples()}
    for record in records:
        if record['cohort'] not in COHORTS or record['split_source'] not in {'train', 'val'}:
            raise ValueError('invalid source cohort or membership')
        safe_id(record['case_id'])
        split = by_split[record['cohort'], record['case_id']]
        if (split.patient_id != record['patient_id'] or split.split not in {'train', 'val', 'test', 'official_val'}
                or (split.split == 'official_val') != (record['split_source'] == 'val')):
            raise ValueError('source/split patient or labeled membership mismatch')
        for key in [*(f'path_{sequence}' for sequence in SEQUENCES), 'path_seg']:
            value = record.get(key)
            if not isinstance(value, str) or not value:
                if key != 'path_seg' or record['split_source'] == 'train':
                    raise ValueError(f'missing original source path: {key}')
                record[key] = ''
                continue
            if key == 'path_seg' and record['split_source'] == 'val':
                raise ValueError('official validation unexpectedly has labels')
            relative = PureWindowsPath(value).relative_to(PureWindowsPath(source_root))
            if '..' in relative.parts:
                raise ValueError('source path traversal')
            path = raw_root.joinpath(*relative.parts).resolve()
            if not path.is_relative_to(raw_root) or not path.is_file():
                raise ValueError(f'original source missing or outside root: {path}')
            record[key] = str(path)
    return records, dict(zip(label_keys, labels, strict=True))


def _write_json(path: Path, payload: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write('\n')


def rebuild(cfg: DataConfig, *, raw_root: Path, output: Path, expected_split_sha256: str,
            reuse_cache: Path | None = None, source_root: str = 'C:/data/brats2023', workers: int = 4) -> dict:
    if workers < 1:
        raise ValueError('workers must be positive')
    raw_root, output = raw_root.resolve(), output.resolve()
    reuse_cache = reuse_cache.resolve() if reuse_cache else None
    if output.exists() or not output.parent.is_dir():
        raise ValueError('output must be a new directory with an existing parent')
    protected = [cfg.cache_root.resolve(), cfg.manifest_dir.resolve(), split_path(cfg).resolve(),
                 *(raw_root / cohort for cohort in COHORTS)]
    if reuse_cache:
        if not reuse_cache.is_dir():
            raise ValueError('reuse cache does not exist')
        protected.append(reuse_cache)
    if raw_root.is_relative_to(output) or any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
        raise ValueError('output overlaps protected input data')
    metadata_paths = {'split_sha256': split_path(cfg), 'manifest_sha256': cfg.manifest_csv,
                      'label_stats_sha256': cfg.label_stats_csv, 'integrity_sha256': cfg.integrity_csv}
    metadata_hashes = {key: sha256_file(path) for key, path in metadata_paths.items()}
    if not expected_split_sha256 or metadata_hashes['split_sha256'] != expected_split_sha256:
        raise ValueError('frozen split SHA256 mismatch')
    records, labels = load_inputs(cfg, raw_root, source_root)
    if shutil.disk_usage(output.parent).free < max(2 * 1024**3, len(records) * 10 * 1024**2):
        raise OSError('insufficient free disk for isolated repair')
    output.mkdir(exist_ok=False)
    cache_root = output / 'cache'
    margin, divisor = int(cfg.preprocess['crop_margin']), int(cfg.preprocess['size_divisor'])
    _write_json(output / 'repair-start.json', {
        'status': 'in_progress', 'expected_cases': len(records), **metadata_hashes,
        'crop_margin': margin, 'size_divisor': divisor,
        'crop_policy': 'image_bbox_expand_only_to_preserve_labels',
        'normalization_mask': 'MRI_nonzero_union_only',
        'reuse_cache': str(reuse_cache) if reuse_cache else None,
    })
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(repair_case, row, labels.get((row['cohort'], row['case_id'])),
                               str(cache_root), str(reuse_cache) if reuse_cache else None, margin, divisor): row
                   for row in records}
        for completed, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failure = {'case_id': row['case_id'], 'cohort': row['cohort'], 'error': f'{type(exc).__name__}: {exc}'}
                failures.append(failure)
                print(json.dumps({'case_failure': failure}), flush=True)
            if completed % 100 == 0 or completed == len(records):
                print(json.dumps({'checked': completed, 'total': len(records), 'failed': len(failures)}), flush=True)
    if any(sha256_file(path) != metadata_hashes[key] for key, path in metadata_paths.items()):
        failures.append({'error': 'metadata changed during repair'})
    report = {'status': 'failed' if failures else 'source_verified', **metadata_hashes,
              'case_count': len(results), 'actions': dict(Counter(row['action'] for row in results)),
              'cases': sorted(results, key=lambda row: row['path']), 'failures': failures}
    if failures:
        _write_json(output / 'repair-report.json', report)
        raise ValueError(f'{len(failures)} failures; no completion audit written')
    print('Running all-case mechanical integrity audit', flush=True)
    try:
        audit = audit_cache(replace(cfg, cache_root=cache_root), expected_split_sha256=expected_split_sha256)
        audited = {entry['path']: entry['sha256'] for entry in audit['files']}
        if audited != {entry['path']: entry['sha256'] for entry in results}:
            raise ValueError('cache bytes changed after original-source verification')
        if any(sha256_file(path) != metadata_hashes[key] for key, path in metadata_paths.items()):
            raise ValueError('metadata changed before completion')
    except Exception as exc:
        report.update(status='failed', failures=[{'error': f'{type(exc).__name__}: {exc}'}])
        _write_json(output / 'repair-report.json', report)
        raise
    report.update(status='passed', dataset_identity=audit['dataset_identity'])
    pd.DataFrame([{key: row[key] for key in ('case_id', 'cohort', 'path', 'action', 'size', 'sha256')}
                  for row in report['cases']]).to_csv(output / 'cache_index.csv', index=False)
    _write_json(output / 'repair-report.json', report)
    _write_json(output / 'cache-audit.json', audit)
    summary = {key: report[key] for key in ('status', 'case_count', 'actions', 'dataset_identity', 'split_sha256')}
    summary['annotations_outside_mri_cases'] = sum(row['roundtrip']['label_voxels_outside_mri_mask'] > 0 for row in results)
    print(json.dumps(summary), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--manifest-source-root', default='C:/data/brats2023')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-cache', type=Path)
    parser.add_argument('--expected-split-sha256', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    rebuild(DataConfig.load(), raw_root=args.raw_root, output=args.output,
            expected_split_sha256=args.expected_split_sha256, reuse_cache=args.reuse_cache,
            source_root=args.manifest_source_root, workers=args.workers)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
