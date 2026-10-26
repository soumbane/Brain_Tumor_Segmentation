"""Preprocess cases into a compact uint8 cache.

Preprocessing is deliberately minimal, matching what every BraTS 2023 top team did:

1. **Stack** the 4 sequences in canonical ``(t1n, t1c, t2w, t2f)`` order.
2. **Crop** to the nonzero brain bounding box plus a margin (~40% volume reduction),
   expanding only if necessary to preserve annotations outside that box. The MRI-only
   mask still defines normalization; annotation values never alter intensities. Round
   up to a multiple of ``size_divisor`` so the encoder's downsampling levels
   and sliding-window inference never meet a ragged edge. Where rounding up overruns
   the volume (the z axis: 155 -> 160) the box extends past the end and is
   zero-padded, so ``crop_stop`` can exceed ``orig_shape``. Cropping asserts that no
   brain or labeled voxel is lost.
3. **Z-score per channel over nonzero voxels only**, leaving background at exactly 0.
   Normalizing over the whole volume including background is a common and costly
   mistake -- the background dominates the statistics.
4. **Convert labels** to 3 nested binary regions ``(ET, TC, WT)``.
5. **No resampling** (already 1 mm isotropic in SRI24) and **no N4 / skull stripping**
   (already done by the FeTS pipeline).

Storage (measured, not estimated). Each case is one ``savez_compressed`` archive:

    on disk, compressed                   ~4.3 MB/case  -> ~11.8 GB for 2755 cases
    in memory, after decompression        ~19.2 MB/case  -> ~52.9 GB
    (4 ch uint8 over the cropped bbox ~15.4 MB + uint8 label map 0..3 ~3.8 MB)

zlib gets ~4.5x on the uint8 codes, so the *on-disk* footprint is ~11.8 GB, well
inside the ``GPU_NV_M`` node's 93.13 GiB local disk -- the raw NIfTI (~40 GB) would
fit alongside it too. Quantization is retained for the resident-memory and decode
cost, which is what actually binds during training, not for the disk budget.

The quantization step is 10 sigma / 254 ~= 0.039 sigma -- far below any signal a
network can exploit, so this is a storage decision, not a modeling one. The label
map is stored as the original 0..3 integers rather than 3 binary channels: it is
3x smaller and the nested regions are derived on load, losing nothing.

Each case becomes one ``.npz`` holding ``img`` (uint8, C x D x H x W), ``seg``
(uint8, D x H x W; absent for official validation cases), and the metadata needed
to place predictions back into the original 240x240x155 grid.

Usage::

    .venv\\Scripts\\python.exe -m brats.data.preprocess --workers 8
    .venv\\Scripts\\python.exe -m brats.data.preprocess --limit 4 --verify
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from brats.config import DataConfig
from brats.constants import (
    QUANT_BACKGROUND_CODE,
    QUANT_CLIP_SIGMA,
    QUANT_MAX_CODE,
    QUANT_MIN_CODE,
    REGION_LABELS,
    REGIONS,
    SEQUENCES,
    quant_scale,
)

log = logging.getLogger("brats.preprocess")


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def quantize(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Map z-scored floats to uint8 codes in ``[QUANT_MIN_CODE, QUANT_MAX_CODE]``.

    Voxels outside ``mask`` become ``QUANT_BACKGROUND_CODE`` (0), a reserved value
    kept distinct from a legitimate -5 sigma in-brain voxel.
    """
    clipped = np.clip(z, -QUANT_CLIP_SIGMA, QUANT_CLIP_SIGMA)
    span = QUANT_MAX_CODE - QUANT_MIN_CODE
    codes = np.rint(
        (clipped + QUANT_CLIP_SIGMA) / (2.0 * QUANT_CLIP_SIGMA) * span
    ).astype(np.int16) + QUANT_MIN_CODE
    codes = np.clip(codes, QUANT_MIN_CODE, QUANT_MAX_CODE).astype(np.uint8)
    return np.where(mask, codes, np.uint8(QUANT_BACKGROUND_CODE))


#: Value an in-brain voxel decodes to when its z-score is exactly 0.0.
IN_BRAIN_ZERO = np.float32(1e-6)


def dequantize(codes: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quantize`; background maps back to exactly 0.0.

    The code grid is symmetric, so code 128 decodes to a z-score of exactly 0.0 -- the same
    value as background. Anything that treats ``image != 0`` as "inside the brain" would
    then punch holes in the brain at every voxel within ~0.02 sigma of the channel mean
    (~1-2% of voxels). Such voxels decode to ``IN_BRAIN_ZERO`` instead, which is 20,000x
    smaller than the quantization step and so changes nothing a network can see, while
    keeping ``image != 0`` an exact brain mask.
    """
    scale = quant_scale()
    z = (codes.astype(np.float32) - QUANT_MIN_CODE) * scale - QUANT_CLIP_SIGMA
    z = np.where(z == 0.0, IN_BRAIN_ZERO, z)
    return np.where(codes == QUANT_BACKGROUND_CODE, np.float32(0.0), z)


def labels_to_regions(seg: np.ndarray) -> np.ndarray:
    """Expand a 0..3 label map into 3 nested binary channels in (ET, TC, WT) order."""
    out = np.zeros((len(REGIONS), *seg.shape), dtype=np.uint8)
    for i, region in enumerate(REGIONS):
        out[i] = np.isin(seg, list(REGION_LABELS[region])).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------


def brain_bbox(
    mask: np.ndarray, margin: int, divisor: int
) -> tuple[slice, slice, slice]:
    """Bounding box of ``mask``, padded by ``margin`` and rounded up to ``divisor``.

    The returned extent is always a multiple of ``divisor`` and always contains every
    voxel of ``mask``. Where the rounded extent fits inside the volume the box is
    shifted inward to stay in-bounds. Where it does not -- the SRI24 z axis is 155
    slices, and ``ceil(155 / 16) * 16 = 160`` -- the box starts at 0 and runs *past*
    the end of the volume, and :func:`crop_pad` fills the overhang with zeros.

    Rounding *down* to fit the volume instead (``155 -> 144``) silently discards the
    top slices of any brain whose margin-padded extent exceeds 144, taking tumor
    labels with it.
    """
    slices: list[slice] = []
    for axis in range(3):
        size = mask.shape[axis]
        others = tuple(a for a in range(3) if a != axis)
        present = np.any(mask, axis=others)
        idx = np.nonzero(present)[0]
        if idx.size == 0:  # empty mask: keep the whole axis
            lo, hi = 0, size
        else:
            lo = max(0, int(idx[0]) - margin)
            hi = min(size, int(idx[-1]) + 1 + margin)

        extent = hi - lo
        target = ((extent + divisor - 1) // divisor) * divisor
        if target > size:
            # Cannot round up inside the volume: pad past the end instead.
            lo, hi = 0, target
        else:
            grow = target - extent
            lo = max(0, lo - grow // 2)
            hi = lo + target
            if hi > size:
                hi = size
                lo = hi - target
        slices.append(slice(lo, hi))
    return tuple(slices)  # type: ignore[return-value]


def crop_pad(arr: np.ndarray, box: tuple[slice, slice, slice]) -> np.ndarray:
    """Crop the trailing three axes of ``arr`` to ``box``, zero-filling overhang.

    ``box`` may extend beyond ``arr`` (see :func:`brain_bbox`); those voxels are 0,
    which is the background code for images and the background label for segmentations.
    """
    spatial = arr.shape[-3:]
    out = np.zeros(
        (*arr.shape[:-3], *(s.stop - s.start for s in box)), dtype=arr.dtype
    )
    src = tuple(
        slice(max(s.start, 0), min(s.stop, n)) for s, n in zip(box, spatial, strict=True)
    )
    dst = tuple(
        slice(a.start - s.start, a.stop - s.start) for a, s in zip(src, box, strict=True)
    )
    out[(..., *dst)] = arr[(..., *src)]
    return out


# ---------------------------------------------------------------------------
# Per-case worker
# ---------------------------------------------------------------------------


def _load_source(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load canonical MRI channels and reject inconsistent geometry or labels."""
    volumes = []
    affine = None
    shape = None
    for sequence in SEQUENCES:
        volume = nib.load(row[f"path_{sequence}"])
        values = np.asanyarray(volume.dataobj).astype(np.float32)
        current_affine = np.asarray(volume.affine, dtype=np.float64)
        if (values.ndim != 3 or not np.isfinite(values).all()
                or not np.isfinite(current_affine).all()
                or abs(np.linalg.det(current_affine[:3, :3])) < 1e-8):
            raise ValueError(f"{row['case_id']}: invalid {sequence} data or affine")
        if affine is None:
            affine, shape = current_affine, values.shape
        elif values.shape != shape or not np.allclose(current_affine, affine, atol=1e-4, rtol=0):
            raise ValueError(f"{row['case_id']}: {sequence} geometry differs from first image")
        volumes.append(values)
    seg = None
    seg_path = row.get("path_seg")
    if isinstance(seg_path, str) and seg_path:
        volume = nib.load(seg_path)
        values = np.asanyarray(volume.dataobj)
        if values.shape != shape or not np.allclose(volume.affine, affine, atol=1e-4, rtol=0):
            raise ValueError(f"{row['case_id']}: seg geometry differs from image")
        # Validate before conversion: rounding/casting can hide fractional or wrapped labels.
        if not np.isin(values, [0, 1, 2, 3]).all():
            raise ValueError(f"{row['case_id']}: segmentation must contain only integer labels 0..3")
        seg = values.astype(np.uint8)
    return np.stack(volumes, axis=0), affine, seg


def lossless_bbox(mask: np.ndarray, seg: np.ndarray | None, margin: int,
                  divisor: int) -> tuple[slice, slice, slice]:
    """Keep the image-only crop unless it would discard an annotation."""
    box = brain_bbox(mask, margin, divisor)
    if seg is not None and np.count_nonzero(seg[box]) != np.count_nonzero(seg):
        box = brain_bbox(mask | (seg > 0), margin, divisor)
    return box


def preprocess_case(
    row: dict, out_dir: str, margin: int, divisor: int, overwrite: bool = False
) -> dict:
    """Preprocess one case into ``out_dir/<cohort>/<case_id>.npz``."""
    case_id = row["case_id"]
    cohort = row["cohort"]
    dest = Path(out_dir) / cohort / f"{case_id}.npz"
    if dest.is_file() and not overwrite:
        return {"case_id": case_id, "cohort": cohort, "status": "skipped",
                "path": str(dest), "bytes": dest.stat().st_size}

    dest.parent.mkdir(parents=True, exist_ok=True)

    # ---- load and stack, asserting geometry agreement -------------------
    img4, affine, seg = _load_source(row)
    orig_shape = img4.shape[1:]
    # Annotation support must not change MRI normalization or background codes.
    mask = np.any(img4 != 0, axis=0)

    # ---- crop ----------------------------------------------------------
    box = lossless_bbox(mask, seg, margin=margin, divisor=divisor)
    n_brain = int(mask.sum())
    n_fg = int((seg > 0).sum()) if seg is not None else 0
    img4 = crop_pad(img4, box)
    mask_c = crop_pad(mask, box)
    if seg is not None:
        seg = crop_pad(seg, box)
    # Cropping must be lossless for everything that is not background. A box that
    # clips brain or tumor would otherwise surface nowhere downstream: the cached
    # image and label are cut identically, and evaluation scores in cropped space.
    if int(mask_c.sum()) != n_brain:
        raise ValueError(
            f"{case_id}: crop dropped {n_brain - int(mask_c.sum())} brain voxels "
            f"(box {box})"
        )
    if seg is not None and int((seg > 0).sum()) != n_fg:
        raise ValueError(
            f"{case_id}: crop dropped {n_fg - int((seg > 0).sum())} labeled voxels "
            f"(box {box})"
        )

    # ---- per-channel z-score over nonzero voxels only ------------------
    codes = np.empty(img4.shape, dtype=np.uint8)
    stats = np.zeros((img4.shape[0], 2), dtype=np.float32)
    for c in range(img4.shape[0]):
        chan = img4[c]
        fg = chan[mask_c]
        if fg.size == 0:
            mean, std = 0.0, 1.0
        else:
            mean = float(fg.mean())
            std = float(fg.std())
            if std < 1e-8:
                std = 1.0
        stats[c] = (mean, std)
        codes[c] = quantize((chan - mean) / std, mask_c)
    del img4

    payload: dict[str, np.ndarray] = {
        "img": codes,
        "affine": affine.astype(np.float32),
        "orig_shape": np.asarray(orig_shape, dtype=np.int32),
        # Crop offsets, needed to paste predictions back into the original grid.
        "crop_start": np.asarray([s.start for s in box], dtype=np.int32),
        "crop_stop": np.asarray([s.stop for s in box], dtype=np.int32),
        "norm_stats": stats,
    }
    if seg is not None:
        payload["seg"] = seg

    np.savez_compressed(dest, **payload)
    return {
        "case_id": case_id,
        "cohort": cohort,
        "status": "written",
        "path": str(dest),
        "bytes": dest.stat().st_size,
        "cropped_shape": "x".join(str(s.stop - s.start) for s in box),
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_roundtrip(npz_path: Path, row: dict, *, margin: int | None = None,
                     divisor: int | None = None) -> dict:
    """Verify cache geometry, normalization and voxelwise fidelity to raw NIfTI.

    Supplying margin/divisor also verifies the current crop policy before reuse.
    Ground truth is inspected only for mechanical integrity, never model tuning.
    """
    with np.load(npz_path, allow_pickle=False) as cache:
        codes = cache["img"]
        stats = cache["norm_stats"]
        start, stop, shape = (cache[key] for key in ("crop_start", "crop_stop", "orig_shape"))
        cached_affine = cache["affine"]
        cached_seg = cache["seg"] if "seg" in cache.files else None
    source, affine, source_seg = _load_source(row)
    if any(vector.shape != (3,) or vector.dtype.kind not in "iu" for vector in (start, stop, shape)):
        raise ValueError("invalid cache geometry vectors")
    if (not np.array_equal(shape, source.shape[1:]) or np.any(start < 0)
            or np.any(start >= shape) or np.any(stop <= start)):
        raise ValueError("cache geometry differs from source")
    if (codes.dtype != np.uint8 or codes.shape != (len(SEQUENCES), *tuple(stop - start))
            or stats.shape != (len(SEQUENCES), 2) or not np.isfinite(stats).all()
            or np.any(stats[:, 1] <= 0)):
        raise ValueError("invalid image codes or normalization statistics")
    box = tuple(slice(int(low), int(high)) for low, high in zip(start, stop, strict=True))
    full_mask = np.any(source != 0, axis=0)
    expected_mask = crop_pad(full_mask, box)
    mask = codes[0] != QUANT_BACKGROUND_CODE
    mask_consistent = all(np.array_equal(codes[index] != QUANT_BACKGROUND_CODE, mask)
                          for index in range(len(SEQUENCES)))
    crop_lossless = np.array_equal(expected_mask, mask) and np.count_nonzero(full_mask) == np.count_nonzero(mask)
    geometry_ok = (cached_affine.shape == (4, 4) and np.isfinite(cached_affine).all()
                   and np.allclose(cached_affine, affine, atol=1e-4, rtol=0))
    policy_ok = True
    if margin is not None or divisor is not None:
        if margin is None or divisor is None:
            raise ValueError("margin and divisor must be supplied together")
        policy_ok = box == lossless_bbox(full_mask, source_seg, margin, divisor)

    worst_z = 0.0
    norm_ok = True
    for index in range(len(SEQUENCES)):
        original = crop_pad(source[index], box)
        foreground = original[expected_mask]
        mean = float(foreground.mean()) if foreground.size else 0.0
        std = float(foreground.std()) if foreground.size else 1.0
        std = 1.0 if std < 1e-8 else std
        norm_ok = norm_ok and np.allclose(stats[index], [mean, std], atol=1e-5, rtol=1e-6)
        expected = np.where(expected_mask, np.clip((original - mean) / std,
                                                  -QUANT_CLIP_SIGMA, QUANT_CLIP_SIGMA), 0.0)
        worst_z = max(worst_z, float(np.abs(dequantize(codes[index]) - expected).max()))

    membership_ok = (source_seg is None) == (cached_seg is None)
    seg_ok = None if source_seg is None and cached_seg is None else False
    if source_seg is not None and cached_seg is not None:
        seg_ok = (cached_seg.dtype == np.uint8 and np.array_equal(crop_pad(source_seg, box), cached_seg)
                  and np.count_nonzero(source_seg) == np.count_nonzero(cached_seg))
    return {
        "case_id": row["case_id"],
        "max_abs_z_error": worst_z,
        "within_quant_step": worst_z <= quant_scale() * 0.51,
        "mask_consistent": bool(mask_consistent),
        "crop_lossless": bool(crop_lossless),
        "seg_lossless": None if seg_ok is None else bool(seg_ok),
        "geometry_ok": bool(geometry_ok),
        "normalization_ok": bool(norm_ok),
        "crop_policy_matches": bool(policy_ok),
        "label_membership_ok": membership_ok,
        "label_voxels_outside_mri_mask": int(np.count_nonzero((source_seg > 0) & ~full_mask)) if source_seg is not None else 0,
        "mb": npz_path.stat().st_size / 1e6,
    }


def roundtrip_passed(check: dict) -> bool:
    return all(check[key] for key in ("within_quant_step", "mask_consistent", "crop_lossless",
                                     "geometry_ok", "normalization_ok", "crop_policy_matches",
                                     "label_membership_ok")) and check["seg_lossless"] is not False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N cases per cohort")
    ap.add_argument("--cohort", action="append", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="round-trip a few cached cases against the source NIfTI",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    cfg = DataConfig.load(args.config) if args.config else DataConfig.load()

    if not cfg.manifest_csv.is_file():
        raise SystemExit(
            f"No manifest at {cfg.manifest_csv}. Run `python -m brats.data.manifest`."
        )
    manifest = pd.read_csv(cfg.manifest_csv)
    if args.cohort:
        manifest = manifest[manifest.cohort.isin(args.cohort)]
    if args.limit:
        manifest = (
            manifest.groupby(["cohort", "split_source"], group_keys=False)
            .head(args.limit)
            .reset_index(drop=True)
        )

    margin = int(cfg.preprocess["crop_margin"])
    divisor = int(cfg.preprocess["size_divisor"])
    out_dir = cfg.cache_root
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = manifest.to_dict("records")
    log.info("Preprocessing %d cases -> %s", len(rows), out_dir)

    results: list[dict] = []
    failures: list[tuple[str, str]] = []
    if args.workers <= 1:
        for i, row in enumerate(rows, 1):
            try:
                results.append(
                    preprocess_case(row, str(out_dir), margin, divisor, args.overwrite)
                )
            except Exception as exc:  # noqa: BLE001
                failures.append((row["case_id"], f"{type(exc).__name__}: {exc}"))
            if i % 100 == 0:
                log.info("  %d/%d", i, len(rows))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(
                    preprocess_case, row, str(out_dir), margin, divisor, args.overwrite
                ): row["case_id"]
                for row in rows
            }
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    results.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    failures.append((futs[fut], f"{type(exc).__name__}: {exc}"))
                if i % 100 == 0:
                    log.info("  %d/%d", i, len(rows))

    res = pd.DataFrame(results)
    if not res.empty:
        total_gb = res["bytes"].sum() / 1e9
        mean_mb = res["bytes"].mean() / 1e6
        print("=" * 64)
        print("Preprocessing cache")
        print("=" * 64)
        print(res.groupby(["cohort", "status"]).size().to_string())
        print(f"\nmean size/case : {mean_mb:.1f} MB")
        print(f"total          : {total_gb:.2f} GB for {len(res)} cases")
        if len(res) < len(manifest):
            print(
                f"projected full : "
                f"{total_gb / max(len(res), 1) * len(manifest):.1f} GB "
                f"for {len(manifest)} cases"
            )
        print("node local disk: 93.13 GiB on GPU_NV_M")
        res.to_csv(cfg.manifest_dir / "cache_index.csv", index=False)

    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for cid, msg in failures[:20]:
            print(f"  {cid}: {msg}")

    if args.verify and not res.empty:
        print("\n" + "=" * 64)
        print("Round-trip verification")
        print("=" * 64)
        by_id = {r["case_id"]: r for r in rows}
        checks = [
            verify_roundtrip(Path(r["path"]), by_id[r["case_id"]])
            for r in res.head(4).to_dict("records")
        ]
        vdf = pd.DataFrame(checks)
        print(vdf.to_string(index=False))
        print(f"\nquantization step = {quant_scale():.5f} sigma")
        if not bool(vdf["within_quant_step"].all()):
            print("VERIFICATION FAILED: error exceeds half a quantization step")
            return 1
        if not bool(vdf["mask_consistent"].all()):
            print("VERIFICATION FAILED: brain mask differs between channels")
            return 1
        if not bool(vdf["crop_lossless"].all()):
            print(
                "VERIFICATION FAILED: the crop dropped brain voxels. This cache was "
                "written before the z-axis fix; re-run with --overwrite."
            )
            return 1
        if (vdf["seg_lossless"].dropna() == False).any():  # noqa: E712
            print("VERIFICATION FAILED: segmentation labels not preserved exactly")
            return 1
        if not all(roundtrip_passed(check) for check in checks):
            print("VERIFICATION FAILED: geometry, normalization, or source membership mismatch")
            return 1
        print("Round-trip OK: intensities within half a step, labels lossless.")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
