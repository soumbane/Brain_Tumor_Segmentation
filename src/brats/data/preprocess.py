"""Preprocess cases into a compact uint8 cache.

Preprocessing is deliberately minimal, matching what every BraTS 2023 top team did:

1. **Stack** the 4 sequences in canonical ``(t1n, t1c, t2w, t2f)`` order.
2. **Crop** to the nonzero brain bounding box plus a margin (~40% volume reduction),
   rounded up to a multiple of ``size_divisor`` so the encoder's downsampling levels
   and sliding-window inference never meet a ragged edge.
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


def dequantize(codes: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quantize`; background maps back to exactly 0.0."""
    scale = quant_scale()
    z = (codes.astype(np.float32) - QUANT_MIN_CODE) * scale - QUANT_CLIP_SIGMA
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

    The box is clamped to the volume, then, if rounding would overflow, shifted back
    inward so the returned extent is always in-bounds and always a multiple of
    ``divisor``.
    """
    slices: list[slice] = []
    for axis in range(3):
        others = tuple(a for a in range(3) if a != axis)
        present = np.any(mask, axis=others)
        idx = np.nonzero(present)[0]
        if idx.size == 0:  # empty mask: keep the whole axis
            lo, hi = 0, mask.shape[axis]
        else:
            lo = max(0, int(idx[0]) - margin)
            hi = min(mask.shape[axis], int(idx[-1]) + 1 + margin)

        extent = hi - lo
        target = ((extent + divisor - 1) // divisor) * divisor
        target = min(target, (mask.shape[axis] // divisor) * divisor) or divisor
        grow = target - extent
        if grow > 0:
            lo = max(0, lo - grow // 2)
            hi = lo + target
            if hi > mask.shape[axis]:
                hi = mask.shape[axis]
                lo = max(0, hi - target)
        elif grow < 0:
            hi = lo + target
        slices.append(slice(lo, hi))
    return tuple(slices)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Per-case worker
# ---------------------------------------------------------------------------


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
    volumes: list[np.ndarray] = []
    affine = None
    orig_shape = None
    for suf in SEQUENCES:
        img = nib.load(row[f"path_{suf}"])
        arr = np.asanyarray(img.dataobj).astype(np.float32)
        if affine is None:
            affine = np.asarray(img.affine, dtype=np.float64)
            orig_shape = arr.shape
        elif arr.shape != orig_shape:
            raise ValueError(f"{case_id}: {suf} shape {arr.shape} != {orig_shape}")
        volumes.append(arr)
    img4 = np.stack(volumes, axis=0)  # C, D, H, W in canonical SEQUENCES order
    del volumes

    # Brain mask = union of nonzero across sequences (data is skull-stripped).
    mask = np.any(img4 != 0, axis=0)

    seg = None
    seg_path = row.get("path_seg") or ""
    if isinstance(seg_path, str) and seg_path:
        seg_img = nib.load(seg_path)
        if not np.allclose(np.asarray(seg_img.affine), affine, atol=1e-4):
            raise ValueError(f"{case_id}: seg affine differs from image affine")
        seg = np.rint(np.asanyarray(seg_img.dataobj)).astype(np.uint8)
        if seg.shape != orig_shape:
            raise ValueError(f"{case_id}: seg shape {seg.shape} != {orig_shape}")
        # Cheap re-assertion of the label-3-not-4 invariant at cache time, so a
        # bad file cannot reach training even if the manifest step was skipped.
        vals = np.unique(seg)
        if vals.max(initial=0) > 3:
            raise ValueError(f"{case_id}: label value {int(vals.max())} > 3")

    # ---- crop ----------------------------------------------------------
    box = brain_bbox(mask, margin=margin, divisor=divisor)
    img4 = img4[(slice(None), *box)]
    mask_c = mask[box]
    if seg is not None:
        seg = seg[box]

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


def verify_roundtrip(npz_path: Path, row: dict) -> dict:
    """Check that a cached case reconstructs the original within quantization error."""
    with np.load(npz_path) as z:
        codes = z["img"]
        stats = z["norm_stats"]
        start = z["crop_start"]
        stop = z["crop_stop"]
        cached_seg = z["seg"] if "seg" in z.files else None

    box = tuple(slice(int(a), int(b)) for a, b in zip(start, stop, strict=True))

    # Recover the brain mask that `preprocess_case` actually used. It is the *union*
    # of nonzero voxels across all four sequences, and `quantize` reserves code 0 for
    # out-of-mask voxels while clamping in-mask codes to >= 1, so any channel recovers
    # the mask exactly. Re-deriving it per channel here (`orig != 0`) would disagree on
    # the handful of voxels that are nonzero in one sequence but zero in another, and
    # would report a ~2 sigma error against a cache that is in fact correct.
    mask = codes[0] != QUANT_BACKGROUND_CODE
    mask_consistent = all(
        bool(((codes[c] != QUANT_BACKGROUND_CODE) == mask).all())
        for c in range(1, codes.shape[0])
    )

    worst_z = 0.0
    for c, suf in enumerate(SEQUENCES):
        orig = np.asanyarray(nib.load(row[f"path_{suf}"]).dataobj).astype(np.float32)[box]
        mean, std = float(stats[c, 0]), float(stats[c, 1])
        expect = np.clip((orig - mean) / std, -QUANT_CLIP_SIGMA, QUANT_CLIP_SIGMA)
        expect = np.where(mask, expect, 0.0)
        got = dequantize(codes[c])
        worst_z = max(worst_z, float(np.abs(got - expect).max()))

    seg_ok = None
    if cached_seg is not None and row.get("path_seg"):
        orig_seg = np.rint(
            np.asanyarray(nib.load(row["path_seg"]).dataobj)
        ).astype(np.uint8)[box]
        seg_ok = bool((orig_seg == cached_seg).all())

    return {
        "case_id": row["case_id"],
        "max_abs_z_error": worst_z,
        "within_quant_step": worst_z <= quant_scale() * 0.51,
        "mask_consistent": mask_consistent,
        "seg_lossless": seg_ok,
        "mb": npz_path.stat().st_size / 1e6,
    }


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
        if (vdf["seg_lossless"].dropna() == False).any():  # noqa: E712
            print("VERIFICATION FAILED: segmentation labels not preserved exactly")
            return 1
        print("Round-trip OK: intensities within half a step, labels lossless.")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
