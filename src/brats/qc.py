"""QC visualization: overlay montages for visual review of segmentations.

This is the deliverable for the official ValidationData, which has no ground truth
and therefore cannot be scored -- segmentation there is reviewed visually.

Two montage modes:

* **With ground truth** (test/val splits): GT and prediction side by side, plus a
  disagreement panel. The disagreement panel is the useful one -- it makes false
  positives visible, and under the lesion-wise metric a single spurious component can
  halve a case's score, which no amount of staring at a Dice number will reveal.
* **Without ground truth** (official validation): prediction overlay only.

One caveat worth respecting when reviewing MEN cases: ~90% have tumor abutting the
edge of the skull-stripped brain, because meningiomas are extra-axial and skull
stripping deleted the extracranial part. Boundary-clipped predictions there are
**correct** and must not be "fixed".
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from brats.constants import REGIONS

log = logging.getLogger("brats.qc")

#: Overlay colors per region (RGB, 0-1). ET is red -- it is the region that decides
#: the leaderboard, so it should be the one that catches the eye.
REGION_COLORS: dict[str, tuple[float, float, float]] = {
    "ET": (1.00, 0.20, 0.20),
    "TC": (0.20, 0.90, 0.30),
    "WT": (0.30, 0.55, 1.00),
}


def _normalize_slice(img: np.ndarray) -> np.ndarray:
    """Robust 1-99 percentile scaling to [0, 1] for display."""
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros_like(img)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((img - lo) / (hi - lo), 0, 1)


def _overlay(
    base: np.ndarray, regions: dict[str, np.ndarray], alpha: float = 0.45
) -> np.ndarray:
    """Blend region masks over a grayscale slice.

    Drawn WT -> TC -> ET so the nested inner regions paint on top and stay visible.
    """
    rgb = np.repeat(_normalize_slice(base)[..., None], 3, axis=2)
    for region in ("WT", "TC", "ET"):
        mask = regions.get(region)
        if mask is None or not mask.any():
            continue
        color = np.asarray(REGION_COLORS[region], dtype=np.float32)
        m = mask.astype(bool)
        rgb[m] = (1 - alpha) * rgb[m] + alpha * color
    return np.clip(rgb, 0, 1)


def _pick_slices(wt: np.ndarray, n: int = 6) -> list[int]:
    """Choose axial slices spanning the tumor, or the brain if no tumor is present."""
    per_slice = wt.reshape(-1, wt.shape[-1]).sum(axis=0)
    nz = np.nonzero(per_slice)[0]
    if nz.size == 0:
        centre = wt.shape[-1] // 2
        half = max(1, wt.shape[-1] // 8)
        return list(np.linspace(centre - half, centre + half, n).astype(int))
    # Bias toward the largest cross-sections rather than uniform spacing.
    lo, hi = int(nz[0]), int(nz[-1])
    return list(np.linspace(lo, hi, n).astype(int))


def case_montage(
    image: np.ndarray,
    pred_regions: dict[str, np.ndarray],
    gt_regions: dict[str, np.ndarray] | None = None,
    case_id: str = "",
    cohort: str = "",
    channel: int = 1,
    n_slices: int = 6,
    out_path: Path | None = None,
    dice_note: str = "",
):
    """Render one case as a montage figure.

    Args:
        image: (C, D, H, W) preprocessed volume.
        channel: Which sequence to display. Default 1 = t1c (post-contrast), the
            sequence enhancing tumor is defined on.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    vol = np.asarray(image)[channel]
    wt_ref = (gt_regions or pred_regions).get("WT", np.zeros_like(vol, dtype=bool))
    slices = _pick_slices(np.asarray(wt_ref), n_slices)

    n_rows = 3 if gt_regions is not None else 1
    fig, axes = plt.subplots(
        n_rows, len(slices), figsize=(2.4 * len(slices), 2.6 * n_rows), squeeze=False
    )

    for col, z in enumerate(slices):
        base = vol[:, :, z]
        pred_sl = {r: np.asarray(pred_regions[r])[:, :, z] for r in REGIONS}

        if gt_regions is None:
            axes[0][col].imshow(np.rot90(_overlay(base, pred_sl)))
            axes[0][col].set_title(f"z={z}", fontsize=8)
        else:
            gt_sl = {r: np.asarray(gt_regions[r])[:, :, z] for r in REGIONS}
            axes[0][col].imshow(np.rot90(_overlay(base, gt_sl)))
            axes[1][col].imshow(np.rot90(_overlay(base, pred_sl)))

            # Disagreement: false positives red, false negatives blue. FPs are the
            # expensive error under the lesion-wise metric, so they get the hot color.
            rgb = np.repeat(_normalize_slice(base)[..., None], 3, axis=2)
            fp = pred_sl["WT"] & ~gt_sl["WT"]
            fn = gt_sl["WT"] & ~pred_sl["WT"]
            rgb[fp] = [1.0, 0.0, 0.0]
            rgb[fn] = [0.0, 0.35, 1.0]
            axes[2][col].imshow(np.rot90(rgb))
            axes[0][col].set_title(f"z={z}", fontsize=8)

        for row in range(n_rows):
            axes[row][col].axis("off")

    if gt_regions is not None:
        for row, label in enumerate(("ground truth", "prediction", "FP red / FN blue")):
            axes[row][0].set_ylabel(label)
            axes[row][0].axis("on")
            axes[row][0].set_xticks([])
            axes[row][0].set_yticks([])

    handles = [
        mpatches.Patch(color=REGION_COLORS[r], label=r) for r in ("WT", "TC", "ET")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)

    title = f"{case_id}  [{cohort}]"
    if dice_note:
        title += f"   {dice_note}"
    if gt_regions is None:
        title += "   (no ground truth -- visual review only)"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def montage_from_predictions(
    predictions,
    out_dir: Path,
    cfg=None,
    max_per_cohort: int = 8,
    worst_first: bool = True,
):
    """Render montages, prioritizing the cases most worth a human's attention.

    ``worst_first`` sorts labeled cases by ascending WT Dice, because the interesting
    review targets are the failures, not the many near-perfect cases. For unlabeled
    cases (official validation) it falls back to cohort-stratified sampling.
    """
    from brats.metrics.lesionwise import legacy_dice
    from brats.postprocess import PostProcessConfig, postprocess

    cfg = cfg or PostProcessConfig()
    out_dir.mkdir(parents=True, exist_ok=True)

    scored: list[tuple[float, dict, dict, dict | None]] = []
    for pred in predictions:
        regions = postprocess(pred["probs"], cohort=pred["cohort"], cfg=cfg)
        gt = None
        score = 1.0
        if "label" in pred:
            gt = {r: pred["label"][i] > 0.5 for i, r in enumerate(REGIONS)}
            score = legacy_dice(gt["WT"], regions["WT"])
        scored.append((score, pred, regions, gt))

    if worst_first:
        scored.sort(key=lambda t: t[0])

    written: dict[str, int] = {}
    paths: list[Path] = []
    for score, pred, regions, gt in scored:
        cohort = pred["cohort"]
        if written.get(cohort, 0) >= max_per_cohort:
            continue
        written[cohort] = written.get(cohort, 0) + 1

        note = f"WT Dice {score:.3f}" if gt is not None else ""
        path = out_dir / cohort / f"{pred['case_id']}.png"
        case_montage(
            image=pred["image"] if "image" in pred else pred["probs"],
            pred_regions=regions,
            gt_regions=gt,
            case_id=pred["case_id"],
            cohort=cohort,
            out_path=path,
            dice_note=note,
        )
        paths.append(path)

    log.info("wrote %d montages to %s", len(paths), out_dir)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/segresnet_base.yaml")
    ap.add_argument("--data-config", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="official_val",
                    choices=["val", "test", "official_val"])
    ap.add_argument("--max-per-cohort", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    import torch

    from brats.config import DataConfig
    from brats.data.transforms import CachedBratsDataset, load_records, val_transforms
    from brats.inference import InferenceConfig, predict_dataset
    from brats.models.multitask import MultiTaskConfig, build_model
    from brats.train import CheckpointManager, TrainConfig

    cfg = TrainConfig.from_yaml(args.config)
    data_cfg = DataConfig.load(args.data_config) if args.data_config else DataConfig.load()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) if args.out_dir else (
        data_cfg.reports_dir / f"qc_{args.split}"
    )

    model = build_model(MultiTaskConfig(pooling=cfg.pooling)).to(device)  # type: ignore[arg-type]
    CheckpointManager(cfg).load(model, path=Path(args.checkpoint))

    labeled = args.split != "official_val"
    ds = CachedBratsDataset(
        load_records(data_cfg, args.split),
        transform=val_transforms(labeled),
        load_label=labeled,
    )

    preds = []
    inf_cfg = InferenceConfig(roi_size=cfg.patch_size, tta_flip_axes=())
    for i, pred in enumerate(
        predict_dataset(model, ds, cfg=inf_cfg, device=device, limit=args.limit)
    ):
        # Carry the image through so the montage can draw on it.
        pred["image"] = np.asarray(ds[i]["image"])
        preds.append(pred)

    montage_from_predictions(preds, out_dir, max_per_cohort=args.max_per_cohort)
    return 0


__all__ = ["case_montage", "montage_from_predictions", "REGION_COLORS"]


if __name__ == "__main__":
    raise SystemExit(main())
