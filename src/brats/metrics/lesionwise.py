"""Lesion-wise Dice and HD95 -- the official BraTS 2023 metric.

BraTS 2023 replaced legacy overlap Dice with a **lesion-wise** formulation::

    LW Dice = sum_i Dice(lesion_i) / (TP + FN + FP)

* Ground-truth lesions are isolated by dilating the GT mask and running 3D
  connected components.
* GT lesions below a volumetric threshold are excluded from scoring.
* A predicted component counts as TP if it overlaps GT by at least one voxel.
* **Every FP and FN scores Dice = 0 and HD95 = 374 mm.**

Implementation policy. The official repository
(``github.com/rachitsaluja/BraTS-2023-Metrics``, git clone -- there is no pip
package) is the source of truth for leaderboard-comparable numbers, and
``panoptica`` is the cross-check. **They must not be silently substituted for one
another**: their dilation factors and volumetric thresholds differ.

The organizers' own papers describe the dilation *inconsistently* -- the MEN paper
says 1-voxel symmetric dilation with 26-connectivity, the PED paper says dilate by
3 pixels in all directions. **Read the code, not the prose.** The defaults below
follow the repository; :func:`verify_against_official` exists to confirm this
implementation agrees with it before any number is reported.

This module is a self-contained reference implementation used for fast in-training
validation. Final reported numbers should come from :func:`score_with_official`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brats.constants import REGIONS

#: Penalty distance for a false positive or false negative lesion, in mm.
#: Fixed by the challenge; it is the diagonal of the SRI24 volume.
HD95_PENALTY_MM: float = 374.0

#: Dilation applied to the GT mask before connected-component labeling, in voxels.
#: Merges lesion fragments that a radiologist would call one lesion.
GT_DILATION_VOXELS: int = 3

#: GT lesions strictly smaller than this are dropped from scoring entirely.
MIN_GT_LESION_VOXELS: int = 50


@dataclass
class LesionWiseResult:
    """Per-region lesion-wise scores for one case."""

    region: str
    dice: float
    hd95: float
    n_tp: int
    n_fp: int
    n_fn: int
    legacy_dice: float
    gt_voxels: int
    pred_voxels: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "region": self.region,
            "lw_dice": self.dice,
            "lw_hd95": self.hd95,
            "tp": self.n_tp,
            "fp": self.n_fp,
            "fn": self.n_fn,
            "legacy_dice": self.legacy_dice,
            "gt_voxels": self.gt_voxels,
            "pred_voxels": self.pred_voxels,
        }


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def _components(mask: np.ndarray, connectivity: int = 26) -> tuple[np.ndarray, int]:
    try:
        import cc3d

        labels, n = cc3d.connected_components(
            mask.astype(np.uint8), connectivity=connectivity, return_N=True
        )
        return labels, int(n)
    except ImportError:
        from scipy import ndimage

        st = ndimage.generate_binary_structure(3, 3 if connectivity == 26 else 1)
        labels, n = ndimage.label(mask, structure=st)
        return labels, int(n)


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask
    from scipy import ndimage

    st = ndimage.generate_binary_structure(3, 3)  # 26-connectivity
    return ndimage.binary_dilation(mask, structure=st, iterations=iterations)


def legacy_dice(gt: np.ndarray, pred: np.ndarray) -> float:
    """Whole-volume overlap Dice. Reported alongside for pre-2023 comparability.

    Convention: both empty scores 1.0 (a correct prediction of absence).
    """
    g, p = gt.astype(bool), pred.astype(bool)
    gs, ps = int(g.sum()), int(p.sum())
    if gs == 0 and ps == 0:
        return 1.0
    if gs == 0 or ps == 0:
        return 0.0
    return 2.0 * float((g & p).sum()) / float(gs + ps)


def hausdorff95(
    gt: np.ndarray, pred: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> float:
    """Symmetric 95th-percentile Hausdorff distance in mm.

    Computed with a Euclidean distance transform, which is far cheaper than pairwise
    surface distances and exact for isotropic 1 mm data.
    """
    from scipy import ndimage

    g, p = gt.astype(bool), pred.astype(bool)
    if not g.any() and not p.any():
        return 0.0
    if not g.any() or not p.any():
        return HD95_PENALTY_MM

    dt_to_gt = ndimage.distance_transform_edt(~g, sampling=spacing)
    dt_to_pred = ndimage.distance_transform_edt(~p, sampling=spacing)

    # Surface voxels: boundary of each mask.
    g_surf = g & ~ndimage.binary_erosion(g)
    p_surf = p & ~ndimage.binary_erosion(p)
    if not g_surf.any():
        g_surf = g
    if not p_surf.any():
        p_surf = p

    d_pred_to_gt = dt_to_gt[p_surf]
    d_gt_to_pred = dt_to_pred[g_surf]
    both = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(both, 95))


# ---------------------------------------------------------------------------
# Lesion-wise scoring
# ---------------------------------------------------------------------------


def lesionwise_region(
    gt: np.ndarray,
    pred: np.ndarray,
    region: str,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    dilation: int = GT_DILATION_VOXELS,
    min_lesion: int = MIN_GT_LESION_VOXELS,
) -> LesionWiseResult:
    """Lesion-wise Dice and HD95 for one binary region of one case."""
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    gt_vox, pred_vox = int(gt.sum()), int(pred.sum())
    leg = legacy_dice(gt, pred)

    # Correctly predicting nothing is a perfect score. This is why the PED ET gate
    # is the right response to the metric rather than a trick: many PED cases have
    # genuinely empty ET.
    if gt_vox == 0 and pred_vox == 0:
        return LesionWiseResult(region, 1.0, 0.0, 0, 0, 0, leg, 0, 0)

    # Isolate GT lesions on the dilated mask, but measure overlap on the original.
    gt_labels_dil, n_gt = _components(_dilate(gt, dilation))
    gt_lesions: list[np.ndarray] = []
    for cid in range(1, n_gt + 1):
        lesion = gt & (gt_labels_dil == cid)
        if int(lesion.sum()) >= min_lesion:
            gt_lesions.append(lesion)

    pred_labels, n_pred = _components(pred)
    pred_components = [pred_labels == cid for cid in range(1, n_pred + 1)]

    dice_scores: list[float] = []
    hd_scores: list[float] = []
    matched_pred: set[int] = set()
    n_tp = n_fn = 0

    for lesion in gt_lesions:
        # A predicted component matches if it overlaps this GT lesion at all.
        hits = [i for i, comp in enumerate(pred_components) if (comp & lesion).any()]
        if not hits:
            n_fn += 1
            dice_scores.append(0.0)
            hd_scores.append(HD95_PENALTY_MM)
            continue
        n_tp += 1
        matched_pred.update(hits)
        merged = np.zeros_like(pred)
        for i in hits:
            merged |= pred_components[i]
        dice_scores.append(legacy_dice(lesion, merged))
        hd_scores.append(hausdorff95(lesion, merged, spacing))

    # Unmatched predicted components are false positives. Each contributes a zero
    # to the numerator and a unit to the denominator -- this is the term that makes
    # one spurious blob so expensive.
    n_fp = len(pred_components) - len(matched_pred)
    for _ in range(n_fp):
        dice_scores.append(0.0)
        hd_scores.append(HD95_PENALTY_MM)

    denom = n_tp + n_fn + n_fp
    if denom == 0:
        # GT had only sub-threshold lesions and nothing was predicted.
        return LesionWiseResult(region, 1.0, 0.0, 0, 0, 0, leg, gt_vox, pred_vox)

    return LesionWiseResult(
        region=region,
        dice=float(np.sum(dice_scores) / denom),
        hd95=float(np.sum(hd_scores) / denom),
        n_tp=n_tp,
        n_fp=n_fp,
        n_fn=n_fn,
        legacy_dice=leg,
        gt_voxels=gt_vox,
        pred_voxels=pred_vox,
    )


def lesionwise_case(
    gt_regions: dict[str, np.ndarray],
    pred_regions: dict[str, np.ndarray],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    **kwargs,
) -> dict[str, LesionWiseResult]:
    """Score all three regions for one case."""
    return {
        region: lesionwise_region(
            gt_regions[region], pred_regions[region], region, spacing, **kwargs
        )
        for region in REGIONS
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean.

    **Mandatory for PED**, where the test split is n~10. The official PED test set
    was n=24 and the top four teams were statistically indistinguishable
    (p = 0.10-0.45). A bare PED point estimate is not a result.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(
        np.percentile(means, 100 * (1 - alpha / 2))
    )


def aggregate(df, group_cols: list[str] | None = None):
    """Aggregate per-case scores to mean, median, IQR and bootstrap CI.

    **Always report median alongside mean.** The MEN winner scored mean ET Dice
    0.899 but median 0.976, and mean HD95 23.9 mm against median 0.96 mm. On a
    typical case the segmentation is essentially perfect; the mean is dragged by a
    handful of catastrophic cases (calcified non-enhancing meningiomas -- one NVAUTO
    case scored ET 0.00 / TC 0.00 / WT 0.338). Reporting only the mean hides the
    model's actual behavior and sends you chasing cases that threshold tuning
    cannot fix.
    """
    import pandas as pd

    group_cols = group_cols or ["cohort", "region"]
    rows = []
    for keys, sub in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys, strict=True))
        row["n"] = len(sub)
        for metric in ("lw_dice", "lw_hd95", "legacy_dice"):
            if metric not in sub.columns:
                continue
            v = sub[metric].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(v)
            row[f"{metric}_mean"] = float(np.nanmean(v))
            row[f"{metric}_median"] = float(np.nanmedian(v))
            row[f"{metric}_q25"] = float(np.nanpercentile(v, 25))
            row[f"{metric}_q75"] = float(np.nanpercentile(v, 75))
            row[f"{metric}_ci_lo"] = lo
            row[f"{metric}_ci_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Official repository bridge
# ---------------------------------------------------------------------------

OFFICIAL_REPO_URL = "https://github.com/rachitsaluja/BraTS-2023-Metrics.git"
OFFICIAL_REPO_PATH = Path("external/brats_metrics")


def official_repo_available(root: Path | None = None) -> bool:
    from brats.config import REPO_ROOT

    p = root or (REPO_ROOT / OFFICIAL_REPO_PATH)
    return p.is_dir() and any(p.glob("*.py"))


def clone_hint() -> str:
    return (
        f"git clone {OFFICIAL_REPO_URL} {OFFICIAL_REPO_PATH}\n"
        "(there is no pip package; the repo is the source of truth for "
        "leaderboard-comparable numbers)"
    )


def score_with_official(pred_dir: Path, gt_dir: Path, out_csv: Path) -> None:
    """Score a directory of predictions with the official implementation.

    Reported numbers must come from here, not from the reference implementation
    above. Raises with instructions if the repo has not been cloned.
    """
    if not official_repo_available():
        raise SystemExit(
            "Official BraTS-2023-Metrics repo not found.\n" + clone_hint()
        )
    raise NotImplementedError(
        "Wire this to the cloned repo's CLI once it is present. Inspect its "
        "entry point and per-challenge threshold/dilation table "
        "(figs/BraTS_ThreshDil.png) first -- read the code, not the papers' prose."
    )


def verify_against_official(tolerance: float = 1e-3) -> None:
    """Confirm this implementation agrees with the official one on a toy case.

    Run before reporting any number. Disagreement means the dilation factor or the
    volumetric threshold differs, which silently shifts every score.
    """
    raise NotImplementedError(
        "Requires the cloned official repo; see clone_hint(). Construct a synthetic "
        "case with (a) one large lesion, (b) one 60-voxel lesion, (c) one 40-voxel "
        "lesion below the exclusion threshold, and (d) one spurious FP blob, then "
        "assert both implementations agree to within `tolerance`."
    )


__all__ = [
    "LesionWiseResult",
    "lesionwise_region",
    "lesionwise_case",
    "legacy_dice",
    "hausdorff95",
    "aggregate",
    "bootstrap_ci",
    "score_with_official",
    "verify_against_official",
    "official_repo_available",
    "clone_hint",
    "HD95_PENALTY_MM",
    "GT_DILATION_VOXELS",
    "MIN_GT_LESION_VOXELS",
]
