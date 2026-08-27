"""Post-processing: the highest-leverage stage in the whole project.

Read this before spending time on architecture. On *identical predictions*,
BiomedMBZ's published ablation (arXiv 2403.09262):

    post-processing            LW Dice    legacy Dice
    none                        78.08        90.60
    + 8-flip TTA                81.25        90.70
    + component size filter     87.74        90.91
    + confidence filter         87.66        91.04
    + both                      88.05        90.94

**Legacy Dice moved +0.34. Lesion-wise Dice moved +9.97.** Architecture choice
among top backbones is worth ~+-0.01. This module deserves more engineering time
than the model and the training recipe combined.

The reason is the metric. BraTS 2023 scores lesion-wise:

    LW Dice = sum_i Dice(lesion_i) / (TP + FN + FP)

Every false positive lands in the denominator *and* contributes a zero to the
numerator, so one spurious 60-voxel blob in a single-lesion case drops that case
from ~0.95 to ~0.475. Under legacy Dice the same blob costs a fraction of a percent.

Two traps, both from Ferreira et al. (arXiv 2402.17317):

* **Validation needs larger thresholds than training.** Never tune small-component
  thresholds on training-fold predictions and ship them.
* **Do not tune to mean Dice -- tune to the BraTS rank.** Their best-mean-Dice
  threshold (WT 1450) was judged too risky and they shipped WT 250. See
  :mod:`brats.metrics.ranking`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brats.constants import REGION_INDEX, REGIONS

# ---------------------------------------------------------------------------
# Published reference values
# ---------------------------------------------------------------------------

#: Per-channel probability thresholds. Not 0.5 everywhere -- BiomedMBZ's final
#: values. ET is lower because ET is under-predicted at 0.5.
DEFAULT_THRESHOLDS: dict[str, float] = {"ET": 0.40, "TC": 0.50, "WT": 0.50}

#: Simple per-region minimum component size, in voxels. Ferreira shipped
#: WT 250 / TC 150 / ET 100; CNMC used 130 (PED) / 110 (MEN).
DEFAULT_MIN_SIZE: dict[str, int] = {"ET": 100, "TC": 150, "WT": 250}


@dataclass
class JointFilterParams:
    """Joint size + mean-confidence component filter (BiomedMBZ ``FilterObjects``).

    Keep a component if::

        (size >= s_upper AND mean_prob >= p_upper)
        OR (s_lower <= size < s_upper AND mean_prob >= p_mid)

    Filtering on size *and* confidence beats size alone, because a size-only cutoff
    discards small-but-confident true lesions -- which under the lesion-wise metric
    are worth as much as large ones.

    Defaults are BiomedMBZ's final published values.
    """

    s_upper: int
    s_lower: int
    p_upper: float
    p_mid: float


DEFAULT_JOINT_FILTER: dict[str, JointFilterParams] = {
    "WT": JointFilterParams(s_upper=2000, s_lower=100, p_upper=0.850, p_mid=0.925),
    "ET": JointFilterParams(s_upper=95, s_lower=70, p_upper=0.710, p_mid=0.500),
    # TC: size-only in the published config (confidence gates set to 0).
    "TC": JointFilterParams(s_upper=350, s_lower=350, p_upper=0.0, p_mid=0.0),
}

#: PED ET/WT volume-ratio gate. If ET occupies less than this fraction of WT,
#: erase ET. Took PED validation ET lesion-wise Dice 0.466 -> 0.733 and ET HD95
#: 158.89 -> 75.93: the largest single improvement in the BraTS 2023 literature.
#:
#: Not a hack. Many PED cases (diffuse midline glioma / DIPG) genuinely do not
#: enhance, and under the lesion-wise metric correctly predicting *nothing* scores
#: 1.0 while hallucinating a small blob scores 0.
PED_ET_WT_RATIO_GATE: float = 0.04


@dataclass
class PostProcessConfig:
    """Full post-processing configuration. Tune on validation only."""

    thresholds: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_THRESHOLDS)
    )
    use_joint_filter: bool = True
    joint_filter: dict[str, JointFilterParams] = field(
        default_factory=lambda: dict(DEFAULT_JOINT_FILTER)
    )
    min_size: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_MIN_SIZE))
    apply_ped_et_gate: bool = True
    ped_et_wt_ratio: float = PED_ET_WT_RATIO_GATE
    #: Enforce ET subset of TC subset of WT after filtering. The regions are nested by
    #: definition; independent sigmoid channels can violate that.
    enforce_nesting: bool = True
    #: 26-connectivity for 3D components, matching the official metric's convention.
    connectivity: int = 26


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------


def _label_components(mask: np.ndarray, connectivity: int = 26) -> tuple[np.ndarray, int]:
    """Label 3D connected components, preferring ``cc3d`` and falling back to scipy."""
    try:
        import cc3d

        labels, n = cc3d.connected_components(
            mask.astype(np.uint8), connectivity=connectivity, return_N=True
        )
        return labels, int(n)
    except ImportError:
        from scipy import ndimage

        structure = ndimage.generate_binary_structure(3, 3 if connectivity == 26 else 1)
        labels, n = ndimage.label(mask, structure=structure)
        return labels, int(n)


def filter_components_joint(
    mask: np.ndarray,
    prob: np.ndarray,
    params: JointFilterParams,
    connectivity: int = 26,
) -> np.ndarray:
    """Apply the joint size + mean-confidence filter to one binary region."""
    if not mask.any():
        return mask

    labels, n = _label_components(mask, connectivity)
    if n == 0:
        return mask

    out = np.zeros_like(mask, dtype=bool)
    # One pass over the labeled volume rather than n boolean comparisons.
    flat_labels = labels.reshape(-1)
    flat_prob = prob.reshape(-1)
    sizes = np.bincount(flat_labels, minlength=n + 1)
    prob_sums = np.bincount(flat_labels, weights=flat_prob, minlength=n + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_probs = np.where(sizes > 0, prob_sums / np.maximum(sizes, 1), 0.0)

    keep = np.zeros(n + 1, dtype=bool)
    for cid in range(1, n + 1):
        size = int(sizes[cid])
        conf = float(mean_probs[cid])
        if size >= params.s_upper and conf >= params.p_upper:
            keep[cid] = True
        elif params.s_lower <= size < params.s_upper and conf >= params.p_mid:
            keep[cid] = True
    keep[0] = False

    out = keep[labels]
    return out


def filter_components_size(
    mask: np.ndarray, min_size: int, connectivity: int = 26
) -> np.ndarray:
    """Baseline: drop components smaller than ``min_size`` voxels."""
    if not mask.any() or min_size <= 1:
        return mask
    labels, n = _label_components(mask, connectivity)
    if n == 0:
        return mask
    sizes = np.bincount(labels.reshape(-1), minlength=n + 1)
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]


# ---------------------------------------------------------------------------
# Nesting
# ---------------------------------------------------------------------------


def enforce_nesting(regions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Force ET subset of TC subset of WT.

    Independent sigmoid channels can predict ET where they did not predict TC. The
    nesting is definitional (TC = NCR + ET, WT = NCR + ED + ET), so widen the
    containing regions rather than erasing the contained ones: an ET voxel is
    evidence of tumor core and of whole tumor.
    """
    et = regions["ET"].astype(bool)
    tc = regions["TC"].astype(bool) | et
    wt = regions["WT"].astype(bool) | tc
    return {"ET": et, "TC": tc, "WT": wt}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def postprocess(
    probs: np.ndarray,
    cohort: str | None = None,
    cfg: PostProcessConfig | None = None,
) -> dict[str, np.ndarray]:
    """Turn sigmoid probabilities into final binary regions.

    Args:
        probs: (3, D, H, W) float probabilities in canonical (ET, TC, WT) order.
        cohort: ``"GLI" | "MEN" | "PED"``. Required for the PED ET gate; without it
            the gate is skipped.
        cfg: Configuration; defaults to the published reference values.

    Returns:
        ``{"ET": bool array, "TC": ..., "WT": ...}``.
    """
    cfg = cfg or PostProcessConfig()
    if probs.ndim != 4 or probs.shape[0] != len(REGIONS):
        raise ValueError(
            f"expected ({len(REGIONS)}, D, H, W) in (ET, TC, WT) order, "
            f"got {probs.shape}"
        )

    # 1. Per-channel thresholds.
    regions: dict[str, np.ndarray] = {}
    for region in REGIONS:
        ch = REGION_INDEX[region]
        regions[region] = probs[ch] >= cfg.thresholds[region]

    # 2. Component filtering, using each region's own probability map.
    for region in REGIONS:
        ch = REGION_INDEX[region]
        if cfg.use_joint_filter:
            regions[region] = filter_components_joint(
                regions[region], probs[ch], cfg.joint_filter[region], cfg.connectivity
            )
        else:
            regions[region] = filter_components_size(
                regions[region], cfg.min_size[region], cfg.connectivity
            )

    # 3. PED ET volume-ratio gate. Applied before nesting so an erased ET does not
    #    get resurrected by the nesting step.
    if cfg.apply_ped_et_gate and cohort == "PED":
        wt_vox = int(regions["WT"].sum())
        if wt_vox > 0:
            ratio = int(regions["ET"].sum()) / wt_vox
            if ratio < cfg.ped_et_wt_ratio:
                # Relabel ET -> non-enhancing core: ET is erased, TC is unchanged
                # (those voxels remain tumor core, just not enhancing).
                regions["TC"] = regions["TC"] | regions["ET"]
                regions["ET"] = np.zeros_like(regions["ET"])

    # 4. Nesting.
    if cfg.enforce_nesting:
        regions = enforce_nesting(regions)

    return regions


def regions_to_label_map(regions: dict[str, np.ndarray]) -> np.ndarray:
    """Collapse nested regions back to a BraTS 2023 label map (0/1/2/3).

    ET -> 3, tumor core minus ET -> 1 (NCR), whole tumor minus core -> 2 (ED).
    Writes **label 3 for enhancing tumor**, matching BraTS 2023. Not 4.
    """
    from brats.constants import LABEL_ED, LABEL_ET, LABEL_NCR

    et = regions["ET"].astype(bool)
    tc = regions["TC"].astype(bool)
    wt = regions["WT"].astype(bool)

    out = np.zeros(et.shape, dtype=np.uint8)
    out[wt & ~tc] = LABEL_ED
    out[tc & ~et] = LABEL_NCR
    out[et] = LABEL_ET
    return out


def paste_into_original(
    volume: np.ndarray,
    orig_shape: tuple[int, int, int],
    crop_start: np.ndarray,
    crop_stop: np.ndarray,
) -> np.ndarray:
    """Place a cropped prediction back into the original 240x240x155 grid.

    Predictions must be submitted and scored in the original geometry; the crop was
    a training-time convenience only.
    """
    out = np.zeros(tuple(int(s) for s in orig_shape), dtype=volume.dtype)
    box = tuple(
        slice(int(a), int(b)) for a, b in zip(crop_start, crop_stop, strict=True)
    )
    out[box] = volume
    return out


__all__ = [
    "PostProcessConfig",
    "JointFilterParams",
    "postprocess",
    "regions_to_label_map",
    "paste_into_original",
    "enforce_nesting",
    "filter_components_joint",
    "filter_components_size",
    "DEFAULT_THRESHOLDS",
    "DEFAULT_JOINT_FILTER",
    "PED_ET_WT_RATIO_GATE",
]
