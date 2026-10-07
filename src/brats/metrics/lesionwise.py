"""Lesion-wise Dice and HD95 -- the official BraTS 2023 metric.

BraTS 2023 replaced legacy overlap Dice with a **lesion-wise** formulation::

    LW Dice = sum_i Dice(lesion_i) / (TP + FN + FP)

This module reproduces the official implementation
(``github.com/rachitsaluja/BraTS-2023-Metrics``, ``metrics.py``; git clone -- there is no
pip package) rule for rule. It was written from the papers first and then corrected
against the code, which differs from the prose in ways that move scores:

* **Dilation and volume threshold are per challenge**: GLI 3 / 50, PED 3 / 50, SSA 3 / 50,
  **MEN 1 / 50**, MET 1 / 2 (:data:`CHALLENGE_PARAMS`). One global value is wrong for MEN.
* GT lesions are the connected components (26-connectivity) of the GT dilated with an
  **18-connected** structure (``generate_binary_structure(3, 2)``), each holding the GT
  voxels inside it. A prediction component matches a lesion if it overlaps the lesion's
  **dilated** region, not just the lesion itself.
* A lesion is scored only if its volume is **strictly greater than** the threshold.
  A prediction component touching a below-threshold lesion is **not** a false positive:
  it is simply not scored. (Treating it as an FP would punish a correct detection.)
* Every FP and FN scores Dice = 0 and HD95 = 374 mm. If nothing is left to score
  (empty GT and empty prediction, or only sub-threshold lesions matched) the case
  scores Dice 1.0 / HD95 0.0.

Known approximation. HD95 here is computed from voxel surfaces with a Euclidean distance
transform, taking the larger of the two directional 95th percentiles. The official code
uses DeepMind's ``surface_distance``: area-weighted surface elements on a half-voxel-shifted
grid. The two agree to within about a voxel; Dice and the TP/FP/FN counts are exact. Run
:func:`verify_against_official` to see the measured gap on your machine, and use
:func:`score_with_official` for any number that will be reported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brats.constants import REGIONS

#: Penalty distance for a false positive or false negative lesion, in mm.
#: Fixed by the challenge; it is the diagonal of the SRI24 volume.
HD95_PENALTY_MM: float = 374.0

#: ``(GT dilation in voxels, GT lesion volume threshold in voxels)`` per challenge, from
#: ``get_LesionWiseResults`` in the official repository.
CHALLENGE_PARAMS: dict[str, tuple[int, int]] = {
    "GLI": (3, 50),
    "SSA": (3, 50),
    "MEN": (1, 50),
    "PED": (3, 50),
    "MET": (1, 2),
}

#: Used only when no cohort is given (e.g. ad-hoc calls); the GLI/PED setting.
DEFAULT_CHALLENGE = "GLI"
GT_DILATION_VOXELS: int = CHALLENGE_PARAMS[DEFAULT_CHALLENGE][0]
MIN_GT_LESION_VOXELS: int = CHALLENGE_PARAMS[DEFAULT_CHALLENGE][1]


def challenge_params(cohort: str | None) -> tuple[int, int]:
    """``(dilation, volume_threshold)`` for a cohort code such as ``"MEN"``."""
    key = (cohort or DEFAULT_CHALLENGE).upper().removeprefix("BRATS-")
    if key not in CHALLENGE_PARAMS:
        raise KeyError(f"no lesion-wise parameters for {cohort!r}; known: {sorted(CHALLENGE_PARAMS)}")
    return CHALLENGE_PARAMS[key]


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
    """Dilate with the official structuring element: 18-connected, ``generate_binary_structure(3, 2)``."""
    if iterations <= 0:
        return mask
    from scipy import ndimage

    st = ndimage.generate_binary_structure(3, 2)
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
    """Robust Hausdorff distance in mm: the larger of the two directional 95th percentiles.

    Distances run between *surface* voxels (a mask minus its erosion), via a Euclidean
    distance transform computed only inside the joint bounding box -- exact, because both
    surfaces lie inside it, and orders of magnitude cheaper than a full-volume transform.
    An approximation of the official area-weighted measure; see the module docstring.
    """
    from scipy import ndimage

    g, p = gt.astype(bool), pred.astype(bool)
    if not g.any() and not p.any():
        return 0.0
    if not g.any() or not p.any():
        return HD95_PENALTY_MM

    idx = np.nonzero(g | p)
    box = tuple(
        slice(max(int(a.min()) - 1, 0), int(a.max()) + 2) for a in idx
    )
    g, p = g[box], p[box]

    g_surf = g & ~ndimage.binary_erosion(g)
    p_surf = p & ~ndimage.binary_erosion(p)
    dt_to_gt = ndimage.distance_transform_edt(~g_surf, sampling=spacing)
    dt_to_pred = ndimage.distance_transform_edt(~p_surf, sampling=spacing)

    # `inverted_cdf` is the nearest-rank percentile, matching the official cumulative-area
    # lookup when every surface element has the same area.
    pred_to_gt = np.percentile(dt_to_gt[p_surf], 95, method="inverted_cdf")
    gt_to_pred = np.percentile(dt_to_pred[g_surf], 95, method="inverted_cdf")
    return float(max(pred_to_gt, gt_to_pred))


# ---------------------------------------------------------------------------
# Lesion-wise scoring
# ---------------------------------------------------------------------------


def lesionwise_region(
    gt: np.ndarray,
    pred: np.ndarray,
    region: str,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    dilation: int | None = None,
    min_lesion: int | None = None,
    cohort: str | None = None,
) -> LesionWiseResult:
    """Lesion-wise Dice and HD95 for one binary region of one case.

    ``dilation`` and ``min_lesion`` default to the cohort's official values
    (:data:`CHALLENGE_PARAMS`); pass them explicitly only to override.
    """
    d0, m0 = challenge_params(cohort)
    dilation = d0 if dilation is None else dilation
    min_lesion = m0 if min_lesion is None else min_lesion

    gt = gt.astype(bool)
    pred = pred.astype(bool)
    gt_vox, pred_vox = int(gt.sum()), int(pred.sum())
    leg = legacy_dice(gt, pred)

    # Correctly predicting nothing is a perfect score. This is why the PED ET gate is
    # the right response to the metric rather than a trick: many PED cases have
    # genuinely empty ET.
    if gt_vox == 0 and pred_vox == 0:
        return LesionWiseResult(region, 1.0, 0.0, 0, 0, 0, leg, 0, 0)

    # GT lesions: components of the dilated GT. A lesion's dilated region is exactly its
    # component of the dilated mask (dilation distributes over union), so one labeling
    # gives both the lesion (GT voxels inside it) and the area predictions may match in.
    roi_labels, n_gt = _components(_dilate(gt, dilation))
    pred_labels, n_pred = _components(pred)

    touched = np.zeros(n_pred + 1, dtype=bool)  # pred components overlapping ANY lesion region
    dice_scores: list[float] = []
    hd_scores: list[float] = []
    n_tp = n_fn = 0

    for cid in range(1, n_gt + 1):
        roi = roi_labels == cid
        hits = np.unique(pred_labels[roi])
        hits = hits[hits != 0]
        touched[hits] = True  # matched even if the lesion is too small to be scored

        lesion = gt & roi
        if int(lesion.sum()) <= min_lesion:
            continue  # official: only lesions strictly larger than the threshold count

        if hits.size == 0:
            n_fn += 1
            dice_scores.append(0.0)
            hd_scores.append(HD95_PENALTY_MM)
            continue
        n_tp += 1
        matched = np.isin(pred_labels, hits)
        dice_scores.append(legacy_dice(lesion, matched))
        hd_scores.append(hausdorff95(lesion, matched, spacing))

    # Prediction components that overlap no lesion region at all are false positives.
    # Each contributes a zero to the numerator and a unit to the denominator -- this is
    # the term that makes one spurious blob so expensive.
    n_fp = int((~touched[1:]).sum())
    dice_scores += [0.0] * n_fp
    hd_scores += [HD95_PENALTY_MM] * n_fp

    denom = n_tp + n_fn + n_fp
    if denom == 0:
        # Nothing scoreable: only sub-threshold lesions, and nothing spurious predicted.
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
    cohort: str | None = None,
    **kwargs,
) -> dict[str, LesionWiseResult]:
    """Score all three regions for one case, with the cohort's official parameters."""
    return {
        region: lesionwise_region(
            gt_regions[region], pred_regions[region], region, spacing, cohort=cohort, **kwargs
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

_official_module = None


def official_repo_path(root: Path | None = None) -> Path:
    """Where the official repository is expected: ``root``, ``$BRATS_METRICS_REPO``, or ``external/brats_metrics``."""
    import os

    from brats.config import REPO_ROOT

    if root is not None:
        return Path(root)
    if os.environ.get("BRATS_METRICS_REPO"):
        return Path(os.environ["BRATS_METRICS_REPO"])
    return REPO_ROOT / OFFICIAL_REPO_PATH


def official_repo_available(root: Path | None = None) -> bool:
    p = official_repo_path(root)
    return (p / "metrics.py").is_file() and (p / "surface_distance").is_dir()


def clone_hint() -> str:
    return (
        f"git clone {OFFICIAL_REPO_URL} {OFFICIAL_REPO_PATH}\n"
        "(there is no pip package; the repo is the source of truth for "
        "leaderboard-comparable numbers)"
    )


def _load_official(root: Path | None = None):
    """Import the official ``metrics.py`` with shims for a modern numpy / pandas.

    The repository pins numpy 1.24 and pandas 1.3 and breaks on current versions in two
    places that do not affect the numbers: it spells infinity ``np.Inf`` (removed in
    numpy 2) and builds a results frame with ``DataFrame.append`` (removed in pandas 2).
    Both are restored here, in-process, without touching the cloned files. Its
    ``get_LesionWiseResults`` still fails on pandas >= 3 for an unrelated reason, so
    :func:`_official_case` calls the numpy-only ``get_LesionWiseScores`` instead.
    """
    global _official_module
    if _official_module is not None:
        return _official_module
    if not official_repo_available(root):
        raise SystemExit("Official BraTS-2023-Metrics repo not found.\n" + clone_hint())

    import importlib.util
    import sys

    import numpy as _np
    import pandas as _pd

    if not hasattr(_np, "Inf"):
        _np.Inf = _np.inf  # type: ignore[attr-defined]
    if not hasattr(_pd.DataFrame, "append"):
        _pd.DataFrame.append = lambda self, other, **kw: _pd.concat([self, other])  # type: ignore[attr-defined]

    repo = str(official_repo_path(root).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)  # the official metrics.py does `import surface_distance`
    spec = importlib.util.spec_from_file_location("brats_official_metrics", Path(repo) / "metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    _official_module = module
    return module


_CASE_COHORT_RE = re.compile(r"BraTS-([A-Z]+)-")

#: Transcribed from ``get_LesionWiseResults`` in the official ``metrics.py`` (the values
#: are inline ``if/elif`` branches there, so they cannot be imported). Kept as a second,
#: independent copy of :data:`CHALLENGE_PARAMS` on purpose: it is the oracle that
#: :func:`verify_against_official` checks our table against.
_OFFICIAL_PARAMS = {
    "BraTS-GLI": (3, 50), "BraTS-SSA": (3, 50), "BraTS-MEN": (1, 50),
    "BraTS-PED": (3, 50), "BraTS-MET": (1, 2),
}


def _official_case(official, pred_file: str, gt_file: str, challenge: str) -> "pd.DataFrame":  # noqa: F821
    """Official per-case scores, one row per region (WT, TC, ET).

    Calls the official ``get_LesionWiseScores`` for the heavy lifting and repeats the
    short aggregation from ``get_LesionWiseResults``. That function cannot be called
    directly on current pandas: it stores arrays in DataFrame cells and ``DataFrame.replace``
    rejects them. The aggregation arithmetic below is a line-for-line port.
    """
    import math

    import pandas as pd

    dil, thresh = _OFFICIAL_PARAMS[challenge]
    rows = []
    for label in ("WT", "TC", "ET"):
        tp, fn, fp, gt_tp, pairs, full_dice, full_hd95, full_gt_vol, _, sens, spec = (
            official.get_LesionWiseScores(pred_file, gt_file, label, dil)
        )
        pairs = [(inter, gtc, vol, d, (374.0 if not math.isfinite(h) else float(h))) for inter, gtc, vol, d, h in pairs]
        fn_sub = sum(1 for inter, _g, vol, _d, _h in pairs if len(inter) == 0 and vol <= thresh)
        gt_tp_sub = sum(1 for inter, _g, vol, _d, _h in pairs if len(inter) != 0 and vol <= thresh)
        scored = [(d, h) for _i, _g, vol, d, h in pairs if vol > thresh]
        denom = len(scored) + len(fp)
        if denom == 0:  # official: 0/0 -> nan -> Dice 1, HD95 0
            lw_dice, lw_hd = 1.0, 0.0
        else:
            lw_dice = sum(d for d, _ in scored) / denom
            lw_hd = (sum(h for _, h in scored) + len(fp) * 374) / denom
        rows.append({
            "Labels": label,
            "Num_TP": len(gt_tp) - gt_tp_sub, "Num_FP": len(fp), "Num_FN": len(fn) - fn_sub,
            "Sensitivity": sens, "Specificity": spec,
            "Legacy_Dice": full_dice,
            "Legacy_HD95": 374.0 if not math.isfinite(full_hd95) else full_hd95,
            "GT_Complete_Volume": full_gt_vol,
            "LesionWise_Score_Dice": lw_dice, "LesionWise_Score_HD95": lw_hd,
        })
    return pd.DataFrame(rows)


def score_with_official(
    pred_dir: Path, gt_dir: Path, out_csv: Path, root: Path | None = None
) -> "pd.DataFrame":  # noqa: F821
    """Score a directory of predictions with the official implementation.

    Reported numbers must come from here, not from the reference implementation above.

    Args:
        pred_dir: ``<case_id>.nii.gz`` label maps (values 0/1/2/3, **3 = enhancing
            tumor**) in the original 240x240x155 grid. :func:`brats.postprocess.regions_to_label_map`
            and :func:`brats.postprocess.paste_into_original` produce exactly that.
        gt_dir: Searched recursively for ``<case_id>-seg.nii.gz``.
        out_csv: One row per (case, region) with the official columns.
    """
    import pandas as pd

    official = _load_official(root)
    gts = {f.name.removesuffix("-seg.nii.gz"): f for f in Path(gt_dir).rglob("*-seg.nii.gz")}
    frames = []
    for pred in sorted(Path(pred_dir).glob("*.nii.gz")):
        case = pred.name.removesuffix(".nii.gz")
        if case not in gts:
            raise SystemExit(f"no ground truth for {case} under {gt_dir}")
        m = _CASE_COHORT_RE.match(case)
        if m is None:
            raise SystemExit(f"cannot infer the challenge from case id {case!r}")
        df = _official_case(official, str(pred), str(gts[case]), f"BraTS-{m.group(1)}")
        df.insert(0, "case_id", case)
        frames.append(df)
    if not frames:
        raise SystemExit(f"no predictions (*.nii.gz) in {pred_dir}")
    out = pd.concat(frames, ignore_index=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def _random_label_map(rng: np.random.Generator, shape=(64, 64, 48)) -> np.ndarray:
    """A BraTS-style label map with several nested blobs, some tiny (sub-threshold)."""
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    seg = np.zeros(shape, np.uint8)
    for _ in range(int(rng.integers(1, 4))):
        c = [int(rng.integers(8, n - 8)) for n in shape]
        r = float(rng.choice([2.0, 2.5, 3.0, 5.0, 7.0]))
        d2 = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2
        seg[d2 <= r * r] = 2
        seg[d2 <= (0.7 * r) ** 2] = 1
        seg[d2 <= (0.35 * r) ** 2] = 3
    return seg


def _perturb(seg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A plausible prediction: shifted/eroded GT, dropped lesions, empty output, spurious blobs."""
    from scipy import ndimage

    if rng.random() < 0.08:
        return np.zeros_like(seg)  # predicts nothing at all
    pred = np.roll(seg, shift=tuple(int(v) for v in rng.integers(-1, 2, 3)), axis=(0, 1, 2))
    if rng.random() < 0.4:
        pred = np.where(ndimage.binary_erosion(pred > 0, iterations=1), pred, 0).astype(np.uint8)
    if rng.random() < 0.35:  # miss a whole lesion
        comps, n = ndimage.label(pred > 0)
        if n:
            pred[comps == int(rng.integers(1, n + 1))] = 0
    for _ in range(int(rng.integers(0, 3))):
        c = [int(rng.integers(4, n - 4)) for n in seg.shape]
        r = int(rng.integers(1, 4))
        zz, yy, xx = np.ogrid[: seg.shape[0], : seg.shape[1], : seg.shape[2]]
        blob = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= r * r
        pred[blob] = int(rng.choice([1, 2, 3]))
    return pred.astype(np.uint8)


def verify_against_official(
    n_cases: int = 24,
    seed: int = 0,
    hd_tolerance_mm: float = 2.0,
    root: Path | None = None,
    workdir: Path | None = None,
) -> dict:
    """Run both implementations on random synthetic cases and compare them.

    Run this before reporting any number. Lesion-wise **Dice and the TP/FP/FN counts must
    match exactly**; a mismatch means a matching rule, dilation or threshold differs and
    every score is shifted. HD95 is allowed ``hd_tolerance_mm`` (see the module docstring).

    Returns a dict with the per-case comparison and the worst gaps; raises
    ``AssertionError`` listing any exact-match violation or HD95 gap above tolerance.
    """
    import tempfile

    import nibabel as nib

    from brats.constants import REGION_LABELS

    official = _load_official(root)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    problems: list[str] = []

    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        tmp = Path(tmp)
        for i in range(n_cases):
            cohort = ("GLI", "MEN", "PED")[i % 3]
            gt = np.zeros((64, 64, 48), np.uint8) if rng.random() < 0.08 else _random_label_map(rng)
            pred = _perturb(gt, rng) if gt.any() else _perturb(_random_label_map(rng), rng)
            nib.save(nib.Nifti1Image(gt, np.eye(4)), str(tmp / "gt.nii.gz"))
            nib.save(nib.Nifti1Image(pred, np.eye(4)), str(tmp / "pred.nii.gz"))
            ref = _official_case(
                official, str(tmp / "pred.nii.gz"), str(tmp / "gt.nii.gz"), f"BraTS-{cohort}"
            ).set_index("Labels")

            ours = lesionwise_case(
                {r: np.isin(gt, list(REGION_LABELS[r])) for r in REGIONS},
                {r: np.isin(pred, list(REGION_LABELS[r])) for r in REGIONS},
                cohort=cohort,
            )
            for region in REGIONS:
                o, r = ours[region], ref.loc[region]
                row = {
                    "case": i, "cohort": cohort, "region": region,
                    "dice_ours": o.dice, "dice_official": float(r["LesionWise_Score_Dice"]),
                    "hd95_ours": o.hd95, "hd95_official": float(r["LesionWise_Score_HD95"]),
                    "tp": (o.n_tp, int(r["Num_TP"])), "fp": (o.n_fp, int(r["Num_FP"])),
                    "fn": (o.n_fn, int(r["Num_FN"])),
                }
                rows.append(row)
                tag = f"case {i} {cohort} {region}"
                if abs(row["dice_ours"] - row["dice_official"]) > 1e-9:
                    problems.append(f"{tag}: LW Dice {row['dice_ours']:.6f} != official {row['dice_official']:.6f}")
                for k in ("tp", "fp", "fn"):
                    if row[k][0] != row[k][1]:
                        problems.append(f"{tag}: {k.upper()} {row[k][0]} != official {row[k][1]}")
                if abs(row["hd95_ours"] - row["hd95_official"]) > hd_tolerance_mm:
                    problems.append(
                        f"{tag}: LW HD95 {row['hd95_ours']:.2f} vs official {row['hd95_official']:.2f} mm"
                    )

    worst_hd = max((abs(r["hd95_ours"] - r["hd95_official"]) for r in rows), default=0.0)
    worst_dice = max((abs(r["dice_ours"] - r["dice_official"]) for r in rows), default=0.0)
    if problems:
        raise AssertionError(
            f"{len(problems)} disagreement(s) with the official metric:\n  " + "\n  ".join(problems[:20])
        )
    return {"n_comparisons": len(rows), "max_abs_dice_gap": worst_dice, "max_abs_hd95_gap_mm": worst_hd, "rows": rows}


__all__ = [
    "LesionWiseResult",
    "CHALLENGE_PARAMS",
    "challenge_params",
    "lesionwise_region",
    "lesionwise_case",
    "legacy_dice",
    "hausdorff95",
    "aggregate",
    "bootstrap_ci",
    "score_with_official",
    "verify_against_official",
    "official_repo_available",
    "official_repo_path",
    "clone_hint",
    "HD95_PENALTY_MM",
    "GT_DILATION_VOXELS",
    "MIN_GT_LESION_VOXELS",
]
