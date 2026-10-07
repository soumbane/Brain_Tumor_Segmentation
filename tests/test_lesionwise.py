"""Lesion-wise metric: behaviour of each official rule, plus parity with the official code.

The parity tests need the official repository (``git clone`` into ``external/brats_metrics``
or point ``$BRATS_METRICS_REPO`` at it) and are skipped without it. The rest do not.
"""

import numpy as np
import pytest

from brats.metrics.lesionwise import (
    CHALLENGE_PARAMS,
    HD95_PENALTY_MM,
    _OFFICIAL_PARAMS,
    challenge_params,
    hausdorff95,
    lesionwise_region,
    official_repo_available,
    score_with_official,
    verify_against_official,
)

SHAPE = (80, 80, 80)


def _cube(mask, lo, size):
    mask[lo[0] : lo[0] + size[0], lo[1] : lo[1] + size[1], lo[2] : lo[2] + size[2]] = True
    return mask


def _one_lesion():
    return _cube(np.zeros(SHAPE, bool), (20, 20, 20), (10, 10, 10))  # 1000 voxels


# -- parameters ---------------------------------------------------------------------------


def test_parameters_are_per_challenge_and_match_the_official_table():
    assert challenge_params("MEN") == (1, 50)
    assert challenge_params("GLI") == challenge_params("PED") == (3, 50)
    assert challenge_params("BraTS-MEN") == (1, 50)
    # Two independent transcriptions of the official table must agree.
    assert {f"BraTS-{k}": v for k, v in CHALLENGE_PARAMS.items()} == _OFFICIAL_PARAMS
    with pytest.raises(KeyError):
        challenge_params("XYZ")


# -- rules --------------------------------------------------------------------------------


def test_perfect_prediction_and_correct_absence_score_one():
    gt = _one_lesion()
    r = lesionwise_region(gt, gt, "WT")
    assert (r.dice, r.hd95, r.n_tp, r.n_fp, r.n_fn) == (1.0, 0.0, 1, 0, 0)

    empty = np.zeros(SHAPE, bool)
    r = lesionwise_region(empty, empty, "ET")
    assert (r.dice, r.hd95) == (1.0, 0.0)


def test_missing_everything_and_predicting_on_nothing_both_score_zero():
    gt = _one_lesion()
    r = lesionwise_region(gt, np.zeros(SHAPE, bool), "WT")
    assert (r.dice, r.hd95, r.n_fn) == (0.0, HD95_PENALTY_MM, 1)

    r = lesionwise_region(np.zeros(SHAPE, bool), gt, "ET")  # hallucinated ET
    assert (r.dice, r.hd95, r.n_fp) == (0.0, HD95_PENALTY_MM, 1)


def test_one_false_positive_blob_halves_a_single_lesion_case():
    gt = _one_lesion()
    pred = gt.copy()
    clean = lesionwise_region(gt, pred, "WT")
    pred = _cube(pred, (60, 60, 60), (4, 4, 4))  # 64-voxel spurious blob, far away

    dirty = lesionwise_region(gt, pred, "WT")

    assert dirty.n_fp == 1 and dirty.dice == pytest.approx(clean.dice / 2)


def test_lesion_threshold_is_strictly_greater_than_50_voxels():
    def score(volume_shape):
        gt = _cube(np.zeros(SHAPE, bool), (20, 20, 20), volume_shape)
        return lesionwise_region(gt, np.zeros(SHAPE, bool), "WT", cohort="GLI")

    assert score((5, 5, 2)).n_fn == 0     # exactly 50 voxels: NOT scored (and so not a miss)
    assert score((5, 5, 2)).dice == 1.0   # nothing left to score
    assert score((3, 3, 6)).n_fn == 1     # 54 voxels: scored, and missed


def test_prediction_on_a_sub_threshold_lesion_is_not_a_false_positive():
    """Official: the component is matched (just not scored). Counting it as an FP punishes a
    correct detection of a lesion the radiologists chose to ignore."""
    big = _one_lesion()
    tiny = _cube(np.zeros(SHAPE, bool), (60, 60, 60), (3, 3, 3))  # 27 voxels
    gt = big | tiny

    r = lesionwise_region(gt, gt.copy(), "WT")

    assert (r.n_tp, r.n_fp, r.n_fn) == (1, 0, 0) and r.dice == 1.0


def test_dilation_is_per_challenge_and_decides_which_fragments_form_one_lesion():
    """Two cubes 5 voxels apart: one lesion under GLI/PED (dilation 3), two under MEN (1)."""
    gt = _cube(np.zeros(SHAPE, bool), (20, 20, 20), (8, 8, 8))
    gt = _cube(gt, (20, 20, 33), (8, 8, 8))  # gap of 5 along the last axis
    pred = _cube(np.zeros(SHAPE, bool), (20, 20, 20), (8, 8, 8))  # finds the first fragment only

    gli = lesionwise_region(gt, pred, "WT", cohort="GLI")
    men = lesionwise_region(gt, pred, "WT", cohort="MEN")

    assert (gli.n_tp, gli.n_fn) == (1, 0)  # fragments merged: one lesion, partly found
    assert (men.n_tp, men.n_fn) == (1, 1)  # separate lesions: one found, one missed
    assert men.dice < gli.dice


def test_prediction_near_but_not_touching_a_lesion_matches_through_the_dilated_region():
    gt = _one_lesion()
    pred = _cube(np.zeros(SHAPE, bool), (20, 20, 32), (10, 10, 4))  # 2 voxels beyond the lesion

    r = lesionwise_region(gt, pred, "WT", cohort="GLI")  # dilation 3 reaches it

    assert (r.n_tp, r.n_fp) == (1, 0)
    assert lesionwise_region(gt, pred, "WT", cohort="MEN").n_fp == 1  # dilation 1 does not


# -- HD95 -----------------------------------------------------------------------------------


def test_hausdorff95_basics():
    a = _one_lesion()
    assert hausdorff95(a, a) == 0.0
    assert hausdorff95(np.zeros(SHAPE, bool), np.zeros(SHAPE, bool)) == 0.0
    assert hausdorff95(a, np.zeros(SHAPE, bool)) == HD95_PENALTY_MM
    shifted = np.roll(a, 3, axis=0)
    assert hausdorff95(a, shifted) == pytest.approx(3.0, abs=0.01)
    assert hausdorff95(a, shifted, spacing=(2.0, 1.0, 1.0)) == pytest.approx(6.0, abs=0.01)


def test_hausdorff95_is_symmetric():
    a = _one_lesion()
    b = _cube(np.zeros(SHAPE, bool), (24, 22, 21), (14, 6, 9))
    assert hausdorff95(a, b) == hausdorff95(b, a)


# -- parity with the official implementation -----------------------------------------------

needs_official = pytest.mark.skipif(
    not official_repo_available(),
    reason="official BraTS-2023-Metrics not found (git clone into external/brats_metrics "
    "or set BRATS_METRICS_REPO)",
)


@needs_official
def test_dice_and_lesion_counts_match_the_official_implementation_exactly():
    # Raises AssertionError on any LW-Dice or TP/FP/FN disagreement; HD95 within 2 mm.
    report = verify_against_official(n_cases=45, seed=0)
    assert report["n_comparisons"] == 135
    assert report["max_abs_dice_gap"] == 0.0
    assert report["max_abs_hd95_gap_mm"] < 2.0


@needs_official
def test_score_with_official_reads_niftis_and_returns_the_official_columns(tmp_path):
    import nibabel as nib

    from brats.metrics.lesionwise import _random_label_map

    rng = np.random.default_rng(3)
    (tmp_path / "gt").mkdir()
    (tmp_path / "pred").mkdir()
    for case in ("BraTS-MEN-00001-000", "BraTS-GLI-00002-000"):
        seg = _random_label_map(rng)
        nib.save(nib.Nifti1Image(seg, np.eye(4)), str(tmp_path / "gt" / f"{case}-seg.nii.gz"))
        nib.save(nib.Nifti1Image(seg, np.eye(4)), str(tmp_path / "pred" / f"{case}.nii.gz"))

    df = score_with_official(tmp_path / "pred", tmp_path / "gt", tmp_path / "out.csv")

    assert len(df) == 6 and set(df.Labels) == {"WT", "TC", "ET"}
    assert {"LesionWise_Score_Dice", "LesionWise_Score_HD95", "Num_TP", "Num_FP", "Num_FN"} <= set(df.columns)
    assert (df.LesionWise_Score_Dice == 1.0).all()  # prediction == ground truth
    assert (tmp_path / "out.csv").is_file()
