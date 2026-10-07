"""Post-processing: config loading, and the first real tests of its logic.

Regression for: ``configs/postproc.yaml`` was read by nothing, so editing the thresholds had
no effect -- ``evaluate`` and ``qc`` always used the defaults hard-coded in the module.
"""

import dataclasses
import textwrap

import numpy as np
import pytest

from brats.constants import LABEL_ED, LABEL_ET, LABEL_NCR, REGION_INDEX
from brats.data.preprocess import labels_to_regions
from brats.postprocess import (
    DEFAULT_POSTPROC_YAML,
    JointFilterParams,
    PostProcessConfig,
    enforce_nesting,
    filter_components_joint,
    filter_components_size,
    postprocess,
    regions_to_label_map,
)

ET, TC, WT = REGION_INDEX["ET"], REGION_INDEX["TC"], REGION_INDEX["WT"]
SHAPE = (48, 48, 48)


def _write(tmp_path, body):
    p = tmp_path / "pp.yaml"
    p.write_text(textwrap.dedent(body))
    return p


# -- config ---------------------------------------------------------------------------


def test_shipped_yaml_agrees_with_the_code_defaults():
    """If these drift apart, one of them is lying about what the pipeline does."""
    assert PostProcessConfig.from_yaml(DEFAULT_POSTPROC_YAML) == PostProcessConfig()


def test_yaml_values_are_actually_used(tmp_path):
    p = _write(tmp_path, """
        postprocess:
          thresholds: {ET: 0.25}
          ped_et_wt_ratio: 0.1
          use_joint_filter: false
          joint_filter:
            WT: {s_upper: 10, s_lower: 5, p_upper: 0.6, p_mid: 0.7}
    """)

    cfg = PostProcessConfig.from_yaml(p)

    assert cfg.thresholds == {"ET": 0.25, "TC": 0.5, "WT": 0.5}  # TC/WT keep defaults
    assert cfg.ped_et_wt_ratio == 0.1 and cfg.use_joint_filter is False
    assert cfg.joint_filter["WT"] == JointFilterParams(10, 5, 0.6, 0.7)
    assert cfg.joint_filter["ET"] == PostProcessConfig().joint_filter["ET"]


@pytest.mark.parametrize(
    "body",
    [
        "postprocess:\n  treshold: {ET: 0.4}\n",             # typo'd key
        "postprocess:\n  thresholds: {XT: 0.4}\n",           # unknown region
        "postprocess:\n  thresholds: {ET: high}\n",          # not a number
        "postprocess:\n  joint_filter:\n    WT: {s_upper: 1}\n",  # incomplete params
    ],
)
def test_bad_yaml_fails_loudly(tmp_path, body):
    with pytest.raises((ValueError, KeyError)):
        PostProcessConfig.from_yaml(_write(tmp_path, body))


# -- helpers --------------------------------------------------------------------------


def _blob(mask, center, r):
    z, y, x = np.ogrid[: mask.shape[0], : mask.shape[1], : mask.shape[2]]
    mask |= (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2 <= r * r
    return mask


def _probs(blobs, p_in=0.95, p_out=0.01):
    """(3,D,H,W) probabilities with the given {(region, center, radius): prob} blobs."""
    probs = np.full((3, *SHAPE), p_out, np.float32)
    for (ch, center, r), p in blobs.items():
        m = _blob(np.zeros(SHAPE, bool), center, r)
        probs[ch][m] = p
    return probs


# -- component filters ----------------------------------------------------------------


def test_size_filter_drops_small_components_only():
    m = _blob(np.zeros(SHAPE, bool), (12, 12, 12), 6)
    m = _blob(m, (36, 36, 36), 2)  # ~33 voxels
    out = filter_components_size(m, min_size=100)
    assert out[12, 12, 12] and not out[36, 36, 36]


def test_joint_filter_follows_the_published_rule():
    params = JointFilterParams(s_upper=500, s_lower=50, p_upper=0.85, p_mid=0.925)
    big, mid, tiny = (12, 12, 12), (36, 12, 12), (12, 36, 36)  # r=6 (~900), r=3 (~123), r=1
    mask = np.zeros(SHAPE, bool)
    for c, r in ((big, 6), (mid, 3), (tiny, 1)):
        _blob(mask, c, r)

    def run(p_big, p_mid):
        prob = np.zeros(SHAPE, np.float32)
        prob[_blob(np.zeros(SHAPE, bool), big, 6)] = p_big
        prob[_blob(np.zeros(SHAPE, bool), mid, 3)] = p_mid
        prob[_blob(np.zeros(SHAPE, bool), tiny, 1)] = 0.99
        return filter_components_joint(mask, prob, params)

    out = run(p_big=0.90, p_mid=0.95)
    assert out[big] and out[mid]                    # big: size+conf >= upper; mid: conf >= p_mid
    assert not out[tiny]                            # below s_lower regardless of confidence
    out = run(p_big=0.80, p_mid=0.90)
    assert not out[big]                             # big but under-confident (< p_upper)
    assert not out[mid]                             # mid-size and conf < p_mid


def test_size_only_tc_rule_ignores_confidence():
    """TC ships as (350, 350, 0, 0): a pure size cutoff."""
    params = PostProcessConfig().joint_filter["TC"]
    mask = _blob(np.zeros(SHAPE, bool), (24, 24, 24), 6)  # ~900 voxels
    low_conf = np.full(SHAPE, 0.01, np.float32)
    assert filter_components_joint(mask, low_conf, params).sum() == mask.sum()


# -- the pipeline ---------------------------------------------------------------------


def test_per_channel_thresholds_differ_for_et():
    probs = _probs({(ET, (24, 24, 24), 5): 0.45, (TC, (24, 24, 24), 5): 0.45, (WT, (24, 24, 24), 5): 0.45})
    cfg = dataclasses.replace(PostProcessConfig(), use_joint_filter=False, min_size={r: 0 for r in "ET TC WT".split()})
    out = postprocess(probs, "GLI", cfg)
    assert out["ET"].any()                          # 0.45 >= 0.40
    # nesting widens TC/WT to contain ET even though 0.45 < 0.50 for those channels
    assert (out["TC"] >= out["ET"]).all() and (out["WT"] >= out["TC"]).all()


def _ped_case(et_radius):
    return _probs({
        (WT, (24, 24, 24), 14): 0.99, (TC, (24, 24, 24), 9): 0.99, (ET, (24, 24, 24), et_radius): 0.99,
    })


def test_ped_et_gate_erases_small_et_but_keeps_it_as_core():
    cfg = dataclasses.replace(PostProcessConfig(), use_joint_filter=False, min_size={r: 0 for r in "ET TC WT".split()})
    probs = _ped_case(et_radius=2)  # ET/WT far below 0.04

    ped, gli = postprocess(probs, "PED", cfg), postprocess(probs, "GLI", cfg)

    assert not ped["ET"].any() and gli["ET"].any()  # gate only applies to PED
    assert ped["TC"].sum() == gli["TC"].sum()       # ET voxels remain tumor core
    assert ped["WT"].sum() == gli["WT"].sum()


def test_ped_et_gate_keeps_large_et():
    cfg = dataclasses.replace(PostProcessConfig(), use_joint_filter=False, min_size={r: 0 for r in "ET TC WT".split()})
    out = postprocess(_ped_case(et_radius=8), "PED", cfg)
    assert out["ET"].sum() / out["WT"].sum() > cfg.ped_et_wt_ratio and out["ET"].any()


def test_nesting_widens_containers_instead_of_erasing_et():
    et = np.zeros(SHAPE, bool); et[5, 5, 5] = True
    out = enforce_nesting({"ET": et, "TC": np.zeros(SHAPE, bool), "WT": np.zeros(SHAPE, bool)})
    assert out["ET"][5, 5, 5] and out["TC"][5, 5, 5] and out["WT"][5, 5, 5]


def test_postprocess_rejects_wrong_channel_count():
    with pytest.raises(ValueError):
        postprocess(np.zeros((4, *SHAPE), np.float32))


# -- label maps -----------------------------------------------------------------------


def test_label_map_round_trip_uses_label_3_for_et_never_4():
    rng = np.random.default_rng(0)
    seg = rng.integers(0, 4, SHAPE).astype(np.uint8)
    regions = {r: labels_to_regions(seg)[i].astype(bool) for i, r in enumerate(["ET", "TC", "WT"])}

    back = regions_to_label_map(regions)

    np.testing.assert_array_equal(back, seg)
    assert back.max() == LABEL_ET == 3
    assert set(np.unique(back)) <= {0, LABEL_NCR, LABEL_ED, LABEL_ET}
