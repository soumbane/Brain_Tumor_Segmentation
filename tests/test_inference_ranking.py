"""Inference (TTA alignment, YAML config) and the BraTS-rank utilities."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from torch import nn  # noqa: E402

from brats.inference import (  # noqa: E402
    InferenceConfig,
    eight_flip_axes,
    inference_config_from_yaml,
    predict_volume,
)
from brats.metrics.ranking import ablation_table, rank_configurations, select_best_config  # noqa: E402


class _Pointwise(nn.Module):
    """Segmentation logits are a fixed per-voxel function of the input: flip-equivariant."""

    def forward(self, x):
        return {"seg": x[:, :3] * 2.0 - 1.0, "cls": torch.zeros(x.shape[0], 3)}


class _Asymmetric(nn.Module):
    """Looks only at the voxel to its left, so it is *not* flip-equivariant."""

    def forward(self, x):
        left = torch.roll(x[:, :3], shifts=1, dims=-1)
        return {"seg": left, "cls": torch.zeros(x.shape[0], 3)}


def _image():
    torch.manual_seed(0)
    return torch.randn(1, 4, 16, 16, 16)


def test_eight_flip_set_is_the_eight_distinct_combinations():
    axes = eight_flip_axes()
    assert len(axes) == len(set(axes)) == 8 and () in axes and (2, 3, 4) in axes


def test_tta_with_a_flip_equivariant_model_changes_nothing():
    """If the un-flip were missing or misaligned, averaging would blur the prediction."""
    img = _image()
    base = predict_volume(_Pointwise(), img, InferenceConfig(roi_size=(16, 16, 16)), device="cpu")
    tta = predict_volume(
        _Pointwise(), img,
        InferenceConfig(roi_size=(16, 16, 16), tta_flip_axes=eight_flip_axes()), device="cpu",
    )
    np.testing.assert_allclose(tta["probs"], base["probs"], atol=1e-5)


def test_tta_actually_averages_for_an_asymmetric_model():
    img = _image()
    cfg = InferenceConfig(roi_size=(16, 16, 16))
    base = predict_volume(_Asymmetric(), img, cfg, device="cpu")["probs"]
    tta = predict_volume(
        _Asymmetric(), img, InferenceConfig(roi_size=(16, 16, 16), tta_flip_axes=eight_flip_axes()),
        device="cpu",
    )["probs"]
    assert not np.allclose(base, tta, atol=1e-3)
    assert tta.shape == base.shape == (3, 16, 16, 16)


def test_inference_config_comes_from_the_yaml():
    cfg = inference_config_from_yaml()
    assert (cfg.overlap, cfg.mode, cfg.sw_batch_size) == (0.5, "gaussian", 4)
    assert cfg.tta_flip_axes == ()
    assert len(inference_config_from_yaml(tta="eight_flip").tta_flip_axes) == 8


def test_training_patch_overrides_the_yaml_roi():
    assert inference_config_from_yaml(roi_size=(96, 96, 96)).roi_size == (96, 96, 96)


def test_bad_tta_value_is_rejected():
    with pytest.raises(ValueError):
        inference_config_from_yaml(tta="sixteen_flip")


# -- ranking --------------------------------------------------------------------------


def _scores():
    rows = []
    for case in range(10):
        for region in ("ET", "TC", "WT"):
            for cfg, dice, hd in (("good", 0.90, 5.0), ("meh", 0.80, 9.0), ("bad", 0.50, 40.0)):
                rows.append(dict(config=cfg, case_id=f"c{case}", region=region, lw_dice=dice, lw_hd95=hd,
                                 legacy_dice=dice, stage=cfg))
    return pd.DataFrame(rows)


def test_rank_orders_configurations_by_per_case_rank():
    table = rank_configurations(_scores())
    assert list(table.config) == ["good", "meh", "bad"]
    assert table.rank_score.round(3).tolist() == [1.0, 2.0, 3.0]


def test_rank_beats_mean_when_one_config_wins_most_cases_but_loses_big_on_one():
    """The point of ranking (Ferreira): a config that is slightly better on nearly every
    case should beat one whose higher *mean* comes from a single case."""
    rows = []
    for case in range(10):
        steady = (0.80, 10.0)
        spiky = (0.79, 10.5) if case else (1.00, 1.0)  # worse on 9 cases, perfect on one
        for cfg, (d, h) in (("steady", steady), ("spiky", spiky)):
            rows.append(dict(config=cfg, case_id=f"c{case}", region="WT", lw_dice=d, lw_hd95=h))
    df = pd.DataFrame(rows)
    assert df.groupby("config").lw_dice.mean().idxmax() == "spiky"   # mean picks the spiky config...
    assert rank_configurations(df).config.iloc[0] == "steady"        # ...rank does not


def test_select_best_config_breaks_ties_towards_the_milder_worst_case():
    rows = []
    for case in range(20):
        for cfg, d in (("safe", 0.80), ("risky", 0.80 if case % 2 else 0.80)):
            worst = 0.2 if (cfg == "risky" and case == 0) else d
            rows.append(dict(config=cfg, case_id=f"c{case}", region="WT", lw_dice=worst, lw_hd95=5.0))
    chosen, table = select_best_config(pd.DataFrame(rows), risk_margin=0.5)
    assert chosen == "safe" and len(table) == 2


def test_ablation_table_reports_deltas_against_the_first_stage():
    t = ablation_table(_scores())
    assert t.stage.tolist() == ["good", "meh", "bad"]
    assert t.lw_dice_delta.iloc[0] == 0.0 and t.lw_dice_delta.iloc[2] == pytest.approx(-0.4)
