"""Cross-module consistency: things that are individually plausible but only matter together.

Channel order, schedule/resume, sliding-window stitching, the cohort aggregation rule, and the
smaller utilities (tracking, classification metrics, confound features) that had no coverage.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from torch import nn  # noqa: E402

from brats.constants import COHORTS, REGION_INDEX, REGION_LABELS, REGIONS  # noqa: E402
from brats.data.preprocess import labels_to_regions  # noqa: E402
from brats.inference import InferenceConfig, predict_volume  # noqa: E402
from brats.postprocess import PostProcessConfig, postprocess, regions_to_label_map  # noqa: E402


# -- channel order (ET, TC, WT) is the same everywhere ---------------------------------------


def _label_volume():
    seg = np.zeros((24, 24, 24), np.uint8)
    seg[4:20, 4:20, 4:20] = 2   # ED
    seg[7:17, 7:17, 7:17] = 1   # NCR
    seg[10:14, 10:14, 10:14] = 3  # ET
    return seg


def test_regions_are_defined_in_et_tc_wt_order_and_match_the_label_semantics():
    assert REGIONS == ("ET", "TC", "WT")
    seg = _label_volume()
    regions = labels_to_regions(seg)
    for i, name in enumerate(REGIONS):
        assert REGION_INDEX[name] == i
        np.testing.assert_array_equal(regions[i].astype(bool), np.isin(seg, list(REGION_LABELS[name])))
    assert regions[0].sum() < regions[1].sum() < regions[2].sum()  # nested: ET < TC < WT


def test_perfect_probabilities_survive_postprocessing_back_to_the_same_label_map():
    seg = _label_volume()
    probs = labels_to_regions(seg).astype(np.float32) * 0.98 + 0.01
    cfg = PostProcessConfig(use_joint_filter=False, min_size={r: 0 for r in REGIONS}, apply_ped_et_gate=False)

    out = postprocess(probs, "GLI", cfg)

    np.testing.assert_array_equal(regions_to_label_map(out), seg)


class _OracleModel(nn.Module):
    """Reads the ground truth from the first three input channels and emits perfect logits.
    ``order`` lets a test deliberately scramble the output channels."""

    def __init__(self, order=(0, 1, 2)):
        super().__init__()
        self.order = list(order)

    def forward(self, x, gt_wt=None):
        seg = x[:, self.order] * 20.0 - 10.0
        return {"seg": seg, "cls": torch.zeros(x.shape[0], 3)}


def _validation_loader():
    seg = _label_volume()
    label = torch.from_numpy(labels_to_regions(seg).astype(np.float32))[None]       # (1,3,D,H,W)
    image = torch.cat([label, torch.zeros(1, 1, *label.shape[2:])], dim=1)            # (1,4,D,H,W)
    batch = {"image": image, "label": label, "cohort": torch.tensor([0])}
    return [batch]


def test_validate_reports_dice_one_per_region_for_an_oracle_and_less_if_channels_are_swapped():
    from brats.train import TrainConfig, validate

    cfg = TrainConfig(patch_size=(16, 16, 16))
    good = validate(_OracleModel(), _validation_loader(), torch.device("cpu"), cfg)
    assert all(good[f"dice_{r}"] == pytest.approx(1.0) for r in REGIONS) and good["dice_avg"] == pytest.approx(1.0)
    assert good["cls_acc"] == 1.0

    # (TC, WT, ET) is the MONAI model-zoo order the project warns about.
    scrambled = validate(_OracleModel(order=(1, 2, 0)), _validation_loader(), torch.device("cpu"), cfg)
    assert scrambled["dice_avg"] < 0.95


# -- sliding window ---------------------------------------------------------------------------


class _Pointwise(nn.Module):
    def forward(self, x, gt_wt=None):
        return {"seg": torch.tanh(x[:, :3]) * 4.0, "cls": torch.zeros(x.shape[0], 3)}


@pytest.mark.parametrize("sw_batch_size", [1, 3, 8])
def test_stitched_prediction_equals_the_direct_one_for_a_pointwise_model(sw_batch_size):
    torch.manual_seed(0)
    x = torch.randn(1, 4, 40, 36, 44)  # larger than, and not a multiple of, the window
    cfg = InferenceConfig(roi_size=(16, 16, 16), overlap=0.5, sw_batch_size=sw_batch_size)

    probs = predict_volume(_Pointwise(), x, cfg, device="cpu")["probs"]

    expected = torch.sigmoid(torch.tanh(x[0, :3]) * 4.0).numpy()
    np.testing.assert_allclose(probs, expected, atol=1e-5)


class _TumorAwareCls(nn.Module):
    """Predicts WT where input channel 2 is positive; says 'GLI' in windows containing tumor
    and 'PED' in windows without any (which carry no real evidence)."""

    def forward(self, x, gt_wt=None):
        wt = (x[:, 2:3] > 0).float() * 20.0 - 10.0
        seg = torch.cat([wt, wt, wt], dim=1)
        has_tumor = (x[:, 2] > 0).flatten(1).any(dim=1).float()
        cls = torch.stack([has_tumor * 5.0, torch.zeros_like(has_tumor), (1 - has_tumor) * 5.0], dim=1)
        return {"seg": seg, "cls": cls}


def test_cohort_vote_is_weighted_by_predicted_tumor_so_empty_windows_do_not_outvote():
    x = torch.zeros(1, 4, 48, 48, 48)
    x[0, 2, 4:12, 4:12, 4:12] = 1.0  # one small tumor in a corner of a mostly empty volume
    cfg = InferenceConfig(roi_size=(16, 16, 16), overlap=0.5, sw_batch_size=4)

    out = predict_volume(_TumorAwareCls(), x, cfg, device="cpu")

    # An unweighted mean over the many tumor-free windows would say PED; the weighted vote says GLI.
    assert int(np.argmax(out["cls_probs"])) == COHORTS.index("GLI")


# -- LR schedule ------------------------------------------------------------------------------


def _scheduler(steps_per_epoch=7, epochs=10, warmup=2):
    from brats.train import TrainConfig, build_scheduler

    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=2e-4)
    return opt, build_scheduler(opt, TrainConfig(epochs=epochs, warmup_epochs=warmup), steps_per_epoch)


def _lrs(opt, sched, n):
    out = []
    for _ in range(n):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return out


def test_schedule_warms_up_linearly_then_cosines_to_zero():
    opt, sched = _scheduler()
    lrs = _lrs(opt, sched, 70)
    warm = 2 * 7
    assert lrs[0] == 0.0
    assert lrs[warm] == pytest.approx(2e-4)
    assert all(a <= b + 1e-12 for a, b in zip(lrs[:warm], lrs[1 : warm + 1]))      # rising
    assert all(a >= b - 1e-12 for a, b in zip(lrs[warm:-1], lrs[warm + 1 :]))      # falling
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)               # ends at ~0


def test_resumed_schedule_continues_exactly_where_it_stopped():
    opt, sched = _scheduler()
    full = _lrs(opt, sched, 50)

    opt1, sched1 = _scheduler()
    _lrs(opt1, sched1, 23)
    state_o, state_s = opt1.state_dict(), sched1.state_dict()
    opt2, sched2 = _scheduler()
    opt2.load_state_dict(state_o)
    sched2.load_state_dict(state_s)
    resumed = _lrs(opt2, sched2, 27)

    np.testing.assert_allclose(resumed, full[23:], rtol=1e-9, atol=0)


# -- classification metrics -------------------------------------------------------------------


def _preds(correct):
    rng = np.random.default_rng(0)
    out = []
    for i in range(30):
        c = COHORTS[i % 3]
        p = np.full(3, 0.05, np.float32)
        p[COHORTS.index(c) if correct else (COHORTS.index(c) + 1) % 3] = 0.9
        out.append({"case_id": f"x{i}", "cohort": c, "cls_probs": p / p.sum()})
    return out


def test_classification_metrics_for_perfect_and_always_wrong_predictions():
    from brats.evaluate import evaluate_classification

    _, good = evaluate_classification(_preds(True))
    assert good["accuracy"] == good["balanced_accuracy"] == good["macro_f1"] == 1.0
    assert good["macro_auc"] == 1.0 and good["confusion"] == [[10, 0, 0], [0, 10, 0], [0, 0, 10]]

    _, bad = evaluate_classification(_preds(False))
    assert bad["accuracy"] == 0.0 and bad["balanced_accuracy"] == 0.0


# -- aggregation -------------------------------------------------------------------------------


def test_aggregate_and_bootstrap_ci_are_sane():
    import pandas as pd

    from brats.metrics.lesionwise import aggregate, bootstrap_ci

    lo, hi = bootstrap_ci(np.array([0.2, 0.4, 0.6, 0.8, 1.0]), n_boot=2000)
    assert 0.2 <= lo < 0.6 < hi <= 1.0
    assert bootstrap_ci(np.array([0.5])) == (0.5, 0.5)
    assert all(np.isnan(bootstrap_ci(np.array([]))))

    df = pd.DataFrame({
        "cohort": ["GLI"] * 4 + ["PED"] * 2, "region": ["WT"] * 6,
        "lw_dice": [0.5, 0.7, 0.9, 1.0, 0.2, 0.4], "lw_hd95": [5, 3, 2, 1, 50, 40], "legacy_dice": [0.5] * 6,
    })
    agg = aggregate(df).set_index("cohort")
    assert agg.loc["GLI", "n"] == 4 and agg.loc["GLI", "lw_dice_mean"] == pytest.approx(0.775)
    assert agg.loc["GLI", "lw_dice_median"] == pytest.approx(0.8) and agg.loc["PED", "n"] == 2


# -- experiment tracking ------------------------------------------------------------------------


def test_disabled_and_non_zero_rank_loggers_are_silent_no_ops():
    from brats.tracking import WandbLogger

    for logger in (WandbLogger(enabled=False), WandbLogger(enabled=True, is_rank_zero=False),
                   WandbLogger(enabled=True, mode="disabled")):
        assert not logger.enabled
        logger.log({"x": 1.0}, step=1)
        logger.summary("k", 1)
        logger.finish()


def test_offline_logger_runs_and_survives_bad_values(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    from brats.tracking import WandbLogger

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WANDB_MODE", "offline")
    logger = WandbLogger(enabled=True, mode="offline", run_name="t", config={"a": 1}, dir=tmp_path / "w")
    assert logger.enabled
    logger.log({"loss": 1.0, "nan": float("nan"), "text": "ignored"}, step=1)
    logger.summary("best", 0.5)
    logger.finish()  # must not raise


# -- confound features ---------------------------------------------------------------------------


def test_confound_features_have_the_documented_shape_and_use_no_tumor_information(tmp_path):
    import nibabel as nib

    from brats.confound.probes import case_features
    from brats.constants import SEQUENCES

    rng = np.random.default_rng(0)
    d = tmp_path / "BraTS-GLI-00001-000"
    d.mkdir()
    brain = np.zeros((40, 44, 36), np.float32)
    brain[5:35, 6:38, 4:32] = rng.uniform(10, 200, (30, 32, 28))
    for s in SEQUENCES:
        nib.save(nib.Nifti1Image(brain, np.eye(4)), str(d / f"{d.name}-{s}.nii.gz"))
    # A segmentation file is deliberately present and deliberately corrupt: features must not read it.
    (d / f"{d.name}-seg.nii.gz").write_bytes(b"not a nifti")

    f = case_features(str(d), "GLI")

    assert f is not None
    n = lambda prefix: sum(k.startswith(prefix) for k in f)  # noqa: E731
    assert (n("hist_"), n("stat_"), n("geom_")) == (128, 44, 19)   # matches the committed pre-screen report
    assert all(np.isfinite(v) for k, v in f.items() if k not in ("case_id", "cohort"))
    assert f["geom_volume"] == float((brain != 0).sum())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_inference_on_an_accelerator_matches_cpu():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 24, 24, 24)
    cfg = InferenceConfig(roi_size=(16, 16, 16), overlap=0.5)
    cpu = predict_volume(_Pointwise(), x, cfg, device="cpu")["probs"]
    mps = predict_volume(_Pointwise(), x, cfg, device="mps")["probs"]
    np.testing.assert_allclose(mps, cpu, atol=1e-5)
