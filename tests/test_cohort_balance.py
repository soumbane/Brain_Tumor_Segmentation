"""Cohort balance is corrected once, and a resumed run keeps *its own* loss configuration.

Regressions for: the weighted sampler and the class-weighted CE both corrected for the minority
cohort (a ~12x over-weighting of PED), and class weights lived in the checkpoint so a resumed run
silently inherited whichever weights the original run used.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from brats.train import TrainConfig, uses_balanced_sampler  # noqa: E402


def test_balanced_sampler_only_in_single_process_runs():
    on = TrainConfig(cohort_balanced_sampling=True)
    off = TrainConfig(cohort_balanced_sampling=False)
    assert uses_balanced_sampler(on, 1)
    assert not uses_balanced_sampler(on, 4)  # DDP: DistributedSampler owns sharding
    assert not uses_balanced_sampler(off, 1)



# -- resume must not import stale loss configuration ----------------------------------------


def _save_ckpt(tmp_path, loss_module, monkeypatch):
    from brats.models.multitask import MultiTaskBraTS, MultiTaskConfig
    from brats.train import CheckpointManager

    monkeypatch.delenv("BRATS_CKPT_ROOT", raising=False)
    model = MultiTaskBraTS(MultiTaskConfig(norm="instance", init_filters=8))  # tiny: real checkpoints are written
    mgr = CheckpointManager(TrainConfig(run_name="r"), root=tmp_path)
    mgr.save(0, model, torch.optim.AdamW(model.parameters()), None, loss_module, {"dice_avg": 0.1})
    return mgr, model


def test_resume_keeps_this_runs_class_weights_not_the_checkpoints(tmp_path, monkeypatch):
    from brats.losses import build_loss

    writer = build_loss(lambda_cls=0.1)                                # inverse-frequency weights
    mgr, model = _save_ckpt(tmp_path, writer, monkeypatch)
    assert "cls_loss.class_weights" not in writer.state_dict()         # not persisted at all

    resumer = build_loss(lambda_cls=0.1, class_weights=torch.ones(3))  # e.g. balanced sampler run
    mgr.load(model, loss_module=resumer)

    assert torch.equal(resumer.cls_loss.class_weights, torch.ones(3))


def test_resume_accepts_old_checkpoints_that_carry_class_weights(tmp_path, monkeypatch):
    from brats.losses import build_loss

    writer = build_loss(lambda_cls=0.1)
    mgr, model = _save_ckpt(tmp_path, writer, monkeypatch)
    state = torch.load(mgr.dir / "last.pt", weights_only=False)
    state["loss_module"]["cls_loss.class_weights"] = torch.tensor([0.2, 0.25, 2.5])  # the old format
    torch.save(state, mgr.dir / "last.pt")

    resumer = build_loss(lambda_cls=0.1, class_weights=torch.ones(3))
    mgr.load(model, loss_module=resumer)                               # must not raise

    assert torch.equal(resumer.cls_loss.class_weights, torch.ones(3))


def test_uncertainty_log_variances_still_resume(tmp_path, monkeypatch):
    from brats.losses import build_loss

    writer = build_loss(weighting="uncertainty")
    with torch.no_grad():
        writer.uncertainty.log_var.copy_(torch.tensor([0.7, -0.3]))
    mgr, model = _save_ckpt(tmp_path, writer, monkeypatch)

    resumer = build_loss(weighting="uncertainty")
    mgr.load(model, loss_module=resumer)

    assert torch.allclose(resumer.uncertainty.log_var, torch.tensor([0.7, -0.3]))
