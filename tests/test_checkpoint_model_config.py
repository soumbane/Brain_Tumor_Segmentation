"""A checkpoint must be loadable without knowing how it was trained.

Regression for: the trainer picks ``norm`` from the effective batch size (batch norm needs
>= 4, otherwise instance norm), but ``evaluate`` / ``qc`` / ``binarize_control`` rebuilt the
model from the YAML with the default ``norm="batch"``. A checkpoint from any run with an
effective batch below 4 -- e.g. the single-GPU SPCS spec -- then failed to load.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from torch import nn  # noqa: E402

from brats.models.multitask import (  # noqa: E402
    MultiTaskBraTS,
    MultiTaskConfig,
    load_model_from_checkpoint,
)
from brats.train import CheckpointManager, TrainConfig  # noqa: E402


# A tiny model: these tests write real checkpoints, and the default 20M-parameter model
# is ~250 MB per file (optimizer state included).
SMALL = dict(init_filters=8)


def _save(tmp_path, model, monkeypatch, name="run"):
    monkeypatch.delenv("BRATS_CKPT_ROOT", raising=False)
    mgr = CheckpointManager(TrainConfig(run_name=name), root=tmp_path)
    mgr.save(
        epoch=0,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters()),
        scheduler=None,
        loss_module=nn.Identity(),
        metrics={"dice_avg": 0.5},
    )
    return mgr.dir / "last.pt"


def test_the_mismatch_is_real():
    """Guards the guard: an instance-norm state dict does not fit a batch-norm model."""
    inst = MultiTaskBraTS(MultiTaskConfig(norm="instance", **SMALL))
    with pytest.raises(RuntimeError):
        MultiTaskBraTS(MultiTaskConfig(norm="batch", **SMALL)).load_state_dict(inst.state_dict())


@pytest.mark.parametrize("norm", ["batch", "instance"])
def test_checkpoint_round_trips_without_a_fallback(tmp_path, monkeypatch, norm):
    model = MultiTaskBraTS(MultiTaskConfig(norm=norm, **SMALL))
    path = _save(tmp_path, model, monkeypatch)

    loaded = load_model_from_checkpoint(path)  # note: no config supplied

    assert loaded.cfg.norm == norm
    for (k, a), (_, b) in zip(
        model.state_dict().items(), loaded.state_dict().items(), strict=True
    ):
        torch.testing.assert_close(a, b, msg=k)


@pytest.mark.parametrize("norm", ["batch", "instance"])
def test_legacy_checkpoint_without_model_config(tmp_path, monkeypatch, norm):
    """Checkpoints that predate ``model_config`` still load: norm is read off the weights."""
    model = MultiTaskBraTS(MultiTaskConfig(norm=norm, **SMALL))
    path = _save(tmp_path, model, monkeypatch)
    state = torch.load(path, weights_only=False)
    del state["model_config"]
    torch.save(state, path)

    loaded = load_model_from_checkpoint(path, fallback=MultiTaskConfig(pooling="tumor_attention", **SMALL))

    assert loaded.cfg.norm == norm
