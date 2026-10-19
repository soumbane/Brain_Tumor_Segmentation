"""No network: every W&B upload surface is captured by a fake SDK."""
import json
import sys
from types import SimpleNamespace

import pytest

from brats.tracking import WandbLogger, build_logger


class FakeConfig(dict):
    def update(self, values, **kwargs):
        super().update(values)


class FakeRun:
    def __init__(self, step=0):
        self.step = step
        self.settings = SimpleNamespace(mode="online")
        self.config = FakeConfig()
        self.summary = {}
        self.events = []
        self.fail_log = self.fail_finish = False

    def log(self, values, **kwargs):
        assert "step" not in kwargs, "optimizer replay must not rewind W&B history"
        if self.fail_log:
            raise RuntimeError("SECRET_MUST_NOT_APPEAR")
        self.events.append((self.step, values))
        self.step += 1

    def finish(self):
        if self.fail_finish:
            raise RuntimeError("SECRET_MUST_NOT_APPEAR")


@pytest.fixture
def sdk(monkeypatch):
    run = FakeRun()
    calls = []
    def init(**kwargs):
        calls.append(kwargs)
        run.config.update(kwargs["config"])
        return run
    module = SimpleNamespace(init=init, Settings=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "wandb", module)
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-test-only")
    return run, calls


def logger(tmp_path, **overrides):
    kwargs = dict(enabled=True, numeric_only=True, required_online=True,
                  mode="online", entity="authorized-entity", run_id="a" * 32,
                  dir=tmp_path)
    kwargs.update(overrides)
    return WandbLogger(**kwargs)


def test_required_online_rejects_missing_key_entity_and_offline(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        logger(tmp_path)
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-test-only")
    with pytest.raises(RuntimeError, match="entity"):
        logger(tmp_path, entity=None)
    with pytest.raises(ValueError, match="online"):
        logger(tmp_path, mode="offline")


def test_numeric_allowlists_every_upload_surface(tmp_path, sdk):
    run, calls = sdk
    tracker = logger(tmp_path, run_name="CASE_PATIENT", group="CASE_PATIENT",
                     tags=["CASE_PATIENT"], config={"case_id": "CASE_PATIENT", "lr": 0.1,
                                                  "optimizer": "CASE_PATIENT", "ckpt_dir": "/private"})
    tracker.log({"train_rank0/loss": 1.2, "CASE_PATIENT": 7,
                 "val/cls_acc": float("inf"), "epoch": "CASE_PATIENT"}, step=4)
    tracker.log_config_update({"effective_batch": 8, "patient_count_CASE_PATIENT": 1,
                               "pooling": "CASE_PATIENT", "patch_size": [128, 128, 128]})
    tracker.summary("case_id", "CASE_PATIENT")
    tracker.summary("best_dice_ET", 0.5)
    tracker.summary("best_dice_TC", "CASE_PATIENT")
    tracker.watch(object())  # fake SDK has no watch/artifact API: must not call them
    tracker.log_artifact("/private/CASE_PATIENT.nii.gz", "CASE_PATIENT")
    assert "CASE_PATIENT" not in json.dumps([calls, run.events, run.config, run.summary])
    assert "private" not in json.dumps(run.config)
    assert calls[0]["name"] == "brats-aaaaaaaaaaaa"
    assert calls[0]["group"] is None and calls[0]["tags"] == ["numeric-only"]
    assert calls[0]["settings"]["console"] == "off"
    assert calls[0]["settings"]["ignore_globs"] == ["*", "**/*"]
    assert calls[0]["settings"]["x_save_requirements"] is False
    assert calls[0]["settings"]["x_disable_meta"] is True
    assert calls[0]["settings"]["host"] == "spcs"
    assert calls[0]["settings"]["git_remote_url"] == ""
    assert calls[0]["resume"] == "never"
    assert run.events[-1][1] == {"train_rank0/loss": 1.2, "train/optimizer_step": 4}
    assert run.summary == {"best_dice_ET": 0.5}


def test_runtime_failure_is_visible_nonfatal_and_local(tmp_path, sdk, caplog):
    run, _ = sdk
    tracker = logger(tmp_path)
    run.fail_log = run.fail_finish = True
    tracker.log({"epoch/loss": 2.0}, step=2)
    assert tracker.enabled and not tracker.tracking_ok
    assert not tracker.finish()
    health = json.loads(tracker.health_path.read_text())
    assert health["errors"] == 2 and not health["flush_ok"]
    assert not health["remote_history_verified"]
    assert "epoch/loss" in tracker.metrics_path.read_text()
    assert "failed" in caplog.text and "SECRET_MUST_NOT_APPEAR" not in caplog.text


def test_resume_uses_server_sdk_high_water_not_checkpoint_counter(tmp_path, sdk):
    run, calls = sdk
    run.step = 120
    tracker = logger(tmp_path, resume=True)
    tracker.log({"epoch": 5}, step=30)  # restored checkpoint older than W&B
    tracker.log({"epoch": 5}, step=31)
    assert [event for event, _ in run.events] == [120, 121, 122]
    assert tracker.state_dict()["logging_sequence"] == 123
    assert calls[0]["resume"] == "must"
    assert tracker.finish()
    assert json.loads(tracker.health_path.read_text())["flush_ok"]
    assert not tracker.remote_history_verified


def test_rank_nonzero_has_no_sdk_or_local_side_effect(tmp_path, sdk):
    _, calls = sdk
    tracker = logger(tmp_path, is_rank_zero=False)
    tracker.log({"epoch": 1})
    assert not calls and tracker.health_path is None


def test_required_init_failure_is_fatal_but_does_not_print_secret(tmp_path, sdk, monkeypatch, caplog):
    def fail(**kwargs):
        raise RuntimeError("SECRET_MUST_NOT_APPEAR")
    monkeypatch.setattr(sys.modules["wandb"], "init", fail)
    with pytest.raises(RuntimeError, match="preflight failed"):
        logger(tmp_path)
    assert "SECRET_MUST_NOT_APPEAR" not in caplog.text


def test_config_strings_cannot_leak_case_names(tmp_path, sdk):
    tracker = logger(tmp_path, config={"optimizer": "adamw", "pooling": ["case"],
                                     "patch_size": [128, "case", 128]})
    assert sdk[0].config == {"optimizer": "adamw"}
    assert tracker.finish()


def test_flush_is_bounded_and_inherited_metadata_removed(tmp_path, sdk, monkeypatch):
    import threading
    import time
    monkeypatch.setenv("WANDB_NOTES", "CASE_PATIENT")
    monkeypatch.setenv("WANDB_CONFIG_PATHS", "/private/patient.json")
    tracker = logger(tmp_path)
    import os
    assert "WANDB_NOTES" not in os.environ and "WANDB_CONFIG_PATHS" not in os.environ
    release = threading.Event()
    sdk[0].finish = lambda: release.wait(10)
    tracker.finish_timeout_seconds = 0.05
    start = time.monotonic()
    try:
        assert not tracker.finish()
        assert time.monotonic() - start < 1
        assert not json.loads(tracker.health_path.read_text())["flush_ok"]
    finally:
        release.set()


def test_mlflow_selection_still_has_priority(monkeypatch):
    class Sentinel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    monkeypatch.setattr("brats.tracking.MLflowLogger", Sentinel)
    cfg = SimpleNamespace(use_mlflow=True, run_name="databricks")
    assert isinstance(build_logger(cfg), Sentinel)
