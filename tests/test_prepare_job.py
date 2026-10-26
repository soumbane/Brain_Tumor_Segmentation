"""Offline launch preparation must never submit, authenticate, or start training."""
import importlib.util
import json
from pathlib import Path
import socket
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
module_spec = importlib.util.spec_from_file_location("prepare_job", ROOT / "spcs/prepare_job.py")
prepare_job = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(prepare_job)


@pytest.fixture
def inputs():
    return dict(mode="smoke", secret_name="CARE_ML_RESEARCH_WKSP.BRAIN_TUMOR_SEG.TEST_WANDB_KEY",
                cache_audit_path="/mnt/output/audits/verified-cache.json", split_sha256="a" * 64,
                runtime_seconds=600)


def container(document):
    return document["spec"]["containers"][0]


def test_smoke_has_real_harness_not_full_trainer(inputs):
    document = prepare_job.prepare_spec(**inputs)
    target = container(document)
    command = target["command"]
    assert command[:11] == prepare_job.LAUNCHER
    assert command[11] == "/app/scripts/gpu_smoke_test.py"
    assert "brats.train" not in command and "--epochs" not in command
    assert command[command.index("--steps") + 1] == "2"
    assert "--acknowledge-paid-gpu" in command
    assert "--resume" not in command
    assert target["env"]["BRATS_RUN_ID"].startswith("smoke-")
    assert command[command.index("--run-id") + 1] == target["env"]["BRATS_RUN_ID"]
    assert target["resources"]["requests"]["nvidia.com/gpu"] == 4
    assert target["env"]["BRATS_WANDB_ENTITY"] == "banerjee-s-corewell"
    assert target["image"].endswith(":spcs-20261026-runtime1")
    assert "REQUIRED_" not in yaml.safe_dump(document)
    assert "WANDB_API_KEY" not in target["env"]
    assert target["secrets"][0]["snowflakeSecret"]["objectName"] == inputs["secret_name"]


def test_full_training_and_runtime_config_contract(inputs, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("monai")
    from brats.train import TrainConfig, build_parser

    inputs["mode"] = "train"
    target = container(prepare_job.prepare_spec(**inputs))
    command = target["command"]
    assert "brats.train" in command and "/app/scripts/gpu_smoke_test.py" not in command
    assert "--resume" not in command
    args = build_parser().parse_args(command[command.index("brats.train") + 1:])
    assert args.config == prepare_job.TRAIN_CONFIG
    cfg = TrainConfig.from_yaml(ROOT / "configs/segresnet_spcs.yaml")
    monkeypatch.setenv("BRATS_CKPT_ROOT", "/work")
    for name in ("run_id", "attempt_id", "cache_audit_path", "expected_split_sha256", "wandb_entity", "max_runtime_seconds"):
        value = target["env"][f"BRATS_{name.upper()}"]
        setattr(cfg, name, int(value) if name == "max_runtime_seconds" else value)
    cfg.validate_config(launch=True)
    assert cfg.epochs == 100 and cfg.val_every == 10
    assert cfg.require_online_tracking and cfg.numeric_only_tracking and not cfg.use_mlflow


def test_resume_and_interruption_are_explicit(inputs):
    with pytest.raises(ValueError, match="existing run-id"):
        prepare_job.prepare_spec(**inputs, resume=True)
    original = container(prepare_job.prepare_spec(**inputs))
    resumed = container(prepare_job.prepare_spec(**inputs, run_id=original["env"]["BRATS_RUN_ID"], resume=True))
    assert "--resume" in resumed["command"]
    assert resumed["env"]["BRATS_RUN_ID"] == original["env"]["BRATS_RUN_ID"]
    assert resumed["env"]["BRATS_ATTEMPT_ID"] != original["env"]["BRATS_ATTEMPT_ID"]
    interrupted = container(prepare_job.prepare_spec(**inputs, interrupt_after_publication=True, smoke_lambda_cls=0.0))
    assert "--interrupt-after-publication" in interrupted["command"]
    assert interrupted["command"][interrupted["command"].index("--lambda-cls") + 1] == "0.0"


@pytest.mark.parametrize("overrides", [
    {"runtime_seconds": 0}, {"runtime_seconds": 120}, {"runtime_seconds": 1.5},
    {"runtime_seconds": True}, {"split_sha256": "REQUIRED_HASH"},
    {"secret_name": "api-key-not-an-object"}, {"secret_name": "OTHER.SCHEMA.KEY"},
    {"secret_name": "CARE_ML_RESEARCH_WKSP.BRAIN_TUMOR_SEG.REQUIRED_WANDB_SECRET"},
    {"cache_audit_path": "/mnt/data/audit.json"}, {"cache_audit_path": "/mnt/output/../data/audit.json"},
    {"cache_audit_path": "/mnt/output/REQUIRED_AUDIT.json"}, {"cache_audit_path": "@OUTPUT_STAGE/audit.json"},
    {"run_id": "../escape"}, {"run_id": "train-wrong-mode"}, {"attempt_id": "REQUIRED_ATTEMPT"},
    {"smoke_steps": 1}, {"smoke_lambda_cls": float("nan")},
    {"mode": "train", "run_id": "smoke-collision"},
    {"mode": "train", "smoke_steps": 2}, {"mode": "train", "interrupt_after_publication": True},
])
def test_invalid_or_unsafe_inputs_are_rejected(inputs, overrides):
    with pytest.raises(ValueError):
        prepare_job.prepare_spec(**{**inputs, **overrides})


def test_input_template_is_not_mutated_and_ids_are_unique(inputs):
    before = prepare_job.TEMPLATE.read_bytes()
    first = container(prepare_job.prepare_spec(**inputs))
    second = container(prepare_job.prepare_spec(**inputs))
    assert first["env"]["BRATS_RUN_ID"] != second["env"]["BRATS_RUN_ID"]
    assert first["env"]["BRATS_ATTEMPT_ID"] != second["env"]["BRATS_ATTEMPT_ID"]
    assert prepare_job.TEMPLATE.read_bytes() == before


def test_cli_only_writes_requested_local_spec(inputs, tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("local preparation attempted network/process execution")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    output = tmp_path / "smoke.yaml"
    arguments = ["--output", str(output)]
    for key, value in inputs.items():
        arguments.extend(["--" + key.replace("_", "-"), str(value)])
    assert prepare_job.main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "prepared_not_submitted"
    assert result["remote_prerequisites_verified"] is False
    assert result["compute_pool"] == "CARE_ML_RSRCH_GPU_M"
    assert result["external_access_integration"] == "CARE_ML_RSRCH_INTERNET_EAI"
    assert container(yaml.safe_load(output.read_text()))["env"]["BRATS_RUN_ID"] == result["run_id"]
    assert len(list(tmp_path.iterdir())) == 1
    with pytest.raises(SystemExit):
        prepare_job.main(arguments)  # do not overwrite, even with freshly generated IDs


def test_no_default_budget_or_secret_and_no_writes_on_missing_inputs(tmp_path):
    with pytest.raises(SystemExit):
        prepare_job.main(["--mode", "smoke", "--output", str(tmp_path / "job.yaml")])
    assert not list(tmp_path.iterdir())


def test_existing_output_symlink_and_stage_destinations_rejected(inputs, tmp_path):
    document = prepare_job.prepare_spec(**inputs)
    existing = tmp_path / "existing.yaml"
    existing.write_text("preserve")
    alias = tmp_path / "alias.yaml"
    alias.symlink_to(existing)
    with pytest.raises(FileExistsError):
        prepare_job.write_spec(document, alias, mode="smoke")
    assert existing.read_text() == "preserve"
    with pytest.raises(ValueError, match="local storage"):
        prepare_job.write_spec(document, Path("/mnt/output/spec.yaml"), mode="smoke")
