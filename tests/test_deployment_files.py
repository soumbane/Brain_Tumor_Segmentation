"""Deployment invariants executable without Snowflake or W&B credentials."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from brats.config import REPO_ROOT
from brats.train import TrainConfig, build_parser

SPEC = yaml.safe_load((REPO_ROOT / "spcs/service_spec.yaml").read_text())
CONTAINER = SPEC["spec"]["containers"][0]


def test_four_rank_fresh_launch_cli():
    command = CONTAINER["command"]
    assert command[:2] == ["python", "/app/spcs/supervise.py"]
    # Module launch keeps both elastic and its rank workers on the image venv.
    assert ["python", "-m", "torch.distributed.run", "--standalone",
            "--nnodes=1", "--nproc_per_node=4"] == command[5:11]
    offset = command.index("brats.train") + 1
    args = build_parser().parse_args(command[offset:])
    assert not args.resume and not args.no_wandb
    assert Path(args.config).name == "segresnet_spcs.yaml"


def test_resources_and_stage_syntax_leave_headroom():
    resources = CONTAINER["resources"]
    assert resources["requests"]["nvidia.com/gpu"] == resources["limits"]["nvidia.com/gpu"] == 4
    assert resources["requests"]["cpu"] == 40
    assert resources["requests"]["memory"] == "144Gi"
    assert resources["limits"]["memory"] == "160Gi"
    volumes = {v["name"]: v for v in SPEC["spec"]["volumes"]}
    assert volumes["shm"] == {"name": "shm", "source": "memory", "size": "16Gi"}
    assert volumes["work"]["source"] == "local"
    for name in ("data", "output"):
        assert volumes[name]["source"] == "stage"
        assert volumes[name]["stageConfig"]["name"].startswith("@CARE_ML_RESEARCH_WKSP.")
        assert "readOnly" not in volumes[name]
    assert "endpoints" not in SPEC["spec"]


def test_stage_and_local_paths_are_separate():
    env = CONTAINER["env"]
    assert env["BRATS_CKPT_ROOT"] == "/work"
    assert env["BRATS_ARTIFACT_ROOT"].startswith("/mnt/output/")
    assert "WANDB_API_KEY" not in env
    secret = CONTAINER["secrets"][0]
    assert secret["envVarName"] == "WANDB_API_KEY" and secret["secretKeyRef"] == "secret_string"
    assert secret["snowflakeSecret"]["objectName"] == (
        "CARE_ML_RESEARCH_WKSP.BRAIN_TUMOR_SEG.WANDB_API_KEY"
    )
    assert "REQUIRED" in env["BRATS_MAX_RUNTIME_SECONDS"]


def test_strict_spcs_recipe_and_legacy_isolation():
    cfg = TrainConfig.from_yaml(REPO_ROOT / "configs/segresnet_spcs.yaml")
    assert cfg.batch_size_per_gpu == 2 and cfg.patch_size == (128, 128, 128)
    assert cfg.expected_world_size == 4 and cfg.num_workers == 4 and cfg.prefetch_factor == 1
    assert cfg.epochs == 100 and cfg.warmup_epochs == 8 and cfg.cls_warmup_epochs == 20
    assert cfg.require_online_tracking and cfg.numeric_only_tracking and not cfg.use_mlflow
    with pytest.raises(ValueError):
        cfg.validate_config(launch=True)
    base = TrainConfig.from_yaml(REPO_ROOT / "configs/segresnet_base.yaml")
    assert base.use_mlflow and base.batch_size_per_gpu == 16


def test_image_pins_and_runtime_packaging():
    text = (REPO_ROOT / "Dockerfile").read_text()
    assert "-runtime@sha256:" in text and "-devel" not in text
    assert "torch.__version__" in text and "pip check" in text
    assert "PYTHONPATH=\"/app/src" in text
    for needed in ("src/", "configs/", "splits/", "scripts/", "spcs/"):
        assert re.search(rf"^COPY\s+{re.escape(needed)}\s", text, re.M)
    for src in re.findall(r"^COPY\s+(\S+)\s+\S+\s*$", text, re.M):
        assert (REPO_ROOT / src).exists()
    requirements = (REPO_ROOT / "spcs/requirements.txt").read_text().splitlines()
    assert all("==" in line for line in requirements if line and not line.startswith("#"))


def test_build_separate_from_approved_push_and_no_latest():
    build = (REPO_ROOT / "spcs/build.sh").read_text()
    push = (REPO_ROOT / "spcs/push.sh").read_text()
    assert "docker push" not in build and "docker build" not in push
    assert "--platform linux/amd64" in build
    assert "--approve-image-push" in push
    assert "BRATS_REMOTE_TAG_ABSENCE_VERIFIED" in push
    assert ":latest" not in CONTAINER["image"]
    assert CONTAINER["image"] == (
        "/CARE_ML_RESEARCH_WKSP/BRAIN_TUMOR_SEG/TRAINING_IMAGES/brats-train:spcs-20261026-runtime1"
    )
    assert CONTAINER["env"]["BRATS_WANDB_ENTITY"] == "banerjee-s-corewell"
    assert "WANDB_BASE_URL" not in CONTAINER["env"]


def test_inference_single_process_validation_only():
    spec = yaml.safe_load((REPO_ROOT / "spcs/inference_spec.yaml").read_text())["spec"]
    container = spec["containers"][0]
    command = container["command"]
    assert "torchrun" not in command and "brats.evaluate" in command
    assert command[command.index("--split") + 1] == "val"
    assert container["resources"]["requests"]["nvidia.com/gpu"] == 1
    assert not spec.get("endpoints")


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("script", ["build.sh", "push.sh"])
def test_shell_syntax(script):
    result = subprocess.run(["bash", "-n", str(REPO_ROOT / "spcs" / script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_yaml_configs_parse():
    for path in (REPO_ROOT / "configs").glob("*.yaml"):
        assert isinstance(yaml.safe_load(path.read_text()), dict), path
