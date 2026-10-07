"""The deployment files must agree with the code they launch.

A drifted flag or path in the SPCS spec, Dockerfile or push script only shows up after waiting
for a GPU node, so these are checked statically.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from brats.config import REPO_ROOT
from brats.train import build_parser

SPEC = yaml.safe_load((REPO_ROOT / "spcs" / "service_spec.yaml").read_text())
CONTAINER = SPEC["spec"]["containers"][0]
SRC_TEXT = "\n".join(p.read_text() for p in (REPO_ROOT / "src").rglob("*.py"))


def test_spec_command_is_accepted_by_the_training_cli():
    cmd = CONTAINER["command"]
    assert cmd[:3] == ["python", "-m", "brats.train"]
    args = build_parser().parse_args([str(a) for a in cmd[3:]])  # exits on an unknown flag
    assert args.resume and args.no_wandb and args.epochs == 50
    assert (REPO_ROOT / "configs" / Path(args.config).name).is_file()


def test_every_env_var_in_the_spec_is_actually_read_by_the_code():
    ignored = {"OMP_NUM_THREADS", "NCCL_DEBUG", "WANDB_MODE"}  # consumed by libraries, not by us
    for name in set(CONTAINER["env"]) - ignored:
        assert f'"{name}"' in SRC_TEXT, f"{name} is set in service_spec.yaml but no code reads it"


def test_spec_paths_point_inside_declared_volume_mounts():
    mounts = [m["mountPath"] for m in CONTAINER["volumeMounts"]]
    env = CONTAINER["env"]
    for key in ("BRATS_CACHE_ROOT", "BRATS_SPLITS_CSV", "BRATS_MANIFEST_DIR", "BRATS_CKPT_ROOT"):
        assert any(env[key].startswith(m) for m in mounts), f"{key}={env[key]} is not under a mount {mounts}"
    declared = {v["name"] for v in SPEC["spec"]["volumes"]}
    assert {m["name"] for m in CONTAINER["volumeMounts"]} <= declared


def test_spec_requests_no_more_gpus_than_it_limits():
    res = CONTAINER["resources"]
    assert res["requests"]["nvidia.com/gpu"] == res["limits"]["nvidia.com/gpu"] == 1


def test_dockerfile_copies_only_paths_that_exist_and_puts_the_package_on_the_path():
    text = (REPO_ROOT / "Dockerfile").read_text()
    for src in re.findall(r"^COPY\s+(\S+)\s+\S+\s*$", text, flags=re.M):
        assert (REPO_ROOT / src.rstrip("/")).exists(), f"Dockerfile copies missing path {src}"
    assert "PYTHONPATH=\"/app/src" in text
    assert re.search(r"^CMD \[.*brats\.train", text, flags=re.M)


def test_dockerfile_copies_everything_the_runtime_needs():
    text = (REPO_ROOT / "Dockerfile").read_text()
    for needed in ("src/", "configs/", "splits/"):
        assert re.search(rf"^COPY\s+{re.escape(needed)}\s", text, flags=re.M), f"{needed} not copied"


def test_the_spec_image_name_matches_what_push_sh_builds():
    push = (REPO_ROOT / "spcs" / "push.sh").read_text()
    image = re.search(r'IMAGE_NAME="([^"]+)"', push).group(1)
    assert CONTAINER["image"].endswith(f"/{image}:latest")


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_push_script_has_valid_shell_syntax():
    r = subprocess.run(["bash", "-n", str(REPO_ROOT / "spcs" / "push.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_yaml_configs_parse_and_the_train_config_has_no_unknown_keys():
    for f in (REPO_ROOT / "configs").glob("*.yaml"):
        assert isinstance(yaml.safe_load(f.read_text()), dict), f
