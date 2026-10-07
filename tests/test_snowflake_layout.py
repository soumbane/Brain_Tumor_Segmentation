"""Snowflake plumbing that can be checked without a Snowflake account.

Nothing here talks to Snowflake. It pins down the *contracts* between the pieces that used
to disagree: where the ML-Job payload puts configs, where the stage upload puts the cache,
and where the loader and smoke test look for it.
"""

import dataclasses
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from brats.config import REPO_ROOT, DataConfig
from brats.data.transforms import load_records
from brats.snowflake import stage_data, submit_job


def _py(code, cwd, env_extra=None):
    import os

    env = {k: v for k, v in os.environ.items() if k != "BRATS_REPO_ROOT"}
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True)


def test_repo_root_is_found_in_the_source_checkout():
    assert (REPO_ROOT / "configs" / "data.yaml").is_file()
    assert (REPO_ROOT / "src" / "brats").is_dir()


def test_payload_contains_package_configs_and_splits_without_bytecode(tmp_path):
    payload = submit_job.build_payload_dir(tmp_path / "payload")

    assert (payload / "brats" / "config.py").is_file()
    assert (payload / "brats" / "snowflake" / "_train_entry.py").is_file()
    assert (payload / "configs" / "segresnet_base.yaml").is_file()
    assert (payload / "configs" / "data.yaml").is_file()
    assert (payload / "splits" / "split_random_seed42.csv").is_file()
    assert not list(payload.rglob("__pycache__")) and not list(payload.rglob("*.pyc"))


def test_package_inside_the_payload_finds_its_own_configs(tmp_path):
    """The failure this guards: `parents[2]` pointed outside the payload."""
    payload = submit_job.build_payload_dir(tmp_path / "payload")
    code = (
        f"import sys; sys.path.insert(0, {str(payload)!r}); "
        "from brats.config import REPO_ROOT, DataConfig; DataConfig.load(); print(REPO_ROOT)"
    )
    out = _py(code, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert Path(out.stdout.strip()).resolve() == payload.resolve()


def test_train_entry_makes_the_package_importable_when_run_as_a_script(tmp_path):
    payload = submit_job.build_payload_dir(tmp_path / "payload")
    entry = payload / submit_job.TRAIN_ENTRY_RELPATH
    code = (
        f"import runpy; runpy.run_path({str(entry)!r}); "
        "import brats.config; print(brats.config.REPO_ROOT)"
    )
    out = _py(code, cwd=tmp_path)  # cwd and sys.path know nothing about the payload
    assert out.returncode == 0, out.stderr
    assert Path(out.stdout.strip()).resolve() == payload.resolve()


def test_env_override_wins_for_the_repo_root(tmp_path):
    out = _py("from brats.config import REPO_ROOT; print(REPO_ROOT)", cwd=tmp_path,
              env_extra={"BRATS_REPO_ROOT": str(tmp_path), "PYTHONPATH": str(REPO_ROOT / "src")})
    assert Path(out.stdout.strip()) == tmp_path


class _RecordingSession:
    """Stands in for a Snowpark session: remembers every PUT."""

    def __init__(self):
        self.puts = []
        self.file = self

    def put(self, local, dest, overwrite=False, auto_compress=True):
        self.puts.append((Path(local), dest))


def test_uploaded_layout_is_what_the_loader_reads(tmp_path):
    """stage_data.upload -> mount at BRATS_CACHE_ROOT -> load_records, end to end."""
    cache = tmp_path / "cache"
    for cohort, case in (("GLI", "BraTS-GLI-00001-000"), ("PED", "BraTS-PED-00002-000")):
        (cache / cohort).mkdir(parents=True)
        (cache / cohort / f"{case}.npz").write_bytes(b"x")
    cfg = dataclasses.replace(DataConfig.load(), cache_root=cache)
    session = _RecordingSession()

    stage_data.upload(cfg, "@DB.SCH.STAGE", session=session)

    # Rebuild the stage tree the job would see when the stage is mounted at /mnt/data.
    mount = tmp_path / "mnt_data"
    for local, dest in session.puts:
        target = mount / dest.removeprefix("@DB.SCH.STAGE").strip("/")
        target.mkdir(parents=True, exist_ok=True)
        (target / local.name).write_bytes(local.read_bytes())

    assert (mount / "GLI" / "BraTS-GLI-00001-000.npz").is_file()
    assert (mount / "splits" / "split_random_seed42.csv").is_file()
    assert not (mount / "cache").exists(), "files must not be nested under cache/"

    splits = pd.DataFrame(
        [dict(case_id="BraTS-GLI-00001-000", cohort="GLI", patient_id="BraTS-GLI-00001", timepoint=0, split="train"),
         dict(case_id="BraTS-PED-00002-000", cohort="PED", patient_id="BraTS-PED-00002", timepoint=0, split="train")]
    )
    splits_csv = mount / "splits" / "mini.csv"
    splits.to_csv(splits_csv, index=False)
    job_cfg = dataclasses.replace(DataConfig.load(), cache_root=mount)  # BRATS_CACHE_ROOT=/mnt/data
    recs = load_records(job_cfg, "train", splits_csv=splits_csv)
    assert sorted(r["case_id"] for r in recs) == ["BraTS-GLI-00001-000", "BraTS-PED-00002-000"]


def test_smoke_test_looks_where_the_cache_is_mounted():
    src = submit_job.SMOKE_TEST_PATH.read_text()
    assert "/mnt/data/cache" not in src
    assert "BRATS_CACHE_ROOT" in src and "**/*.npz" in src


def test_there_is_exactly_one_copy_of_each_payload_script():
    text = Path(submit_job.__file__).read_text()
    assert "TRAIN_ENTRY_SOURCE" not in text and "def smoke_test_source" not in text
    assert submit_job.SMOKE_TEST_PATH.is_file()


# -- the shape of the calls we make to Snowflake (the Snowflake side itself is not exercised) --


class _FakeJobs:
    """Stands in for ``snowflake.ml.jobs``; records how we call it."""

    def __init__(self):
        self.calls = []

    def _job(self, name, **kw):
        self.calls.append((name, kw))
        return type("Job", (), {"id": "JOB_1"})()

    def submit_directory(self, path, compute_pool, **kw):
        # Snapshot now: the payload is a temp dir.
        kw["_payload_files"] = sorted(str(p.relative_to(path)) for p in Path(path).rglob("*") if p.is_file())
        return self._job("submit_directory", path=path, compute_pool=compute_pool, **kw)

    def submit_file(self, path, compute_pool, **kw):
        return self._job("submit_file", path=path, compute_pool=compute_pool, **kw)


@pytest.fixture
def fake_jobs(monkeypatch):
    import types

    fake = _FakeJobs()
    mod = types.ModuleType("snowflake.ml.jobs")
    mod.submit_directory, mod.submit_file = fake.submit_directory, fake.submit_file
    for name in ("snowflake", "snowflake.ml"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "snowflake.ml.jobs", mod)
    return fake


def test_training_submission_ships_the_payload_and_forwards_the_cli_flags(fake_jobs):
    submit_job.submit_training(epochs=7, lambda_cls=0.0, run_name="abl", resume=True, session=object())

    (name, kw), = fake_jobs.calls
    assert name == "submit_directory" and kw["entrypoint"] == submit_job.TRAIN_ENTRY_RELPATH
    files = kw["_payload_files"]
    assert "configs/segresnet_base.yaml" in files and "splits/split_random_seed42.csv" in files
    assert "brats/snowflake/_train_entry.py" in files and "brats/train.py" in files
    args = kw["args"]
    assert args[args.index("--epochs") + 1] == "7" and args[args.index("--lambda-cls") + 1] == "0.0"
    assert "--resume" in args and args[args.index("--run-name") + 1] == "abl"
    assert args[args.index("--stage-uri") + 1] == submit_job.CKPT_STAGE
    assert kw["external_access_integrations"] == ["PYPI_EAI"] and kw["spec_overrides"] is None


def test_online_wandb_adds_the_integration_and_injects_the_key_as_a_secret(fake_jobs):
    submit_job.submit_training(wandb_mode="online", wandb_api_key_secret="DB.SCH.KEY", session=object())

    (_, kw), = fake_jobs.calls
    assert "WANDB_EAI" in kw["external_access_integrations"]
    secret = kw["spec_overrides"]["spec"]["containers"][0]["secrets"][0]
    assert secret["snowflakeSecret"] == "DB.SCH.KEY" and secret["envVarName"] == "WANDB_API_KEY"
    assert "DB.SCH.KEY" not in " ".join(kw["args"])  # the key never goes through job arguments


def test_every_flag_forwarded_to_the_entry_point_is_accepted_by_it(fake_jobs):
    import ast

    submit_job.submit_training(epochs=1, lambda_cls=0.1, run_name="x", resume=True, session=object())
    (_, kw), = fake_jobs.calls
    entry = Path(submit_job.__file__).with_name("_train_entry.py")
    declared = {
        n.args[0].value for n in ast.walk(ast.parse(entry.read_text()))
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument" and n.args
    }
    sent = {a for a in kw["args"] if a.startswith("--")}
    # argparse's parse_args() exits on any flag it does not know: this was a crash at launch.
    assert "--wandb-mode" in sent and sent <= declared, f"entry point does not accept {sent - declared}"


def test_smoke_test_submission_uploads_the_real_script(fake_jobs):
    submit_job.submit_smoke_test(session=object())
    (name, kw), = fake_jobs.calls
    assert name == "submit_file" and Path(kw["path"]) == submit_job.SMOKE_TEST_PATH
