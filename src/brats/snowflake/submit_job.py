"""Submit training / inference to the GPU_NV_M compute pool as a Snowflake ML Job.

Design notes that matter:

* ``GPU_NV_M`` is **one node with 4 A10G GPUs**, so this is single-node multi-GPU:
  ``num_nodes=1, num_workers_per_node=4``, one GPU per worker, NCCL.
* **Do not use ``ShardedDataConnector``.** That API shards a Snowflake *table*; our
  data is ``.npz`` files on a stage. Sharding is handled by ``DistributedSampler``
  inside :mod:`brats.train`.
* ``pip_requirements`` needs an external access integration for PyPI. Without
  ``PYPI_EAI`` (see ``scripts/admin_grants.sql``) the payload cannot install MONAI.
* Checkpoints go to a stage every epoch. SPCS maintenance windows can cancel a
  running job service and Snowflake will **not** restart it, so ``--resume`` against
  the staged checkpoint is the recovery path, not a convenience.

The smoke test is the Phase 2 exit criterion: confirm 4 visible GPUs, working bf16,
and that one cached case loads with the expected shape and label histogram.

Usage::

    .venv\\Scripts\\python.exe -m brats.snowflake.submit_job --smoke-test
    .venv\\Scripts\\python.exe -m brats.snowflake.submit_job --train --epochs 50
    .venv\\Scripts\\python.exe -m brats.snowflake.submit_job --status <job_id>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from brats.config import REPO_ROOT

log = logging.getLogger("brats.snowflake.submit_job")

COMPUTE_POOL = "BRATS_GPU_NV_M"
PAYLOAD_STAGE = "PAYLOAD"
CKPT_STAGE = "@BRATS_MRI.CORE.CKPT"
EAI = ["PYPI_EAI"]
#: Separate integration for W&B egress. Only attached when wandb_mode="online";
#: offline runs need no network at all.
WANDB_EAI = "WANDB_EAI"

#: GPU_NV_M: 4x A10G. One worker per GPU.
GPUS_PER_NODE = 4

#: Torch must be installed from its own index and BEFORE anything else, because the
#: default PyPI wheel is CPU-only and the CUDA suffix changes between releases.
#: Verify the current suffix at pytorch.org/get-started/locally.
PIP_REQUIREMENTS = [
    "monai[nibabel,tqdm,einops]>=1.4",
    "nibabel>=5.2",
    "connected-components-3d>=3.12",
    "scikit-learn>=1.4",
    "scipy>=1.11",
    "pandas>=2.1",
    "pyyaml>=6.0",
    "wandb>=0.17",
]


#: The smoke test and the entry point are real files, not strings embedded here. Earlier
#: versions kept a second copy in this module and regenerated the file from it on every
#: submit, so editing the file had no lasting effect.
SMOKE_TEST_PATH = REPO_ROOT / "scripts" / "gpu_smoke_test.py"
TRAIN_ENTRY_RELPATH = "brats/snowflake/_train_entry.py"


def build_payload_dir(dest: Path | None = None) -> Path:
    """Assemble the ML-Job payload: the ``brats`` package plus ``configs/`` and ``splits/``.

    The payload root must contain ``brats/`` (so the entry point can import it) *and*
    ``configs/data.yaml`` (which :data:`brats.config.REPO_ROOT` searches for upward from
    ``brats/config.py``). Uploading ``src/`` alone left both ``configs/`` and ``splits/``
    behind, so the job could not find its own configuration.
    """
    import shutil
    import tempfile

    dest = Path(dest) if dest else Path(tempfile.mkdtemp(prefix="brats_payload_"))
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(REPO_ROOT / "src" / "brats", dest / "brats", ignore=ignore, dirs_exist_ok=True)
    for name in ("configs", "splits"):
        shutil.copytree(REPO_ROOT / name, dest / name, ignore=ignore, dirs_exist_ok=True)
    return dest


def _session():
    from snowflake.snowpark import Session

    return Session.builder.getOrCreate()


def submit_smoke_test(compute_pool: str = COMPUTE_POOL, session=None):
    """Run the smoke test on one GPU of the pool."""
    from snowflake.ml.jobs import submit_file

    if not SMOKE_TEST_PATH.is_file():
        raise SystemExit(f"smoke test not found: {SMOKE_TEST_PATH}")
    session = session or _session()
    job = submit_file(
        str(SMOKE_TEST_PATH),
        compute_pool,
        stage_name=PAYLOAD_STAGE,
        pip_requirements=PIP_REQUIREMENTS,
        external_access_integrations=EAI,
        session=session,
    )
    log.info("submitted smoke test: %s", job.id)
    return job


def submit_training(
    compute_pool: str = COMPUTE_POOL,
    config: str = "configs/segresnet_base.yaml",
    epochs: int | None = None,
    lambda_cls: float | None = None,
    run_name: str | None = None,
    resume: bool = False,
    wandb_mode: str = "offline",
    wandb_api_key_secret: str | None = None,
    session=None,
):
    """Submit distributed training across the node's 4 GPUs.

    ``PyTorchDistributor`` sets ``RANK``/``LOCAL_RANK``/``WORLD_SIZE``, which
    :func:`brats.train.setup_distributed` reads to initialize NCCL. The training
    function is a thin shim so the real logic stays in :mod:`brats.train` and remains
    runnable with plain ``torchrun``.

    Args:
        wandb_mode: ``"offline"`` (default) writes run data to the container's disk
            for a later ``wandb sync``; nothing leaves the account. ``"online"``
            streams live but needs egress to ``api.wandb.ai`` through an external
            access integration **and** an API key -- see ``wandb_api_key_secret``.
        wandb_api_key_secret: Name of a Snowflake SECRET holding the W&B API key.
            Only needed for ``wandb_mode="online"``. The key is injected as an env
            var so it never appears in the payload or in job arguments.
    """
    from snowflake.ml.jobs import submit_directory

    args: list[str] = ["--config", config, "--stage-uri", CKPT_STAGE,
                       "--wandb-mode", wandb_mode]
    if epochs is not None:
        args += ["--epochs", str(epochs)]
    if lambda_cls is not None:
        args += ["--lambda-cls", str(lambda_cls)]
    if run_name:
        args += ["--run-name", run_name]
    if resume:
        args += ["--resume"]

    eai = list(EAI)
    spec_overrides = None
    if wandb_mode == "online":
        eai.append(WANDB_EAI)
        if wandb_api_key_secret:
            spec_overrides = {
                "spec": {
                    "containers": [
                        {
                            "name": "main",
                            "secrets": [
                                {
                                    "snowflakeSecret": wandb_api_key_secret,
                                    "envVarName": "WANDB_API_KEY",
                                    "secretKeyRef": "secret_string",
                                }
                            ],
                        }
                    ]
                }
            }
        else:
            log.warning(
                "wandb_mode=online without wandb_api_key_secret: the run will fail to "
                "authenticate. Pass the secret name, or use offline mode and sync later."
            )

    session = session or _session()
    payload = build_payload_dir()
    log.info("payload assembled at %s", payload)
    job = submit_directory(
        str(payload),
        compute_pool,
        entrypoint=TRAIN_ENTRY_RELPATH,
        stage_name=PAYLOAD_STAGE,
        args=args,
        pip_requirements=PIP_REQUIREMENTS,
        external_access_integrations=eai,
        session=session,
        spec_overrides=spec_overrides,
        env_vars={
            # 4 ranks share 44 vCPU; cap intra-op threads so the DataLoader workers
            # are not fighting the compute threads for cores.
            "OMP_NUM_THREADS": "8",
            "NCCL_DEBUG": "WARN",
            "WANDB_MODE": wandb_mode,
        },
    )
    log.info("submitted training job: %s", job.id)
    log.info("checkpoints -> %s (resume with --resume if SPCS cancels the job)", CKPT_STAGE)
    if wandb_mode == "offline":
        log.info(
            "wandb is OFFLINE: run data stays in the container under runs/wandb/. "
            "To view curves, either use --wandb-mode online (needs %s) or retrieve "
            "the run dir and `wandb sync` it locally.",
            WANDB_EAI,
        )
    return job


def job_status(job_id: str, tail_logs: bool = True, session=None) -> None:
    from snowflake.ml.jobs import get_job

    session = session or _session()
    job = get_job(job_id, session=session)
    print(f"status: {job.status}")
    if tail_logs:
        print("-" * 60)
        print(job.get_logs())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compute-pool", default=COMPUTE_POOL)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--status", default=None, metavar="JOB_ID")
    ap.add_argument("--config", default="configs/segresnet_base.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lambda-cls", type=float, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--wandb-mode", default="offline", choices=["online", "offline", "disabled"]
    )
    ap.add_argument(
        "--wandb-secret",
        default=None,
        help="name of a Snowflake SECRET holding WANDB_API_KEY (online mode only)",
    )
    ap.add_argument(
        "--write-payloads",
        action="store_true",
        help="assemble the training payload directory locally and print its path, "
        "without submitting anything",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    if args.write_payloads:
        print(build_payload_dir())
        return 0

    if args.status:
        job_status(args.status)
        return 0

    if args.smoke_test:
        job = submit_smoke_test(args.compute_pool)
        print(f"job id: {job.id}")
        print(f"poll with: python -m brats.snowflake.submit_job --status {job.id}")
        return 0

    if args.train:
        job = submit_training(
            compute_pool=args.compute_pool,
            config=args.config,
            epochs=args.epochs,
            lambda_cls=args.lambda_cls,
            run_name=args.run_name,
            resume=args.resume,
            wandb_mode=args.wandb_mode,
            wandb_api_key_secret=args.wandb_secret,
        )
        print(f"job id: {job.id}")
        return 0

    ap.print_help()
    return 1


__all__ = [
    "submit_smoke_test",
    "submit_training",
    "job_status",
    "build_payload_dir",
    "SMOKE_TEST_PATH",
    "TRAIN_ENTRY_RELPATH",
    "COMPUTE_POOL",
    "PIP_REQUIREMENTS",
]


if __name__ == "__main__":
    raise SystemExit(main())
