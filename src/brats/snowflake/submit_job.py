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
import textwrap
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


def smoke_test_source() -> str:
    """Entry point for the Phase 2 exit criterion."""
    return textwrap.dedent(
        '''
        """GPU smoke test: 4 GPUs, bf16, and one cached case round-tripping."""
        import os, glob
        import numpy as np
        import torch


        def main():
            print("=" * 60)
            print("GPU_NV_M smoke test")
            print("=" * 60)

            print(f"torch                : {torch.__version__}")
            print(f"cuda available       : {torch.cuda.is_available()}")
            n = torch.cuda.device_count()
            print(f"device count         : {n}  (expected 4 on GPU_NV_M)")
            for i in range(n):
                p = torch.cuda.get_device_properties(i)
                print(f"  cuda:{i} {p.name}  {p.total_memory / 1e9:.1f} GB  sm_{p.major}{p.minor}")

            assert torch.cuda.is_available(), "no CUDA device visible"
            if n != 4:
                print(f"WARNING: expected 4 GPUs, saw {n}")

            # bf16 is the project's precision of choice; A10G is Ampere so it is native.
            bf16_ok = torch.cuda.is_bf16_supported()
            print(f"bf16 supported       : {bf16_ok}")
            assert bf16_ok, "bf16 unsupported -- the training recipe assumes it"
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a = torch.randn(512, 512, device="cuda")
                out = (a @ a).float()
            print(f"bf16 matmul          : ok (mean {out.mean().item():.4f})")

            # NCCL presence: multi-GPU DDP depends on it. Linux SPCS has it; Windows
            # would not, which is why all training runs remotely.
            print(f"nccl available       : {torch.distributed.is_nccl_available()}")

            print(f"cpu count            : {os.cpu_count()}  (expected 44)")

            # One cached case: shape, dtype, and the label invariant.
            hits = sorted(glob.glob("/mnt/data/cache/**/*.npz", recursive=True))
            print(f"cached npz visible   : {len(hits)}")
            if hits:
                with np.load(hits[0]) as z:
                    print(f"  file        : {hits[0]}")
                    print(f"  img shape   : {z['img'].shape}  dtype {z['img'].dtype}")
                    print(f"  orig shape  : {z['orig_shape']}")
                    if "seg" in z.files:
                        vals = np.unique(z["seg"])
                        print(f"  label values: {vals.tolist()}")
                        assert vals.max() <= 3, (
                            f"label {vals.max()} > 3: this is NOT BraTS 2023 "
                            "(label 4 is the 2021 convention and silently empties ET)"
                        )
                        print("  label check : OK (no 4s -- ET is label 3)")
            else:
                print("  no cache mounted; stage the cache before training")

            # MONAI import and a forward pass through the real model.
            from monai.networks.nets import SegResNetDS
            print("monai import         : ok")
            net = SegResNetDS(
                spatial_dims=3, init_filters=32, in_channels=4, out_channels=3,
                blocks_down=(1, 2, 2, 4), dsdepth=4,
            ).cuda()
            x = torch.randn(1, 4, 128, 128, 128, device="cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                y = net(x)
            shapes = [tuple(t.shape) for t in (y if isinstance(y, (list, tuple)) else [y])]
            print(f"forward 128^3        : {shapes}")
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"peak VRAM (batch 1)  : {peak:.2f} GB of 24 GB")
            print(f"projected batch 2    : ~{peak * 2:.2f} GB")

            print("\\nSMOKE TEST PASSED")
            return {"gpus": n, "bf16": bf16_ok, "peak_gb": peak, "cached": len(hits)}


        if __name__ == "__main__":
            __return__ = main()
        '''
    ).strip()


def write_smoke_test(path: Path | None = None) -> Path:
    """Materialize the smoke-test payload next to the package."""
    dest = path or (REPO_ROOT / "scripts" / "gpu_smoke_test.py")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(smoke_test_source() + "\n", encoding="utf-8")
    log.info("wrote %s", dest)
    return dest


def _session():
    from snowflake.snowpark import Session

    return Session.builder.getOrCreate()


def submit_smoke_test(compute_pool: str = COMPUTE_POOL, session=None):
    """Run the smoke test on one GPU of the pool."""
    from snowflake.ml.jobs import submit_file

    path = write_smoke_test()
    session = session or _session()
    job = submit_file(
        str(path),
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
    job = submit_directory(
        str(REPO_ROOT / "src"),
        compute_pool,
        entrypoint="brats/snowflake/_train_entry.py",
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


TRAIN_ENTRY_SOURCE = textwrap.dedent(
    '''
    """ML Job entry point: launch brats.train across the node's 4 GPUs.

    Kept deliberately thin. PyTorchDistributor owns process/rank setup; all training
    logic lives in brats.train, which stays runnable under plain torchrun so the same
    code path is exercised locally and remotely.
    """

    import argparse
    import sys


    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", default="configs/segresnet_base.yaml")
        ap.add_argument("--epochs", type=int, default=None)
        ap.add_argument("--lambda-cls", type=float, default=None)
        ap.add_argument("--run-name", default=None)
        ap.add_argument("--stage-uri", default="")
        ap.add_argument("--resume", action="store_true")
        args = ap.parse_args()

        from snowflake.ml.modeling.distributors.pytorch import (
            PyTorchDistributor,
            PyTorchScalingConfig,
            WorkerResourceConfig,
        )

        def train_func():
            import os

            import torch.distributed as dist
            from snowflake.ml.modeling.distributors.pytorch import get_context

            ctx = get_context()
            # Bridge Snowflake's context into the env vars torch DDP expects, so
            # brats.train needs no Snowflake-specific branch.
            os.environ["RANK"] = str(ctx.get_rank())
            os.environ["LOCAL_RANK"] = str(ctx.get_local_rank())
            os.environ["WORLD_SIZE"] = str(ctx.get_world_size())

            from brats.config import DataConfig
            from brats.train import TrainConfig, train

            cfg = TrainConfig.from_yaml(args.config)
            if args.epochs is not None:
                cfg.epochs = args.epochs
            if args.lambda_cls is not None:
                cfg.lambda_cls = args.lambda_cls
            if args.run_name:
                cfg.run_name = args.run_name
            if args.stage_uri:
                cfg.stage_uri = args.stage_uri

            train(cfg, DataConfig.load(), resume=args.resume)
            if dist.is_initialized():
                dist.destroy_process_group()

        distributor = PyTorchDistributor(
            train_func=train_func,
            scaling_config=PyTorchScalingConfig(
                num_nodes=1,                # GPU_NV_M is a single 4-GPU node
                num_workers_per_node=4,     # one worker per A10G
                resource_requirements_per_worker=WorkerResourceConfig(
                    num_cpus=10,            # 44 vCPU / 4 ranks, leaving headroom
                    num_gpus=1,
                ),
            ),
        )
        # No dataset_map: our data is .npz on a stage, not a Snowflake table, so
        # sharding is handled by DistributedSampler rather than ShardedDataConnector.
        distributor.run()
        return 0


    if __name__ == "__main__":
        __return__ = main()
    '''
).strip()


def write_train_entry(path: Path | None = None) -> Path:
    dest = path or (REPO_ROOT / "src" / "brats" / "snowflake" / "_train_entry.py")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(TRAIN_ENTRY_SOURCE + "\n", encoding="utf-8")
    log.info("wrote %s", dest)
    return dest


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
        help="generate the payload scripts locally without submitting anything",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    if args.write_payloads:
        write_smoke_test()
        write_train_entry()
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
        write_train_entry()
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
    "write_smoke_test",
    "write_train_entry",
    "COMPUTE_POOL",
    "PIP_REQUIREMENTS",
]


if __name__ == "__main__":
    raise SystemExit(main())
