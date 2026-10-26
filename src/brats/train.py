"""DDP training loop for the multi-task model on GPU_NV_M (4x A10G 24 GB).

Compute shape. ``GPU_NV_M`` is one node: 4x NVIDIA A10G 24 GB, 44 vCPU, 178 GiB RAM,
93.13 GiB local disk. At 128^3 with bf16, SegResNetDS runs batch 2/GPU, giving an
**effective batch of 8** across 4 ranks -- the batch the recipe wants, without
needing a 96 GB card. 44 vCPU / 4 ranks leaves ~11 vCPU per rank, enough to keep
``num_workers=8`` fed; 3D augmentation is CPU-bound and starves the GPUs first.

A10G is Ampere (sm_86), so **bf16 is native**: no ``GradScaler``, and far more
forgiving for Dice-family losses whose small denominators are a known fp16 NaN
source. NCCL over PCIe (no NVLink) is fine for a ~20M-parameter model.

Batch 8 is off-literature. Every published BraTS recipe used batch 1-5 under 16-48 GB
limits, so **the literature's LR does not transfer** -- treat LR as an early sweep
around AdamW 2e-4 rather than a settled value.

**Resumability is not optional.** SPCS maintenance windows can cancel a running job
service and Snowflake will not restart it. Checkpoints are written every epoch to a
Snowflake stage, and ``--resume`` restores model, optimizer, scheduler, scaler, RNG
state and epoch counter. Phase 3's exit criterion is a clean resume after a forced
kill.

Checkpoint selection follows NVAUTO: keep best-average **and** best-ET, best-TC,
best-WT separately, because *"the best average checkpoint may not be the best in all
3 sub-regions."*

Launch (inside an ML Job, one process per GPU)::

    torchrun --nproc_per_node=4 -m brats.train --config configs/segresnet_base.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler

from brats.config import REPO_ROOT, DataConfig, load_yaml
from brats.constants import COHORT_INDEX, COHORTS, REGIONS
from brats.data.transforms import (
    CachedBratsDataset,
    collate_metadata,
    load_records,
    seed_worker,
    train_transforms,
    val_transforms,
)
from brats.losses import build_loss
from brats.metrics.lesionwise import legacy_dice
from brats.models.multitask import MultiTaskConfig, build_model, count_parameters
from brats.tracking import build_logger

log = logging.getLogger("brats.train")
_STOP_REQUESTED = False


def _request_stop(signum, frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def check_stop(device: torch.device, deadline: float | None = None) -> None:
    stop = _STOP_REQUESTED or (deadline is not None and time.monotonic() >= deadline)
    flag = torch.tensor(int(stop), device=device)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item():
        raise InterruptedError("coordinated stop: partial epoch discarded; resume last complete publication")


def host_metrics() -> dict[str, float]:
    metrics = {}
    try:
        import psutil
        process = psutil.Process()
        metrics["rss_bytes_rank0"] = process.memory_info().rss
        metrics["cpu_percent_rank0"] = process.cpu_percent(interval=0.05)
        metrics["host_available_ram_bytes_rank0"] = psutil.virtual_memory().available
    except ImportError:
        pass
    if hasattr(os, "getloadavg"):
        metrics["host_load1_rank0"] = os.getloadavg()[0]
    return metrics


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Training hyperparameters. See ``configs/segresnet_base.yaml``."""

    run_name: str = "segresnet_base"
    patch_size: tuple[int, int, int] = (128, 128, 128)
    batch_size_per_gpu: int = 2
    epochs: int = 150
    warmup_epochs: int = 8

    lr: float = 2e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"  # "adamw" | "sgd"
    momentum: float = 0.99  # SGD/nnU-Net lineage only
    grad_clip: float = 12.0
    #: Abort after this many *consecutive* optimizer steps whose gradient is NaN/Inf
    #: (bf16/fp32 only; fp16 loss scaling produces such steps by design). Each is
    #: skipped and logged, so one bad batch cannot destroy the weights, but a run that
    #: keeps producing them is dead and should stop rather than burn GPU hours.
    max_nonfinite_steps: int = 20

    seg_loss: str = "dice_ce"  # "dice_ce" | "dice_focal"
    lambda_cls: float = 0.1  # set 0.0 for the required ablation
    loss_weighting: str = "fixed"  # "fixed" | "uncertainty"
    batch_dice: bool = True

    pooling: str = "tumor_attention"  # "gap" (ablation) | "tumor_attention" | "tafe"
    cls_warmup_epochs: int = 20

    num_workers: int = 8
    amp_dtype: str = "bfloat16"
    #: Cohort-balanced sampling. Cohorts are imbalanced 12.6:10.1:1, so without this
    #: PED (n=99 against GLI's 1251) is effectively invisible to the classifier.
    cohort_balanced_sampling: bool = True

    val_every: int = 5
    val_max_cases: int = 60  # subset for in-training validation speed
    seed: int = 42

    ckpt_dir: str = "checkpoints"
    #: Snowflake stage to mirror checkpoints to, e.g. "@BRATS_MRI.CORE.CKPT".
    #: Empty disables staging (local runs).
    stage_uri: str = ""

    # -- experiment tracking ------------------------------------------------
    use_mlflow: bool = False  # Databricks native tracking
    mlflow_experiment: str = ""  # MLflow experiment name; empty = default
    wandb: bool = False
    wandb_project: str = "brats2023"
    wandb_entity: str = ""
    wandb_group: str = ""
    #: "offline" is the safe default in an SPCS container: online mode needs egress
    #: to api.wandb.ai via an external access integration. Offline writes to disk for
    #: a later `wandb sync`.
    wandb_mode: str = "offline"
    #: Log gradient histograms. Off by default: real overhead on a 20M-param 3D CNN
    #: for information that rarely changes a decision.
    wandb_watch: bool = False

    execution_profile: str = "legacy"
    run_id: str = ""
    attempt_id: str = ""
    artifact_root: str = ""
    cache_audit_path: str = ""
    expected_split_sha256: str = ""
    expected_world_size: int = 4
    prefetch_factor: int = 1
    val_num_workers: int | None = None
    val_sw_batch_size: int = 4
    checkpoint_retries: int = 3
    require_online_tracking: bool = False
    numeric_only_tracking: bool = False
    wandb_run_id: str = ""
    wandb_resume: bool = False
    max_runtime_seconds: int = 0
    shutdown_grace_seconds: int = 120
    extra: dict[str, Any] = field(default_factory=dict)

    def validate_config(self, *, launch: bool = False) -> None:
        if self.execution_profile not in {"legacy", "spcs"}:
            raise ValueError("execution_profile must be legacy or spcs")
        if self.execution_profile != "spcs":
            return
        positive = ("batch_size_per_gpu", "epochs", "grad_clip", "lr", "val_every",
                    "max_nonfinite_steps", "expected_world_size", "prefetch_factor",
                    "val_sw_batch_size", "checkpoint_retries", "shutdown_grace_seconds")
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("warmup_epochs", "cls_warmup_epochs", "num_workers", "val_max_cases", "seed", "max_runtime_seconds"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("batch_size_per_gpu", "epochs", "val_every", "max_nonfinite_steps", "expected_world_size", "prefetch_factor", "val_sw_batch_size", "checkpoint_retries", "shutdown_grace_seconds"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        for name in ("weight_decay", "lambda_cls", "momentum"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        for name in ("wandb", "require_online_tracking", "numeric_only_tracking", "wandb_watch", "use_mlflow", "batch_dice", "cohort_balanced_sampling"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.val_num_workers is not None and (type(self.val_num_workers) is not int or self.val_num_workers < 0):
            raise ValueError("val_num_workers must be a nonnegative integer")
        if len(self.patch_size) != 3 or any(type(value) is not int or value < 16 or value % 16 for value in self.patch_size):
            raise ValueError("patch_size must contain three positive multiples of 16")
        if self.optimizer not in {"adamw", "sgd"} or self.amp_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("unsupported optimizer or precision")
        if self.seg_loss not in {"dice_ce", "dice_focal"} or self.loss_weighting not in {"fixed", "uncertainty"} or self.pooling not in {"gap", "tumor_attention", "tafe"}:
            raise ValueError("unsupported loss or pooling")
        if self.optimizer == "sgd" and not 0 < self.momentum < 1:
            raise ValueError("SGD nesterov momentum must be between zero and one")
        if self.wandb_project != "brats2023":
            raise ValueError("SPCS requires the approved brats2023 W&B project")
        if self.warmup_epochs > self.epochs or self.stage_uri or self.extra:
            raise ValueError("invalid warmup, legacy stage_uri, or extra settings in SPCS profile")
        if not self.require_online_tracking or not self.numeric_only_tracking or self.use_mlflow or self.wandb_watch or not self.wandb or self.wandb_mode != "online":
            raise ValueError("SPCS requires online numeric-only W&B without watch or MLflow")
        if launch:
            from brats.artifacts import safe_id
            safe_id(self.run_id)
            safe_id(self.attempt_id)
            if not all((self.artifact_root, self.cache_audit_path, self.expected_split_sha256, self.wandb_entity)):
                raise ValueError("SPCS launch requires artifact root, cache audit, frozen split hash and W&B entity")
            if self.max_runtime_seconds <= self.shutdown_grace_seconds:
                raise ValueError("SPCS requires a runtime deadline longer than shutdown grace")
            local = (Path(os.environ.get("BRATS_CKPT_ROOT", str(REPO_ROOT))) / self.ckpt_dir / self.run_id).resolve()
            durable = (Path(self.artifact_root) / self.run_id).resolve()
            if local.is_relative_to(durable) or durable.is_relative_to(local):
                raise ValueError("mutable checkpoint root must be separate from stage artifacts")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        raw = load_yaml(path)
        tr = raw.get("train", raw)
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(tr) - known)
        if unknown and tr.get("execution_profile") == "spcs":
            raise ValueError(f"unknown SPCS config keys: {unknown}")
        if unknown:
            # Dropping these silently turns a typo (`lamda_cls`) into a default value.
            log.warning("ignoring unknown keys in %s: %s", path, unknown)
        kwargs = {k: v for k, v in tr.items() if k in known}
        if "patch_size" in kwargs:
            kwargs["patch_size"] = tuple(kwargs["patch_size"])
        cfg = cls(**kwargs)
        cfg.validate_config()
        return cfg


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def setup_distributed() -> tuple[int, int, int]:
    """Initialize NCCL from torchrun/PyTorchDistributor env vars.

    Returns ``(rank, local_rank, world_size)``. Falls back to single-process when the
    env vars are absent, so the same script runs locally.
    """
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        # Validation and checkpointing run on rank 0 only while the other ranks wait in a
        # barrier, so the collective timeout must outlast the slowest of them. The default
        # (10 min for NCCL) is tight for ~60 full-volume sliding-window passes.
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=60))
    return rank, local_rank, world_size


def is_main(rank: int) -> bool:
    return rank == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / dist.get_world_size())


def set_seed(seed: int, rank: int = 0) -> None:
    # Offset by rank so augmentation differs across ranks while staying reproducible.
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def rank_zero_call(action):
    """Broadcast success/error of rank-zero I/O before peers enter another collective."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    outcome = [None]
    if rank == 0:
        try:
            outcome[0] = {"value": action(), "error": None}
        except Exception as exc:
            outcome[0] = {"value": None, "error": f"{type(exc).__name__}: {exc}"}
    if dist.is_initialized():
        dist.broadcast_object_list(outcome, src=0)
    if outcome[0]["error"]:
        raise RuntimeError(outcome[0]["error"])
    return outcome[0]["value"]


def collective_call(action):
    """Each rank acknowledges its own load/preflight; any error fails every peer."""
    value, error = None, None
    try:
        value = action()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(f"collective operation failed: {errors}")
    return value


def capture_rng_state() -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state().cpu() if torch.cuda.is_available() else None}


def restore_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"].cpu())


def gather_rng_states() -> list[dict]:
    state = capture_rng_state()
    states = [state]
    if dist.is_initialized():
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, state)
    return states


def seed_loader_epoch(loader: DataLoader, seed: int, epoch: int, rank: int, *, validation: bool = False) -> None:
    """Separate sampler/worker/zero-worker MONAI streams at replayable boundaries."""
    derived = int(np.random.SeedSequence([seed, epoch, rank, int(validation)]).generate_state(1)[0])
    if loader.generator is not None:
        loader.generator.manual_seed(derived)
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)
    elif getattr(loader.sampler, "generator", None) is not None:
        loader.sampler.generator.manual_seed(derived)
    if loader.num_workers == 0 and hasattr(loader.dataset, "set_random_state"):
        loader.dataset.set_random_state(seed=derived)


def _cohort_sample_weights(records: list[dict]) -> list[float]:
    """Inverse-frequency weights so each cohort is equally likely per batch."""
    counts: dict[str, int] = {}
    for r in records:
        counts[r["cohort"]] = counts.get(r["cohort"], 0) + 1
    return [1.0 / counts[r["cohort"]] for r in records]


def uses_balanced_sampler(cfg: TrainConfig, world_size: int) -> bool:
    """Whether a cohort-balancing sampler is actually in effect.

    Only single-process runs get one: under DDP the ``DistributedSampler`` owns sharding
    and mixing in a ``WeightedRandomSampler`` would double-count cases. When it is in
    effect the classification loss must NOT also be class-weighted, or the minority
    cohort is corrected twice (a ~12x over-weighting of PED).
    """
    return bool(cfg.cohort_balanced_sampling) and world_size == 1


def build_loaders(
    cfg: TrainConfig, data_cfg: DataConfig, rank: int, world_size: int
) -> tuple[DataLoader, DataLoader, list[dict]]:
    train_recs = load_records(data_cfg, "train")
    val_recs = load_records(data_cfg, "val")
    if cfg.val_max_cases and len(val_recs) > cfg.val_max_cases:
        # Stratify the validation subset so all three cohorts stay represented.
        per = max(1, cfg.val_max_cases // len(COHORTS))
        subset: list[dict] = []
        for cohort in COHORTS:
            subset.extend([r for r in val_recs if r["cohort"] == cohort][:per])
        val_recs = subset

    train_ds = CachedBratsDataset(
        train_recs, transform=train_transforms(cfg.patch_size), load_label=True
    )
    val_ds = CachedBratsDataset(
        val_recs, transform=val_transforms(True), load_label=True
    )

    train_generator = torch.Generator().manual_seed(cfg.seed + rank)
    val_generator = torch.Generator().manual_seed(cfg.seed + rank + 1000000)
    if uses_balanced_sampler(cfg, world_size):
        sampler: Any = WeightedRandomSampler(
            _cohort_sample_weights(train_recs), num_samples=len(train_recs), replacement=True,
            generator=train_generator
        )
        shuffle = False
    else:
        # Under DDP, DistributedSampler owns sharding. Cohort balance is then carried
        # by the class-weighted classification loss rather than by the sampler; mixing
        # a WeightedRandomSampler with DDP sharding would double-count cases.
        sampler = (
            DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed)
            if world_size > 1
            else None
        )
        shuffle = sampler is None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size_per_gpu,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0 and cfg.execution_profile != "spcs",
        collate_fn=collate_metadata,
        worker_init_fn=seed_worker,
        generator=train_generator,
        **({"multiprocessing_context": "spawn", "prefetch_factor": cfg.prefetch_factor}
           if cfg.execution_profile == "spcs" and cfg.num_workers else {}),
    )
    # Validation runs full volumes through sliding-window inference: batch size 1.
    val_workers = cfg.val_num_workers if cfg.val_num_workers is not None else max(2, cfg.num_workers // 2)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        collate_fn=collate_metadata,
        worker_init_fn=seed_worker,
        generator=val_generator,
        **({"multiprocessing_context": "spawn", "prefetch_factor": cfg.prefetch_factor}
           if cfg.execution_profile == "spcs" and val_workers else {}),
    )
    if not len(train_loader):
        raise ValueError("no training batches: dataset must cover batch size times world size")
    return train_loader, val_loader, val_recs


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

#: Checkpoints tracked separately. NVAUTO: "the best average checkpoint may not be
#: the best in all 3 sub-regions."
TRACKED = ("avg", *REGIONS)


class CheckpointManager:
    """Writes per-epoch checkpoints locally and mirrors them to a Snowflake stage."""

    def __init__(self, cfg: TrainConfig, root: Path | None = None) -> None:
        self.cfg = cfg
        env_root = os.environ.get("BRATS_CKPT_ROOT")
        base = Path(env_root) if env_root else (root or REPO_ROOT)
        self.dir = base / cfg.ckpt_dir / (cfg.run_id if cfg.execution_profile == "spcs" else cfg.run_name)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.best: dict[str, float] = {k: -1.0 for k in TRACKED}
        self.best_epochs: dict[str, int] = {}
        self.best_references: dict[str, dict | str] = {}
        self.identity: dict = {}
        self.counters: dict[str, int] = {"attempted": 0, "successful": 0, "skipped": 0}
        self.run_root = Path(cfg.artifact_root) / cfg.run_id if cfg.execution_profile == "spcs" else None
        self.selected_manifest: dict | None = None

    def prepare_run(self, resume: bool, *, dataset_identity: str, world_size: int,
                    steps_per_epoch: int, model_config: dict) -> None:
        from brats.artifacts import canonical_hash, initialize_run, resolve_bundle, sha256_file
        from importlib.metadata import version
        # Runtime destinations/attempt IDs may change at restart, research settings may not.
        excluded = {"artifact_root", "ckpt_dir", "run_name", "run_id", "attempt_id", "cache_audit_path",
                    "wandb_resume", "wandb_run_id", "max_runtime_seconds", "shutdown_grace_seconds"}
        research = {key: value for key, value in asdict(self.cfg).items() if key not in excluded}
        self.identity = {"dataset": dataset_identity, "world_size": world_size,
                         "steps_per_epoch": steps_per_epoch, "model_config": model_config,
                         "config_hash": canonical_hash(research)}
        if self.run_root is not None:
            source_root = Path(__file__).parent
            self.identity["source_hash"] = canonical_hash({path.relative_to(source_root).as_posix(): sha256_file(path)
                                                           for path in sorted(source_root.rglob("*.py"))})
            self.identity["dependencies"] = {name: version(name) for name in ("torch", "monai", "numpy")}
            identity = rank_zero_call(lambda: initialize_run(
                self.run_root, run_id=self.cfg.run_id, attempt_id=self.cfg.attempt_id,
                resume=resume, wandb_run_id=self.cfg.wandb_run_id))
            self.cfg.wandb_run_id = identity["wandb_run_id"]
            self.cfg.wandb_resume = resume
            if resume:
                self.selected_manifest = rank_zero_call(lambda: resolve_bundle(
                    self.run_root, retries=self.cfg.checkpoint_retries))
                if self.selected_manifest["metadata"].get("run_id") != self.cfg.run_id:
                    raise ValueError("checkpoint manifest run identity mismatch")

    def load_collective(self, model, optimizer, scheduler, loss_module, scaler, rank: int) -> int:
        from brats.artifacts import materialize_bundle, sha256_file
        if self.run_root is not None:
            selected = self.selected_manifest
            if selected is None:
                raise ValueError("resume checkpoint was not selected")
            def load_exact():
                files = materialize_bundle(self.run_root, selected, self.dir / f"restore-rank-{rank}")
                epoch = self.load(model, optimizer, scheduler, loss_module, path=files["checkpoint.pt"], scaler=scaler, rank=rank)
                if rank == 0:
                    # Replace any unpublished local tail with the selected bundle's history.
                    temporary = self.dir / "metrics.jsonl.restore.tmp"
                    shutil.copyfile(files["metrics.jsonl"], temporary)
                    temporary.replace(self.dir / "metrics.jsonl")
                return epoch
            return collective_call(load_exact)
        selected = rank_zero_call(lambda: {"path": str(self.dir / "last.pt"), "sha256": sha256_file(self.dir / "last.pt")})
        def load_local():
            if sha256_file(selected["path"]) != selected["sha256"]:
                raise ValueError("ranks do not see the same checkpoint bytes")
            return self.load(model, optimizer, scheduler, loss_module, path=Path(selected["path"]), scaler=scaler, rank=rank)
        return collective_call(load_local)

    # -- staging ---------------------------------------------------------

    def _stage(self, path: Path) -> None:
        """Mirror one checkpoint to the configured Snowflake stage.

        Non-fatal on failure: losing a stage upload must not kill a training run,
        but it is logged loudly because it is the resumability guarantee.
        """
        if not self.cfg.stage_uri:
            return
        try:
            from snowflake.snowpark import Session

            session = Session.builder.getOrCreate()
            session.file.put(
                str(path), self.cfg.stage_uri, overwrite=True, auto_compress=False
            )
            log.info("staged %s -> %s", path.name, self.cfg.stage_uri)
        except Exception as exc:  # noqa: BLE001
            log.error("FAILED to stage %s: %s -- run is not resumable from the stage",
                      path.name, exc)

    def tracking_files(self, tracker: Any) -> dict[str, Path]:
        """Snapshot only explicit closed local numeric logs, never SDK directories."""
        files = {}
        if tracker is not None:
            for attribute in ("health_path", "metrics_path"):
                source = getattr(tracker, attribute, None)
                if source is not None and Path(source).is_file():
                    destination = self.dir / "tracking-snapshot" / Path(source).name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
                    files[f"tracking/{destination.name}"] = destination
        return files

    def publish_tracking(self, tracker: Any, epoch: int) -> None:
        if self.run_root is None:
            return
        files = self.tracking_files(tracker)
        if files:
            from brats.artifacts import publish_bundle
            publish_bundle(self.run_root, files, {"kind": "tracking-final", "run_id": self.cfg.run_id,
                                                "epoch": epoch, "attempt_id": self.cfg.attempt_id},
                           attempt_id=self.cfg.attempt_id, retries=self.cfg.checkpoint_retries)

    # -- save / load -----------------------------------------------------

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        loss_module: nn.Module,
        metrics: dict[str, float],
        tracker: Any = None,
        *, scaler: Any = None, rng_states: list[dict] | None = None,
    ) -> None:
        net = model.module if isinstance(model, DDP) else model
        improved = []
        for key in TRACKED:
            value = metrics.get("dice_avg" if key == "avg" else f"dice_{key}")
            if value is not None and np.isfinite(value) and value > self.best[key]:
                self.best[key] = float(value)
                self.best_epochs[key] = epoch
                improved.append(key)
        references = dict(self.best_references)
        for key in improved:
            references[key] = "self"
        state = {
            "epoch": epoch,
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "loss_module": loss_module.state_dict(),
            "metrics": metrics,
            "config": asdict(self.cfg),
            # The trainer picks `norm` at run time from the effective batch size, so
            # the architecture must travel with the weights: see
            # brats.models.multitask.load_model_from_checkpoint.
            "model_config": asdict(net.cfg),
            "best": dict(self.best),
            "best_epochs": dict(self.best_epochs),
            "best_references": references,
            "identity": self.identity,
            "counters": dict(self.counters),
            "run_id": self.cfg.run_id,
            "attempt_id": self.cfg.attempt_id,
            "wandb_run_id": self.cfg.wandb_run_id,
            "tracking_state": tracker.state_dict() if tracker is not None and hasattr(tracker, "state_dict") else {},
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng_by_rank": rng_states if rng_states is not None else [capture_rng_state()],
            "rng": capture_rng_state(),
        }

        # Atomic write: a checkpoint truncated by a mid-write kill is worse than none.
        last = self.dir / "last.pt"
        tmp = self.dir / "last.pt.tmp"
        minimum_free = max(last.stat().st_size * 3 if last.exists() else 0,
                           sum(parameter.numel() * parameter.element_size() for parameter in net.parameters()) * 16)
        if shutil.disk_usage(self.dir).free < minimum_free:
            raise OSError("insufficient local disk for checkpoint and retained best references")
        with tmp.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(last)
        self._stage(last)

        for key in improved:
            dest = self.dir / f"best_{key}.pt"
            shutil.copyfile(last, dest)
            self._stage(dest)
            if tracker is not None:
                tracker.summary(f"best_dice_{key}", self.best[key])
                tracker.summary(f"best_dice_{key}_epoch", epoch)

        with (self.dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, **metrics}) + "\n")
        if self.run_root is not None:
            from brats.artifacts import publish_bundle, sha256_file
            files = {"checkpoint.pt": last, "metrics.jsonl": self.dir / "metrics.jsonl"}
            files.update(self.tracking_files(tracker))
            completion = publish_bundle(self.run_root, files,
                           {"kind": "checkpoint", "epoch": epoch, "run_id": self.cfg.run_id,
                            "best_epochs": self.best_epochs, "best_references": references, "identity": self.identity},
                           attempt_id=self.cfg.attempt_id, retries=self.cfg.checkpoint_retries)
            self.best_references = {key: ({"manifest_path": completion.relative_to(self.run_root.resolve()).as_posix(),
                                         "manifest_sha256": sha256_file(completion)} if value == "self" else value)
                                    for key, value in references.items()}

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        loss_module: nn.Module | None = None,
        path: Path | None = None,
        map_location: str = "cpu",
        *, scaler: Any = None, rank: int = 0,
    ) -> int:
        """Restore a checkpoint. Returns the epoch to resume *from*."""
        ckpt_path = path or (self.dir / "last.pt")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"explicit resume checkpoint is missing: {ckpt_path}")
        # Trusted project artifacts only: hashes are integrity checks, not authentication.
        state = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        if self.run_root is not None:
            if state.get("identity") != self.identity or state.get("run_id") != self.cfg.run_id:
                raise ValueError("incompatible checkpoint: architecture/schedule/world size/data/config changed")
            if state.get("wandb_run_id") != self.cfg.wandb_run_id:
                raise ValueError("checkpoint W&B identity mismatch")
            if len(state.get("rng_by_rank", [])) != self.identity["world_size"]:
                raise ValueError("checkpoint missing per-rank RNG state")
        net = model.module if isinstance(model, DDP) else model
        net.load_state_dict(state["model"])
        if optimizer is not None and state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        if loss_module is not None and state.get("loss_module"):
            # Only parameters, e.g. uncertainty log-variances. Checkpoints written before
            # `class_weights` became non-persistent carry that key; drop what this module
            # does not own rather than failing (or, worse, importing stale weights).
            own = loss_module.state_dict()
            loss_module.load_state_dict({k: v for k, v in state["loss_module"].items() if k in own})
        self.best = state.get("best", self.best)
        self.best_epochs = state.get("best_epochs", {})
        self.best_references = state.get("best_references", {})
        if self.selected_manifest is not None:
            from brats.artifacts import safe_path, read_manifest
            selected_ref = {"manifest_path": Path(self.selected_manifest["manifest_path"]).relative_to(self.run_root.resolve()).as_posix(),
                            "manifest_sha256": self.selected_manifest["manifest_sha256"]}
            self.best_references = {key: selected_ref if value == "self" else value for key, value in self.best_references.items()}
            # Retained best references are exact immutable selections, not epoch guesses.
            for reference in self.best_references.values():
                manifest = read_manifest(self.run_root, safe_path(self.run_root, reference["manifest_path"]))
                if manifest["manifest_sha256"] != reference["manifest_sha256"]:
                    raise ValueError("retained best checkpoint reference changed")
        self.counters = state.get("counters", self.counters)
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        states = state.get("rng_by_rank")
        rng = states[rank] if states and rank < len(states) else state.get("rng")
        if rng:
            restore_rng_state(rng)
        epoch = int(state["epoch"]) + 1
        log.info("resumed from %s at epoch %d", ckpt_path, epoch)
        return epoch


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
) -> dict[str, float]:
    """Cheap in-training validation: legacy overlap Dice plus cohort accuracy.

    Deliberately *not* the lesion-wise metric. Lesion-wise scoring needs connected
    components and distance transforms per case, which is far too slow to run every
    few epochs, and it is dominated by post-processing that is not yet tuned. Use
    legacy Dice to steer training; report lesion-wise only in real evaluation.
    """
    from brats.inference import InferenceConfig, predict_volume

    net = model.module if isinstance(model, DDP) else model
    net.eval()

    inf_cfg = InferenceConfig(
        roi_size=cfg.patch_size,
        sw_batch_size=cfg.val_sw_batch_size,
        overlap=0.5,
        tta_flip_axes=(),  # no TTA during training validation
        amp_dtype={"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.amp_dtype],
    )

    per_region: dict[str, list[float]] = {r: [] for r in REGIONS}
    per_cohort = {cohort: {region: [] for region in REGIONS} for cohort in COHORTS}
    cls_correct = cls_total = 0

    for batch in loader:
        image = batch["image"][0]
        label = batch["label"][0].numpy()
        out = predict_volume(net, image, cfg=inf_cfg, device=device)
        probs = out["probs"]
        true_cohort = int(batch["cohort"][0])
        for i, region in enumerate(REGIONS):
            score = legacy_dice(label[i] > 0.5, probs[i] >= 0.5)
            per_region[region].append(score)
            per_cohort[COHORTS[true_cohort]][region].append(score)

        pred_cohort = int(np.argmax(out["cls_probs"]))
        cls_correct += int(pred_cohort == true_cohort)
        cls_total += 1

    metrics = {
        f"dice_{r}": float(np.mean(v)) if v else 0.0 for r, v in per_region.items()
    }
    metrics["dice_avg"] = float(np.mean(list(metrics.values())))
    metrics["cls_acc"] = cls_correct / max(cls_total, 1)
    for cohort, regions in per_cohort.items():
        for region, scores in regions.items():
            if scores:
                metrics[f"legacy_dice_{cohort}_{region}"] = float(np.mean(scores))
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def sync_grads(params) -> None:
    """Average gradients of parameters that DDP does not manage, across ranks.

    ``DDP`` only synchronises the wrapped model. Parameters owned by the loss module
    (the uncertainty-weighting log-variances) get a gradient from each rank's *local*
    batch, so without this the ranks drift apart -- and, because they all enter the
    clipping norm, so does the clip coefficient applied to the (synchronised) model.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    world = dist.get_world_size()
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)  # every rank must join the collective
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world)


def optimizer_step(
    loss: Tensor,
    params: list[nn.Parameter],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    grad_clip: float,
    unmanaged: list[nn.Parameter] | tuple = (),
) -> tuple[float, bool]:
    """Backward, clip, and step -- refusing to apply a non-finite gradient.

    ``clip_grad_norm_`` multiplies every gradient by ``clip / (norm + eps)``; if the norm
    is NaN so is every gradient, and one optimizer step turns every weight into NaN.
    This was the failure mode of the bf16 attention-pooling bug: one bad batch silently
    destroyed the model. A non-finite norm now skips the update instead.

    Under DDP the norm is computed on already-synchronised gradients, so every rank
    reaches the same verdict and skips together.

    Returns:
        ``(grad_norm, stepped)``. With an fp16 ``GradScaler`` the scaler owns skipping
        (non-finite steps are expected while it calibrates) and ``stepped`` only reports
        whether the norm was finite.
    """
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        sync_grads(unmanaged)
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
        scaler.step(optimizer)
        scaler.update()
        return float(norm), bool(torch.isfinite(norm))

    loss.backward()
    sync_grads(unmanaged)
    norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
    if not bool(torch.isfinite(norm)):
        return float(norm), False
    optimizer.step()
    return float(norm), True


def build_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    """Linear warmup then cosine annealing to zero, stepped per iteration."""
    from torch.optim.lr_scheduler import LambdaLR

    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = max(1, cfg.warmup_epochs * steps_per_epoch)

    def fn(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    return LambdaLR(optimizer, fn)


def train(cfg: TrainConfig, data_cfg: DataConfig, resume: bool = False) -> None:
    cfg.validate_config(launch=True)
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    handlers = {}
    if cfg.execution_profile == "spcs":
        handlers = {sig: signal.signal(sig, _request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    cleanup: dict[str, Any] = {}
    try:
        _train(cfg, data_cfg, resume, cleanup=cleanup)
    finally:
        # No collectives here: a peer may already have failed. Torch elastic and
        # the parent supervisor bound stalled ranks; close only this rank's logger.
        tracker = cleanup.get("tracker")
        if tracker is not None and is_main(int(os.environ.get("RANK", "0"))):
            tracker.finish()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if dist.is_initialized():
            dist.destroy_process_group()


def _train(cfg: TrainConfig, data_cfg: DataConfig, resume: bool = False, *, cleanup: dict | None = None) -> None:
    rank, local_rank, world_size = setup_distributed()
    deadline = time.monotonic() + cfg.max_runtime_seconds - cfg.shutdown_grace_seconds if cfg.execution_profile == "spcs" else None
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed, rank)
    dataset_identity = "legacy-unverified"
    if cfg.execution_profile == "spcs":
        from brats.artifacts import safe_id, safe_path
        safe_id(cfg.run_id)
        safe_id(cfg.attempt_id)
        self_root = safe_path(cfg.artifact_root, cfg.run_id)
        if data_cfg.cache_root.resolve().is_relative_to(self_root) or self_root.is_relative_to(data_cfg.cache_root.resolve()):
            raise ValueError("artifact publication must not overlap the read-only dataset cache")
        def check_devices():
            if world_size != cfg.expected_world_size or device.type != "cuda" or torch.cuda.device_count() != cfg.expected_world_size:
                raise ValueError("SPCS GPU/rank count differs from the approved recipe")
            if cfg.amp_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise ValueError("bf16 is not supported; refusing to change precision")
        collective_call(check_devices)
        from brats.data.cache_check import verify_audit_identity
        dataset_identity = rank_zero_call(lambda: verify_audit_identity(
            data_cfg, cfg.cache_audit_path, expected_split_sha256=cfg.expected_split_sha256))

    if is_main(rank):
        log.info("world_size=%d device=%s", world_size, device)
        log.info(
            "effective batch = %d (%d per GPU x %d ranks)",
            cfg.batch_size_per_gpu * world_size, cfg.batch_size_per_gpu, world_size,
        )
        if cfg.lambda_cls == 0.0:
            log.info("lambda_cls=0: this is the ABLATION run (segmentation only)")

    # -- model ----------------------------------------------------------
    model_cfg = MultiTaskConfig(
        pooling=cfg.pooling,  # type: ignore[arg-type]
        warmup_epochs=cfg.cls_warmup_epochs,
        norm="batch" if cfg.batch_size_per_gpu * world_size >= 4 else "instance",
    )
    model = build_model(model_cfg).to(device)
    if is_main(rank):
        total, trainable = count_parameters(model)
        log.info("parameters: %.1fM total, %.1fM trainable", total / 1e6, trainable / 1e6)

    if world_size > 1:
        # SyncBatchNorm: with batch 2/GPU, per-rank BN statistics are too noisy. It is
        # CUDA-only, so CPU (gloo) debugging runs keep ordinary per-rank BatchNorm.
        if device.type == "cuda":
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        elif is_main(rank):
            log.warning("SyncBatchNorm needs CUDA; using per-rank BatchNorm on %s", device.type)
        # When lambda_cls == 0 the classification head still runs in forward (so its
        # loss stays reportable and the ablation stays comparable) but receives no
        # gradient. DDP rejects unused parameters by default and fails on the SECOND
        # iteration with "Expected to have finished reduction in the prior iteration".
        # Verified: 7 params (classifier + pooling temperature) go grad-less.
        needs_unused = cfg.lambda_cls == 0.0 and cfg.loss_weighting != "uncertainty"
        if needs_unused and is_main(rank):
            log.info(
                "lambda_cls=0: enabling find_unused_parameters (classification head "
                "runs for reporting but takes no gradient)"
            )
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=needs_unused,
        )

    balanced_sampler = uses_balanced_sampler(cfg, world_size)
    if is_main(rank):
        if cfg.cohort_balanced_sampling and not balanced_sampler:
            log.warning(
                "cohort_balanced_sampling is ignored under DDP (world_size=%d); cohort "
                "balance is carried by the class-weighted classification loss instead",
                world_size,
            )
        log.info(
            "cohort balance via %s",
            "weighted sampler (CE unweighted)" if balanced_sampler else "class-weighted CE",
        )
    loss_module = build_loss(
        seg_kind=cfg.seg_loss,  # type: ignore[arg-type]
        lambda_cls=cfg.lambda_cls,
        weighting=cfg.loss_weighting,  # type: ignore[arg-type]
        batch_dice=cfg.batch_dice,
        # Sampler already balances the batches: weighting the CE as well would correct
        # the minority cohort twice.
        class_weights=torch.ones(len(COHORTS)) if balanced_sampler else None,
    ).to(device)

    params = list(model.parameters()) + list(loss_module.parameters())
    if cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            params, lr=cfg.lr, momentum=cfg.momentum,
            weight_decay=cfg.weight_decay, nesterov=True,
        )

    train_loader, val_loader, _ = build_loaders(cfg, data_cfg, rank, world_size)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.amp_dtype]
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16 and device.type == "cuda"))
    ckpt = CheckpointManager(cfg)
    ckpt.prepare_run(resume, dataset_identity=dataset_identity, world_size=world_size,
                     steps_per_epoch=len(train_loader), model_config=asdict(model_cfg))
    start_epoch = ckpt.load_collective(model, optimizer, scheduler, loss_module, scaler, rank) if resume else 0

    resume_rng = capture_rng_state() if resume else None
    # Logger preflight failure is acknowledged by every rank before any training.
    tracker = collective_call(lambda: build_logger(cfg, is_rank_zero=is_main(rank)))
    if cleanup is not None:
        cleanup["tracker"] = tracker
    if tracker.enabled:
        total, trainable = count_parameters(model.module if isinstance(model, DDP) else model)
        tracker.log_config_update(
            {
                "world_size": world_size,
                "effective_batch": cfg.batch_size_per_gpu * world_size,
                "params_total_m": round(total / 1e6, 2),
                "steps_per_epoch": len(train_loader),
                "train_cases": len(train_loader.dataset),
                "val_cases": len(val_loader.dataset),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            }
        )
        if cfg.wandb_watch:
            tracker.watch(model)

    # The tracker must not advance model RNG on a resumed boundary.
    if resume_rng is not None:
        restore_rng_state(resume_rng)

    # Loss-module parameters (uncertainty log-variances) sit outside DDP; see sync_grads.
    unmanaged = list(loss_module.parameters())

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        net = model.module if isinstance(model, DDP) else model
        alpha = net.set_warmup_alpha(epoch)
        seed_loader_epoch(train_loader, cfg.seed, epoch, rank)
        seed_loader_epoch(val_loader, cfg.seed, epoch, rank, validation=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        t0 = time.time()
        running = {"total": 0.0, "seg": 0.0, "cls": 0.0}
        n_steps = 0
        n_ok = 0           # steps whose update was applied (and enter the epoch mean)
        n_skipped = 0      # steps skipped for a non-finite gradient
        consecutive_bad = 0
        global_step = ckpt.counters["attempted"]
        data_seconds = step_seconds = 0.0
        last_batch_end = time.perf_counter()
        grad_norm = 0.0

        for batch in train_loader:
            data_seconds += time.perf_counter() - last_batch_end
            step_started = time.perf_counter()
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            cohort = batch["cohort"].to(device, non_blocking=True)
            # WT channel as the GT attention mask for the pooling warm-start.
            gt_wt = label[:, REGIONS.index("WT")].unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
            ):
                out = model(image, gt_wt=gt_wt)
                losses = loss_module(out["seg"], label, out["cls"], cohort)

            loss = losses["total"]
            grad_norm, stepped = optimizer_step(
                loss, params, optimizer, scaler, cfg.grad_clip, unmanaged=unmanaged,
            )
            # Keep the legacy attempted-step LR schedule; report skipped updates explicitly.
            scheduler.step()
            n_steps += 1
            ckpt.counters["attempted"] += 1
            ckpt.counters["successful"] += int(stepped)
            ckpt.counters["skipped"] += int(not stepped)
            n_skipped += int(not stepped)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_seconds += time.perf_counter() - step_started
            last_batch_end = time.perf_counter()
            if cfg.execution_profile == "spcs":
                check_stop(device, deadline)

            if stepped:
                consecutive_bad = 0
                for k in running:
                    running[k] += float(losses[k].detach())
                n_ok += 1
            elif not scaler.is_enabled():
                consecutive_bad += 1
                if is_main(rank):
                    log.warning(
                        "non-finite gradient at epoch %d step %d -- update SKIPPED "
                        "(loss %.4g seg %.4g cls %.4g, cases %s)",
                        epoch + 1, n_steps, float(losses["total"].detach()),
                        float(losses["seg"]), float(losses["cls"]),
                        batch.get("case_id"),
                    )
                if consecutive_bad >= cfg.max_nonfinite_steps:
                    # Same verdict on every rank (the norm is synchronised), so all
                    # raise together. last.pt from the previous epoch stays intact.
                    raise RuntimeError(
                        f"{consecutive_bad} consecutive non-finite gradient steps at "
                        f"epoch {epoch + 1}; aborting. Resume from the last checkpoint "
                        "after fixing the cause."
                    )

            # Per-step logging on rank 0. Every 20 steps keeps the curve readable
            # without flooding the run; LR matters because batch 8 is off-literature
            # and the schedule is being swept.
            if is_main(rank) and tracker.enabled and n_steps % 20 == 0:
                prefix = "train_rank0" if cfg.execution_profile == "spcs" else "train"
                tracker.log(
                    {
                        f"{prefix}/loss": float(losses["total"].detach()),
                        f"{prefix}/loss_seg": float(losses["seg"]),
                        f"{prefix}/loss_cls": float(losses["cls"]),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/warmup_alpha": alpha,
                        "epoch": epoch + 1,
                    },
                    step=global_step + n_steps,
                )

        for k in running:
            running[k] = all_reduce_mean(running[k] / max(n_ok, 1), device)

        if is_main(rank):
            log.info(
                "epoch %3d/%d  loss %.4f (seg %.4f cls %.4f)  lr %.2e  alpha %.2f  %.1fs",
                epoch + 1, cfg.epochs, running["total"], running["seg"], running["cls"],
                scheduler.get_last_lr()[0], alpha, time.time() - t0,
            )
            if n_skipped:
                log.warning(
                    "  %d/%d steps this epoch were skipped for a non-finite gradient",
                    n_skipped, n_steps,
                )

        # All ranks enter telemetry collectives, then rank zero validates unwrapped.
        do_val = ((epoch + 1) % cfg.val_every == 0) or (epoch + 1 == cfg.epochs)
        metrics = dict(running)
        metrics.update({"attempted_updates": n_steps, "successful_updates": n_ok,
                        "skipped_steps": n_skipped, "effective_batch": cfg.batch_size_per_gpu * world_size,
                        "data_seconds_mean_rank": all_reduce_mean(data_seconds, device),
                        "step_seconds_mean_rank": all_reduce_mean(step_seconds, device),
                        "train_seconds_mean_rank": all_reduce_mean(time.time() - t0, device)})
        peaks = [float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0]
        if dist.is_initialized():
            gathered = [None] * world_size
            dist.all_gather_object(gathered, peaks[0])
            peaks = gathered
        for index, peak in enumerate(peaks):
            metrics[f"peak_vram_bytes_rank_{index}"] = peak
        metrics["samples_per_second"] = n_steps * cfg.batch_size_per_gpu * world_size / max(metrics["train_seconds_mean_rank"], 1e-9)
        val_started = time.perf_counter()
        if do_val:
            metrics.update(rank_zero_call(lambda: validate(model, val_loader, device, cfg)))
        metrics["validation_seconds"] = time.perf_counter() - val_started if do_val else 0.0
        metrics["epoch_seconds"] = time.time() - t0
        metrics["disk_free_bytes_rank0"] = rank_zero_call(lambda: shutil.disk_usage(ckpt.dir).free)
        metrics.update(rank_zero_call(host_metrics))
        last_norm = rank_zero_call(lambda: grad_norm if np.isfinite(grad_norm) else None)
        if last_norm is not None:
            metrics["grad_norm_rank0_last"] = last_norm
        def log_epoch():
            payload = {"epoch": epoch + 1, "epoch/loss": running["total"],
                       "epoch/loss_seg": running["seg"], "epoch/loss_cls": running["cls"],
                       "epoch/lr": scheduler.get_last_lr()[0], "epoch/warmup_alpha": alpha,
                       "epoch/seconds": metrics["epoch_seconds"]}
            for key, value in metrics.items():
                if key.startswith("dice_") or key.startswith("legacy_dice_") or key == "cls_acc":
                    payload[f"val/{key}"] = value
                elif key not in running:
                    payload[f"runtime/{key}"] = value
            if loss_module.uncertainty is not None:
                weights = loss_module.uncertainty.weights()
                payload["loss_weight/seg"], payload["loss_weight/cls"] = weights
            tracker.log(payload, step=ckpt.counters["attempted"])
        rank_zero_call(log_epoch)
        # Validation/logging may consume random numbers. Capture after both, and
        # restore after publication so incidental serialization/tracking cannot drift it.
        rng_states = gather_rng_states()
        checkpoint_started = time.perf_counter()
        rank_zero_call(lambda: ckpt.save(epoch, model, optimizer, scheduler, loss_module, metrics,
                                        tracker=tracker, scaler=scaler, rng_states=rng_states))
        rank_zero_call(lambda: tracker.log({"runtime/checkpoint_seconds": time.perf_counter() - checkpoint_started,
                                          "epoch": epoch + 1}, step=ckpt.counters["attempted"]))
        restore_rng_state(rng_states[rank])
        if cfg.execution_profile == "spcs":
            check_stop(device, deadline)

    rank_zero_call(lambda: tracker.finish())
    rank_zero_call(lambda: ckpt.publish_tracking(tracker, cfg.epochs - 1))
    if cleanup is not None:
        cleanup.pop("tracker", None)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train the multi-task BraTS model")
    ap.add_argument("--config", default="configs/segresnet_base.yaml")
    ap.add_argument("--data-config", default=None)
    ap.add_argument("--resume", action="store_true", help="resume from last.pt")
    ap.add_argument("--epochs", type=int, default=None, help="override epoch count")
    ap.add_argument(
        "--lambda-cls", type=float, default=None,
        help="override; pass 0.0 for the segmentation-only ablation",
    )
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--attempt-id", default=None)
    ap.add_argument("--artifact-root", default=None)
    ap.add_argument("--cache-audit-path", default=None)
    ap.add_argument("--expected-split-sha256", default=None)
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--max-runtime-seconds", type=int, default=None)
    ap.add_argument(
        "--stage-uri",
        default=None,
        help="Snowflake stage for per-epoch checkpoints, e.g. @BRATS_MRI.CORE.CKPT",
    )
    ap.add_argument("--wandb", action="store_true", help="enable W&B tracking")
    ap.add_argument("--no-wandb", action="store_true", help="disable W&B tracking")
    ap.add_argument(
        "--wandb-mode",
        default=None,
        choices=["online", "offline", "disabled"],
        help="offline (default) writes to disk for a later `wandb sync`; online needs "
        "egress to api.wandb.ai via an external access integration",
    )
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-group", default=None, help="group ablation arms together")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )
    cfg = TrainConfig.from_yaml(args.config)
    for name in ("run_id", "attempt_id", "artifact_root", "cache_audit_path", "expected_split_sha256", "wandb_entity", "max_runtime_seconds"):
        value = getattr(args, name) or os.environ.get(f"BRATS_{name.upper()}")
        if value is not None:
            setattr(cfg, name, int(value) if name == "max_runtime_seconds" else value)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lambda_cls is not None:
        cfg.lambda_cls = args.lambda_cls
    if args.run_name:
        cfg.run_name = args.run_name
    if args.stage_uri is not None:
        cfg.stage_uri = args.stage_uri
    if args.wandb:
        cfg.wandb = True
    if args.no_wandb:
        cfg.wandb = False
    if args.wandb_mode:
        cfg.wandb_mode = args.wandb_mode
    if args.wandb_project:
        cfg.wandb_project = args.wandb_project
    if args.wandb_group:
        cfg.wandb_group = args.wandb_group

    data_cfg = DataConfig.load(args.data_config) if args.data_config else DataConfig.load()
    train(cfg, data_cfg, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
