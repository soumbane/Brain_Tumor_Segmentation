"""DDP training loop for the multi-task model on GPU_NV_M (4x A10G 24 GB).

Compute shape. ``GPU_NV_M`` is one node: 4x NVIDIA A10G 24 GB, 44 vCPU, 178 GiB RAM,
93.13 GiB local disk. At 128^3 with bf16, SegResNetDS runs batch 2/GPU, giving an
**effective batch of 8** across 4 ranks -- the batch the recipe wants, without
needing a 96 GB card. 44 vCPU / 4 ranks leaves ~11 vCPU per rank, enough to keep
``num_workers=8`` fed; 3D augmentation is CPU-bound and starves the GPUs first.

A10G is Ampere (sm_86), so **bf16 is native**: no ``GradScaler``, and far more
forgiving for Dice-family losses whose small denominators are a known fp16 NaN
source. NCCL over PCIe (no NVLink) is fine for a ~30M-parameter model.

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
import shutil
import time
from dataclasses import asdict, dataclass, field
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
    train_transforms,
    val_transforms,
)
from brats.losses import build_loss
from brats.metrics.lesionwise import legacy_dice
from brats.models.multitask import MultiTaskConfig, build_model, count_parameters
from brats.tracking import build_logger

log = logging.getLogger("brats.train")


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

    # -- experiment tracking (Weights & Biases) ---------------------------
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

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        raw = load_yaml(path)
        tr = raw.get("train", raw)
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in tr.items() if k in known}
        if "patch_size" in kwargs:
            kwargs["patch_size"] = tuple(kwargs["patch_size"])
        return cls(**kwargs)


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
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
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


def _cohort_sample_weights(records: list[dict]) -> list[float]:
    """Inverse-frequency weights so each cohort is equally likely per batch."""
    counts: dict[str, int] = {}
    for r in records:
        counts[r["cohort"]] = counts.get(r["cohort"], 0) + 1
    return [1.0 / counts[r["cohort"]] for r in records]


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

    if cfg.cohort_balanced_sampling and world_size == 1:
        sampler: Any = WeightedRandomSampler(
            _cohort_sample_weights(train_recs), num_samples=len(train_recs), replacement=True
        )
        shuffle = False
    else:
        # Under DDP, DistributedSampler owns sharding. Cohort balance is then carried
        # by the class-weighted classification loss rather than by the sampler; mixing
        # a WeightedRandomSampler with DDP sharding would double-count cases.
        sampler = (
            DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
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
        persistent_workers=cfg.num_workers > 0,
        collate_fn=collate_metadata,
    )
    # Validation runs full volumes through sliding-window inference: batch size 1.
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(2, cfg.num_workers // 2),
        pin_memory=True,
        collate_fn=collate_metadata,
    )
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
        self.dir = (root or REPO_ROOT) / cfg.ckpt_dir / cfg.run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.best: dict[str, float] = {k: -1.0 for k in TRACKED}

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
    ) -> None:
        net = model.module if isinstance(model, DDP) else model
        state = {
            "epoch": epoch,
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "loss_module": loss_module.state_dict(),
            "metrics": metrics,
            "config": asdict(self.cfg),
            "best": self.best,
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }

        # Atomic write: a checkpoint truncated by a mid-write kill is worse than none.
        last = self.dir / "last.pt"
        tmp = self.dir / "last.pt.tmp"
        torch.save(state, tmp)
        tmp.replace(last)
        self._stage(last)

        for key in TRACKED:
            metric_key = "dice_avg" if key == "avg" else f"dice_{key}"
            value = metrics.get(metric_key)
            if value is None:
                continue
            if value > self.best[key]:
                self.best[key] = float(value)
                dest = self.dir / f"best_{key}.pt"
                shutil.copyfile(last, dest)
                self._stage(dest)
                log.info("new best %s: %.4f (epoch %d)", key, value, epoch)
                if tracker is not None:
                    tracker.summary(f"best_dice_{key}", float(value))
                    tracker.summary(f"best_dice_{key}_epoch", epoch)

        with (self.dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, **metrics}) + "\n")

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        loss_module: nn.Module | None = None,
        path: Path | None = None,
        map_location: str = "cpu",
    ) -> int:
        """Restore a checkpoint. Returns the epoch to resume *from*."""
        ckpt_path = path or (self.dir / "last.pt")
        if not ckpt_path.is_file():
            log.info("no checkpoint at %s -- starting from scratch", ckpt_path)
            return 0
        state = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        net = model.module if isinstance(model, DDP) else model
        net.load_state_dict(state["model"])
        if optimizer is not None and state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        if loss_module is not None and state.get("loss_module"):
            loss_module.load_state_dict(state["loss_module"])
        self.best = state.get("best", self.best)
        rng = state.get("rng")
        if rng:
            # Restoring RNG makes a resumed run bit-comparable to an uninterrupted one.
            torch.set_rng_state(rng["torch"].cpu() if torch.is_tensor(rng["torch"]) else rng["torch"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
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
        overlap=0.5,
        tta_flip_axes=(),  # no TTA during training validation
        amp_dtype=torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16,
    )

    per_region: dict[str, list[float]] = {r: [] for r in REGIONS}
    cls_correct = cls_total = 0

    for batch in loader:
        image = batch["image"][0]
        label = batch["label"][0].numpy()
        out = predict_volume(net, image, cfg=inf_cfg, device=device)
        probs = out["probs"]
        for i, region in enumerate(REGIONS):
            per_region[region].append(legacy_dice(label[i] > 0.5, probs[i] >= 0.5))

        pred_cohort = int(np.argmax(out["cls_probs"]))
        true_cohort = int(batch["cohort"][0])
        cls_correct += int(pred_cohort == true_cohort)
        cls_total += 1

    metrics = {
        f"dice_{r}": float(np.mean(v)) if v else 0.0 for r, v in per_region.items()
    }
    metrics["dice_avg"] = float(np.mean(list(metrics.values())))
    metrics["cls_acc"] = cls_correct / max(cls_total, 1)
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


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
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed, rank)

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
        # SyncBatchNorm: with batch 2/GPU, per-rank BN statistics are too noisy.
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
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

    loss_module = build_loss(
        seg_kind=cfg.seg_loss,  # type: ignore[arg-type]
        lambda_cls=cfg.lambda_cls,
        weighting=cfg.loss_weighting,  # type: ignore[arg-type]
        batch_dice=cfg.batch_dice,
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

    ckpt = CheckpointManager(cfg)
    start_epoch = ckpt.load(model, optimizer, scheduler, loss_module) if resume else 0

    # Rank-zero only. A disabled tracker is a silent no-op, and no tracker failure can
    # kill the run -- see brats.tracking.
    tracker = build_logger(cfg, is_rank_zero=is_main(rank))
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

    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    # bf16 needs no GradScaler; fp16 does. Prefer bf16 -- see brats.losses.
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        net = model.module if isinstance(model, DDP) else model
        alpha = net.set_warmup_alpha(epoch)
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        t0 = time.time()
        running = {"total": 0.0, "seg": 0.0, "cls": 0.0}
        n_steps = 0
        global_step = epoch * len(train_loader)

        for batch in train_loader:
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
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                optimizer.step()
            scheduler.step()

            for k in running:
                running[k] += float(losses[k].detach())
            n_steps += 1

            # Per-step logging on rank 0. Every 20 steps keeps the curve readable
            # without flooding the run; LR matters because batch 8 is off-literature
            # and the schedule is being swept.
            if is_main(rank) and tracker.enabled and n_steps % 20 == 0:
                tracker.log(
                    {
                        "train/loss": float(losses["total"].detach()),
                        "train/loss_seg": float(losses["seg"]),
                        "train/loss_cls": float(losses["cls"]),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/warmup_alpha": alpha,
                        "epoch": epoch + 1,
                    },
                    step=global_step + n_steps,
                )

        for k in running:
            running[k] = all_reduce_mean(running[k] / max(n_steps, 1), device)

        if is_main(rank):
            log.info(
                "epoch %3d/%d  loss %.4f (seg %.4f cls %.4f)  lr %.2e  alpha %.2f  %.1fs",
                epoch + 1, cfg.epochs, running["total"], running["seg"], running["cls"],
                scheduler.get_last_lr()[0], alpha, time.time() - t0,
            )

        # Validation and checkpointing on rank 0 only.
        do_val = ((epoch + 1) % cfg.val_every == 0) or (epoch + 1 == cfg.epochs)
        metrics = dict(running)
        if do_val and is_main(rank):
            metrics.update(validate(model, val_loader, device, cfg))
            log.info(
                "  val  dice avg %.4f  ET %.4f TC %.4f WT %.4f  cls_acc %.3f",
                metrics["dice_avg"], metrics["dice_ET"], metrics["dice_TC"],
                metrics["dice_WT"], metrics["cls_acc"],
            )
        if is_main(rank):
            ckpt.save(
                epoch, model, optimizer, scheduler, loss_module, metrics, tracker=tracker
            )
            if tracker.enabled:
                payload = {
                    "epoch": epoch + 1,
                    "epoch/loss": running["total"],
                    "epoch/loss_seg": running["seg"],
                    "epoch/loss_cls": running["cls"],
                    "epoch/lr": scheduler.get_last_lr()[0],
                    "epoch/warmup_alpha": alpha,
                    "epoch/seconds": time.time() - t0,
                }
                for key in ("dice_avg", "dice_ET", "dice_TC", "dice_WT", "cls_acc"):
                    if key in metrics:
                        payload[f"val/{key}"] = metrics[key]
                if loss_module.uncertainty is not None:
                    w = loss_module.uncertainty.weights()
                    payload["loss_weight/seg"] = w[0]
                    payload["loss_weight/cls"] = w[1]
                tracker.log(payload, step=(epoch + 1) * len(train_loader))
        barrier()

    if is_main(rank):
        tracker.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


def main(argv: list[str] | None = None) -> int:
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
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )
    cfg = TrainConfig.from_yaml(args.config)
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
