"""Opt-in four-GPU full-patch DDP/recovery gate; never run automatically.

Run via torchrun with four ranks, a separately approved supervisor deadline,
and a distinct smoke-* run prefix. Run once for lambda>0 and separately for
lambda=0. --interrupt-after-publication exits 75 after a completed smoke epoch;
resume in a fresh container/local directory using the same run and new attempt.
This is NOT a full-cache audit, a full training run, or proof of W&B remote inventory.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from brats.config import DataConfig
from brats.constants import REGIONS
from brats.data.cache_check import verify_audit_identity
from brats.data.transforms import CachedBratsDataset, collate_metadata, load_records, seed_worker, train_transforms, val_transforms
from brats.losses import build_loss
from brats.models.multitask import MultiTaskConfig, build_model
from brats.tracking import build_logger
from brats.train import (CheckpointManager, TrainConfig, build_scheduler, collective_call,
                         gather_rng_states, optimizer_step, rank_zero_call, seed_loader_epoch,
                         set_seed, setup_distributed, validate)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/segresnet_spcs.yaml")
    parser.add_argument("--data-config")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--interrupt-after-publication", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lambda-cls", type=float, default=0.1)
    parser.add_argument("--acknowledge-paid-gpu", action="store_true")
    args = parser.parse_args(argv)
    if not args.acknowledge_paid_gpu:
        parser.error("requires separately approved GPU execution and --acknowledge-paid-gpu")
    if args.steps < 2 or not args.run_id.startswith("smoke-"):
        parser.error("at least two updates per alpha and a distinct smoke-* prefix are required")
    cfg = TrainConfig.from_yaml(args.config)
    cfg.run_id, cfg.attempt_id = args.run_id, args.attempt_id
    cfg.lambda_cls = args.lambda_cls
    for name in ("artifact_root", "cache_audit_path", "expected_split_sha256", "wandb_entity", "max_runtime_seconds"):
        value = os.environ.get(f"BRATS_{name.upper()}")
        if value:
            setattr(cfg, name, int(value) if name == "max_runtime_seconds" else value)
    cfg.validate_config(launch=True)
    if cfg.execution_profile != "spcs" or cfg.patch_size != (128, 128, 128) or cfg.batch_size_per_gpu != 2 or cfg.amp_dtype != "bfloat16":
        raise ValueError("smoke must exercise the approved full128/batch2/bf16 SPCS recipe")
    rank, local_rank, world = setup_distributed()
    try:
        if world != 4 or torch.cuda.device_count() != 4 or not torch.cuda.is_bf16_supported():
            raise RuntimeError("smoke requires four CUDA ranks/devices with bf16")
        device = torch.device(f"cuda:{local_rank}")
        probe = torch.tensor(1, device=device)
        dist.all_reduce(probe)
        if probe.item() != 4:
            raise RuntimeError("NCCL collective failed")
        set_seed(cfg.seed, rank)
        data_cfg = DataConfig.load(args.data_config) if args.data_config else DataConfig.load()
        identity = rank_zero_call(lambda: verify_audit_identity(data_cfg, cfg.cache_audit_path,
                                                              expected_split_sha256=cfg.expected_split_sha256))
        count = args.steps * cfg.batch_size_per_gpu * world
        records = load_records(data_cfg, "train")[:count]
        if len(records) != count:
            raise ValueError("insufficient distinct training records for two real updates per rank")
        dataset = CachedBratsDataset(records, transform=train_transforms(cfg.patch_size))
        loader = DataLoader(dataset, batch_size=2, drop_last=True,
                            sampler=DistributedSampler(dataset, num_replicas=world, rank=rank, seed=cfg.seed),
                            num_workers=cfg.num_workers, collate_fn=collate_metadata, worker_init_fn=seed_worker,
                            generator=torch.Generator(), pin_memory=True,
                            **({"multiprocessing_context": "spawn", "prefetch_factor": cfg.prefetch_factor} if cfg.num_workers else {}))
        val_records = load_records(data_cfg, "val")[:1]
        val_loader = DataLoader(CachedBratsDataset(val_records, transform=val_transforms()),
                                batch_size=1, collate_fn=collate_metadata, generator=torch.Generator())
        model_cfg = MultiTaskConfig(pooling=cfg.pooling, warmup_epochs=cfg.cls_warmup_epochs, norm="batch")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(build_model(model_cfg).to(device))
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=cfg.lambda_cls == 0 and cfg.loss_weighting != "uncertainty")
        loss_module = build_loss(seg_kind=cfg.seg_loss, lambda_cls=cfg.lambda_cls,
                                 weighting=cfg.loss_weighting, batch_dice=cfg.batch_dice).to(device)
        params = list(model.parameters()) + list(loss_module.parameters())
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = build_scheduler(optimizer, cfg, args.steps * 3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        manager = CheckpointManager(cfg)
        if args.resume and (manager.dir / "last.pt").exists():
            raise ValueError("recovery gate requires a clean local checkpoint directory")
        manager.prepare_run(args.resume, dataset_identity=identity, world_size=world,
                            steps_per_epoch=args.steps * 3, model_config=asdict(model_cfg))
        epoch = manager.load_collective(model, optimizer, scheduler, loss_module, scaler, rank) if args.resume else 0
        tracker = collective_call(lambda: build_logger(cfg, is_rank_zero=rank == 0))
        completed = 0
        torch.cuda.reset_peak_memory_stats(device)
        metrics = {}
        for phase, alpha in enumerate((1.0, 0.5, 0.0)):
            seed_loader_epoch(loader, cfg.seed, epoch * 3 + phase, rank)
            model.train()
            model.module._alpha = alpha
            phase_updates = 0
            for batch in loader:
                image, label, cohort = (batch[key].to(device) for key in ("image", "label", "cohort"))
                if tuple(image.shape) != (2, 4, 128, 128, 128):
                    raise ValueError("smoke loader did not produce full128 batch2")
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(image, gt_wt=label[:, REGIONS.index("WT")].unsqueeze(1))
                    losses = loss_module(outputs["seg"], label, outputs["cls"], cohort)
                _, stepped = optimizer_step(losses["total"], params, optimizer, scaler, cfg.grad_clip,
                                             unmanaged=list(loss_module.parameters()))
                if not stepped:
                    raise RuntimeError("smoke optimizer update was skipped")
                scheduler.step()
                phase_updates += 1
                completed += 1
            if phase_updates < 2:
                raise RuntimeError("smoke did not complete two optimizer updates")
            # Validation followed by further DDP training is an explicit gate.
            if phase == 0:
                metrics.update(rank_zero_call(lambda: validate(model, val_loader, device, cfg)))
        peaks = [None] * world
        dist.all_gather_object(peaks, torch.cuda.max_memory_allocated(device))
        metrics.update({f"peak_vram_bytes_rank_{index}": peak for index, peak in enumerate(peaks)})
        manager.counters["attempted"] += completed
        manager.counters["successful"] += completed
        rank_zero_call(lambda: tracker.log({"runtime/successful_updates": completed, "epoch": epoch + 1}))
        states = gather_rng_states()
        rank_zero_call(lambda: manager.save(epoch, model, optimizer, scheduler, loss_module, metrics,
                                            tracker=tracker, scaler=scaler, rng_states=states))
        rank_zero_call(lambda: tracker.finish())
        rank_zero_call(lambda: manager.publish_tracking(tracker, epoch))
        if rank == 0:
            print(json.dumps({"status": "gpu_compute_and_checkpoint_gate_passed", "resume": args.resume,
                              "successful_updates_per_rank": completed, "peak_vram_bytes": peaks,
                              "python": platform.python_version(), "torch": torch.__version__, "monai": version("monai"),
                              "remaining_gates": ["remote W&B inventory/readback", "own-job cancellation", "operator deadline"]}))
        if args.interrupt_after_publication:
            return 75
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
