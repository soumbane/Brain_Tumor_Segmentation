"""CPU restart, worker-stream and collective failure regression tests."""
import json
import random
from dataclasses import asdict, replace
from datetime import timedelta

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import torch.multiprocessing as mp
from brats.artifacts import ArtifactError, resolve_bundle
from brats.train import (CheckpointManager, TrainConfig, capture_rng_state, restore_rng_state,
                         seed_loader_epoch, rank_zero_call, collective_call, gather_rng_states,
                         optimizer_step)
from brats.data.transforms import seed_worker
from brats.models.multitask import MultiTaskConfig


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = MultiTaskConfig(init_filters=8)
        self.net = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.4), nn.Linear(4, 1))

    def forward(self, inputs):
        return self.net(inputs)


class RandomDataset(Dataset):
    def __init__(self):
        self.rng = np.random.RandomState(0)

    def set_random_state(self, seed):
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return torch.tensor([index, self.rng.uniform(), self.rng.uniform(), self.rng.uniform()], dtype=torch.float32)


def _loader(workers):
    return DataLoader(RandomDataset(), batch_size=2, shuffle=True, num_workers=workers,
                      generator=torch.Generator(), worker_init_fn=seed_worker,
                      **({"multiprocessing_context": "spawn"} if workers else {}))


def _epoch(model, loader, optimizer, epoch):
    seed_loader_epoch(loader, 42, epoch, 0)
    model.train()
    seen = []
    for batch in loader:
        seen.append(batch.clone())
        optimizer.zero_grad()
        model(batch).square().mean().backward()
        optimizer.step()
    return seen


@pytest.mark.parametrize("workers", [0, 2])
def test_next_epoch_order_augmentation_dropout_match_after_validation_resume(tmp_path, monkeypatch, workers):
    monkeypatch.delenv("BRATS_CKPT_ROOT", raising=False)
    torch.manual_seed(7)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = _loader(workers)
    _epoch(model, loader, optimizer, 0)
    # Separate validation generator and random-consuming finalization precede boundary.
    validation = _loader(0)
    seed_loader_epoch(validation, 42, 0, 0, validation=True)
    list(validation)
    torch.rand(3)
    manager = CheckpointManager(TrainConfig(), root=tmp_path)
    manager.save(0, model, optimizer, None, nn.Identity(), {"dice_avg": 0.6},
                 scaler=torch.amp.GradScaler("cuda", enabled=False), rng_states=[capture_rng_state()])
    expected_batches = _epoch(model, loader, optimizer, 1)
    expected_weights = {key: value.clone() for key, value in model.state_dict().items()}
    restored = TinyModel()
    new_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    fresh_loader = _loader(workers)
    assert manager.load(restored, new_opt) == 1
    actual_batches = _epoch(restored, fresh_loader, new_opt, 1)
    for expected, actual in zip(expected_batches, actual_batches, strict=True):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(expected_weights[key], value, rtol=0, atol=0)


def test_best_scores_serialized_before_last_and_strict_missing_resume(tmp_path, monkeypatch):
    monkeypatch.delenv("BRATS_CKPT_ROOT", raising=False)
    model = TinyModel()
    manager = CheckpointManager(TrainConfig(), root=tmp_path)
    with pytest.raises(FileNotFoundError):
        manager.load(model)
    manager.save(4, model, torch.optim.SGD(model.parameters(), lr=0.1), None, nn.Identity(),
                 {"dice_avg": 0.8, "dice_ET": 0.9})
    state = torch.load(manager.dir / "last.pt", weights_only=False)
    assert state["best"]["avg"] == 0.8 and state["best"]["ET"] == 0.9
    assert state["best_epochs"] == {"avg": 4, "ET": 4}


def test_stage_resume_into_clean_local_root_and_incompatibility_is_not_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("BRATS_CKPT_ROOT", raising=False)
    cfg = TrainConfig(execution_profile="spcs", run_id="r", attempt_id="a", artifact_root=str(tmp_path / "stage"))
    model = TinyModel()
    opt = torch.optim.AdamW(model.parameters())
    manager = CheckpointManager(cfg, root=tmp_path / "first")
    manager.prepare_run(False, dataset_identity="exact-npz-hash", world_size=1, steps_per_epoch=4, model_config=asdict(model.cfg))
    manager.save(0, model, opt, None, nn.Identity(), {"dice_avg": 0.5}, rng_states=[capture_rng_state()])
    manager.save(1, model, opt, None, nn.Identity(), {"dice_avg": 0.4}, rng_states=[capture_rng_state()])
    fresh_cfg = replace(cfg, attempt_id="b")
    fresh = CheckpointManager(fresh_cfg, root=tmp_path / "fresh")
    fresh.prepare_run(True, dataset_identity="exact-npz-hash", world_size=1, steps_per_epoch=4, model_config=asdict(model.cfg))
    assert fresh.load_collective(model, opt, None, nn.Identity(), None, 0) == 2
    assert fresh.best_epochs["avg"] == 0
    mismatch = CheckpointManager(replace(cfg, attempt_id="c", epochs=999), root=tmp_path / "mismatch")
    mismatch.prepare_run(True, dataset_identity="exact-npz-hash", world_size=1, steps_per_epoch=4, model_config=asdict(model.cfg))
    with pytest.raises(RuntimeError, match="incompatible"):
        mismatch.load_collective(model, opt, None, nn.Identity(), None, 0)
    assert resolve_bundle(manager.run_root)["metadata"]["epoch"] == 1


def test_scaler_state_and_rank_specific_rng(tmp_path, monkeypatch):
    monkeypatch.delenv("BRATS_CKPT_ROOT", raising=False)
    model = TinyModel()
    manager = CheckpointManager(TrainConfig(), root=tmp_path)
    scaler = torch.amp.GradScaler("cpu", init_scale=32)
    torch.manual_seed(101)
    rank0 = capture_rng_state()
    torch.manual_seed(202)
    rank1 = capture_rng_state()
    expected = torch.rand(4)
    manager.save(0, model, torch.optim.AdamW(model.parameters()), None, nn.Identity(), {},
                 scaler=scaler, rng_states=[rank0, rank1])
    restored_scaler = torch.amp.GradScaler("cpu", init_scale=1024)
    manager.load(model, scaler=restored_scaler, rank=1)
    assert restored_scaler.get_scale() == 32
    torch.testing.assert_close(expected, torch.rand(4))


def test_spcs_rejects_unknown_and_privacy_contradictions_but_legacy_warns(tmp_path, caplog):
    path = tmp_path / "config.yaml"
    path.write_text("train:\n  execution_profile: spcs\n  tyop: 3\n")
    with pytest.raises(ValueError, match="unknown"):
        TrainConfig.from_yaml(path)
    path.write_text("train:\n  tyop: 3\n")
    TrainConfig.from_yaml(path)
    assert "unknown" in caplog.text
    with pytest.raises(ValueError, match="online"):
        TrainConfig(execution_profile="spcs").validate_config()


def test_loader_safety_is_scoped_to_spcs_without_changing_legacy_defaults(monkeypatch):
    import brats.train as training

    records = [{"cohort": "GLI", "case_id": f"case-{index}"} for index in range(8)]
    monkeypatch.setattr(training, "load_records", lambda *args: records)
    legacy = TrainConfig(num_workers=2)
    train_loader, val_loader, _ = training.build_loaders(legacy, None, 0, 1)
    assert train_loader.multiprocessing_context is None
    assert val_loader.multiprocessing_context is None
    assert train_loader.prefetch_factor == 2 and train_loader.persistent_workers
    assert legacy.val_sw_batch_size == 4
    spcs = replace(legacy, execution_profile="spcs", prefetch_factor=1)
    train_loader, val_loader, _ = training.build_loaders(spcs, None, 0, 1)
    assert train_loader.multiprocessing_context.get_start_method() == "spawn"
    assert val_loader.multiprocessing_context.get_start_method() == "spawn"
    assert train_loader.prefetch_factor == 1 and not train_loader.persistent_workers


def _collective_worker(rank, init_file, result_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(100 + rank)
        states = gather_rng_states()
        assert not torch.equal(states[0]["torch"], states[1]["torch"])
        failures = []
        for operation in (lambda: rank_zero_call(lambda: (_ for _ in ()).throw(OSError("publish failed"))),
                          lambda: collective_call(lambda: (_ for _ in ()).throw(ValueError("rank read failure")) if rank == 1 else True)):
            try:
                operation()
            except RuntimeError as exc:
                failures.append(str(exc))
        assert len(failures) == 2
        # The real multitask model must survive a second DDP reduction at all
        # attention phases, including classifier-disabled and uncertainty modes.
        from torch.nn.parallel import DistributedDataParallel as DDP
        from brats.models.multitask import build_model
        from brats.losses import build_loss
        for coefficient, weighting in ((0.1, "fixed"), (0.0, "fixed"), (0.1, "uncertainty")):
            model = DDP(build_model(MultiTaskConfig(init_filters=2, blocks_down=(1, 1), dsdepth=1, cls_hidden=4)),
                        find_unused_parameters=coefficient == 0)
            loss_module = build_loss(lambda_cls=coefficient, weighting=weighting)
            params = list(model.parameters()) + list(loss_module.parameters())
            optimizer = torch.optim.AdamW(params, lr=0.001)
            for alpha in (1.0, 0.5, 0.0):
                model.module._alpha = alpha
                for _ in range(2):
                    image = torch.randn(2, 4, 8, 8, 8)
                    target = torch.ones(2, 3, 8, 8, 8)
                    optimizer.zero_grad()
                    result = model(image, gt_wt=target[:, 2:3])
                    loss = loss_module(result["seg"], target, result["cls"], torch.tensor([rank, rank + 1]))
                    _, stepped = optimizer_step(loss["total"], params, optimizer,
                                                 torch.amp.GradScaler("cuda", enabled=False), 12,
                                                 unmanaged=list(loss_module.parameters()))
                    assert stepped
                # Rank-zero unwrapped validation, then further DDP training.
                def evaluate():
                    model.module.eval()
                    with torch.no_grad():
                        model.module(torch.randn(1, 4, 8, 8, 8))
                rank_zero_call(evaluate)
                model.train()
            if weighting == "uncertainty":
                values = [None, None]
                dist.all_gather_object(values, loss_module.uncertainty.log_var.detach().tolist())
                assert values[0] == values[1]
        from pathlib import Path
        # Rank zero chooses once; rank one must never independently resolve a
        # different epoch. Both acknowledge exact bytes and their own RNG slot.
        root = Path(result_dir)
        cfg = TrainConfig(execution_profile="spcs", run_id="collective", attempt_id="first", artifact_root=str(root / "stage"))
        small = TinyModel()
        opt = torch.optim.AdamW(small.parameters())
        writer = CheckpointManager(cfg, root=root / "local-first")
        writer.prepare_run(False, dataset_identity="same-data", world_size=2, steps_per_epoch=2, model_config=asdict(small.cfg))
        states = gather_rng_states()
        rank_zero_call(lambda: writer.save(0, small, opt, None, nn.Identity(), {}, rng_states=states))
        reader = CheckpointManager(replace(cfg, attempt_id="resume"), root=root / "local-clean")
        import brats.artifacts as artifact_module
        actual_resolve = artifact_module.resolve_bundle
        if rank == 1:
            def forbidden_resolve(*args, **kwargs):
                raise AssertionError("peer independently selected a checkpoint")
            artifact_module.resolve_bundle = forbidden_resolve
        try:
            reader.prepare_run(True, dataset_identity="same-data", world_size=2, steps_per_epoch=2, model_config=asdict(small.cfg))
            assert reader.load_collective(small, opt, None, nn.Identity(), None, rank) == 1
            torch.testing.assert_close(torch.get_rng_state(), states[rank]["torch"])
        finally:
            artifact_module.resolve_bundle = actual_resolve
        Path(result_dir, f"{rank}.json").write_text(json.dumps(failures))
    finally:
        dist.destroy_process_group()


def test_rank_zero_publication_and_peer_load_failure_propagate(tmp_path):
    ctx = mp.get_context("spawn")
    workers = [ctx.Process(target=_collective_worker, args=(rank, str(tmp_path / "rdzv"), str(tmp_path))) for rank in range(2)]
    for worker in workers:
        worker.start()
    try:
        for worker in workers:
            worker.join(timeout=120)
            assert not worker.is_alive() and worker.exitcode == 0
        assert json.loads((tmp_path / "0.json").read_text()) == json.loads((tmp_path / "1.json").read_text())
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
                worker.join()
