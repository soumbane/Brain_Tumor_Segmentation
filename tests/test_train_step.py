"""The optimizer step must never apply a non-finite gradient, and DDP must stay in sync.

Regression for: ``clip_grad_norm_`` with a NaN norm multiplies *every* gradient by NaN, so a
single bad batch turned every weight in the model into NaN with no warning.
"""


import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from torch import nn  # noqa: E402

from brats.train import optimizer_step, sync_grads  # noqa: E402


def _scaler_off():
    # Same construction as train.py for bf16/fp32; "cuda" with enabled=False is a no-op.
    return torch.amp.GradScaler("cuda", enabled=False)


def _tiny():
    torch.manual_seed(0)
    net = nn.Linear(4, 2)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    return net, opt


def test_finite_gradient_updates_the_weights():
    net, opt = _tiny()
    before = [p.detach().clone() for p in net.parameters()]
    loss = net(torch.randn(8, 4)).pow(2).mean()

    norm, stepped = optimizer_step(loss, list(net.parameters()), opt, _scaler_off(), 12.0)

    assert stepped and norm == norm and norm < float("inf")
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters(), strict=True))


@pytest.mark.parametrize("poison", [float("nan"), float("inf")])
def test_non_finite_gradient_leaves_weights_and_optimizer_untouched(poison):
    net, opt = _tiny()
    # Give the optimizer some history first so we can check its state survives too.
    optimizer_step(net(torch.randn(8, 4)).pow(2).mean(), list(net.parameters()), opt, _scaler_off(), 12.0)
    weights = [p.detach().clone() for p in net.parameters()]
    state = {k: v.clone() for k, v in opt.state[next(net.parameters())].items() if torch.is_tensor(v)}

    loss = net(torch.randn(8, 4)).sum() * poison
    norm, stepped = optimizer_step(loss, list(net.parameters()), opt, _scaler_off(), 12.0)

    assert not stepped
    assert all(torch.equal(a, b) for a, b in zip(weights, net.parameters(), strict=True))
    assert all(torch.isfinite(p).all() for p in net.parameters())
    after = opt.state[next(net.parameters())]
    assert all(torch.equal(v, after[k]) for k, v in state.items())


def test_training_recovers_after_a_skipped_step():
    net, opt = _tiny()
    params = list(net.parameters())
    optimizer_step(net(torch.randn(8, 4)).sum() * float("nan"), params, opt, _scaler_off(), 12.0)
    opt.zero_grad(set_to_none=True)  # the loop does this at the top of each iteration

    _, stepped = optimizer_step(net(torch.randn(8, 4)).pow(2).mean(), params, opt, _scaler_off(), 12.0)

    assert stepped and all(torch.isfinite(p).all() for p in net.parameters())


# -- DDP: loss-module parameters live outside DDP and must be averaged by hand --------------


def _worker(rank, world, init_file, out):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    p = nn.Parameter(torch.zeros(3))
    p.grad = torch.full((3,), float(rank + 1))  # rank 0 -> 1, rank 1 -> 2, ...
    sync_grads([p])
    out.put((rank, p.grad.tolist()))
    dist.destroy_process_group()


def test_sync_grads_averages_across_ranks(tmp_path):
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(r, 2, str(tmp_path / "rdzv"), out)) for r in range(2)
    ]
    for p in procs:
        p.start()
    results = dict(out.get(timeout=120) for _ in procs)
    for p in procs:
        p.join(timeout=60)

    assert results[0] == results[1] == [1.5, 1.5, 1.5]


def test_sync_grads_is_a_noop_without_a_process_group():
    p = nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([1.0, 2.0])
    sync_grads([p])
    assert p.grad.tolist() == [1.0, 2.0]
