"""bf16 autocast must not poison the model through the tumor-attention pool.

Regression for: ``sigmoid`` of a bf16 WT logit saturates to exactly 1.0 once the logit
exceeds ~6.2; the old ``logit(clamp(p))`` round trip then produced ``inf``, the backward
pass produced ``0 * inf = NaN`` on the learned temperature, and ``clip_grad_norm_`` spread
that NaN to every parameter in a single optimizer step.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from brats.constants import REGION_INDEX  # noqa: E402
from brats.models.multitask import MultiTaskBraTS, MultiTaskConfig  # noqa: E402
from brats.models.pooling import TumorAttentionPool  # noqa: E402

WT = REGION_INDEX["WT"]


def _model_with_confident_wt_head(pooling: str) -> MultiTaskBraTS:
    """A model whose WT logits are large everywhere, as after a few trained epochs."""
    torch.manual_seed(0)
    model = MultiTaskBraTS(MultiTaskConfig(pooling=pooling, warmup_epochs=0))
    model.train()
    with torch.no_grad():
        for level in model.backbone.up_layers:
            head = level["head"]
            head.bias.zero_()
            head.bias[WT] = 8.0
    return model


@pytest.mark.parametrize("pooling", ["tumor_attention", "tafe"])
def test_one_bf16_step_leaves_all_parameters_finite(pooling):
    model = _model_with_confident_wt_head(pooling)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    x = torch.randn(2, 4, 32, 32, 32)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out["cls"].float(), torch.tensor([0, 1]))
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
    opt.step()

    assert torch.isfinite(grad_norm), f"grad norm is {grad_norm}"
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    assert not bad, f"non-finite parameters after one step: {bad[:5]}"


def test_pool_gradient_is_finite_for_saturated_logits():
    pool = TumorAttentionPool(learn_temperature=True)
    feat = torch.randn(2, 8, 4, 4, 4, dtype=torch.bfloat16, requires_grad=True)
    wt_logit = torch.full((2, 1, 16, 16, 16), 20.0, dtype=torch.bfloat16)

    pool(feat, wt_logit=wt_logit).float().sum().backward()

    assert torch.isfinite(pool.log_temperature.grad).all()
    assert torch.isfinite(feat.grad.float()).all()


def test_pool_matches_reference_weighted_mean():
    """Temperature scales the logit, then sigmoid, then a normalised weighted mean."""
    torch.manual_seed(1)
    pool = TumorAttentionPool(floor=0.0, learn_temperature=True)
    with torch.no_grad():
        pool.log_temperature.fill_(0.5)  # temperature e^0.5

    feat = torch.randn(2, 5, 4, 4, 4)
    logit = torch.randn(2, 1, 4, 4, 4) * 2  # same resolution: no interpolation involved

    weight = torch.sigmoid(logit * torch.tensor(0.5).exp())
    expected = (feat * weight).sum((2, 3, 4)) / weight.sum((2, 3, 4))

    torch.testing.assert_close(pool(feat, wt_logit=logit), expected)
