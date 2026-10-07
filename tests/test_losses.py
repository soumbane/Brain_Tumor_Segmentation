"""The segmentation loss must be per-channel sigmoid, not a softmax across regions.

Regression for: ``seg_loss: dice_ce`` used MONAI's ``DiceCELoss``, whose CE term is
``nn.CrossEntropyLoss`` -- a softmax over the (ET, TC, WT) channels -- when there is more
than one channel. For nested regions a background voxel then contributed exactly zero loss.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

import torch.nn.functional as F  # noqa: E402
from monai.losses import DiceLoss  # noqa: E402

from brats.losses import MultiTaskLoss, SegLoss  # noqa: E402

# One voxel per row: background, ED only (WT), core (TC+WT), enhancing (ET+TC+WT).
NESTED = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.float32)


def _batch(logit_fn):
    target = NESTED.T.reshape(1, 3, 4, 1, 1)  # (B=1, C=(ET,TC,WT), 4 voxels)
    return logit_fn(target), target


def test_dice_ce_equals_dice_plus_per_channel_bce():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 8, 8, 8) * 3
    target = (torch.rand(2, 3, 8, 8, 8) > 0.7).float()

    got = SegLoss("dice_ce", batch_dice=True)(logits, target)

    dice = DiceLoss(include_background=True, sigmoid=True, batch=True, smooth_dr=1e-5)(logits, target)
    expected = dice + F.binary_cross_entropy_with_logits(logits, target)
    torch.testing.assert_close(got, expected)


def test_loss_is_near_zero_for_the_perfect_prediction():
    logits, target = _batch(lambda t: t * 20 - 10)
    assert float(SegLoss("dice_ce")(logits, target)) < 1e-3


def test_background_false_positives_are_penalised():
    """With softmax-CE a background voxel (target [0,0,0]) cost exactly nothing."""
    target = torch.zeros(1, 3, 4, 4, 4)
    target[:, :, :2] = 1.0  # half the volume is tumor so Dice is well defined
    off = (target * 20 - 10)
    on_in_background = off.clone()
    on_in_background[:, :, 2:] = 10.0  # predict all three regions ON where the truth is background

    loss = SegLoss("dice_ce")
    assert float(loss(on_in_background, target)) > float(loss(off, target)) + 1.0


@pytest.mark.parametrize("kind", ["dice_ce", "dice_focal"])
def test_deep_supervision_is_a_weighted_sum_of_levels(kind):
    torch.manual_seed(2)
    target = (torch.rand(1, 3, 16, 16, 16) > 0.6).float()
    levels = [torch.randn(1, 3, 16, 16, 16), torch.randn(1, 3, 8, 8, 8), torch.randn(1, 3, 4, 4, 4)]
    loss = SegLoss(kind)

    got = loss(levels, target)

    w = torch.tensor([1.0, 0.5, 0.25])
    w = w / w.sum()
    expected = sum(
        wi * loss(lv, F.interpolate(target, size=lv.shape[2:], mode="nearest"))
        for wi, lv in zip(w, levels, strict=True)
    )
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("kind", ["dice_ce", "dice_focal"])
def test_bf16_logits_give_a_float32_loss_and_finite_gradients(kind):
    torch.manual_seed(3)
    logits = (torch.randn(2, 3, 8, 8, 8) * 20).to(torch.bfloat16).requires_grad_(True)
    target = (torch.rand(2, 3, 8, 8, 8) > 0.5).float()

    out = SegLoss(kind)(logits, target)
    out.backward()

    assert out.dtype == torch.float32 and torch.isfinite(out)
    assert torch.isfinite(logits.grad.float()).all()


def test_uniform_class_weights_give_plain_cross_entropy():
    """The trainer passes ones when a balancing sampler is active (no double correction)."""
    logits = torch.randn(6, 3)
    target = torch.tensor([0, 0, 1, 1, 2, 2])
    seg = torch.randn(1, 3, 4, 4, 4)
    seg_t = (torch.rand(1, 3, 4, 4, 4) > 0.5).float()

    unweighted = MultiTaskLoss(lambda_cls=1.0, class_weights=torch.ones(3))
    out = unweighted(seg, seg_t, logits, target)

    torch.testing.assert_close(out["cls"], F.cross_entropy(logits, target))
    weighted = MultiTaskLoss(lambda_cls=1.0)  # default: inverse-frequency weights
    assert not torch.allclose(weighted(seg, seg_t, logits, target)["cls"], out["cls"])
