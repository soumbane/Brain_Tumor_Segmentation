"""Losses for the multi-task objective.

    L_total = L_seg + lambda * L_cls

``L_seg`` -- Dice + CE on 3 nested sigmoid channels, summed over deep-supervision
levels with ``1/2^i`` weights. Both Dice+CE and Dice+Focal podiumed in BraTS 2023
with no clear winner (NVAUTO and BiomedMBZ used Dice+Focal; CNMC/nnU-Net used
Dice+CE), so both are available. **Batch Dice** rather than per-sample Dice: the
winners used it, and it is what keeps cases with genuinely empty ET from producing
a degenerate gradient. Many PED cases have empty ET -- diffuse midline gliomas
frequently do not enhance at all.

``L_cls`` -- cross-entropy with class weights inverse to cohort frequency
(GLI:MEN:PED = 12.6:10.1:1).

``lambda`` weighting. Dice and CE live on different scales, so this matters:

1. **Fixed lambda**, small sweep {0.05, 0.1, 0.3, 1.0}. Start here.
2. **Homoscedastic uncertainty weighting** (Kendall, Gal & Cipolla, arXiv 1705.07115),
   learning a per-task log-variance. This is what GMMAS (arXiv 2501.17758) adopted
   for exactly this task family.
3. GradNorm is deliberately not implemented: a direct three-way comparison found it
   consistently underperformed on one of two tasks at higher compute and memory cost.

``lambda = 0`` is a **required ablation**, not an afterthought -- it is the only way
to answer whether the classification head costs segmentation Dice, and no published
multi-task paper answers that for us.

Numerics: prefer **bf16** over fp16. A10G is Ampere, so bf16 is native; it needs no
``GradScaler`` and is far more forgiving for Dice-family losses, whose small
denominators are a known fp16 NaN source. Mixing a Dice-scale and a CE-scale loss
under fp16 makes the scaler's job harder still.
"""

from __future__ import annotations

from typing import Literal, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from brats.constants import COHORTS, EXPECTED_TRAIN_COUNTS, N_COHORTS

SegLossKind = Literal["dice_ce", "dice_focal"]


def cohort_class_weights(
    counts: dict[str, int] | None = None, normalize: bool = True
) -> Tensor:
    """Class weights inverse to cohort frequency, in ``COHORTS`` order."""
    counts = counts or EXPECTED_TRAIN_COUNTS
    n = torch.tensor([float(counts[c]) for c in COHORTS])
    w = n.sum() / (len(COHORTS) * n)
    return w / w.mean() if normalize else w


def deep_supervision_weights(n_levels: int, device=None) -> Tensor:
    """``1/2^i`` weights for deep-supervision levels, normalized to sum to 1."""
    w = torch.tensor([0.5**i for i in range(n_levels)], dtype=torch.float32, device=device)
    return w / w.sum()


class SegLoss(nn.Module):
    """Dice + (CE | Focal) over 3 nested sigmoid channels, with deep supervision.

    Args:
        kind: ``"dice_ce"`` or ``"dice_focal"``.
        batch_dice: Aggregate Dice over the whole batch instead of per sample.
            Stabilizes cases with an empty region -- with per-sample Dice, an empty
            GT channel gives a near-constant loss and a useless gradient.
        include_background: Always False here; there is no background channel, the
            3 outputs are independent nested foreground regions.
    """

    def __init__(
        self,
        kind: SegLossKind = "dice_ce",
        batch_dice: bool = True,
        smooth_nr: float = 0.0,
        smooth_dr: float = 1e-5,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        from monai.losses import DiceCELoss, DiceFocalLoss

        common = dict(
            include_background=True,  # all 3 channels are foreground regions
            sigmoid=True,             # nested regions -> sigmoid, never softmax
            to_onehot_y=False,        # targets are already 3 binary channels
            batch=batch_dice,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            lambda_dice=lambda_dice,
        )
        if kind == "dice_ce":
            self.loss = DiceCELoss(**common, lambda_ce=lambda_ce)
        elif kind == "dice_focal":
            self.loss = DiceFocalLoss(**common, lambda_focal=lambda_ce, gamma=focal_gamma)
        else:
            raise ValueError(f"unknown seg loss kind: {kind!r}")
        self.kind = kind

    def forward(self, logits: Tensor | Sequence[Tensor], target: Tensor) -> Tensor:
        """
        Args:
            logits: Either a single (B,3,D,H,W) tensor or a list of them, finest
                first, when deep supervision is active.
            target: (B,3,D,H,W) binary targets in (ET, TC, WT) order.
        """
        if isinstance(logits, Tensor):
            return self.loss(logits, target)

        weights = deep_supervision_weights(len(logits), device=target.device)
        total = logits[0].new_zeros(())
        for w, lg in zip(weights, logits, strict=True):
            if lg.shape[2:] != target.shape[2:]:
                # Nearest-neighbour downsampling keeps the targets binary; any
                # interpolation would create fractional labels.
                tgt = F.interpolate(target.float(), size=lg.shape[2:], mode="nearest")
            else:
                tgt = target.float()
            total = total + w * self.loss(lg, tgt)
        return total


class ClsLoss(nn.Module):
    """Class-weighted cross-entropy over the 3 cohorts, with optional smoothing."""

    def __init__(
        self,
        class_weights: Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        w = class_weights if class_weights is not None else cohort_class_weights()
        self.register_buffer("class_weights", w)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            logits,
            target.long(),
            weight=self.class_weights.to(logits.dtype),
            label_smoothing=self.label_smoothing,
        )


class UncertaintyWeighting(nn.Module):
    """Homoscedastic uncertainty weighting (Kendall, Gal & Cipolla, arXiv 1705.07115).

    Learns a per-task log-variance ``s_i`` and forms
    ``sum_i exp(-s_i) * L_i + s_i``. Parameterizing in log space keeps the variance
    positive without a constraint, and the ``+ s_i`` term stops the trivial solution
    of driving every weight to zero.
    """

    def __init__(self, n_tasks: int = 2) -> None:
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: Sequence[Tensor]) -> Tensor:
        if len(losses) != self.log_var.numel():
            raise ValueError(f"expected {self.log_var.numel()} losses, got {len(losses)}")
        total = losses[0].new_zeros(())
        for i, loss in enumerate(losses):
            s = self.log_var[i]
            total = total + torch.exp(-s) * loss + s
        return total

    def weights(self) -> list[float]:
        """Current effective task weights, for logging."""
        return torch.exp(-self.log_var).detach().cpu().tolist()


class MultiTaskLoss(nn.Module):
    """Combine segmentation and classification losses.

    Args:
        lambda_cls: Fixed classification weight. **Set to 0.0 for the required
            ablation** that measures whether the classification head costs Dice.
        weighting: ``"fixed"`` or ``"uncertainty"``.
    """

    def __init__(
        self,
        seg_kind: SegLossKind = "dice_ce",
        lambda_cls: float = 0.1,
        weighting: Literal["fixed", "uncertainty"] = "fixed",
        batch_dice: bool = True,
        class_weights: Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.seg_loss = SegLoss(kind=seg_kind, batch_dice=batch_dice)
        self.cls_loss = ClsLoss(class_weights=class_weights, label_smoothing=label_smoothing)
        self.lambda_cls = float(lambda_cls)
        self.weighting = weighting
        self.uncertainty = UncertaintyWeighting(2) if weighting == "uncertainty" else None

    @property
    def classification_enabled(self) -> bool:
        return self.weighting == "uncertainty" or self.lambda_cls > 0.0

    def forward(
        self,
        seg_logits: Tensor | Sequence[Tensor],
        seg_target: Tensor,
        cls_logits: Tensor | None = None,
        cls_target: Tensor | None = None,
    ) -> dict[str, Tensor]:
        l_seg = self.seg_loss(seg_logits, seg_target)

        want_cls = (
            self.classification_enabled
            and cls_logits is not None
            and cls_target is not None
        )
        if not want_cls:
            # lambda = 0 ablation: report the cls loss for comparability but keep it
            # out of the graph so it cannot influence a single gradient.
            l_cls = (
                self.cls_loss(cls_logits, cls_target).detach()
                if cls_logits is not None and cls_target is not None
                else l_seg.new_zeros(())
            )
            return {"total": l_seg, "seg": l_seg.detach(), "cls": l_cls}

        assert cls_logits is not None and cls_target is not None
        l_cls = self.cls_loss(cls_logits, cls_target)

        if self.uncertainty is not None:
            total = self.uncertainty([l_seg, l_cls])
        else:
            total = l_seg + self.lambda_cls * l_cls

        return {"total": total, "seg": l_seg.detach(), "cls": l_cls.detach()}


def build_loss(
    seg_kind: SegLossKind = "dice_ce",
    lambda_cls: float = 0.1,
    weighting: Literal["fixed", "uncertainty"] = "fixed",
    **kwargs,
) -> MultiTaskLoss:
    return MultiTaskLoss(
        seg_kind=seg_kind, lambda_cls=lambda_cls, weighting=weighting, **kwargs
    )


__all__ = [
    "SegLoss",
    "ClsLoss",
    "UncertaintyWeighting",
    "MultiTaskLoss",
    "build_loss",
    "cohort_class_weights",
    "deep_supervision_weights",
    "N_COHORTS",
]
