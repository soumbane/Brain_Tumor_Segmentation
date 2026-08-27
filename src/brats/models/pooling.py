"""Pooling strategies for the cohort classification head.

The plan's central design decision. Three options, in increasing robustness:

**A. Plain GAP** -- global average pool over the encoder bottleneck. High confound
risk: at a 128^3 patch the bottleneck is ~8^3, so a flat average is dominated by
whole-brain / background statistics, which is exactly the channel through which
cohort identity leaks. Tinauer et al. (arXiv 2501.15831) showed a 3D CNN
classifying Alzheimer's kept full accuracy on *binarized* images, relying on brain
contours introduced by skull stripping. Kept here only as an ablation arm -- it is
the measurement of how much the confound is worth.

**B. Soft tumor-attention pooling** (default). Pool encoder features weighted by
the predicted whole-tumor probability, restricting the classifier's receptive field
to tumor voxels and closing the dominant leakage path (brain contour, FOV,
skull-strip geometry).

  A correction to the original design, which specified pooling over the *predicted
  binary mask*: at epoch 0 that mask is noise, so the pooling region is random and
  the classifier gets no usable gradient -- it does not train. Two changes fix it
  without weakening the confound argument:

  * Use ``sigmoid(WT logit)`` as a *soft, differentiable* weight rather than a
    thresholded binary mask.
  * **Warm-start** from the downsampled ground-truth WT mask, then anneal linearly
    to the predicted one. Nothing leaks at inference, where GT is unavailable and
    ``alpha = 0`` by construction.

**C. TAFE-style multi-scale attention** (MTS-UNET, arXiv 2503.06828) -- better
accuracy, more complexity. Implemented as a later upgrade path.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

PoolingKind = Literal["gap", "tumor_attention", "tafe"]


def _downsample_to(mask: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Resize a (B,1,D,H,W) soft mask to ``shape`` with trilinear interpolation."""
    if tuple(mask.shape[2:]) == tuple(shape):
        return mask
    return F.interpolate(mask, size=shape, mode="trilinear", align_corners=False)


class GlobalAveragePool(nn.Module):
    """Option A. Flat mean over spatial dims. The confound-exposed baseline."""

    def forward(
        self, feat: Tensor, wt_prob: Tensor | None = None, gt_wt: Tensor | None = None,
        alpha: float = 0.0,
    ) -> Tensor:
        return feat.mean(dim=tuple(range(2, feat.ndim)))


class TumorAttentionPool(nn.Module):
    """Option B. Soft tumor-weighted average pool with GT warm-start.

    Args:
        floor: Minimum attention weight everywhere. Keeps a trickle of gradient
            flowing when the predicted tumor probability collapses to zero, and
            prevents a divide-by-tiny in the normalization.
        learn_temperature: If set, scale the WT logits by a learned temperature
            before the sigmoid, letting the model sharpen or soften its own
            attention rather than being stuck with the segmentation head's
            calibration.
    """

    def __init__(self, floor: float = 0.01, learn_temperature: bool = True) -> None:
        super().__init__()
        self.floor = float(floor)
        self.log_temperature = (
            nn.Parameter(torch.zeros(1)) if learn_temperature else None
        )

    def forward(
        self,
        feat: Tensor,
        wt_prob: Tensor | None = None,
        gt_wt: Tensor | None = None,
        alpha: float = 0.0,
    ) -> Tensor:
        """Pool ``feat`` (B,C,D,H,W) using a tumor-derived attention weight.

        Args:
            feat: Encoder bottleneck features.
            wt_prob: Predicted WT *probability* (B,1,d,h,w) at any resolution.
            gt_wt: Ground-truth WT binary mask (B,1,D,H,W), training only.
            alpha: Weight on the GT mask. ``1.0`` = pure teacher forcing,
                ``0.0`` = pure prediction. Annealed by the training loop; must be
                0 at inference.
        """
        spatial = tuple(feat.shape[2:])

        if wt_prob is None and gt_wt is None:
            # Nothing to attend with; degrade to a mean rather than fail.
            return feat.mean(dim=tuple(range(2, feat.ndim)))

        weight: Tensor | None = None
        if wt_prob is not None:
            w = wt_prob
            if self.log_temperature is not None:
                # Re-sharpen in logit space; clamp keeps the logit finite.
                logit = torch.logit(w.clamp(1e-6, 1 - 1e-6))
                w = torch.sigmoid(logit * self.log_temperature.exp())
            weight = _downsample_to(w, spatial)

        if gt_wt is not None and alpha > 0.0:
            gt = _downsample_to(gt_wt.to(feat.dtype), spatial)
            weight = gt if weight is None else (alpha * gt + (1.0 - alpha) * weight)

        assert weight is not None
        weight = weight.to(feat.dtype).clamp_min(self.floor)

        # Weighted mean over spatial dims.
        num = (feat * weight).sum(dim=tuple(range(2, feat.ndim)))
        den = weight.sum(dim=tuple(range(2, weight.ndim))).clamp_min(1e-6)
        return num / den


class TAFEPool(nn.Module):
    """Option C. Multi-scale tumor-focused attention pooling (TAFE-style).

    Concatenates tumor-attention-pooled features from several encoder levels, each
    projected to a common width, then fuses them. Ablations in MTS-UNET
    (arXiv 2503.06828) report this is necessary for their accuracy, not decorative.
    """

    def __init__(self, in_channels: list[int], out_channels: int, floor: float = 0.01):
        super().__init__()
        self.pool = TumorAttentionPool(floor=floor, learn_temperature=True)
        self.projections = nn.ModuleList(
            nn.Sequential(nn.Linear(c, out_channels), nn.GELU()) for c in in_channels
        )
        self.fuse = nn.Sequential(
            nn.Linear(out_channels * len(in_channels), out_channels),
            nn.GELU(),
        )

    def forward(
        self,
        feats: list[Tensor],
        wt_prob: Tensor | None = None,
        gt_wt: Tensor | None = None,
        alpha: float = 0.0,
    ) -> Tensor:
        if len(feats) != len(self.projections):
            raise ValueError(
                f"expected {len(self.projections)} feature levels, got {len(feats)}"
            )
        pooled = [
            proj(self.pool(f, wt_prob=wt_prob, gt_wt=gt_wt, alpha=alpha))
            for proj, f in zip(self.projections, feats, strict=True)
        ]
        return self.fuse(torch.cat(pooled, dim=1))


def build_pooling(kind: PoolingKind, **kwargs) -> nn.Module:
    """Factory for the pooling strategies."""
    if kind == "gap":
        return GlobalAveragePool()
    if kind == "tumor_attention":
        return TumorAttentionPool(
            floor=kwargs.get("floor", 0.01),
            learn_temperature=kwargs.get("learn_temperature", True),
        )
    if kind == "tafe":
        return TAFEPool(
            in_channels=kwargs["in_channels"],
            out_channels=kwargs.get("out_channels", 256),
            floor=kwargs.get("floor", 0.01),
        )
    raise ValueError(f"unknown pooling kind: {kind!r}")
