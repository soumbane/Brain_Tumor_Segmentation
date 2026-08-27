"""Multi-task network: SegResNetDS backbone, 3-channel sigmoid segmentation
decoder, and a cohort classification head on the encoder bottleneck.

Why SegResNetDS is the sole backbone. Every BraTS 2023 podium finish across
GLI/MEN/PED was a CNN encoder-decoder; no transformer-primary method won anything,
and Swin UNETR standalone was far worse than nnU-Net under matched conditions
(CNMC's MEN validation ET: 0.640 vs 0.818). MONAI's Auto3DSeg/SegResNet took 1st
MEN, 2nd GLI, 2nd PED, 1st METS and 1st BraTS-Africa on an essentially unchanged
recipe. Beyond accuracy, SegResNet is the only strong BraTS backbone whose
reference implementation already places a second head on the encoder bottleneck
(the VAE regularization branch, arXiv 1810.11654) -- architecturally the exact
thing the classification head needs, in the exact network.

Output contract, which the rest of the codebase relies on:

* Segmentation logits have **3 channels in (ET, TC, WT) order**, activated with
  **sigmoid**, because the regions are nested (ET subset of TC subset of WT). Softmax
  over 4 classes is wrong here. The MONAI model-zoo bundle's (TC, WT, ET) order must
  never leak in.
* In training mode with deep supervision, ``seg`` is a *list* of logit tensors from
  finest to coarsest. In eval mode it is a single tensor.
* ``cls`` is 3 logits over (GLI, MEN, PED), to be softmaxed by the loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from brats.constants import N_COHORTS, N_INPUT_CHANNELS, N_REGIONS, REGION_INDEX
from brats.models.pooling import PoolingKind, build_pooling

WT_CHANNEL = REGION_INDEX["WT"]


@dataclass
class MultiTaskConfig:
    """Architecture configuration."""

    in_channels: int = N_INPUT_CHANNELS
    seg_channels: int = N_REGIONS
    n_cohorts: int = N_COHORTS

    # SegResNetDS geometry. init_filters=32 with 5 encoder levels is the
    # NVAUTO/Auto3DSeg lineage; at 128^3 with bf16 this fits batch 2 on a 24 GB A10G.
    init_filters: int = 32
    blocks_down: tuple[int, ...] = (1, 2, 2, 4)
    dsdepth: int = 4  # number of deep-supervision heads
    norm: str = "batch"  # batch is viable because DDP gives an effective batch of 8
    act: str = "relu"

    # Classification head.
    pooling: PoolingKind = "tumor_attention"
    cls_hidden: int = 256
    cls_dropout: float = 0.3

    # GT-mask warm-start for tumor-attention pooling. Alpha starts at 1.0 (pure
    # teacher forcing) and anneals to 0.0 over this many epochs. Fixes the
    # cold-start failure where a garbage predicted mask yields no usable gradient.
    warmup_epochs: int = 20

    extra: dict[str, Any] = field(default_factory=dict)


class MultiTaskBraTS(nn.Module):
    """SegResNetDS with a cohort classification head on the bottleneck.

    The backbone is instantiated from MONAI. ``SegResNetDS`` exposes ``encoder`` and
    ``up_layers`` but has **no** ``decoder`` attribute -- decoding happens inline in
    its ``_forward``. So :meth:`_encode_decode` below mirrors that loop exactly, in
    order to capture the bottleneck features on the way through. Verified against
    MONAI 1.6.0.

    Mirroring rather than calling has one further benefit: it always returns the full
    deep-supervision list, whereas ``SegResNetDS.forward`` collapses to a single
    tensor in ``eval()`` mode.
    """

    def __init__(self, cfg: MultiTaskConfig | None = None) -> None:
        super().__init__()
        from monai.networks.nets import SegResNetDS

        self.cfg = cfg or MultiTaskConfig()
        c = self.cfg

        self.backbone = SegResNetDS(
            spatial_dims=3,
            init_filters=c.init_filters,
            in_channels=c.in_channels,
            out_channels=c.seg_channels,
            blocks_down=c.blocks_down,
            norm=c.norm,
            act=c.act,
            dsdepth=c.dsdepth,
        )

        bottleneck_ch = c.init_filters * 2 ** (len(c.blocks_down) - 1)
        self.bottleneck_channels = bottleneck_ch

        if c.pooling == "tafe":
            # Pool the deepest three encoder levels.
            levels = [
                c.init_filters * 2**i
                for i in range(len(c.blocks_down) - 3, len(c.blocks_down))
            ]
            self.pool = build_pooling(
                "tafe", in_channels=levels, out_channels=c.cls_hidden
            )
            cls_in = c.cls_hidden
        else:
            self.pool = build_pooling(c.pooling)
            cls_in = bottleneck_ch

        self.classifier = nn.Sequential(
            nn.LayerNorm(cls_in),
            nn.Linear(cls_in, c.cls_hidden),
            nn.GELU(),
            nn.Dropout(c.cls_dropout),
            nn.Linear(c.cls_hidden, c.n_cohorts),
        )

        #: Current GT-mask blend weight. Owned by the training loop via
        #: :meth:`set_warmup_alpha`; always 0 in eval.
        self._alpha: float = 1.0 if c.warmup_epochs > 0 else 0.0

    # -- warm-start schedule ---------------------------------------------

    def set_warmup_alpha(self, epoch: int) -> float:
        """Linearly anneal the GT-mask blend weight from 1.0 to 0.0."""
        w = self.cfg.warmup_epochs
        self._alpha = 0.0 if w <= 0 else max(0.0, 1.0 - epoch / float(w))
        return self._alpha

    @property
    def warmup_alpha(self) -> float:
        return self._alpha

    # -- backbone plumbing -------------------------------------------------

    def _encode_decode(self, x: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        """Run encoder and decoder, returning (encoder_levels, seg_logits).

        Mirrors ``SegResNetDS._forward`` (MONAI 1.6.0). ``encoder_levels`` is
        shallowest-first; ``seg_logits`` is finest-first, one entry per
        deep-supervision head.
        """
        net = self.backbone
        if net.preprocess is not None:
            x = net.preprocess(x)
        if not net.is_valid_shape(x):
            raise ValueError(
                f"input spatial dims {tuple(x.shape[2:])} must be divisible by "
                f"{net.shape_factor()}"
            )

        # Shallowest-first; keep an unmutated copy for the classification head.
        levels: list[Tensor] = list(net.encoder(x))

        # Decode from the bottleneck upward. Copy-and-reverse so `levels` survives.
        skips = list(reversed(levels))
        h = skips.pop(0)
        if not skips:
            skips = [torch.zeros(1, device=h.device, dtype=h.dtype)]

        outputs: list[Tensor] = []
        n_up = len(net.up_layers)
        for i, level in enumerate(net.up_layers):
            h = level["upsample"](h)
            h = h + skips.pop(0)  # not in-place: keeps autograd graph clean
            h = level["blocks"](h)
            if n_up - i <= net.dsdepth:
                outputs.append(level["head"](h))
        outputs.reverse()  # finest first
        return levels, outputs

    # -- forward ----------------------------------------------------------

    def forward(
        self, x: Tensor, gt_wt: Tensor | None = None
    ) -> dict[str, Tensor | list[Tensor]]:
        """Run both heads.

        Args:
            x: Input volume (B, 4, D, H, W), channels in canonical
                ``(t1n, t1c, t2w, t2f)`` order.
            gt_wt: Optional ground-truth WT mask (B, 1, D, H, W) used only to
                warm-start attention pooling during training. Ignored in eval.

        Returns:
            ``{"seg": logits or [logits...], "cls": (B, 3) logits}``. Segmentation
            logits are **unactivated**, in (ET, TC, WT) channel order.
        """
        feats, seg_list = self._encode_decode(x)
        bottleneck = feats[-1]
        finest = seg_list[0]

        # Attention weight from the model's own WT prediction, detached so the
        # classification loss cannot corrupt the segmentation decoder through the
        # attention path -- the classifier adapts to the segmentation, not the
        # reverse.
        wt_prob = torch.sigmoid(finest[:, WT_CHANNEL : WT_CHANNEL + 1].detach())

        alpha = self._alpha if (self.training and gt_wt is not None) else 0.0
        pool_input = feats[-3:] if self.cfg.pooling == "tafe" else bottleneck
        pooled = self.pool(
            pool_input,
            wt_prob=wt_prob,
            gt_wt=gt_wt if self.training else None,
            alpha=alpha,
        )
        cls_logits = self.classifier(pooled)

        seg: Tensor | list[Tensor]
        seg = seg_list if (self.training and len(seg_list) > 1) else finest
        return {"seg": seg, "cls": cls_logits}


def build_model(cfg: MultiTaskConfig | None = None) -> MultiTaskBraTS:
    return MultiTaskBraTS(cfg)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
