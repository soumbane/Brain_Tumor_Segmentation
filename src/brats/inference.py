"""Sliding-window inference, TTA, and volume-level classification aggregation.

Segmentation. ``sliding_window_inference`` with window = training patch size,
overlap 0.5, ``mode="gaussian"``.

TTA is a **sweep, not an assumption**. The published evidence is genuinely split:
BiomedMBZ gained +3.17 lesion-wise Dice from 8-flip TTA, while Ferreira found that
*disabling* nnU-Net's TTA gave better validation results and was ~8x faster, letting
them ensemble more models instead. Measure it on validation.

Classification. The head is trained on patches but the prediction is per volume.
Resolution: average classifier logits over sliding-window positions **weighted by
the predicted tumor volume in each window**, so empty windows cannot vote. Patch
location statistics themselves leak cohort information, which is why unweighted
averaging over all positions would be the wrong choice.

Do **not** ``torch.compile`` this path: ``mode="reduce-overhead"`` uses CUDA graphs,
which break on the dynamic shapes sliding-window inference produces.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn

from brats.constants import N_COHORTS, N_REGIONS, REGION_INDEX

log = logging.getLogger("brats.inference")

WT_CHANNEL = REGION_INDEX["WT"]


@dataclass
class InferenceConfig:
    """Inference settings."""

    roi_size: tuple[int, int, int] = (128, 128, 128)
    sw_batch_size: int = 4
    overlap: float = 0.5
    mode: str = "gaussian"
    #: Flip axes for TTA. ``()`` disables TTA; the full 8-flip set is the three
    #: spatial axes and their combinations. Sweep this.
    tta_flip_axes: tuple[tuple[int, ...], ...] = ()
    amp_dtype: torch.dtype = torch.bfloat16
    #: Run sliding-window on CPU to keep the full-volume probability map off the GPU.
    #: A 240x240x155x3 float32 map is ~107 MB, so GPU is usually fine.
    device_for_output: str = "cpu"

    extra: dict = field(default_factory=dict)


def eight_flip_axes() -> tuple[tuple[int, ...], ...]:
    """All 8 flip combinations over the 3 spatial axes, including the identity."""
    combos: list[tuple[int, ...]] = []
    for r in range(4):
        combos.extend(itertools.combinations((2, 3, 4), r))
    return tuple(combos)


class _SegOnly(nn.Module):
    """Adapter exposing only segmentation logits, as ``sliding_window_inference`` expects.

    Also records per-window classification logits and predicted tumor volume so the
    volume-level cohort prediction can be aggregated in the same pass rather than
    requiring a second sweep over the volume.
    """

    def __init__(self, model: nn.Module, collect_cls: bool = True) -> None:
        super().__init__()
        self.model = model
        self.collect_cls = collect_cls
        self.cls_logits: list[Tensor] = []
        self.cls_weights: list[Tensor] = []

    def reset(self) -> None:
        self.cls_logits.clear()
        self.cls_weights.clear()

    def forward(self, x: Tensor) -> Tensor:
        out = self.model(x)
        seg = out["seg"]
        if isinstance(seg, (list, tuple)):
            seg = seg[0]  # finest level only at inference

        if self.collect_cls and "cls" in out:
            # Weight each window by its predicted WT volume: windows with no tumor
            # carry no cohort evidence beyond acquisition signature.
            wt_prob = torch.sigmoid(seg[:, WT_CHANNEL])
            weight = wt_prob.flatten(1).sum(dim=1)
            self.cls_logits.append(out["cls"].detach().float().cpu())
            self.cls_weights.append(weight.detach().float().cpu())

        return seg


@torch.no_grad()
def predict_volume(
    model: nn.Module,
    image: Tensor,
    cfg: InferenceConfig | None = None,
    device: torch.device | str = "cuda",
) -> dict[str, np.ndarray]:
    """Predict segmentation probabilities and a cohort distribution for one volume.

    Args:
        model: A :class:`~brats.models.multitask.MultiTaskBraTS` (or DDP-wrapped).
        image: (1, 4, D, H, W) or (4, D, H, W) preprocessed volume.
        cfg: Inference configuration.
        device: Compute device.

    Returns:
        ``{"probs": (3, D, H, W) float32, "cls_probs": (3,) float32,
        "cls_logits": (3,) float32}`` -- segmentation probabilities in canonical
        (ET, TC, WT) order.
    """
    from monai.inferers import sliding_window_inference

    cfg = cfg or InferenceConfig()
    net = model.module if hasattr(model, "module") else model
    net.eval()
    # Attention pooling must never use ground truth at inference.
    if hasattr(net, "set_warmup_alpha"):
        net._alpha = 0.0  # noqa: SLF001 - explicit and intentional

    if image.ndim == 4:
        image = image.unsqueeze(0)
    if image.ndim != 5:
        raise ValueError(f"expected (B,4,D,H,W) or (4,D,H,W), got {tuple(image.shape)}")

    image = image.to(device)
    wrapper = _SegOnly(net, collect_cls=True).to(device)

    flips = cfg.tta_flip_axes or ((),)
    accum: Tensor | None = None
    cls_logit_sum = torch.zeros(N_COHORTS, dtype=torch.float64)
    cls_weight_sum = 0.0

    use_amp = torch.device(device).type == "cuda"
    for axes in flips:
        wrapper.reset()
        x = torch.flip(image, dims=list(axes)) if axes else image
        with torch.autocast(
            device_type="cuda", dtype=cfg.amp_dtype, enabled=use_amp
        ):
            logits = sliding_window_inference(
                inputs=x,
                roi_size=cfg.roi_size,
                sw_batch_size=cfg.sw_batch_size,
                predictor=wrapper,
                overlap=cfg.overlap,
                mode=cfg.mode,
                device=torch.device(cfg.device_for_output),
                progress=False,
            )
        logits = logits.float()
        if axes:  # undo the flip before averaging
            logits = torch.flip(logits, dims=list(axes))
        probs = torch.sigmoid(logits)
        accum = probs if accum is None else accum + probs

        # Tumor-volume-weighted average of per-window classification logits.
        if wrapper.cls_logits:
            lg = torch.cat(wrapper.cls_logits, dim=0).double()
            w = torch.cat(wrapper.cls_weights, dim=0).double()
            if float(w.sum()) <= 0:
                w = torch.ones_like(w)  # no tumor predicted anywhere: fall back
            cls_logit_sum += (lg * w.unsqueeze(1)).sum(dim=0)
            cls_weight_sum += float(w.sum())

    assert accum is not None
    probs = (accum / len(flips)).squeeze(0).cpu().numpy().astype(np.float32)
    if probs.shape[0] != N_REGIONS:
        raise RuntimeError(f"expected {N_REGIONS} region channels, got {probs.shape[0]}")

    if cls_weight_sum > 0:
        cls_logits = (cls_logit_sum / cls_weight_sum).float()
    else:
        cls_logits = torch.zeros(N_COHORTS)
    cls_probs = torch.softmax(cls_logits, dim=0)

    return {
        "probs": probs,
        "cls_logits": cls_logits.numpy().astype(np.float32),
        "cls_probs": cls_probs.numpy().astype(np.float32),
    }


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    dataset,
    cfg: InferenceConfig | None = None,
    device: torch.device | str = "cuda",
    limit: int | None = None,
):
    """Yield per-case predictions for a dataset.

    Yields dicts with ``case_id``, ``cohort``, ``probs``, ``cls_probs``, the crop
    geometry needed to paste back into the original grid, and ``label`` when present.
    """
    n = len(dataset) if limit is None else min(limit, len(dataset))
    for i in range(n):
        item = dataset[i]
        out = predict_volume(model, item["image"], cfg=cfg, device=device)
        rec = {
            "case_id": item["case_id"],
            "cohort": item["cohort_name"],
            "probs": out["probs"],
            "cls_probs": out["cls_probs"],
            "cls_logits": out["cls_logits"],
            "crop_start": np.asarray(item["crop_start"]),
            "crop_stop": np.asarray(item["crop_stop"]),
            "orig_shape": np.asarray(item["orig_shape"]),
        }
        if "label" in item:
            lab = item["label"]
            rec["label"] = lab.numpy() if isinstance(lab, Tensor) else np.asarray(lab)
        yield rec
        if (i + 1) % 25 == 0:
            log.info("inference %d/%d", i + 1, n)


__all__ = [
    "InferenceConfig",
    "predict_volume",
    "predict_dataset",
    "eight_flip_axes",
]
