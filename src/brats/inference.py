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
from pathlib import Path

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

    def __post_init__(self) -> None:
        if len(self.roi_size) != 3 or any(int(v) != v or v <= 0 for v in self.roi_size):
            raise ValueError("roi_size must contain three positive integers")
        if self.sw_batch_size <= 0 or not 0 <= self.overlap < 1:
            raise ValueError("sw_batch_size must be positive and overlap in [0, 1)")
        if self.mode not in {"constant", "gaussian"}:
            raise ValueError("inference mode must be constant or gaussian")
        if self.amp_dtype not in {torch.bfloat16, torch.float16, torch.float32}:
            raise ValueError("inference precision must be bfloat16, float16 or float32")


def inference_config_from_checkpoint(
    state: dict,
    path: str | Path | None = None,
    *,
    fallback: dict | None = None,
    override: bool = False,
    tta: str | None = None,
) -> InferenceConfig:
    """Use checkpoint patch/precision; legacy fallback or override must be explicit.

    ``state`` is a trusted, locally loaded checkpoint. A partial checkpoint config
    may use only explicitly supplied fallback values, never implicit YAML defaults.
    """
    saved = state.get("config") or {}
    effective = dict(fallback or {}) if override else {**(fallback or {}), **saved}
    missing = {"patch_size", "amp_dtype"} - effective.keys()
    if missing:
        raise ValueError(f"checkpoint lacks {sorted(missing)}; explicit legacy config required")
    precision = effective["amp_dtype"]
    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if precision not in dtypes:
        raise ValueError(f"unsupported checkpoint amp_dtype: {precision!r}")
    cfg = inference_config_from_yaml(path, roi_size=tuple(effective["patch_size"]), tta=tta)
    cfg.amp_dtype = dtypes[precision]
    cfg.extra["training_settings_source"] = (
        "explicit_override" if override else
        "checkpoint" if {"patch_size", "amp_dtype"} <= saved.keys() else "explicit_legacy_fallback"
    )
    cfg.__post_init__()
    return cfg


def require_test_acknowledgement(split: str, acknowledged: bool) -> None:
    if split == "test" and not acknowledged:
        raise SystemExit(
            "Refusing to open the internal test split without "
            "--i-understand-this-opens-the-test-split. Freeze all settings on validation first."
        )


def validate_geometry(pred: dict) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray]:
    """Require invertible original-grid metadata and the metric's orthogonal 1-mm grid."""
    vectors = []
    for key in ("orig_shape", "crop_start", "crop_stop"):
        value = np.asarray(pred[key])
        if value.shape != (3,) or not np.isfinite(value).all() or not np.equal(value, np.floor(value)).all():
            raise ValueError(f"invalid {key}: expected three finite integers")
        vectors.append(value.astype(np.int64))
    shape, start, stop = vectors
    if np.any(shape <= 0) or np.any(stop <= start):
        raise ValueError("invalid original shape or crop extents")
    if np.any(np.minimum(stop, shape) <= np.maximum(start, 0)):
        raise ValueError("crop does not intersect the original grid")
    probs = np.asarray(pred["probs"])
    if probs.shape != (N_REGIONS, *tuple(stop - start)):
        raise ValueError("probability shape does not match crop extents")
    if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if "label" in pred and np.asarray(pred["label"]).shape != probs.shape:
        raise ValueError("ground-truth shape does not match probabilities")
    if "affine" not in pred:
        raise ValueError("original affine is required for export; no identity fallback")
    affine = np.asarray(pred["affine"], dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("affine must be a finite 4x4 matrix")
    if not np.allclose(affine[3], (0, 0, 0, 1)) or abs(np.linalg.det(affine[:3, :3])) < 1e-8:
        raise ValueError("affine must be nonsingular and homogeneous")
    if not np.allclose(affine[:3, :3].T @ affine[:3, :3], np.eye(3), atol=1e-4, rtol=1e-4):
        raise ValueError("postprocessing and HD95 require an orthogonal 1-mm voxel grid")
    return tuple(int(v) for v in shape), start, stop, affine


def restore_channels(volume: np.ndarray, pred: dict) -> np.ndarray:
    """Discard padding while restoring channel-first data to the original grid."""
    from brats.postprocess import paste_into_original

    volume = np.asarray(volume)
    expected_shape = tuple(np.asarray(pred["crop_stop"]) - np.asarray(pred["crop_start"]))
    if volume.ndim != 4 or volume.shape[1:] != expected_shape:
        raise ValueError("channel-first volume shape does not match crop geometry")
    shape = tuple(int(v) for v in pred["orig_shape"])
    restored = np.empty((volume.shape[0], *shape), dtype=volume.dtype)
    for index, channel in enumerate(volume):
        restored[index] = paste_into_original(channel, shape, pred["crop_start"], pred["crop_stop"])
    return restored


def canonical_label_map(probs: np.ndarray, cohort: str, cfg) -> tuple[np.ndarray, dict]:
    """Postprocess original-grid probabilities and score the representable export masks.

    Label-map serialization necessarily nests regions, even when a research ablation
    disables nesting. Deriving masks from the final map prevents export/score drift.
    """
    from brats.data.preprocess import labels_to_regions
    from brats.constants import REGIONS
    from brats.postprocess import postprocess, regions_to_label_map

    label_map = regions_to_label_map(postprocess(probs, cohort=cohort, cfg=cfg))
    regions = labels_to_regions(label_map).astype(bool)
    return label_map, dict(zip(REGIONS, regions, strict=True))


def inference_config_from_yaml(
    path: str | Path | None = None,
    roi_size: tuple[int, int, int] | None = None,
    tta: str | None = None,
) -> InferenceConfig:
    """Build an :class:`InferenceConfig` from the ``inference:`` block of ``postproc.yaml``.

    Args:
        roi_size: The window the model was *trained* on. It takes precedence over the
            YAML value, which is only a default: a different window than training is a
            mistake, so a disagreement is logged.
        tta: ``"none"`` or ``"eight_flip"``; overrides the YAML value (the CLI flag).
    """
    from brats.config import load_yaml
    from brats.postprocess import DEFAULT_POSTPROC_YAML

    raw = dict(load_yaml(path or DEFAULT_POSTPROC_YAML).get("inference") or {})
    unknown = set(raw) - {"roi_size", "overlap", "mode", "sw_batch_size", "tta"}
    if unknown:
        raise ValueError(f"unknown inference keys {sorted(unknown)}")

    yaml_roi = tuple(int(v) for v in raw.get("roi_size", (128, 128, 128)))
    if roi_size is not None and tuple(roi_size) != yaml_roi:
        log.warning(
            "inference.roi_size %s in the YAML differs from the training patch %s; "
            "using the training patch", yaml_roi, tuple(roi_size),
        )
    mode = tta if tta is not None else raw.get("tta", "none")
    if mode not in ("none", "eight_flip"):
        raise ValueError(f"tta must be 'none' or 'eight_flip', got {mode!r}")
    return InferenceConfig(
        roi_size=tuple(roi_size) if roi_size is not None else yaml_roi,  # type: ignore[arg-type]
        sw_batch_size=int(raw.get("sw_batch_size", 4)),
        overlap=float(raw.get("overlap", 0.5)),
        mode=str(raw.get("mode", "gaussian")),
        tta_flip_axes=eight_flip_axes() if mode == "eight_flip" else (),
    )


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

    use_amp = torch.device(device).type == "cuda" and cfg.amp_dtype != torch.float32
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


def predict_dataset(
    model: nn.Module,
    dataset,
    cfg: InferenceConfig | None = None,
    device: torch.device | str = "cuda",
    limit: int | None = None,
    include_image: bool = False,
):
    """Yield per-case predictions for a dataset.

    Yields dicts with ``case_id``, ``cohort``, ``probs``, ``cls_probs``, the crop
    geometry needed to paste back into the original grid, and ``label`` when present.
    ``predict_volume`` owns no-grad; decorating this generator would retain its
    previously yielded volume while requesting the next item.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    n = len(dataset) if limit is None else min(limit, len(dataset))
    for i in range(n):
        item = dataset[i]
        if "affine" not in item:
            raise ValueError("cached case is missing its original affine")
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
        rec["affine"] = np.asarray(item["affine"])
        if include_image:
            rec["image"] = np.asarray(item["image"])
        if "label" in item:
            lab = item["label"]
            rec["label"] = lab.numpy() if isinstance(lab, Tensor) else np.asarray(lab)
            del lab
        del item, out
        yield rec
        del rec  # release the preceding volume before loading the next case
        if (i + 1) % 25 == 0:
            log.info("inference %d/%d", i + 1, n)


__all__ = [
    "InferenceConfig",
    "inference_config_from_yaml",
    "inference_config_from_checkpoint",
    "validate_geometry",
    "restore_channels",
    "canonical_label_map",
    "require_test_acknowledgement",
    "predict_volume",
    "predict_dataset",
    "eight_flip_axes",
]
