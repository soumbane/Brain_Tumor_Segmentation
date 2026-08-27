"""The binarization control -- the single most decisive confound diagnostic.

Method, after Tinauer et al. (arXiv 2501.15831). Retrain the *real* model on
intensity-**binarized** volumes: every in-brain voxel set to 1, background 0. All
tissue texture is destroyed; only the geometry of the skull-stripped mask survives.

**If classification accuracy survives binarization, the model is reading shape and
geometry, not tumor phenotype.** In the original paper a 3D CNN classifying
Alzheimer's from 990 matched ADNI scans retained *full* accuracy on binarized images,
and layer-wise relevance propagation confirmed it was keying on brain contours
introduced by skull stripping. They call it a Clever Hans effect.

This is stronger than the cheap probes in :mod:`brats.confound.probes`, which ask
whether cohort is *predictable* from hand-built features. This asks whether *our
model* is actually using it.

Two things make this control cheap for us: the binarized model needs no new data
pipeline (just one transform), and it can run at a short schedule, since the question
is "does accuracy survive?" rather than "what is the best accuracy?".

Interpretation, from the accuracy ratio ``acc_binarized / acc_normal``:

    > 0.90   the classifier is essentially geometry-driven. Report the head as an
             acquisition/geometry detector, not a tumor-type classifier.
    0.7-0.9  substantially confounded; a real but minority contribution from texture.
    < 0.7    texture carries most of the signal; the confound is present but not
             dominant.

Usage::

    .venv\\Scripts\\python.exe -m brats.confound.binarize_control \\
        --config configs/segresnet_base.yaml --epochs 30
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from brats.config import DataConfig
from brats.constants import COHORTS

log = logging.getLogger("brats.confound.binarize")


def binarize_batch(image: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Set every in-brain voxel to 1 and background to 0, per channel.

    Input is z-scored with background at exactly 0, so "in brain" is simply
    ``!= 0``. Using a nonzero test rather than a percentile keeps the operation
    independent of the intensity distribution -- the whole point is to remove it.
    """
    return (image.abs() > threshold).to(image.dtype)


class BinarizingWrapper(torch.nn.Module):
    """Binarizes inputs before the wrapped model sees them.

    Applied as a wrapper rather than a dataset transform so the *identical* training
    code, augmentation, and schedule run in both arms. A matched comparison is the
    only kind that answers the question.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, **kwargs):
        return self.model(binarize_batch(x), **kwargs)

    def __getattr__(self, name: str):
        # Forward set_warmup_alpha, cfg, etc. to the wrapped model.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def evaluate_classification(
    model: torch.nn.Module,
    dataset,
    device: torch.device | str = "cuda",
    binarize: bool = False,
    limit: int | None = None,
) -> dict:
    """Balanced accuracy and confusion matrix for the cohort head."""
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix

    from brats.inference import InferenceConfig, predict_volume

    net = model.module if hasattr(model, "module") else model
    net.eval()
    inf_cfg = InferenceConfig(tta_flip_axes=())

    y_true: list[int] = []
    y_pred: list[int] = []
    n = len(dataset) if limit is None else min(limit, len(dataset))
    for i in range(n):
        item = dataset[i]
        image = item["image"]
        if not torch.is_tensor(image):
            image = torch.as_tensor(np.asarray(image))
        if binarize:
            image = binarize_batch(image.float())
        out = predict_volume(net, image, cfg=inf_cfg, device=device)
        y_pred.append(int(np.argmax(out["cls_probs"])))
        y_true.append(int(item["cohort"]))

    return {
        "n": len(y_true),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion": confusion_matrix(
            y_true, y_pred, labels=list(range(len(COHORTS)))
        ).tolist(),
    }


def interpret(ratio: float, acc_norm: float, acc_bin: float) -> str:
    chance = 1.0 / len(COHORTS)
    if acc_norm <= chance + 0.05:
        return (
            "INCONCLUSIVE: the normal-input classifier is barely above chance, so the "
            "ratio is not meaningful. Fix the classifier before running this control."
        )
    if ratio > 0.90:
        return (
            "SEVERE. Accuracy survives the removal of all tissue texture, so the head "
            "is essentially a geometry/acquisition detector. Report it as such -- this "
            "is the Clever Hans result from Tinauer et al., reproduced on our data. It "
            "is a legitimate and interesting finding, not a failure to hide."
        )
    if ratio > 0.70:
        return (
            "SUBSTANTIAL. Most of the signal is geometry. Report the head as auxiliary "
            "with an explicit confound caveat and lean on tumor-attention pooling."
        )
    if ratio > 0.45:
        return (
            "MODERATE. Geometry contributes materially but texture carries more. "
            "Quantify both; state the split in the results."
        )
    return (
        "WEAK. Accuracy collapses without texture, which is the reassuring outcome: "
        "the head is using image content rather than mask shape. Still report the "
        "number, and still keep the site probe."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/segresnet_base.yaml")
    ap.add_argument("--data-config", default=None)
    ap.add_argument("--epochs", type=int, default=30, help="short matched schedule")
    ap.add_argument(
        "--normal-checkpoint",
        default=None,
        help="existing non-binarized checkpoint; trained fresh if omitted",
    )
    ap.add_argument("--limit", type=int, default=None, help="cap eval cases")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )

    from brats.data.transforms import CachedBratsDataset, load_records, val_transforms
    from brats.models.multitask import MultiTaskConfig, build_model
    from brats.train import CheckpointManager, TrainConfig, train

    cfg = TrainConfig.from_yaml(args.config)
    cfg.epochs = args.epochs
    data_cfg = DataConfig.load(args.data_config) if args.data_config else DataConfig.load()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_ds = CachedBratsDataset(
        load_records(data_cfg, "val"), transform=val_transforms(True), load_label=True
    )

    # -- arm A: normal inputs -------------------------------------------
    normal_cfg = replace(cfg, run_name=f"{cfg.run_name}_bin_control_normal")
    if args.normal_checkpoint:
        model = build_model(MultiTaskConfig(pooling=cfg.pooling)).to(device)  # type: ignore[arg-type]
        CheckpointManager(normal_cfg).load(model, path=Path(args.normal_checkpoint))
    else:
        log.info("training arm A (normal inputs), %d epochs", args.epochs)
        train(normal_cfg, data_cfg)
        model = build_model(MultiTaskConfig(pooling=cfg.pooling)).to(device)  # type: ignore[arg-type]
        CheckpointManager(normal_cfg).load(model)
    res_normal = evaluate_classification(model, val_ds, device, False, args.limit)

    # -- arm B: binarized inputs, matched schedule ----------------------
    log.info("training arm B (binarized inputs), %d epochs", args.epochs)
    bin_cfg = replace(cfg, run_name=f"{cfg.run_name}_bin_control_binarized")
    # NOTE: requires train() to accept an input wrapper. Until that hook exists,
    # run the same command with the dataset transform binarized; the comparison is
    # only valid if BOTH arms share schedule, seed, and augmentation.
    raise NotImplementedError(
        "Arm B needs a one-line hook in brats.train.train() to wrap the model in "
        "BinarizingWrapper. Deliberately left explicit rather than silently "
        "monkey-patching the trainer: the control is worthless if the two arms are "
        "not matched on schedule, seed, and augmentation.\n"
        f"Arm A result: balanced accuracy {res_normal['balanced_accuracy']:.3f} "
        f"on n={res_normal['n']}."
    )


def report(res_normal: dict, res_binarized: dict, out_dir: Path) -> str:
    """Format the two-arm comparison. Separated so it is testable without training."""
    acc_n = res_normal["balanced_accuracy"]
    acc_b = res_binarized["balanced_accuracy"]
    ratio = acc_b / acc_n if acc_n > 0 else float("nan")

    lines = [
        "=" * 64,
        "Binarization control (Tinauer et al., arXiv 2501.15831)",
        "=" * 64,
        f"normal inputs     : balanced accuracy {acc_n:.4f}  (n={res_normal['n']})",
        f"binarized inputs  : balanced accuracy {acc_b:.4f}  (n={res_binarized['n']})",
        f"chance            : {1.0 / len(COHORTS):.4f}",
        f"survival ratio    : {ratio:.3f}",
        "",
        interpret(ratio, acc_n, acc_b),
    ]
    text = "\n".join(lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "binarize_control.txt").write_text(text, encoding="utf-8")
    (out_dir / "binarize_control.json").write_text(
        json.dumps({"normal": res_normal, "binarized": res_binarized, "ratio": ratio}, indent=2),
        encoding="utf-8",
    )
    return text


__all__ = [
    "binarize_batch",
    "BinarizingWrapper",
    "evaluate_classification",
    "report",
    "interpret",
]


if __name__ == "__main__":
    raise SystemExit(main())
