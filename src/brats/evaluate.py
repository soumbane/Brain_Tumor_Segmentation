"""Final evaluation: held-out test split and the official ValidationData.

Two distinct evaluations, and the distinction matters:

**1. Internal test split (~235 labeled cases).** Full quantitative segmentation
metrics -- lesion-wise Dice and HD95 per cohort per region, mean *and* median, IQR,
and bootstrap CIs. **Opened exactly once**, after every hyperparameter and
post-processing threshold is frozen on validation. Every peek invalidates it.

**2. Official ValidationData (405 cases).** These have no ``seg.nii.gz``, so:

* **Segmentation: visual only.** No ground truth, therefore no Dice. Overlay montages
  for qualitative review (see :mod:`brats.qc`).
* **Classification: fully quantitative.** The cohort label comes from which archive a
  case shipped in, not from ``seg.nii.gz``. So these 405 never-trained cases are a
  legitimate, larger, and cleaner classification test set than the internal split --
  and closer to a true external test, since they were assembled as a separate release.

Post-processing is ablated in stages so the contrast reproduces BiomedMBZ's table: on
identical predictions legacy Dice moved +0.34 while lesion-wise Dice moved +9.97. If
our table does not show a similar divergence, the post-processing is not working.

Usage::

    # validation (repeatable, for tuning)
    .venv\\Scripts\\python.exe -m brats.evaluate --split val --checkpoint <path>

    # the one-shot test evaluation
    .venv\\Scripts\\python.exe -m brats.evaluate --split test --checkpoint <path> \\
        --i-understand-this-opens-the-test-split

    # official validation: classification metrics + QC montages
    .venv\\Scripts\\python.exe -m brats.evaluate --split official_val --checkpoint <path>
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from brats.config import DataConfig
from brats.constants import COHORTS, REGIONS
from brats.metrics.lesionwise import aggregate, lesionwise_case
from brats.metrics.ranking import ablation_table
from brats.postprocess import PostProcessConfig, postprocess

log = logging.getLogger("brats.evaluate")

#: Post-processing stages, ablated cumulatively. Mirrors BiomedMBZ's table.
ABLATION_STAGES: dict[str, dict] = {
    "raw_0.5": dict(
        thresholds={r: 0.5 for r in REGIONS},
        use_joint_filter=False,
        min_size={r: 0 for r in REGIONS},
        apply_ped_et_gate=False,
    ),
    "tuned_thresholds": dict(use_joint_filter=False, min_size={r: 0 for r in REGIONS},
                             apply_ped_et_gate=False),
    "plus_size_filter": dict(use_joint_filter=False, apply_ped_et_gate=False),
    "plus_joint_filter": dict(use_joint_filter=True, apply_ped_et_gate=False),
    "plus_ped_et_gate": dict(use_joint_filter=True, apply_ped_et_gate=True),
}


def _stage_config(stage: str, base: PostProcessConfig) -> PostProcessConfig:
    return replace(base, **ABLATION_STAGES[stage])


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def evaluate_segmentation(
    predictions,
    base_cfg: PostProcessConfig | None = None,
    stages: list[str] | None = None,
) -> pd.DataFrame:
    """Score predictions at each post-processing stage.

    Args:
        predictions: Iterable of dicts from :func:`brats.inference.predict_dataset`,
            each carrying ``probs``, ``label``, ``cohort``, ``case_id``.

    Returns:
        Long-form frame: one row per (stage, case, region).
    """
    base_cfg = base_cfg or PostProcessConfig()
    stages = stages or list(ABLATION_STAGES)
    rows: list[dict] = []

    for pred in predictions:
        if "label" not in pred:
            continue  # unlabeled (official validation): segmentation is visual only
        gt = {r: pred["label"][i] > 0.5 for i, r in enumerate(REGIONS)}

        for stage in stages:
            regions = postprocess(
                pred["probs"], cohort=pred["cohort"], cfg=_stage_config(stage, base_cfg)
            )
            scored = lesionwise_case(gt, regions)
            for region, res in scored.items():
                rows.append(
                    {
                        "stage": stage,
                        "case_id": pred["case_id"],
                        "cohort": pred["cohort"],
                        **res.as_dict(),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def evaluate_classification(predictions) -> tuple[pd.DataFrame, dict]:
    """Balanced accuracy, macro-F1, per-class AUC and confusion matrix.

    Balanced accuracy and macro-F1 rather than raw accuracy: with a 12.6:10.1:1
    cohort imbalance, raw accuracy is dominated by GLI and MEN and would hide total
    failure on PED.
    """
    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )

    rows = [
        {
            "case_id": p["case_id"],
            "cohort": p["cohort"],
            "y_true": COHORTS.index(p["cohort"]),
            "y_pred": int(np.argmax(p["cls_probs"])),
            **{f"prob_{c}": float(p["cls_probs"][i]) for i, c in enumerate(COHORTS)},
        }
        for p in predictions
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {}

    y_true = df.y_true.to_numpy()
    y_pred = df.y_pred.to_numpy()
    probs = df[[f"prob_{c}" for c in COHORTS]].to_numpy()

    metrics: dict = {
        "n": int(len(df)),
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion": confusion_matrix(
            y_true, y_pred, labels=list(range(len(COHORTS)))
        ).tolist(),
        "per_class": classification_report(
            y_true, y_pred, labels=list(range(len(COHORTS))),
            target_names=list(COHORTS), output_dict=True, zero_division=0,
        ),
    }
    # One-vs-rest AUC needs at least two classes present.
    if len(np.unique(y_true)) > 1:
        try:
            metrics["macro_auc"] = float(
                roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
            )
            for i, c in enumerate(COHORTS):
                metrics[f"auc_{c}"] = float(
                    roc_auc_score((y_true == i).astype(int), probs[:, i])
                )
        except ValueError as exc:  # noqa: BLE001
            log.warning("AUC unavailable: %s", exc)
    return df, metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(
    split: str,
    seg_scores: pd.DataFrame,
    cls_metrics: dict,
    final_stage: str = "plus_ped_et_gate",
) -> str:
    lines = ["=" * 76, f"Evaluation: {split}", "=" * 76]

    if not seg_scores.empty:
        lines += ["", "-- Post-processing ablation (mean over all cases) " + "-" * 25]
        lines.append(ablation_table(seg_scores).to_string(index=False))
        lines += [
            "",
            "  Reference: BiomedMBZ moved legacy Dice +0.34 and lesion-wise +9.97 on",
            "  identical predictions. A similar divergence here means post-processing",
            "  is doing its job; a flat lesion-wise column means it is not.",
        ]

        final = seg_scores[seg_scores.stage == final_stage]
        lines += ["", f"-- Final segmentation ({final_stage}) " + "-" * 36]
        agg = aggregate(final, ["cohort", "region"])
        cols = [
            "cohort", "region", "n",
            "lw_dice_mean", "lw_dice_median", "lw_dice_ci_lo", "lw_dice_ci_hi",
            "lw_hd95_mean", "lw_hd95_median", "legacy_dice_mean",
        ]
        lines.append(agg[[c for c in cols if c in agg.columns]].round(4).to_string(index=False))
        lines += [
            "",
            "  Median is reported next to mean deliberately. The MEN winner scored mean",
            "  ET Dice 0.899 but median 0.976 (mean HD95 23.9 mm, median 0.96 mm): on a",
            "  typical case segmentation is near-perfect and the mean is dragged by a few",
            "  catastrophic cases. If median >= 0.95 and mean ~ 0.85, we are in the same",
            "  regime as the winner and the residual loss is concentrated in pathological",
            "  cases -- recognize that state and stop optimizing.",
        ]
        ped = agg[agg.cohort == "PED"]
        if not ped.empty:
            n_ped = int(ped["n"].iloc[0])
            lines += [
                "",
                f"  PED n={n_ped}: CIs above are mandatory, not decoration. Treat any",
                "  difference under 0.05 Dice as noise -- the official PED test set was",
                "  n=24 and the top four teams were indistinguishable (p=0.10-0.45).",
            ]

    if cls_metrics:
        lines += ["", "-- Classification " + "-" * 57]
        lines.append(f"n                 : {cls_metrics['n']}")
        lines.append(f"accuracy          : {cls_metrics['accuracy']:.4f}")
        lines.append(f"balanced accuracy : {cls_metrics['balanced_accuracy']:.4f}")
        lines.append(f"macro F1          : {cls_metrics['macro_f1']:.4f}")
        if "macro_auc" in cls_metrics:
            lines.append(f"macro AUC         : {cls_metrics['macro_auc']:.4f}")
        lines.append(f"confusion (rows=true {list(COHORTS)}):")
        for name, row in zip(COHORTS, cls_metrics["confusion"], strict=True):
            lines.append(f"  {name}: {row}")
        lines += [
            "",
            "  READ THIS WITH THE CONFOUND DIAGNOSTICS. In pooled BraTS 2023 the",
            "  tumour-type label IS the cohort folder: different consortia, scanners,",
            "  protocols, and patient populations (PED is defined by age). A near-perfect",
            "  accuracy here is the EXPECTED outcome and the LEAST interesting result.",
            "  Report this head as auxiliary/regularizing with an explicit caveat, never",
            "  as a clinical tumour-typing claim. See brats/confound/.",
        ]
        if split == "official_val":
            lines += [
                "",
                "  Note: on official_val the cohort label comes from the source archive,",
                "  not from seg.nii.gz, so these classification numbers are valid on all",
                "  405 never-trained cases. Segmentation here is visual-only (no GT).",
            ]

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/segresnet_base.yaml")
    ap.add_argument("--data-config", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test", "official_val"])
    ap.add_argument("--tta", default="none", choices=["none", "eight_flip"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--i-understand-this-opens-the-test-split",
        action="store_true",
        help="required for --split test; the test split may be opened only once",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )

    if args.split == "test" and not getattr(
        args, "i_understand_this_opens_the_test_split"
    ):
        raise SystemExit(
            "Refusing to evaluate on the test split without the explicit flag.\n"
            "The test split is opened EXACTLY ONCE, after all hyperparameters and "
            "post-processing thresholds are frozen on validation. Every peek "
            "invalidates it.\nRe-run with "
            "--i-understand-this-opens-the-test-split if that is genuinely the case."
        )

    from brats.data.transforms import CachedBratsDataset, load_records, val_transforms
    from brats.inference import InferenceConfig, eight_flip_axes, predict_dataset
    from brats.models.multitask import MultiTaskConfig, build_model
    from brats.train import CheckpointManager, TrainConfig

    cfg = TrainConfig.from_yaml(args.config)
    data_cfg = DataConfig.load(args.data_config) if args.data_config else DataConfig.load()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) if args.out_dir else (
        data_cfg.reports_dir / f"eval_{args.split}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(
        MultiTaskConfig(pooling=cfg.pooling, warmup_epochs=cfg.cls_warmup_epochs)  # type: ignore[arg-type]
    ).to(device)
    CheckpointManager(cfg).load(model, path=Path(args.checkpoint))

    labeled = args.split != "official_val"
    ds = CachedBratsDataset(
        load_records(data_cfg, args.split),
        transform=val_transforms(labeled),
        load_label=labeled,
    )

    inf_cfg = InferenceConfig(
        roi_size=cfg.patch_size,
        tta_flip_axes=eight_flip_axes() if args.tta == "eight_flip" else (),
    )
    log.info("predicting %d cases (tta=%s)", len(ds), args.tta)
    predictions = list(
        predict_dataset(model, ds, cfg=inf_cfg, device=device, limit=args.limit)
    )

    seg_scores = (
        evaluate_segmentation(predictions) if labeled else pd.DataFrame()
    )
    cls_df, cls_metrics = evaluate_classification(predictions)

    if not seg_scores.empty:
        seg_scores.to_csv(out_dir / "segmentation_scores.csv", index=False)
        aggregate(seg_scores[seg_scores.stage == "plus_ped_et_gate"]).to_csv(
            out_dir / "segmentation_summary.csv", index=False
        )
    if not cls_df.empty:
        cls_df.to_csv(out_dir / "classification_predictions.csv", index=False)
        (out_dir / "classification_metrics.json").write_text(
            json.dumps(cls_metrics, indent=2), encoding="utf-8"
        )

    text = format_report(args.split, seg_scores, cls_metrics)
    print(text)
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    log.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
