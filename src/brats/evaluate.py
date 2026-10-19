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
import dataclasses
import json
import logging
import os
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from brats.config import DataConfig
from brats.constants import COHORTS, REGIONS
from brats.metrics.lesionwise import aggregate, lesionwise_case
from brats.metrics.ranking import ablation_table
from brats.postprocess import DEFAULT_POSTPROC_YAML, PostProcessConfig
from brats.inference import (
    canonical_label_map,
    require_test_acknowledgement,
    restore_channels,
    validate_geometry,
)

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
    "ablation_joint_filter": dict(use_joint_filter=True, apply_ped_et_gate=False),
    "ablation_joint_and_ped_gate": dict(use_joint_filter=True, apply_ped_et_gate=True),
    "final_export": {},
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
        if "label" in pred:
            validate_geometry(pred)
            probs = restore_channels(pred["probs"], pred)
            gt = _ground_truth(pred)
            rows.extend(_score_stages(pred, probs, gt, base_cfg, stages))
            del probs, gt
        del pred
    return pd.DataFrame(rows)


def _ground_truth(pred: dict) -> dict | None:
    if "label" not in pred:
        return None
    labels = np.asarray(pred["label"])
    if not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("ground truth must contain binary region masks")
    masks = restore_channels(labels.astype(bool), pred)
    return dict(zip(REGIONS, masks, strict=True))


def _score_stages(pred, probs, gt, cfg, stages, final_regions=None) -> list[dict]:
    rows = []
    for stage in stages:
        regions = final_regions if stage == "final_export" else None
        if regions is None:
            _, regions = canonical_label_map(probs, pred["cohort"], _stage_config(stage, cfg))
        scored = lesionwise_case(gt, regions, cohort=pred["cohort"])
        rows.extend({
            "stage": stage, "case_id": pred["case_id"], "cohort": pred["cohort"],
            "geometry": "original_grid_label_map", "hd95_method": "approximate_voxel_surface",
            **result.as_dict(),
        } for result in scored.values())
    return rows


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def evaluate_classification(predictions, *, compute_metrics: bool = True) -> tuple[pd.DataFrame, dict]:
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
    if df.empty or not compute_metrics:
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
    final_stage: str = "final_export",
) -> str:
    lines = [
        "=" * 76, f"Evaluation: {split}", "=" * 76,
        "Research metrics: lw_hd95 is approximate voxel-surface HD95, not the official scorer.",
        "legacy_dice is whole-volume overlap; final_export scores the exact exported masks.",
    ]

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
                "  not from seg.nii.gz. These are classification holdout results for the",
                "  cases actually processed. Segmentation is visual-only (no GT).",
            ]

    return "\n".join(lines)


def _case_key(record: dict) -> tuple[str, str]:
    cohort, case_id = str(record["cohort"]), str(record["case_id"])
    if cohort not in COHORTS or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case_id):
        raise ValueError("invalid cohort or unsafe case identifier")
    return cohort, case_id


def _json_write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


class _StagePublisher:
    """Per-file verified bundles plus a final aggregate index; one authorized writer."""

    def __init__(self, root: Path):
        import uuid
        from brats.artifacts import ArtifactError, safe_id

        self.root = root.resolve()
        if self.root.exists():
            raise ArtifactError("fresh export prefix already exists; choose a new run ID")
        self.attempt_id = uuid.uuid4().hex
        self.identity = {"run_id": safe_id(self.root.name), "attempt_id": self.attempt_id}
        # mkdir is a collision guard, not a stage lock; the operator supplies one writer.
        self.root.mkdir(parents=True, exist_ok=False)
        _json_write(self.root / "export-run.json", self.identity)

    def publish_file(self, source: Path, relative: str) -> dict:
        from brats.artifacts import publish_bundle, read_manifest

        manifest_path = publish_bundle(
            self.root, {relative: source}, {"kind": "inference_file"}, attempt_id=self.attempt_id,
        )
        manifest = read_manifest(self.root, manifest_path)
        return {
            "name": relative, **manifest["files"][relative],
            "manifest": manifest_path.relative_to(self.root).as_posix(),
            "manifest_sha256": manifest["manifest_sha256"],
        }

    def publish_completion(self, files: list[dict], metadata: dict) -> Path:
        import tempfile
        from brats.artifacts import ArtifactError, publish_bundle, read_manifest, safe_path

        for entry in files:
            manifest = read_manifest(self.root, safe_path(self.root, entry["manifest"]))
            if manifest["manifest_sha256"] != entry["manifest_sha256"]:
                raise ArtifactError("inference file completion changed before aggregate publication")
            expected = {key: entry[key] for key in ("path", "size", "sha256")}
            if manifest["files"].get(entry["name"]) != expected:
                raise ArtifactError("inference file reference mismatch")
        with tempfile.TemporaryDirectory(prefix="brats-evaluation-") as temporary:
            index = Path(temporary) / "completion.json"
            _json_write(index, {"status": "complete", "metadata": metadata, "files": files})
            return publish_bundle(
                self.root, {"completion.json": index},
                {"kind": "evaluation", "status": "complete", "n_cases": metadata["n_cases"],
                 "run_id": self.identity["run_id"], "attempt_id": self.attempt_id},
                attempt_id=self.attempt_id,
            )


def _make_publisher(root: Path) -> _StagePublisher:
    return _StagePublisher(root)


def run_evaluation(
    predictions,
    expected_cases,
    out_dir: Path,
    *,
    split: str,
    postprocess_cfg: PostProcessConfig,
    inference_cfg=None,
    provenance: dict | None = None,
    max_per_cohort: int = 8,
    score: bool = True,
    ablations: bool = True,
    test_acknowledged: bool = False,
    publisher=None,
) -> dict:
    """Consume each volume once; complete only after exact intended case coverage.

    Local output must be a new directory. Published runs use closed immutable files;
    completion metadata is written last. No MRI or identifiers go to online trackers.
    """
    import nibabel as nib
    from brats.qc import case_montage

    require_test_acknowledgement(split, test_acknowledged)
    if split not in {"val", "test", "official_val"}:
        raise ValueError(f"unsupported evaluation split {split!r}")
    if max_per_cohort < 0:
        raise ValueError("max_per_cohort must be nonnegative")
    intended = [_case_key(record) for record in expected_cases]
    expected = set(intended)
    if not intended or len(expected) != len(intended):
        raise ValueError("intended case set must be nonempty with no duplicates")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    files = []

    def publish(path: Path, *, discard_local: bool = False):
        relative = path.relative_to(out_dir).as_posix()
        if publisher is not None:
            descriptor = publisher.publish_file(path, relative)
            if discard_local:
                path.unlink()
        else:
            from brats.artifacts import sha256_file
            descriptor = {"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)}
        files.append(descriptor)

    seen = set()
    cls_records = []
    seg_rows = []
    qc_counts = {cohort: 0 for cohort in COHORTS}
    stages = list(ABLATION_STAGES) if ablations else ["final_export"]
    for pred in predictions:
        key = _case_key(pred)
        if key not in expected or key in seen:
            raise ValueError(f"unexpected or duplicate prediction: {key}")
        if split == "official_val" and "label" in pred:
            raise ValueError("official_val must not contain segmentation ground truth")
        if split != "official_val" and "label" not in pred:
            raise ValueError(f"labeled split missing ground truth: {key}")
        _, _, _, affine = validate_geometry(pred)
        probs = restore_channels(pred["probs"], pred)
        label_map, regions = canonical_label_map(probs, key[0], postprocess_cfg)
        gt = _ground_truth(pred)
        path = out_dir / "nifti" / key[0] / f"{key[1]}-seg.nii.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        nifti = nib.Nifti1Image(label_map, affine)
        nifti.header.set_xyzt_units("mm")
        nifti.set_sform(affine, code=1)
        nifti.set_qform(affine, code=1)
        nib.save(nifti, path)
        publish(path, discard_local=True)
        del nifti, label_map

        if score and gt is not None:
            seg_rows.extend(_score_stages(pred, probs, gt, postprocess_cfg, stages, regions))
        cls_probs = np.asarray(pred["cls_probs"], dtype=np.float64)
        if cls_probs.shape != (len(COHORTS),) or not np.isfinite(cls_probs).all():
            raise ValueError("classification probabilities must be a finite cohort vector")
        if np.any(cls_probs < 0) or not np.isclose(cls_probs.sum(), 1, atol=1e-5):
            raise ValueError("classification probabilities must sum to one")
        cls_records.append({"case_id": key[1], "cohort": key[0], "cls_probs": cls_probs.copy()})

        if qc_counts[key[0]] < max_per_cohort:
            if "image" not in pred:
                raise ValueError("QC requires the MRI image, not a probability-map substitute")
            image = restore_channels(pred["image"], pred)
            path = out_dir / "qc" / key[0] / f"{key[1]}.png"
            case_montage(image, regions, gt, case_id=key[1], cohort=key[0], out_path=path)
            publish(path, discard_local=True)
            qc_counts[key[0]] += 1
            del image
        seen.add(key)
        del pred, probs, regions, gt, cls_probs

    if seen != expected:
        raise ValueError(f"incomplete inference: missing {len(expected - seen)} intended cases")
    seg_scores = pd.DataFrame(seg_rows)
    cls_df, cls_metrics = evaluate_classification(cls_records, compute_metrics=score)
    effective_inference = dataclasses.asdict(inference_cfg) if inference_cfg is not None else None
    if effective_inference is not None:
        effective_inference["amp_dtype"] = str(inference_cfg.amp_dtype)
    metadata = {
        **(provenance or {}),
        "split": split, "n_cases": len(seen),
        "intended_cases": [{"cohort": cohort, "case_id": case_id} for cohort, case_id in intended],
        "inference": effective_inference,
        "postprocess": dataclasses.asdict(postprocess_cfg),
        "ablation_configs": {stage: dataclasses.asdict(_stage_config(stage, postprocess_cfg))
                             for stage in stages} if score and split != "official_val" else {},
        "final_stage": "final_export", "label_encoding": [0, 1, 2, 3],
        "geometry": "original_grid_label_map", "hd95_method": "approximate_voxel_surface",
        "classification_target": "source_cohort_confounded_not_clinical_tumor_type",
        "classification_holdout": split == "official_val",
        "segmentation_scored": bool(score and split != "official_val"),
        "qc_selection": "first_n_per_cohort_in_split_order", "qc_counts": qc_counts,
        "metrics_enabled": score,
    }
    _json_write(out_dir / "provenance.json", metadata)
    _json_write(out_dir / "postprocess_config.json", metadata["postprocess"])
    cls_df.to_csv(out_dir / "classification_predictions.csv", index=False)
    if score:
        _json_write(out_dir / "classification_metrics.json", cls_metrics)
    if not seg_scores.empty:
        seg_scores.to_csv(out_dir / "segmentation_scores.csv", index=False)
        aggregate(seg_scores[seg_scores.stage == "final_export"]).to_csv(
            out_dir / "segmentation_summary.csv", index=False,
        )
    text = format_report(split, seg_scores, cls_metrics)
    if not score:
        text += "\nQC-only run: metrics were not scored."
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    for path in sorted(out_dir.iterdir()):
        if path.is_file():
            publish(path)
    completion = {"status": "complete", "metadata": metadata, "files": files}
    if publisher is not None:
        publisher.publish_completion(files, metadata)
    _json_write(out_dir / "completion.json", completion)
    return completion


def _load_trusted_checkpoint(path=None, *, run_root=None, epoch=None) -> tuple[dict, dict]:
    """Resolve a verified project bundle, or load an explicitly supplied local file.

    Hashes validate integrity, not provenance. Both inputs must belong to a trusted
    project run because training state uses PyTorch's pickle-capable loader.
    """
    import tempfile
    from brats.artifacts import materialize_bundle, resolve_bundle, sha256_file

    if run_root is not None:
        manifest = resolve_bundle(run_root, kind="checkpoint", epoch=epoch)
        with tempfile.TemporaryDirectory(prefix="brats-inference-checkpoint-") as temporary:
            files = materialize_bundle(run_root, manifest, temporary)
            checkpoint = files["checkpoint.pt"]
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        return state, {
            "checkpoint_sha256": manifest["files"]["checkpoint.pt"]["sha256"],
            "checkpoint_manifest_sha256": manifest["manifest_sha256"],
            "checkpoint_run_id": manifest["metadata"].get("run_id"),
            "checkpoint_source": "verified_project_bundle",
        }
    if epoch is not None:
        raise ValueError("--checkpoint-epoch requires --checkpoint-run-dir")
    checkpoint = Path(path)
    checkpoint_hash = sha256_file(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if sha256_file(checkpoint) != checkpoint_hash:
        raise ValueError("checkpoint changed while loading")
    return state, {"checkpoint_sha256": checkpoint_hash, "checkpoint_source": "trusted_local_file"}


def main(argv: list[str] | None = None, *, qc_only: bool = False) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/segresnet_base.yaml")
    ap.add_argument("--data-config", default=None)
    checkpoint_group = ap.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", help="trusted local project checkpoint, never an untrusted pickle")
    checkpoint_group.add_argument("--checkpoint-run-dir", help="trusted training run prefix; resolves a verified bundle")
    ap.add_argument("--checkpoint-epoch", type=int, help="select a completed epoch from --checkpoint-run-dir")
    ap.add_argument("--allow-legacy-config", action="store_true",
                    help="explicitly allow --config to fill missing checkpoint settings")
    ap.add_argument("--override-inference-config", action="store_true",
                    help="explicitly use --config patch/precision instead of checkpoint values")
    ap.add_argument("--split", default="official_val" if qc_only else "val",
                    choices=["val", "test", "official_val"])
    ap.add_argument(
        "--postproc-config", default=None,
        help="post-processing / inference YAML (default: configs/postproc.yaml)",
    )
    ap.add_argument(
        "--tta", default=None, choices=["none", "eight_flip"],
        help="override inference.tta from the post-processing YAML",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=None, help="new unique local output directory")
    ap.add_argument("--publish-dir", default=None, help="new run prefix on the OUTPUT_STAGE mount")
    ap.add_argument("--max-per-cohort", type=int, default=8)
    ap.add_argument("--no-ablations", action="store_true", help="score only the exact exported mask")
    ap.add_argument(
        "--i-understand-this-opens-the-test-split",
        action="store_true",
        help="required for --split test; the test split may be opened only once",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )

    require_test_acknowledgement(args.split, args.i_understand_this_opens_the_test_split)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise SystemExit("inference/export requires one process; do not use multi-rank torchrun")
    if args.limit is not None and args.limit <= 0:
        ap.error("--limit must be positive")
    if args.max_per_cohort < 0:
        ap.error("--max-per-cohort must be nonnegative")

    from brats.data.transforms import CachedBratsDataset, load_records, val_transforms
    from brats.inference import inference_config_from_checkpoint, predict_dataset
    from brats.models.multitask import MultiTaskConfig, build_model, config_from_checkpoint
    from brats.train import TrainConfig

    state, checkpoint_provenance = _load_trusted_checkpoint(
        args.checkpoint, run_root=args.checkpoint_run_dir, epoch=args.checkpoint_epoch,
    )
    fallback_cfg = (TrainConfig.from_yaml(args.config)
                    if args.allow_legacy_config or args.override_inference_config else None)
    if not state.get("model_config") and not args.allow_legacy_config:
        raise ValueError("legacy architecture requires --allow-legacy-config")
    inf_cfg = inference_config_from_checkpoint(
        state, args.postproc_config,
        fallback=dataclasses.asdict(fallback_cfg) if fallback_cfg else None,
        override=args.override_inference_config, tta=args.tta,
    )
    fallback_model = (MultiTaskConfig(pooling=fallback_cfg.pooling, warmup_epochs=fallback_cfg.cls_warmup_epochs)
                      if fallback_cfg is not None else None)
    model_cfg = config_from_checkpoint(state, fallback_model)
    model = build_model(model_cfg)
    model.load_state_dict(state["model"])
    provenance = {
        **checkpoint_provenance,
        "checkpoint_epoch": state.get("epoch"),
        "checkpoint_data_identity": state.get("identity"),
        "model_config": dataclasses.asdict(model_cfg),
        "limit": args.limit,
        "precision_on_device": "float32" if not torch.cuda.is_available() else str(inf_cfg.amp_dtype),
    }
    del state
    data_cfg = DataConfig.load(args.data_config) if args.data_config else DataConfig.load()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    out_dir = Path(args.out_dir) if args.out_dir else (
        data_cfg.reports_dir / f"{'qc' if qc_only else 'eval'}_{args.split}"
    )
    if out_dir.exists():
        raise FileExistsError(f"choose a new --out-dir; refusing to mix runs at {out_dir}")
    records = load_records(data_cfg, args.split)
    if args.limit is not None:
        records = records[:args.limit]
    labeled = args.split != "official_val"
    dataset = CachedBratsDataset(records, transform=val_transforms(labeled), load_label=labeled)
    pp_cfg = PostProcessConfig.from_yaml(args.postproc_config or DEFAULT_POSTPROC_YAML)
    publisher = None
    if args.publish_dir:
        publish_dir = Path(args.publish_dir).resolve()
        local_dir = out_dir.resolve()
        if local_dir.is_relative_to(publish_dir) or publish_dir.is_relative_to(local_dir):
            raise ValueError("--out-dir local workspace must be separate from --publish-dir")
        publisher = _make_publisher(publish_dir)
        provenance["run_id"] = publisher.identity["run_id"]
        provenance["attempt_id"] = publisher.attempt_id
    predictions = predict_dataset(
        model, dataset, cfg=inf_cfg, device=device, include_image=args.max_per_cohort > 0,
    )
    run_evaluation(
        predictions, records, out_dir, split=args.split, postprocess_cfg=pp_cfg,
        inference_cfg=inf_cfg, provenance=provenance, max_per_cohort=args.max_per_cohort,
        score=not qc_only, ablations=not args.no_ablations,
        test_acknowledged=args.i_understand_this_opens_the_test_split, publisher=publisher,
    )
    log.info("completed %d cases; wrote %s", len(records), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
