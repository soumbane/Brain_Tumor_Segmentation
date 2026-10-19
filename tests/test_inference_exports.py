"""Streaming inference, canonical export geometry, and fail-closed publication."""

import gc
import json
import weakref
from dataclasses import replace

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
import torch

from brats.constants import COHORTS, REGIONS
from brats.data.preprocess import labels_to_regions
from brats.evaluate import (
    _make_publisher,
    _stage_config,
    evaluate_segmentation,
    run_evaluation,
)
from brats.inference import (
    inference_config_from_checkpoint,
    predict_dataset,
    restore_channels,
    validate_geometry,
)
from brats.metrics.lesionwise import lesionwise_case
from brats.postprocess import PostProcessConfig


NO_FILTER = PostProcessConfig(
    thresholds={region: 0.5 for region in REGIONS}, use_joint_filter=False,
    min_size={region: 0 for region in REGIONS}, apply_ped_et_gate=False,
)


def prediction(case_id="case-1", cohort="MEN", labeled=True):
    # Nonzero XY crop offsets; the final two z slices are padding, not anatomy.
    start = np.array([2, 3, 0])
    stop = np.array([12, 13, 12])
    shape = np.array([16, 16, 10])
    probs = np.zeros((3, 10, 10, 12), dtype=np.float32)
    probs[:, 1:7, 1:7, 4:10] = 0.9
    probs[:, 8:, 8:, 10:] = 1.0  # padding-only false positive
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    affine[:3, 3] = [13, -9, 5]
    pred = {
        "case_id": case_id, "cohort": cohort,
        "probs": probs, "cls_probs": np.eye(3, dtype=np.float32)[COHORTS.index(cohort)],
        "crop_start": start, "crop_stop": stop, "orig_shape": shape, "affine": affine,
        "image": np.ones((4, 10, 10, 12), dtype=np.float32),
    }
    if labeled:
        pred["label"] = (probs > 0.5).astype(np.float32)
        pred["label"][..., 10:] = 0
    return pred


def expected(*ids):
    return [{"case_id": case_id, "cohort": "MEN"} for case_id in ids]


def run(predictions, path, cases=None, **kwargs):
    return run_evaluation(
        predictions, expected("case-1") if cases is None else cases, path,
        split=kwargs.pop("split", "val"),
        postprocess_cfg=kwargs.pop("postprocess_cfg", NO_FILTER),
        max_per_cohort=kwargs.pop("max_per_cohort", 0), ablations=False, **kwargs,
    )


def test_original_grid_nifti_and_reloaded_scores_agree(tmp_path):
    pred = prediction()
    run(iter([pred]), tmp_path / "output")
    image = nib.load(tmp_path / "output/nifti/MEN/case-1-seg.nii.gz")
    assert image.shape == tuple(pred["orig_shape"])
    np.testing.assert_allclose(image.affine, pred["affine"])
    assert image.header.get_xyzt_units()[0] == "mm"
    labels = np.asanyarray(image.dataobj)
    assert labels.dtype == np.uint8 and set(np.unique(labels)) <= {0, 1, 2, 3}
    assert labels[3:9, 4:10, 4:10].min() == 3
    assert labels[:, :, -1].any()  # genuine edge lesion survives
    regions = dict(zip(REGIONS, labels_to_regions(labels).astype(bool), strict=True))
    gt = dict(zip(REGIONS, restore_channels(pred["label"], pred).astype(bool), strict=True))
    scores = pd.read_csv(tmp_path / "output/segmentation_scores.csv").set_index("region")
    for region, result in lesionwise_case(gt, regions, cohort="MEN").items():
        for key, value in result.as_dict().items():
            if key != "region":
                assert scores.loc[region, key] == pytest.approx(value)
        assert scores.loc[region, "legacy_dice"] == 1
    assert set(scores.stage) == {"final_export"}
    assert set(scores.hd95_method) == {"approximate_voxel_surface"}


def test_all_brats2023_labels_survive_export(tmp_path):
    pred = prediction()
    pred["probs"][:] = 0
    pred["probs"][2, 1, 1, 1] = 1  # ED
    pred["probs"][1:, 2, 2, 2] = 1  # NCR
    pred["probs"][:, 3, 3, 3] = 1  # ET
    pred["label"] = pred["probs"].copy()
    run([pred], tmp_path / "out")
    labels = nib.load(tmp_path / "out/nifti/MEN/case-1-seg.nii.gz").get_fdata()
    assert set(np.unique(labels)) == {0, 1, 2, 3}
    assert labels[3, 4, 1] == 2 and labels[4, 5, 2] == 1 and labels[5, 6, 3] == 3


def test_padding_cannot_inflate_connected_component_before_filter(tmp_path):
    pred = prediction()
    pred["probs"][:] = 0
    pred["label"][:] = 0
    # In-grid component: four voxels. Adjacent padding: four more. A size-six
    # filter must discard it; postprocessing before removing padding keeps it.
    pred["probs"][:, :2, :2, 9:11] = 1
    cfg = replace(NO_FILTER, min_size={region: 6 for region in REGIONS})
    run([pred], tmp_path / "output", postprocess_cfg=cfg)
    labels = nib.load(tmp_path / "output/nifti/MEN/case-1-seg.nii.gz").get_fdata()
    assert not labels.any()


def test_final_config_never_forces_ablated_flags_and_scores_serialized_nesting(tmp_path):
    pred = prediction()
    pred["probs"][1:] = 0  # impossible un-nested probabilities
    cfg = replace(NO_FILTER, enforce_nesting=False)
    assert not _stage_config("final_export", cfg).use_joint_filter
    assert not _stage_config("final_export", cfg).apply_ped_et_gate
    assert _stage_config("ablation_joint_and_ped_gate", cfg).apply_ped_et_gate
    run([pred], tmp_path / "output", postprocess_cfg=cfg)
    scores = pd.read_csv(tmp_path / "output/segmentation_scores.csv")
    assert (scores.legacy_dice == 1).all()  # map serialization restores TC/WT
    meta = json.loads((tmp_path / "output/provenance.json").read_text())
    assert meta["postprocess"]["enforce_nesting"] is False
    assert meta["postprocess"]["use_joint_filter"] is False


@pytest.mark.parametrize("broken", ["missing", "nan", "singular", "spacing", "shear", "extent"])
def test_rejects_invalid_affine_and_geometry(broken):
    pred = prediction()
    if broken == "missing":
        del pred["affine"]
    elif broken == "nan":
        pred["affine"][0, 0] = np.nan
    elif broken == "singular":
        pred["affine"][0, 0] = 0
    elif broken == "spacing":
        pred["affine"][0, 0] = 2
    elif broken == "shear":
        pred["affine"][0, 1] = 0.1
    else:
        pred["crop_stop"][0] -= 1
    with pytest.raises(ValueError):
        validate_geometry(pred)


def test_negative_crop_offsets_discard_leading_padding():
    pred = prediction()
    pred["crop_start"] = np.array([-2, -3, 0])
    pred["crop_stop"] = np.array([8, 7, 12])
    validate_geometry(pred)
    restored = restore_channels(pred["probs"], pred)
    np.testing.assert_array_equal(restored[:, :8, :7], pred["probs"][:, 2:, 3:, :10])
    assert not restored[:, 8:].any() and not restored[:, :, 7:].any()


def test_checkpoint_patch_and_precision_override_yaml_only_explicitly():
    state = {"config": {"patch_size": (32, 48, 64), "amp_dtype": "float32"}}
    fallback = {"patch_size": (16, 16, 16), "amp_dtype": "float16"}
    cfg = inference_config_from_checkpoint(state, fallback=fallback)
    assert cfg.roi_size == (32, 48, 64) and cfg.amp_dtype is torch.float32
    cfg = inference_config_from_checkpoint(state, fallback=fallback, override=True)
    assert cfg.roi_size == (16, 16, 16) and cfg.amp_dtype is torch.float16
    assert cfg.extra["training_settings_source"] == "explicit_override"
    with pytest.raises(ValueError, match="explicit legacy"):
        inference_config_from_checkpoint({})
    cfg = inference_config_from_checkpoint({}, fallback=fallback)
    assert cfg.extra["training_settings_source"] == "explicit_legacy_fallback"


@pytest.mark.parametrize("module", ["brats.evaluate", "brats.qc"])
def test_all_export_clis_gate_internal_test_before_loading(module):
    import importlib

    with pytest.raises(SystemExit, match="Refusing to open"):
        importlib.import_module(module).main(["--split", "test", "--checkpoint", "missing.pt"])


def test_api_test_gate_precedes_iteration(tmp_path):
    def never():
        raise AssertionError("test dataset must not be consumed")
        yield

    with pytest.raises(SystemExit, match="Refusing to open"):
        run(never(), tmp_path / "out", split="test")
    assert not (tmp_path / "out").exists()


def test_single_pass_releases_volume_before_requesting_next(tmp_path):
    def stream():
        for index in range(4):
            pred = prediction(f"case-{index}")
            refs = [weakref.ref(pred[key]) for key in ("probs", "label", "image")]
            yield pred
            del pred
            gc.collect()
            assert all(ref() is None for ref in refs), "previous volume retained"

    run(stream(), tmp_path / "out", cases=expected(*(f"case-{index}" for index in range(4))))


def test_predict_dataset_reads_once_and_releases_generator_locals(monkeypatch):
    import brats.inference as inference

    class Dataset:
        def __init__(self):
            self.reads = 0
            self.refs = []

        def __len__(self):
            return 3

        def __getitem__(self, index):
            gc.collect()
            assert all(ref() is None for ref in self.refs)
            self.reads += 1
            item = prediction(f"case-{index}")
            item["cohort_name"] = item["cohort"]
            self.refs = [weakref.ref(item[key]) for key in ("image", "label")]
            return item

    monkeypatch.setattr(inference, "predict_volume", lambda model, image, **kwargs: {
        "probs": image[:3].copy(), "cls_probs": np.array([0, 1, 0]), "cls_logits": np.array([0, 1, 0]),
    })
    dataset = Dataset()
    for pred in predict_dataset(None, dataset, device="cpu", include_image=True):
        assert "affine" in pred and "image" in pred
        del pred
    assert dataset.reads == 3


@pytest.mark.parametrize("mode", ["missing", "duplicate", "unexpected", "raise"])
def test_partial_or_invalid_coverage_never_completes(tmp_path, mode):
    def stream():
        yield prediction()
        if mode == "duplicate":
            yield prediction()
        elif mode == "unexpected":
            yield prediction("other")
        elif mode == "raise":
            raise RuntimeError("inference failed")

    with pytest.raises((ValueError, RuntimeError)):
        run(stream(), tmp_path / "out", cases=expected("case-1", "case-2"))
    for name in ("completion.json", "classification_metrics.json", "segmentation_summary.csv"):
        assert not (tmp_path / "out" / name).exists()


def test_duplicate_intended_cases_fail_before_output(tmp_path):
    with pytest.raises(ValueError, match="duplicates"):
        run([], tmp_path / "out", cases=expected("case-1", "case-1"))
    assert not (tmp_path / "out").exists()


def test_official_validation_classification_only(tmp_path):
    run([prediction(labeled=False)], tmp_path / "out", split="official_val")
    assert not (tmp_path / "out/segmentation_scores.csv").exists()
    assert json.loads((tmp_path / "out/classification_metrics.json").read_text())["accuracy"] == 1
    meta = json.loads((tmp_path / "out/provenance.json").read_text())
    assert meta["classification_holdout"] and not meta["segmentation_scored"]
    with pytest.raises(ValueError, match="must not contain"):
        run([prediction()], tmp_path / "bad", split="official_val")


def test_bounded_cohort_qc_is_streamed_without_reloading(tmp_path, monkeypatch):
    import brats.qc as qc

    calls = []
    def montage(image, regions, gt, **kwargs):
        calls.append(kwargs["cohort"])
        assert image.shape == (4, 16, 16, 10)
        kwargs["out_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["out_path"].write_bytes(b"png")

    monkeypatch.setattr(qc, "case_montage", montage)
    cases = [{"case_id": f"{cohort}-{index}", "cohort": cohort}
             for cohort in COHORTS for index in range(3)]
    stream = (prediction(record["case_id"], record["cohort"], labeled=False) for record in cases)
    run(stream, tmp_path / "out", cases=cases, split="official_val", max_per_cohort=1, score=False)
    assert calls == list(COHORTS)
    assert not (tmp_path / "out/classification_metrics.json").exists()


def test_stage_publication_closes_files_and_completes_last(tmp_path):
    from brats.artifacts import read_manifest, resolve_bundle

    publisher = _make_publisher(tmp_path / "stage-run")
    result = run([prediction()], tmp_path / "local", publisher=publisher)
    assert not (tmp_path / "local/nifti/MEN/case-1-seg.nii.gz").exists()
    final = resolve_bundle(tmp_path / "stage-run", kind="evaluation")
    assert final["metadata"]["status"] == "complete"
    for entry in result["files"]:
        manifest = read_manifest(tmp_path / "stage-run", tmp_path / "stage-run" / entry["manifest"])
        assert manifest["manifest_sha256"] == entry["manifest_sha256"]
    with pytest.raises(Exception, match="already exists"):
        _make_publisher(tmp_path / "stage-run")


def test_stage_partial_failure_has_no_aggregate_completion(tmp_path):
    from brats.artifacts import ArtifactError, resolve_bundle

    publisher = _make_publisher(tmp_path / "stage-run")
    with pytest.raises(ValueError, match="incomplete"):
        run([prediction()], tmp_path / "local", cases=expected("case-1", "case-2"), publisher=publisher)
    with pytest.raises(ArtifactError, match="no verified evaluation"):
        resolve_bundle(tmp_path / "stage-run", kind="evaluation", retries=1)
    assert not (tmp_path / "local/completion.json").exists()


def test_final_publication_failure_does_not_mark_local_complete(tmp_path, monkeypatch):
    publisher = _make_publisher(tmp_path / "stage-run")
    def fail(files, metadata):
        raise OSError("failed publication")
    monkeypatch.setattr(publisher, "publish_completion", fail)
    with pytest.raises(OSError, match="failed publication"):
        run([prediction()], tmp_path / "local", publisher=publisher)
    assert not (tmp_path / "local/completion.json").exists()


def test_segmentation_api_streams_original_geometry():
    scores = evaluate_segmentation(iter([prediction()]), NO_FILTER, stages=["final_export"])
    assert (scores.legacy_dice == 1).all()


def test_checkpoint_bundle_reader_records_verified_identity(tmp_path):
    from brats.artifacts import initialize_run, publish_bundle, sha256_file
    from brats.evaluate import _load_trusted_checkpoint

    root = tmp_path / "training-run"
    initialize_run(root, run_id="training-run", attempt_id="attempt-1", resume=False)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"epoch": 3, "config": {"patch_size": [16, 16, 16], "amp_dtype": "float32"}}, checkpoint)
    publish_bundle(root, {"checkpoint.pt": checkpoint},
                   {"kind": "checkpoint", "epoch": 3, "run_id": "training-run"}, attempt_id="attempt-1")
    state, provenance = _load_trusted_checkpoint(run_root=root, epoch=3)
    assert state["epoch"] == 3
    assert provenance["checkpoint_sha256"] == sha256_file(checkpoint)
    assert provenance["checkpoint_run_id"] == "training-run"
    assert provenance["checkpoint_manifest_sha256"]
    local_state, local_provenance = _load_trusted_checkpoint(checkpoint)
    assert local_state == state
    assert local_provenance["checkpoint_source"] == "trusted_local_file"


def test_cached_affine_survives_validation_transforms(tmp_path):
    pytest.importorskip("monai")
    from brats.data.transforms import CachedBratsDataset, val_transforms

    pred = prediction()
    path = tmp_path / "cache.npz"
    np.savez_compressed(
        path, img=np.ones((4, 10, 10, 12), dtype=np.uint8),
        seg=np.zeros((10, 10, 12), dtype=np.uint8),
        **{key: pred[key] for key in ("crop_start", "crop_stop", "orig_shape", "affine")},
    )
    dataset = CachedBratsDataset(
        [{"case_id": "case-1", "cohort": "MEN", "cache_path": path}],
        transform=val_transforms(True),
    )
    np.testing.assert_array_equal(dataset[0]["affine"], pred["affine"])


def test_actual_montage_renders_from_restored_mri(tmp_path):
    pytest.importorskip("matplotlib")
    run([prediction(labeled=False)], tmp_path / "out", split="official_val", max_per_cohort=1)
    image = tmp_path / "out/qc/MEN/case-1.png"
    assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
