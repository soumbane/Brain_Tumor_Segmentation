"""The data-validation gates, exercised with real (synthetic) NIfTI files.

The manifest is the project's defence against its most dangerous silent failure -- a
dataset whose enhancing tumor is label 4 (BraTS <= 2021) instead of 3 -- and had no test.
"""

import zipfile

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from brats.constants import SEQUENCES
from brats.data import extract, manifest

SHAPE = (240, 240, 155)  # the nominal BraTS grid; cases are mostly empty so files stay tiny


def _write_case(root, cohort, case_id, split_source="train", seg_values=(1, 2, 3), shape=SHAPE,
                affine_shift=0.0, drop=None):
    d = root / case_id
    d.mkdir(parents=True)
    brain = np.zeros(shape, np.float32)
    brain[40:200, 40:200, 5:150] = 100.0
    for suf in SEQUENCES:
        if suf == drop:
            continue
        aff = np.eye(4)
        if suf == "t2f":
            aff[0, 3] = affine_shift
        nib.save(nib.Nifti1Image(brain, aff), str(d / f"{case_id}-{suf}.nii.gz"))
    if split_source == "train":
        seg = np.zeros(shape, np.uint8)
        for i, v in enumerate(seg_values):
            seg[80 + 10 * i : 90 + 10 * i, 80:90, 60:70] = v
        nib.save(nib.Nifti1Image(seg, np.eye(4)), str(d / f"{case_id}-seg.nii.gz"))
    return d


def _inspect(d, split_source="train"):
    return manifest.inspect_case(str(d), "GLI", split_source)


# -- the label-3-vs-4 gate ------------------------------------------------------------------


def test_a_clean_case_passes(tmp_path):
    rec = _inspect(_write_case(tmp_path, "GLI", "BraTS-GLI-00001-000"))
    assert rec["errors"] == [] and rec["labels_valid"] and not rec["has_label_4"]
    assert rec["vox_et"] == 1000 and rec["vox_ncr"] == 1000 and rec["vox_ed"] == 1000


def test_label_4_is_a_hard_error(tmp_path):
    rec = _inspect(_write_case(tmp_path, "GLI", "BraTS-GLI-00002-000", seg_values=(1, 2, 4)))
    assert rec["has_label_4"] and "LABEL_4_PRESENT" in rec["errors"]


def test_any_other_unexpected_label_is_flagged(tmp_path):
    rec = _inspect(_write_case(tmp_path, "GLI", "BraTS-GLI-00003-000", seg_values=(1, 2, 5)))
    assert not rec["labels_valid"] and any(e.startswith("unexpected_labels") for e in rec["errors"])


def test_summarize_turns_label_4_into_a_failing_gate(tmp_path):
    rows = [
        _inspect(_write_case(tmp_path, "GLI", "BraTS-GLI-00001-000")),
        _inspect(_write_case(tmp_path, "GLI", "BraTS-GLI-00002-000", seg_values=(1, 2, 4))),
    ]
    df = pd.DataFrame(rows)
    df["errors"] = df["errors"].apply(lambda e: ";".join(e))
    df["ok"] = df["errors"] == ""

    _, hard = manifest.summarize(df)

    assert any("label 4" in h for h in hard), hard


# -- other integrity checks -----------------------------------------------------------------


def test_missing_sequence_is_reported(tmp_path):
    rec = _inspect(_write_case(tmp_path, "GLI", "BraTS-GLI-00004-000", drop="t1c"))
    assert "missing:t1c" in rec["errors"]


def test_affine_mismatch_between_sequences_is_reported(tmp_path):
    rec = _inspect(_write_case(tmp_path, "GLI", "BraTS-GLI-00005-000", affine_shift=3.0))
    assert "affine_mismatch" in rec["errors"]


def test_non_nominal_shape_is_logged_not_resampled(tmp_path):
    rec = _inspect(_write_case(tmp_path, "GLI", "BraTS-GLI-00006-000", shape=(200, 200, 120)))
    assert any(e.startswith("nonnominal_shape") for e in rec["errors"])


def test_validation_cases_have_no_segmentation_and_no_label_checks(tmp_path):
    rec = _inspect(_write_case(tmp_path, "GLI", "BraTS-GLI-00007-000", split_source="val"), "val")
    assert rec["errors"] == [] and rec["path_seg"] == "" and not rec["has_label_4"]


def test_patient_and_timepoint_are_parsed(tmp_path):
    rec = _inspect(_write_case(tmp_path, "GLI", "BraTS-GLI-00324-007"))
    assert (rec["patient_id"], rec["timepoint"]) == ("BraTS-GLI-00324", "007")
    assert manifest.patient_id_of("BraTS-PED-00001-000") == "BraTS-PED-00001"


def test_empty_et_and_et_wt_ratio_are_recorded(tmp_path):
    rec = _inspect(_write_case(tmp_path, "GLI", "BraTS-GLI-00008-000", seg_values=(1, 2)))
    assert rec["empty_et"] and rec["vox_et"] == 0 and rec["et_wt_ratio"] == 0.0
    rec = _inspect(_write_case(tmp_path, "GLI", "BraTS-GLI-00009-000", seg_values=(1, 2, 3)))
    assert rec["et_wt_ratio"] == pytest.approx(1000 / 3000)


# -- extraction safety ---------------------------------------------------------------------


@pytest.mark.parametrize("marker", ["OneDrive - Corewell Health", "Dropbox", "Google Drive"])
def test_extraction_refuses_cloud_synced_paths(tmp_path, marker):
    with pytest.raises(SystemExit):
        extract._refuse_synced_path(tmp_path / marker / "brats")
    extract._refuse_synced_path(tmp_path / "plain" / "brats")  # a normal path is fine


def test_extraction_blocks_path_traversal(tmp_path):
    from brats.config import DataConfig
    import dataclasses

    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escaped.txt", "x")
    cfg = dataclasses.replace(
        DataConfig.load(),
        archive_root=tmp_path,
        extract_root=tmp_path / "out",
        archives={"GLI": {"train": "evil.zip", "val": "evil.zip"}},
    )

    with pytest.raises(SystemExit):
        extract.extract_one(cfg, "GLI", "train", force=True)
    assert not (tmp_path / "escaped.txt").exists()


def test_extraction_check_is_strict_about_sibling_directories_sharing_a_prefix(tmp_path):
    """Members extract into <root>/GLI/train. `../train2/x` resolves to <root>/GLI/train2/x,
    whose string starts with '<root>/GLI/train' -- a bare `startswith` waves it through (and
    zipfile then silently rewrites it to <root>/GLI/train/train2/x). It must be rejected."""
    from brats.config import DataConfig
    import dataclasses

    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../train2/escaped.txt", "x")
    cfg = dataclasses.replace(
        DataConfig.load(),
        archive_root=tmp_path,
        extract_root=tmp_path / "out",
        archives={"GLI": {"train": "evil.zip", "val": "evil.zip"}},
    )

    with pytest.raises(SystemExit):
        extract.extract_one(cfg, "GLI", "train", force=True)
    assert not list((tmp_path / "out").rglob("escaped.txt")), "the member was extracted anyway"
