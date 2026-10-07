"""The preprocessing crop must never drop brain or label voxels.

Regression for: the crop target was capped at ``(155 // 16) * 16 = 144`` along z, so any
brain whose margin-padded extent exceeded 144 slices lost its top slices -- image and
label alike, with no error and no way to see it downstream.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nibabel")
pytest.importorskip("pandas")

import nibabel as nib  # noqa: E402

from brats.constants import SEQUENCES  # noqa: E402
from brats.data.preprocess import (  # noqa: E402
    brain_bbox,
    crop_pad,
    preprocess_case,
    verify_roundtrip,
)
from brats.postprocess import paste_into_original  # noqa: E402

MARGIN, DIVISOR = 8, 16


def _mask(shape, lo, hi):
    m = np.zeros(shape, bool)
    m[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = True
    return m


@pytest.mark.parametrize("z_lo", [0, 1, 3, 10])
@pytest.mark.parametrize("z_len", [60, 120, 136, 137, 144, 145, 150, 155])
def test_bbox_covers_the_whole_mask_along_z(z_lo, z_len):
    z_hi = min(z_lo + z_len, 155)
    mask = _mask((240, 240, 155), (40, 30, z_lo), (200, 210, z_hi))

    box = brain_bbox(mask, MARGIN, DIVISOR)

    for s, (first, last) in zip(box, [(40, 199), (30, 209), (z_lo, z_hi - 1)], strict=True):
        assert s.start >= 0
        assert (s.stop - s.start) % DIVISOR == 0
        assert s.start <= first and s.stop > last, f"{s} does not cover {first}..{last}"
    # The whole brain survives the crop.
    assert crop_pad(mask, box).sum() == mask.sum()


def test_bbox_pads_past_the_end_instead_of_clipping():
    mask = _mask((240, 240, 155), (40, 30, 0), (200, 210, 150))
    box = brain_bbox(mask, MARGIN, DIVISOR)
    assert (box[2].start, box[2].stop) == (0, 160)

    vol = np.ones((240, 240, 155), np.float32)
    cropped = crop_pad(vol, box)
    assert cropped.shape[2] == 160
    assert cropped[..., 155:].sum() == 0  # overhang is zero-filled
    assert cropped.sum() == vol[box[0], box[1]].sum()


def test_empty_mask_keeps_the_whole_axis():
    box = brain_bbox(np.zeros((32, 32, 155), bool), MARGIN, DIVISOR)
    assert [(s.start, s.stop) for s in box] == [(0, 32), (0, 32), (0, 160)]


def test_paste_into_original_inverts_a_padded_crop():
    rng = np.random.default_rng(0)
    mask = _mask((48, 48, 155), (4, 4, 0), (44, 44, 155))
    vol = (mask * rng.integers(1, 4, mask.shape)).astype(np.uint8)
    box = brain_bbox(mask, MARGIN, DIVISOR)
    start = np.array([s.start for s in box])
    stop = np.array([s.stop for s in box])
    assert stop[2] > 155  # the interesting case

    restored = paste_into_original(crop_pad(vol, box), (48, 48, 155), start, stop)

    np.testing.assert_array_equal(restored, vol)


def _write_case(tmp_path, shape=(48, 48, 155)):
    """A synthetic case whose brain and tumor both reach the top z slices."""
    rng = np.random.default_rng(0)
    affine = np.eye(4)
    brain = _mask(shape, (4, 4, 0), (44, 44, shape[2]))
    paths = {}
    for suf in SEQUENCES:
        arr = np.where(brain, rng.uniform(50, 500, shape), 0).astype(np.float32)
        p = tmp_path / f"case-{suf}.nii.gz"
        nib.save(nib.Nifti1Image(arr, affine), str(p))
        paths[f"path_{suf}"] = str(p)

    seg = np.zeros(shape, np.uint8)
    seg[12:36, 12:36, 130:155] = 2
    seg[18:30, 18:30, 140:155] = 1
    seg[22:26, 22:26, 148:155] = 3  # enhancing core in the slices the old cap removed
    p = tmp_path / "case-seg.nii.gz"
    nib.save(nib.Nifti1Image(seg, affine), str(p))
    paths["path_seg"] = str(p)
    return {"case_id": "case", "cohort": "GLI", **paths}, seg


def test_preprocess_case_keeps_every_label_voxel(tmp_path):
    row, seg = _write_case(tmp_path)

    res = preprocess_case(row, str(tmp_path / "cache"), MARGIN, DIVISOR)

    with np.load(res["path"]) as z:
        cached = z["seg"]
        assert cached.shape[2] == 160
        assert int(z["crop_stop"][2]) > int(z["orig_shape"][2])
        for label in (1, 2, 3):
            assert int((cached == label).sum()) == int((seg == label).sum())

    check = verify_roundtrip(Path(res["path"]), row)
    assert check["crop_lossless"]
    assert check["seg_lossless"]
    assert check["within_quant_step"]
    assert check["mask_consistent"]
