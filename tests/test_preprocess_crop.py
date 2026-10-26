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
    lossless_bbox,
    roundtrip_passed,
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


def _outside_label_case(tmp_path, coordinate=(47, 20, 150)):
    shape = (48, 48, 155)
    brain = _mask(shape, (12, 12, 20), (28, 28, 90))
    paths = {}
    rng = np.random.default_rng(11)
    for sequence in SEQUENCES:
        values = np.where(brain, rng.uniform(10, 100, shape), 0).astype(np.float32)
        path = tmp_path / f'outside-{sequence}.nii.gz'
        nib.save(nib.Nifti1Image(values, np.eye(4)), path)
        paths[f'path_{sequence}'] = str(path)
    seg = np.zeros(shape, np.uint8)
    seg[20, 20, 50] = 2
    seg[coordinate] = 3
    path = tmp_path / 'outside-seg.nii.gz'
    nib.save(nib.Nifti1Image(seg, np.eye(4)), path)
    paths['path_seg'] = str(path)
    return {'case_id': 'outside', 'cohort': 'MEN', **paths}, brain, seg


@pytest.mark.parametrize('coordinate', [(47, 20, 150), (0, 47, 0), (20, 0, 154)])
def test_annotation_outside_image_crop_is_preserved_without_altering_mask(tmp_path, coordinate):
    row, brain, seg = _outside_label_case(tmp_path, coordinate)
    result = preprocess_case(row, str(tmp_path / 'cache'), MARGIN, DIVISOR)
    with np.load(result['path'], allow_pickle=False) as cache:
        restored = paste_into_original(cache['seg'], seg.shape, cache['crop_start'], cache['crop_stop'])
        np.testing.assert_array_equal(restored, seg)
        restored_mask = paste_into_original(cache['img'][0] != 0, brain.shape, cache['crop_start'], cache['crop_stop'])
        np.testing.assert_array_equal(restored_mask, brain)
        assert not restored_mask[coordinate]
    check = verify_roundtrip(Path(result['path']), row, margin=MARGIN, divisor=DIVISOR)
    assert roundtrip_passed(check)
    assert check['label_voxels_outside_mri_mask'] == 1


def test_ordinary_and_unlabeled_crops_remain_image_only():
    brain = _mask((48, 48, 155), (12, 12, 20), (28, 28, 90))
    seg = np.zeros(brain.shape, np.uint8)
    seg[20, 20, 50] = 3
    assert lossless_bbox(brain, seg, MARGIN, DIVISOR) == brain_bbox(brain, MARGIN, DIVISOR)
    assert lossless_bbox(brain, None, MARGIN, DIVISOR) == brain_bbox(brain, MARGIN, DIVISOR)


@pytest.mark.parametrize('corruption', ['affine', 'normalization', 'mask_move', 'label_move', 'missing_label'])
def test_source_roundtrip_rejects_count_preserving_corruption(tmp_path, corruption):
    row, _, _ = _outside_label_case(tmp_path)
    result = preprocess_case(row, str(tmp_path / 'cache'), MARGIN, DIVISOR)
    path = Path(result['path'])
    with np.load(path, allow_pickle=False) as cache:
        arrays = {key: cache[key] for key in cache.files}
    if corruption == 'affine':
        arrays['affine'][0, 3] += 1
    elif corruption == 'normalization':
        arrays['norm_stats'][0, 0] += 1
    elif corruption == 'mask_move':
        coordinate = tuple(np.argwhere(arrays['img'][0] != 0)[0])
        arrays['img'][:, 0, 0, 0] = arrays['img'][(slice(None), *coordinate)]
        arrays['img'][(slice(None), *coordinate)] = 0
    elif corruption == 'label_move':
        coordinate = tuple(np.argwhere(arrays['seg'] == 3)[0])
        arrays['seg'][0, 0, 0] = 3
        arrays['seg'][coordinate] = 0
    else:
        arrays.pop('seg')
    np.savez_compressed(path, **arrays)
    assert not roundtrip_passed(verify_roundtrip(path, row, margin=MARGIN, divisor=DIVISOR))


@pytest.mark.parametrize('bad_value', [0.5, -1, 4, 256, np.nan])
def test_source_labels_are_validated_before_uint8_conversion(tmp_path, bad_value):
    row, _, seg = _outside_label_case(tmp_path)
    values = seg.astype(np.float32)
    values[20, 20, 50] = bad_value
    nib.save(nib.Nifti1Image(values, np.eye(4)), row['path_seg'])
    with pytest.raises(ValueError, match='integer labels'):
        preprocess_case(row, str(tmp_path / 'cache'), MARGIN, DIVISOR)


def test_sequence_affine_mismatch_rejected(tmp_path):
    row, _, _ = _outside_label_case(tmp_path)
    path = row['path_t2w']
    volume = nib.load(path)
    affine = np.array(volume.affine)
    affine[0, 3] += 1
    nib.save(nib.Nifti1Image(np.asanyarray(volume.dataobj), affine), path)
    with pytest.raises(ValueError, match='geometry'):
        preprocess_case(row, str(tmp_path / 'cache'), MARGIN, DIVISOR)
