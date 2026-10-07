"""Background handling: the cache decodes background to exactly 0 and nothing else does, and
augmentation leaves the background alone."""

import numpy as np
import pytest

from brats.constants import QUANT_BACKGROUND_CODE, quant_scale
from brats.data.preprocess import IN_BRAIN_ZERO, dequantize, quantize

# -- dequantize: background is the only thing that decodes to exactly 0 ---------------------


def test_only_the_background_code_decodes_to_zero():
    codes = np.arange(256, dtype=np.uint8)
    z = dequantize(codes)
    assert (z == 0).nonzero()[0].tolist() == [QUANT_BACKGROUND_CODE]
    assert z[128] == IN_BRAIN_ZERO  # the code that used to decode to exactly 0.0


def test_image_nonzero_is_an_exact_brain_mask_after_a_round_trip():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(32, 32, 32)).astype(np.float32)
    z[:4] = 0.0  # force plenty of voxels into the old collision band around z = 0
    z[4:8] = rng.uniform(-0.03, 0.03, z[4:8].shape)
    mask = rng.random(z.shape) > 0.3

    back = dequantize(quantize(z, mask))

    np.testing.assert_array_equal(back != 0, mask)


def test_the_epsilon_is_far_below_the_quantization_step():
    assert float(IN_BRAIN_ZERO) < quant_scale() / 1000
    rng = np.random.default_rng(1)
    z = np.clip(rng.normal(size=100_000), -5, 5).astype(np.float32)
    err = np.abs(dequantize(quantize(z, np.ones(z.shape, bool))) - z).max()
    assert err <= quant_scale() / 2 + 1e-4


def test_binarize_control_no_longer_punches_holes():
    torch = pytest.importorskip("torch")
    from brats.confound.binarize_control import binarize_batch

    rng = np.random.default_rng(2)
    mask = rng.random((2, 4, 16, 16, 16)) > 0.4
    img = torch.from_numpy(dequantize(quantize(rng.normal(size=mask.shape).astype(np.float32), mask)))

    assert torch.equal(binarize_batch(img).bool(), torch.from_numpy(mask))


# -- augmentation leaves the background alone ----------------------------------------------


def _item(seed=0):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(seed)
    img = np.zeros((4, 24, 24, 24), np.float32)
    brain = np.zeros((24, 24, 24), bool)
    brain[6:18, 6:18, 6:18] = True
    img[:, brain] = rng.normal(size=(4, int(brain.sum()))).astype(np.float32) + 0.5
    return {"image": torch.from_numpy(img)}, brain


def test_intensity_augmentation_keeps_the_background_exactly_zero():
    pytest.importorskip("monai")
    from monai import transforms as T

    from brats.data.transforms import intensity_augmentations

    item, brain = _item()
    aug = T.Compose(intensity_augmentations(p_scale=1.0, p_shift=1.0, p_noise=1.0, p_smooth=1.0))
    out = aug(item)["image"].numpy()

    assert not out[:, ~brain].any(), "background was modified"
    assert not np.allclose(out[:, brain], item["image"].numpy()[:, brain]), "tissue was not augmented"
    assert "_foreground" not in aug(item)


def test_the_unmasked_pipeline_would_have_leaked_into_the_background():
    """Control: MONAI's shift / noise applied directly do touch the background."""
    pytest.importorskip("monai")
    from monai import transforms as T

    item, brain = _item()
    raw = T.Compose([
        T.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        T.RandGaussianNoised(keys="image", prob=1.0, std=0.05),
    ])
    assert raw(item)["image"].numpy()[:, ~brain].any()


def test_full_training_pipeline_still_runs_and_keeps_shapes():
    pytest.importorskip("monai")
    from brats.data.preprocess import labels_to_regions
    from brats.data.transforms import train_transforms

    item, brain = _item()
    seg = np.zeros((24, 24, 24), np.uint8)
    seg[9:15, 9:15, 9:15] = 2
    seg[10:13, 10:13, 10:13] = 3
    item |= {"label": labels_to_regions(seg).astype(np.float32), "cohort": np.int64(1)}

    out = train_transforms((16, 16, 16))(item)
    out = out[0] if isinstance(out, list) else out

    assert tuple(out["image"].shape) == (4, 16, 16, 16) and tuple(out["label"].shape) == (3, 16, 16, 16)
    assert "_foreground" not in out and np.isfinite(out["image"].numpy()).all()
