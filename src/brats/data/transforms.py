"""Dataset and MONAI transform pipelines over the uint8 cache.

Reads the ``.npz`` files written by :mod:`brats.data.preprocess`, dequantizes to
float32, expands the 0..3 label map into 3 nested binary channels in (ET, TC, WT)
order, and applies augmentation.

Augmentation follows the winners' recipes: patch crop, flips on all three axes,
random affine, intensity scale/shift, Gaussian noise and blur. Nothing exotic --
the accuracy in BraTS 2023 came from post-processing, not augmentation cleverness.

Two details that matter:

* The classification head is trained **only on patches containing tumor**, which
  ``RandCropByPosNegLabeld`` with a high positive ratio delivers. A patch of pure
  background carries no cohort evidence beyond acquisition signature, i.e. exactly
  the confound.
* ``num_workers`` should be ~8 per rank on ``GPU_NV_M`` (44 vCPU / 4 ranks). 3D
  augmentation is CPU-bound and is the resource that starves the GPUs first.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from brats.config import REPO_ROOT, DataConfig
from brats.constants import COHORT_INDEX, REGIONS, SEQUENCES
from brats.data.preprocess import dequantize, labels_to_regions

log = logging.getLogger("brats.transforms")


class CachedBratsDataset(Dataset):
    """Loads preprocessed ``.npz`` cases and applies a MONAI transform chain.

    Each item is a dict with:
        ``image``  float32 (4, D, H, W), z-scored, background exactly 0
        ``label``  float32 (3, D, H, W) nested binary regions (absent if unlabeled)
        ``cohort`` int64 scalar class index
        ``case_id``, ``cohort_name`` metadata carried through for reporting
    """

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        transform=None,
        load_label: bool = True,
    ) -> None:
        self.records = list(records)
        self.transform = transform
        self.load_label = load_label

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, rec: dict[str, Any]) -> dict[str, Any]:
        with np.load(rec["cache_path"]) as z:
            img = dequantize(z["img"]).astype(np.float32)
            seg = z["seg"] if ("seg" in z.files and self.load_label) else None
            crop_start = z["crop_start"].copy()
            crop_stop = z["crop_stop"].copy()
            orig_shape = z["orig_shape"].copy()

        item: dict[str, Any] = {
            "image": img,
            "cohort": np.int64(COHORT_INDEX[rec["cohort"]]),
            "case_id": rec["case_id"],
            "cohort_name": rec["cohort"],
            "crop_start": crop_start,
            "crop_stop": crop_stop,
            "orig_shape": orig_shape,
        }
        if seg is not None:
            item["label"] = labels_to_regions(seg).astype(np.float32)
        return item

    def __getitem__(self, idx: int):
        item = self._load(self.records[idx])
        if self.transform is not None:
            item = self.transform(item)
            # ``RandCropByPosNegLabeld`` returns ``list[dict]`` -- one entry per
            # requested sample -- *even when num_samples == 1*, and ``Compose``
            # propagates that list through every later transform. Unwrap the
            # single-sample case so the common path yields a plain dict; leave a
            # genuine multi-sample list intact for ``collate_metadata`` to flatten.
            if isinstance(item, list) and len(item) == 1:
                item = item[0]
        return item


# ---------------------------------------------------------------------------
# Transform pipelines
# ---------------------------------------------------------------------------


def train_transforms(
    patch_size: tuple[int, int, int] = (128, 128, 128),
    pos_ratio: float = 0.8,
    samples_per_case: int = 1,
):
    """Augmentation for training.

    ``pos_ratio`` is high on purpose: the classification head only sees tumor-bearing
    patches, and rare regions (PED ET) need to appear often enough to learn.
    """
    from monai import transforms as T

    return T.Compose(
        [
            T.EnsureChannelFirstd(keys=["image", "label"], channel_dim=0),
            # Patch selection biased toward tumor. Keyed on WT (channel index 2),
            # the largest region, so a patch is anchored on tumor of some kind.
            T.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=pos_ratio,
                neg=1.0 - pos_ratio,
                num_samples=samples_per_case,
                image_key="image",
                image_threshold=0,
                allow_smaller=True,
            ),
            # Pad in case a cropped volume was smaller than the patch on some axis.
            T.SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            T.RandAffined(
                keys=["image", "label"],
                prob=0.25,
                rotate_range=(0.26, 0.26, 0.26),  # ~15 degrees
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),  # never interpolate labels
                padding_mode="zeros",
            ),
            T.RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
            T.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
            T.RandGaussianNoised(keys="image", prob=0.15, mean=0.0, std=0.05),
            T.RandGaussianSmoothd(
                keys="image",
                prob=0.15,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
            ),
            T.EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )


def val_transforms(labeled: bool = True):
    """Validation/inference: no augmentation, no cropping.

    Full volumes go through ``sliding_window_inference``, so nothing is cropped here.
    """
    from monai import transforms as T

    keys = ["image", "label"] if labeled else ["image"]
    return T.Compose(
        [
            T.EnsureChannelFirstd(keys=keys, channel_dim=0),
            T.EnsureTyped(keys=keys, dtype=torch.float32),
        ]
    )


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------


def load_records(
    cfg: DataConfig,
    split: str,
    splits_csv: str | Path | None = None,
    require_cache: bool = True,
) -> list[dict[str, Any]]:
    """Build dataset records for one split by joining the split CSV to the cache.

    Args:
        split: ``"train"``, ``"val"``, ``"test"``, or ``"official_val"``.
        require_cache: Raise if a case has no ``.npz``. Set False to tolerate a
            partially built cache during development.
    """
    path = Path(splits_csv) if splits_csv else REPO_ROOT / cfg.splits["output"]
    if not path.is_file():
        raise SystemExit(f"No split file at {path}. Run `python -m brats.data.splits`.")
    df = pd.read_csv(path)
    sub = df[df.split == split]
    if sub.empty:
        raise SystemExit(f"Split {split!r} is empty in {path}")

    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in sub.itertuples(index=False):
        npz = cfg.cache_root / row.cohort / f"{row.case_id}.npz"
        if not npz.is_file():
            missing.append(row.case_id)
            continue
        records.append(
            {
                "case_id": row.case_id,
                "cohort": row.cohort,
                "patient_id": row.patient_id,
                "cache_path": str(npz),
            }
        )

    if missing:
        msg = (
            f"{len(missing)} of {len(sub)} cases in split {split!r} are missing from "
            f"the cache at {cfg.cache_root} (e.g. {missing[:3]})"
        )
        if require_cache:
            raise SystemExit(msg + "\nRun `python -m brats.data.preprocess`.")
        log.warning(msg)

    log.info("split %s: %d cases", split, len(records))
    return records


def build_datasets(
    cfg: DataConfig,
    patch_size: tuple[int, int, int] = (128, 128, 128),
    splits_csv: str | Path | None = None,
    require_cache: bool = True,
) -> dict[str, CachedBratsDataset]:
    """Construct train / val / test / official_val datasets."""
    out: dict[str, CachedBratsDataset] = {}
    for split in ("train", "val", "test"):
        try:
            recs = load_records(cfg, split, splits_csv, require_cache)
        except SystemExit:
            if require_cache:
                raise
            continue
        tf = train_transforms(patch_size) if split == "train" else val_transforms(True)
        out[split] = CachedBratsDataset(recs, transform=tf, load_label=True)

    try:
        recs = load_records(cfg, "official_val", splits_csv, require_cache)
        out["official_val"] = CachedBratsDataset(
            recs, transform=val_transforms(False), load_label=False
        )
    except SystemExit:
        if require_cache:
            raise
    return out


def collate_metadata(batch: list[Any]) -> dict[str, Any]:
    """Collate that keeps string metadata as lists instead of trying to stack it.

    Also flattens nested lists, so a dataset configured with
    ``samples_per_case > 1`` (where each item is a list of patches) collates into
    one batch of ``batch_size * samples_per_case`` patches.
    """
    from torch.utils.data import default_collate

    flat: list[dict[str, Any]] = []
    for b in batch:
        flat.extend(b) if isinstance(b, list) else flat.append(b)

    meta_keys = {"case_id", "cohort_name"}
    tensor_part = [{k: v for k, v in b.items() if k not in meta_keys} for b in flat]
    out = default_collate(tensor_part)
    for k in meta_keys:
        if k in flat[0]:
            out[k] = [b[k] for b in flat]
    return out


__all__ = [
    "CachedBratsDataset",
    "train_transforms",
    "val_transforms",
    "load_records",
    "build_datasets",
    "collate_metadata",
    "REGIONS",
    "SEQUENCES",
]
