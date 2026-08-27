"""Canonical project constants.

Every invariant that, if violated silently, would corrupt results lives here and
nowhere else. Two of them are the highest-severity risks in the project:

1. **Label 3 is enhancing tumor (ET) in BraTS 2023, not label 4.** BraTS 2021 and
   earlier used 4. Any code, pretrained weight, or post-processing rule carried
   over from those releases that keys ET off ``label == 4`` produces a *silently
   empty ET channel*. ET is the hardest and most heavily weighted region, so the
   failure looks like a bad model rather than a bug.

2. **Region channel order is (ET, TC, WT).** The MONAI model-zoo
   ``brats_mri_segmentation`` bundle emits (TC, WT, ET). Mixing the two scrambles
   per-region metrics without raising anything.

Import these names; do not re-declare the values locally.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# MRI sequences
# ---------------------------------------------------------------------------

#: Canonical input channel order. Fixed once, asserted at load time, never
#: reordered implicitly. t1n = T1 native, t1c = T1 post-gadolinium,
#: t2w = T2 weighted, t2f = T2 FLAIR.
SEQUENCES: Final[tuple[str, ...]] = ("t1n", "t1c", "t2w", "t2f")

N_INPUT_CHANNELS: Final[int] = len(SEQUENCES)

#: Filename suffix of the annotation volume (training cases only).
SEG_SUFFIX: Final[str] = "seg"

# ---------------------------------------------------------------------------
# Label semantics (BraTS 2023)
# ---------------------------------------------------------------------------

LABEL_BACKGROUND: Final[int] = 0

#: NCR (necrotic / non-enhancing core) for GLI; NETC for MEN;
#: NC (non-enhancing + cystic + necrosis, collapsed) for PED.
LABEL_NCR: Final[int] = 1

#: ED (peritumoral edema) for GLI/PED; SNFH (surrounding non-enhancing FLAIR
#: hyperintensity) for MEN.
LABEL_ED: Final[int] = 2

#: **Enhancing tumor. Three, not four.** See module docstring.
LABEL_ET: Final[int] = 3

#: The complete set of permitted values in ``*-seg.nii.gz``. Asserted per case
#: during manifest validation; a 4 here means the dataset is not BraTS 2023.
VALID_LABELS: Final[frozenset[int]] = frozenset(
    {LABEL_BACKGROUND, LABEL_NCR, LABEL_ED, LABEL_ET}
)

# ---------------------------------------------------------------------------
# Evaluation regions
# ---------------------------------------------------------------------------

#: Canonical region order for the 3 sigmoid output channels and for every
#: reported metric. BraTS reporting convention.
REGIONS: Final[tuple[str, ...]] = ("ET", "TC", "WT")

N_REGIONS: Final[int] = len(REGIONS)

#: Which integer labels compose each region. The regions are *nested*
#: (ET subset of TC subset of WT), which is why the task is 3-channel multi-label
#: with sigmoid activation rather than 4-class softmax.
REGION_LABELS: Final[dict[str, frozenset[int]]] = {
    "ET": frozenset({LABEL_ET}),
    "TC": frozenset({LABEL_NCR, LABEL_ET}),
    "WT": frozenset({LABEL_NCR, LABEL_ED, LABEL_ET}),
}

#: Index of each region in the channel dimension.
REGION_INDEX: Final[dict[str, int]] = {r: i for i, r in enumerate(REGIONS)}

# ---------------------------------------------------------------------------
# Cohorts (== the classification target)
# ---------------------------------------------------------------------------

#: Cohort codes in fixed class-index order. NOTE: the classification label is
#: derived from which archive a case shipped in, which is exactly why the task is
#: confounded -- see the confound protocol. Treat the head as auxiliary.
COHORTS: Final[tuple[str, ...]] = ("GLI", "MEN", "PED")

N_COHORTS: Final[int] = len(COHORTS)

COHORT_INDEX: Final[dict[str, int]] = {c: i for i, c in enumerate(COHORTS)}

COHORT_LONG_NAME: Final[dict[str, str]] = {
    "GLI": "adult glioma",
    "MEN": "intracranial meningioma",
    "PED": "pediatric glioma",
}

#: Expected labeled (TrainingData) case counts per cohort. Used as a soft check
#: during manifest construction: a mismatch means the extraction is incomplete,
#: not that the constant is wrong.
EXPECTED_TRAIN_COUNTS: Final[dict[str, int]] = {"GLI": 1251, "MEN": 1000, "PED": 99}

#: Expected official ValidationData counts (no ``seg.nii.gz``).
EXPECTED_VAL_COUNTS: Final[dict[str, int]] = {"GLI": 219, "MEN": 141, "PED": 45}

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

#: Nominal volume shape after the FeTS pipeline: co-registered to SRI24,
#: resampled to 1 mm isotropic, skull-stripped. Deviations are logged, never
#: silently resampled.
NOMINAL_SHAPE: Final[tuple[int, int, int]] = (240, 240, 155)

#: Voxel spacing in mm. Already isotropic, so no resampling is applied anywhere.
NOMINAL_SPACING: Final[tuple[float, float, float]] = (1.0, 1.0, 1.0)

# ---------------------------------------------------------------------------
# Intensity cache quantization
# ---------------------------------------------------------------------------
# The GPU_NV_M node has only 93.13 GiB of local disk. A float16 cache of
# brain-cropped 4-channel volumes is ~40 MB/case -> ~94 GB for 2350 cases, which
# does not fit. Quantizing z-scored intensities to uint8 over a fixed +-5 sigma
# range gives ~22 MB/case -> ~52 GB, with headroom.
#
# Step size is 10 sigma / 255 ~= 0.039 sigma, far below any signal a network can
# exploit, so this is a storage decision and not a modeling one.

#: Clip bound in standard deviations before quantizing.
QUANT_CLIP_SIGMA: Final[float] = 5.0

#: Reserved uint8 code for "outside the brain mask" (exact zero after z-scoring).
#: Kept distinct so background is never confused with a legitimate -5 sigma voxel.
QUANT_BACKGROUND_CODE: Final[int] = 0

#: Usable code range for in-mask intensities.
QUANT_MIN_CODE: Final[int] = 1
QUANT_MAX_CODE: Final[int] = 255


def quant_scale() -> float:
    """Sigma per uint8 code step."""
    return (2.0 * QUANT_CLIP_SIGMA) / (QUANT_MAX_CODE - QUANT_MIN_CODE)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def _validate() -> None:
    """Guard the invariants that this module exists to protect."""
    assert LABEL_ET == 3, "BraTS 2023 encodes enhancing tumor as label 3, not 4"
    assert 4 not in VALID_LABELS, "label 4 is a BraTS 2021 artifact"
    assert REGIONS == ("ET", "TC", "WT"), (
        "channel order is (ET, TC, WT); the MONAI model-zoo bundle's "
        "(TC, WT, ET) order must never leak in"
    )
    # Nesting: ET subset of TC subset of WT.
    assert REGION_LABELS["ET"] < REGION_LABELS["TC"] < REGION_LABELS["WT"]
    assert set(SEQUENCES) == {"t1n", "t1c", "t2w", "t2f"}
    assert len(COHORTS) == N_COHORTS == 3
    assert QUANT_MIN_CODE > QUANT_BACKGROUND_CODE


_validate()
