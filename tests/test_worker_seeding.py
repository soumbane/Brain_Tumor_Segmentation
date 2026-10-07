"""DataLoader workers must not replay one shared augmentation random stream.

Regression for: with the plain torch DataLoader every worker received a copy of the same MONAI
``RandomState``, so all workers drew identical flips / crop offsets / affine parameters, and
``torch.manual_seed`` never reached the augmentation at all (runs were not reproducible).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from monai import transforms as T  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from brats.data.transforms import CachedBratsDataset, collate_metadata, seed_worker  # noqa: E402

N, WORKERS = 32, 4
# Linux (SPCS, Databricks) starts workers with fork; spawn would add ~20 s of imports per loader here.
_CTX = "fork" if "fork" in torch.multiprocessing.get_all_start_methods() else None


class _RampDataset(CachedBratsDataset):
    """The real dataset class with in-memory samples; a flipped ramp starts high, so a flip is visible."""

    def _load(self, rec):
        ramp = np.tile(np.arange(8, dtype=np.float32), (1, 8, 8, 1))
        return {"image": ramp, "cohort": np.int64(0), "case_id": rec["case_id"], "cohort_name": "GLI",
                "crop_start": np.zeros(3), "crop_stop": np.zeros(3), "orig_shape": np.zeros(3)}


def _flip_sequence(seed, init_fn):
    tf = T.Compose([
        T.EnsureChannelFirstd(keys=["image"], channel_dim=0),
        T.RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
    ])
    ds = _RampDataset([{"case_id": f"c{i}", "cohort": "GLI"} for i in range(N)], transform=tf, load_label=False)
    torch.manual_seed(seed)
    dl = DataLoader(ds, batch_size=1, num_workers=WORKERS, shuffle=False,
                    collate_fn=collate_metadata, worker_init_fn=init_fn, multiprocessing_context=_CTX)
    return [bool(b["image"][0, 0, 0, 0, 0] > 0) for b in dl]


def _per_worker(seq):
    return [seq[w::WORKERS] for w in range(WORKERS)]  # samples are served round-robin


def test_each_worker_gets_its_own_stream():
    workers = _per_worker(_flip_sequence(0, seed_worker))
    assert not any(workers[0] == w for w in workers[1:]), "workers replay the same random numbers"


def test_without_the_hook_all_workers_share_one_stream():
    """Control: this is the behaviour the fix removes (guards the test above from being vacuous)."""
    workers = _per_worker(_flip_sequence(0, None))
    assert all(workers[0] == w for w in workers[1:])


def test_augmentation_is_reproducible_for_a_fixed_seed_and_varies_with_it():
    a, b = _flip_sequence(7, seed_worker), _flip_sequence(7, seed_worker)
    c = _flip_sequence(8, seed_worker)
    assert a == b
    assert a != c
