"""Splits must be a pure function of (manifest, seed), and the committed CSV must be sound.

Regression for: the per-cohort RNG was seeded from ``hash((seed, cohort))``. ``hash()`` of a
str is randomised per process, so regenerating the splits gave a different answer every run.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

import brats
from brats.constants import COHORTS
from brats.data.splits import SPLITS, assign_splits, verify

REPO = Path(__file__).resolve().parents[1]
#: Wherever `brats` was imported from, so the subprocess below tests the same code.
SRC = Path(brats.__file__).resolve().parents[1]
FRACS = dict(train_frac=0.8, val_frac=0.1, test_frac=0.1)


def _manifest(cohorts=COHORTS):
    rows = []
    for cohort, n_patients in (("GLI", 60), ("MEN", 50), ("PED", 20)):
        if cohort not in cohorts:
            continue
        for p in range(n_patients):
            for t in range(1 + (p % 3 == 0)):  # some patients have a second timepoint
                rows.append(
                    dict(
                        case_id=f"BraTS-{cohort}-{p:05d}-{t:03d}",
                        cohort=cohort,
                        split_source="train",
                        patient_id=f"BraTS-{cohort}-{p:05d}",
                        timepoint=f"{t:03d}",
                    )
                )
    return pd.DataFrame(rows)


def test_assignment_is_deterministic_in_process():
    m = _manifest()
    a = assign_splits(m, seed=42, **FRACS)
    b = assign_splits(m, seed=42, **FRACS)
    pd.testing.assert_frame_equal(a, b)


def test_assignment_does_not_depend_on_pythonhashseed():
    code = textwrap.dedent(
        f"""
        import json, sys
        sys.path.insert(0, {str(SRC)!r})
        sys.path.insert(0, {str(Path(__file__).parent)!r})
        from test_splits import _manifest, FRACS
        from brats.data.splits import assign_splits
        df = assign_splits(_manifest(), seed=42, **FRACS)
        print(json.dumps(dict(zip(df.case_id, df.split))))
        """
    )

    def run(hashseed):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        return json.loads(out.stdout)

    assert run("1") == run("2") == run("12345")


def test_other_seeds_give_other_splits():
    m = _manifest()
    a = assign_splits(m, seed=42, **FRACS)
    b = assign_splits(m, seed=43, **FRACS)
    assert not a.split.equals(b.split)


def test_adding_a_cohort_does_not_reshuffle_the_others():
    both = assign_splits(_manifest(("GLI", "MEN")), seed=42, **FRACS).set_index("case_id").split
    all3 = assign_splits(_manifest(), seed=42, **FRACS).set_index("case_id").split
    assert (all3.loc[both.index] == both).all()


def test_no_patient_straddles_splits_and_every_cohort_is_in_every_split():
    out = assign_splits(_manifest(), seed=42, **FRACS)
    assert verify(out) == []


def test_committed_split_csv_is_sound():
    path = REPO / "splits" / "split_random_seed42.csv"
    if not path.is_file():
        pytest.skip("splits CSV not present")
    df = pd.read_csv(path)

    assert verify(df) == []
    assert len(df) == 2755
    counts = df.groupby(["cohort", "split"]).size()
    assert counts[("PED", "test")] == 10 and counts[("PED", "val")] == 10
    assert set(df.split) == {*SPLITS, "official_val"}
