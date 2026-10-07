"""Confound probes must cross-validate by patient, not by case."""

import numpy as np
import pytest


def _twin_dataset(n_patients=60, dim=100, seed=0):
    """Two near-identical timepoints per patient; cohort is random per patient, so no
    feature carries any real cohort information -- only patient identity."""
    rng = np.random.default_rng(seed)
    cohort = np.array(["GLI", "MEN", "PED"])[np.arange(n_patients) % 3]
    base = rng.normal(size=(n_patients, dim))
    rows, labels, ids = [], [], []
    for p in range(n_patients):
        for t in range(2):
            rows.append(base[p] + rng.normal(scale=0.01, size=dim))
            labels.append(cohort[p])
            ids.append(f"BraTS-{cohort[p]}-{p:05d}-{t:03d}")
    return np.asarray(rows), np.asarray(labels), ids


def test_probe_cv_is_grouped_by_patient():
    pd = pytest.importorskip("pandas")
    from brats.confound.probes import run_probe

    X, y, ids = _twin_dataset()
    df = pd.DataFrame(X, columns=[f"hist_{i}" for i in range(X.shape[1])])
    df["case_id"], df["cohort"] = ids, y

    res = run_probe(df, "hist_", "twin control")

    # With nothing but patient identity to go on, a leak-free probe sits near chance (1/3).
    assert res["balanced_accuracy"] < 0.6, res["balanced_accuracy"]


def test_embedding_probe_groups_remove_the_twin_leak():
    from brats.confound.embedding_probe import embedding_probe, patient_groups, permutation_control

    X, y, ids = _twin_dataset()
    labels = np.array([{"GLI": 0, "MEN": 1, "PED": 2}[c] for c in y])
    groups = patient_groups(ids)

    leaky = embedding_probe(X, labels)["balanced_accuracy"]            # twins straddle folds
    clean = embedding_probe(X, labels, groups=groups)["balanced_accuracy"]

    assert leaky > 0.9 > 0.6 > clean, (leaky, clean)
    # And the null built from group-level permutations is centred on chance.
    null = permutation_control(X, labels, n_permutations=5, groups=groups)
    assert null["null_mean"] < 0.6


def test_patient_groups_strip_the_timepoint():
    from brats.confound.embedding_probe import patient_groups

    assert patient_groups(["BraTS-GLI-00324-000", "BraTS-GLI-00324-007", "BraTS-PED-00001-000"]).tolist() == [
        "BraTS-GLI-00324", "BraTS-GLI-00324", "BraTS-PED-00001",
    ]
