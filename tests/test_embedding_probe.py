"""The embedding probe must run on the scikit-learn we actually get from pip.

Regression for: ``LogisticRegression(multi_class="multinomial")`` raised ``TypeError`` on
scikit-learn >= 1.8, where that argument no longer exists.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from brats.confound.embedding_probe import embedding_probe, permutation_control  # noqa: E402


def _separable_embeddings(n_per_class=20, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(3), n_per_class)
    centers = rng.normal(scale=6.0, size=(3, dim))
    return centers[y] + rng.normal(size=(len(y), dim)), y


def test_probe_recovers_separable_cohorts():
    X, y = _separable_embeddings()

    res = embedding_probe(X, y)

    assert res["balanced_accuracy"] > 0.9
    assert 0.0 <= res["confounding_index"] <= 1.0
    assert res["n_features"] == X.shape[1]


def test_permutation_null_sits_near_chance():
    X, y = _separable_embeddings()

    res = permutation_control(X, y, n_permutations=5)

    assert res["observed"] > res["null_p95"]
    assert res["null_mean"] < 0.6
    assert 0.0 < res["p_value"] <= 1.0
