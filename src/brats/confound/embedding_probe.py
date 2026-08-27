"""Site/cohort probe on the learned encoder embedding, and the permutation control.

**Embedding probe.** Train a simple classifier on the *learned representation* to
predict cohort. Near-perfect predictability from the embedding means the tumor-type
head is very likely reading cohort identity rather than tumor phenotype. This is the
standard failure metric in the harmonization literature (DLEST, arXiv 2402.06875).

The distinction from :mod:`brats.confound.probes`: that module asks whether cohort is
predictable from hand-built features of the *input*. This asks whether the network
has *chosen* to encode it. A model could in principle discard an available shortcut;
this measures whether ours did.

**Permutation control** (the "Same Analysis Approach", arXiv 1703.06670). Re-run the
entire pipeline with cohort labels randomly permuted. Any accuracy above chance is
pipeline-induced optimism -- leakage, a selection artifact, or an evaluation bug --
because with permuted labels there is nothing real to learn. This catches classes of
error that no amount of careful reasoning will.

**Confounding Index** (arXiv 1905.08871) gives a single reportable scalar instead of
a narrative caveat, which is what belongs in a results table.
"""

from __future__ import annotations

import logging

import numpy as np

from brats.constants import COHORTS

log = logging.getLogger("brats.confound.probes_embedding")


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


def extract_embeddings(
    model,
    dataset,
    device="cuda",
    limit: int | None = None,
    use_attention_pooling: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pool encoder bottleneck features per case.

    Args:
        use_attention_pooling: If True, pool the way the classifier does (tumor
            attention). If False, use plain GAP. Running both is informative: a large
            gap between them localizes *where* the cohort information sits -- inside
            the tumor region, or in the surrounding geometry.

    Returns:
        ``(X, y, case_ids)`` with ``X`` of shape (n_cases, n_features).
    """
    import torch

    net = model.module if hasattr(model, "module") else model
    net.eval()
    if hasattr(net, "_alpha"):
        net._alpha = 0.0  # noqa: SLF001 - never use GT at probe time

    feats: list[np.ndarray] = []
    labels: list[int] = []
    ids: list[str] = []

    n = len(dataset) if limit is None else min(limit, len(dataset))
    with torch.no_grad():
        for i in range(n):
            item = dataset[i]
            image = item["image"]
            if not torch.is_tensor(image):
                image = torch.as_tensor(np.asarray(image))
            x = image.unsqueeze(0).float().to(device)

            # _encode_decode mirrors SegResNetDS's inline decoder (it has no
            # `.decoder` attribute) and returns encoder levels plus seg logits.
            levels, seg_list = net._encode_decode(x)  # noqa: SLF001
            bottleneck = levels[-1]

            if use_attention_pooling:
                from brats.constants import REGION_INDEX

                wt_ch = REGION_INDEX["WT"]
                wt = torch.sigmoid(seg_list[0][:, wt_ch : wt_ch + 1])
                pooled = net.pool(bottleneck, wt_prob=wt, gt_wt=None, alpha=0.0)
            else:
                pooled = bottleneck.mean(dim=tuple(range(2, bottleneck.ndim)))

            feats.append(pooled.squeeze(0).float().cpu().numpy())
            labels.append(int(item["cohort"]))
            ids.append(str(item["case_id"]))
            if (i + 1) % 25 == 0:
                log.info("embedded %d/%d", i + 1, n)

    return np.stack(feats), np.asarray(labels), ids


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def embedding_probe(
    X: np.ndarray, y: np.ndarray, seed: int = 42, n_splits: int = 5
) -> dict:
    """Cross-validated linear probe for cohort from the embedding.

    Deliberately *linear*: a nonlinear probe measures the capacity of the probe as
    much as the content of the representation. Linear separability is the standard
    and the more conservative claim.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, multi_class="multinomial", random_state=seed),
    )
    cv = StratifiedKFold(n_splits=min(n_splits, int(np.bincount(y).min())), shuffle=True,
                         random_state=seed)
    preds = np.empty_like(y)
    for tr, te in cv.split(X, y):
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])

    bal = float(balanced_accuracy_score(y, preds))
    chance = 1.0 / len(np.unique(y))
    return {
        "balanced_accuracy": bal,
        "chance": chance,
        "confounding_index": max(0.0, (bal - chance) / (1.0 - chance)),
        "confusion": confusion_matrix(y, preds).tolist(),
        "n": int(len(y)),
        "n_features": int(X.shape[1]),
    }


def permutation_control(
    X: np.ndarray, y: np.ndarray, n_permutations: int = 100, seed: int = 42
) -> dict:
    """Null distribution of the embedding probe under permuted labels.

    Any systematic gap above chance in the null means the *evaluation itself* is
    optimistic. Returns an empirical p-value for the observed accuracy.
    """
    rng = np.random.default_rng(seed)
    observed = embedding_probe(X, y, seed=seed)["balanced_accuracy"]

    null: list[float] = []
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        null.append(embedding_probe(X, y_perm, seed=seed + i + 1)["balanced_accuracy"])
    null_arr = np.asarray(null)

    # +1 smoothing: with finite permutations a p-value of exactly 0 is not supported.
    p = float((np.sum(null_arr >= observed) + 1) / (n_permutations + 1))
    return {
        "observed": observed,
        "null_mean": float(null_arr.mean()),
        "null_std": float(null_arr.std()),
        "null_p95": float(np.percentile(null_arr, 95)),
        "p_value": p,
        "n_permutations": n_permutations,
    }


def report(
    attention_probe: dict,
    gap_probe: dict | None = None,
    permutation: dict | None = None,
) -> str:
    """Format the embedding-probe findings."""
    lines = [
        "=" * 64,
        "Site/cohort probe on the learned embedding",
        "=" * 64,
        f"tumor-attention pooled embedding : balanced acc "
        f"{attention_probe['balanced_accuracy']:.4f}  "
        f"(chance {attention_probe['chance']:.3f})",
    ]
    if gap_probe:
        lines.append(
            f"GAP pooled embedding             : balanced acc "
            f"{gap_probe['balanced_accuracy']:.4f}"
        )
        delta = gap_probe["balanced_accuracy"] - attention_probe["balanced_accuracy"]
        lines.append(f"GAP minus attention              : {delta:+.4f}")
        if delta > 0.10:
            lines.append(
                "  -> Cohort information sits largely OUTSIDE the tumor region. "
                "Tumor-attention pooling is doing real work; GAP would have leaked."
            )
        elif delta < -0.05:
            lines.append(
                "  -> Attention pooling is MORE cohort-predictive than GAP, i.e. the "
                "tumor region itself carries cohort identity. Expected to some degree "
                "(tumor types genuinely differ), but it caps how much the confound can "
                "be engineered away."
            )
        else:
            lines.append(
                "  -> Comparable. Pooling choice is not the dominant factor here."
            )

    ci = attention_probe["confounding_index"]
    lines += [
        "",
        f"Confounding Index (arXiv 1905.08871): {ci:.3f}",
        "  0 = cohort not linearly recoverable; 1 = perfectly recoverable.",
    ]

    if permutation:
        lines += [
            "",
            "-- Permutation control (Same Analysis Approach, arXiv 1703.06670) ---",
            f"observed        : {permutation['observed']:.4f}",
            f"null mean +- sd : {permutation['null_mean']:.4f} "
            f"+- {permutation['null_std']:.4f}",
            f"null 95th pct   : {permutation['null_p95']:.4f}",
            f"empirical p     : {permutation['p_value']:.4f}",
        ]
        if permutation["null_mean"] > 1.0 / len(COHORTS) + 0.05:
            lines.append(
                "  WARNING: the null sits above chance. The evaluation pipeline itself "
                "is optimistic -- suspect leakage or a selection artifact. Fix this "
                "before interpreting any accuracy."
            )
        else:
            lines.append("  Null is centered at chance: the pipeline is not self-inflating.")

    lines += [
        "",
        "-- How this gets reported " + "-" * 38,
        "  The classification head is an auxiliary/regularizing task with an explicit",
        "  confound caveat, NOT a clinical tumor-typing claim. If the signal turns out",
        "  to be largely acquisition-driven, that is a legitimate and interesting",
        "  finding to report, not a failure to hide. Reporting a high accuracy without",
        "  these diagnostics would be scientifically indefensible -- and a high number",
        "  is exactly what this dataset will hand us.",
    ]
    return "\n".join(lines)


__all__ = [
    "extract_embeddings",
    "embedding_probe",
    "permutation_control",
    "report",
]
