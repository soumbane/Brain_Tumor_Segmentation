"""EDA plus the two cheap confound probes.

The probes are the point. Before any deep model is trained, answer with numbers:
**how separable are the three cohorts from intensity statistics and brain-mask
geometry alone?**

In pooled BraTS 2023 the tumor-type label *is* the cohort folder. The three cohorts
were assembled by different consortia, at different institutions, on different
scanners, with different protocols, on different patient populations (PED is defined
by age), and passed through defacing/skull-stripping with cohort-specific behavior.

If a logistic regression on 4 intensity histograms separates the cohorts, or if a
random forest on brain-mask shape alone does, then a near-perfect deep classification
accuracy is evidence of confounding rather than evidence of a good model. That is not
hedging -- the mechanisms are documented for this exact dataset:

* **Tinauer et al. (arXiv 2501.15831)**: a 3D CNN classifying Alzheimer's from 990
  matched ADNI T1w scans kept *full accuracy on binarized images*, proving it was not
  using tissue texture. LRP showed it relied on brain contours introduced by skull
  stripping. The authors call it a Clever Hans effect.
* **The BraTS 2023 meningioma analysis paper (arXiv 2405.09787)**: 1286 of 1424 MEN
  cases (90.3%) have tumor voxels abutting the edge of the skull-stripped image, and
  the organizers explicitly call for "further investigation into optimal pre-processing
  face anonymization steps."

These two probes cost minutes and are decisive, which is why they run before training
rather than after.

Usage::

    .venv\\Scripts\\python.exe -m brats.confound.probes --max-cases 300
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from brats.config import DataConfig
from brats.constants import COHORTS, SEQUENCES

log = logging.getLogger("brats.confound.probes")

#: Histogram bins per sequence, over the in-brain intensity range.
N_BINS = 32


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def case_features(case_dir: str, cohort: str) -> dict | None:
    """Intensity-histogram and mask-geometry features for one case.

    Deliberately contains **no tumor information at all** -- only whole-brain
    intensity statistics and the geometry of the skull-stripped mask. Anything these
    features can predict is, by construction, not tumor phenotype.
    """
    import nibabel as nib

    d = Path(case_dir)
    case_id = d.name
    paths = {s: d / f"{case_id}-{s}.nii.gz" for s in SEQUENCES}
    if not all(p.is_file() for p in paths.values()):
        return None

    feats: dict[str, float | str] = {"case_id": case_id, "cohort": cohort}
    mask: np.ndarray | None = None

    for seq in SEQUENCES:
        arr = np.asanyarray(nib.load(str(paths[seq])).dataobj).astype(np.float32)
        m = arr != 0
        mask = m if mask is None else (mask | m)
        fg = arr[m]
        if fg.size == 0:
            continue

        # Normalized histogram over the case's own robust range, so the features
        # describe *shape* of the distribution rather than absolute scanner units.
        lo, hi = np.percentile(fg, [0.5, 99.5])
        if hi <= lo:
            hi = lo + 1.0
        hist, _ = np.histogram(fg, bins=N_BINS, range=(float(lo), float(hi)))
        hist = hist.astype(np.float64) / max(hist.sum(), 1)
        for i, v in enumerate(hist):
            feats[f"hist_{seq}_{i}"] = float(v)

        feats[f"stat_{seq}_mean"] = float(fg.mean())
        feats[f"stat_{seq}_std"] = float(fg.std())
        feats[f"stat_{seq}_skew"] = float(
            ((fg - fg.mean()) ** 3).mean() / (fg.std() ** 3 + 1e-8)
        )
        feats[f"stat_{seq}_kurt"] = float(
            ((fg - fg.mean()) ** 4).mean() / (fg.std() ** 4 + 1e-8)
        )
        for q in (1, 5, 25, 50, 75, 95, 99):
            feats[f"stat_{seq}_p{q}"] = float(np.percentile(fg, q))

    # -- brain-mask geometry: the Tinauer pathway -------------------------
    assert mask is not None
    feats["geom_volume"] = float(mask.sum())
    idx = np.nonzero(mask)
    extents = []
    for axis in range(3):
        lo, hi = int(idx[axis].min()), int(idx[axis].max())
        extents.append(hi - lo + 1)
        feats[f"geom_lo_{axis}"] = float(lo)
        feats[f"geom_hi_{axis}"] = float(hi)
        feats[f"geom_extent_{axis}"] = float(hi - lo + 1)
        feats[f"geom_centroid_{axis}"] = float(idx[axis].mean())
    feats["geom_bbox_volume"] = float(np.prod(extents))
    feats["geom_fill_ratio"] = feats["geom_volume"] / max(feats["geom_bbox_volume"], 1)
    feats["geom_aspect_01"] = extents[0] / max(extents[1], 1)
    feats["geom_aspect_02"] = extents[0] / max(extents[2], 1)

    # Surface area via a 6-neighbour boundary count -- cheap proxy for mask
    # smoothness, which is exactly what skull-stripping alters per cohort.
    surface = np.zeros_like(mask)
    for axis in range(3):
        surface |= mask & ~np.roll(mask, 1, axis=axis)
        surface |= mask & ~np.roll(mask, -1, axis=axis)
    feats["geom_surface"] = float(surface.sum())
    feats["geom_sphericity"] = float(
        feats["geom_volume"] ** (2 / 3) / max(feats["geom_surface"], 1)
    )
    return feats


def build_feature_table(
    cfg: DataConfig, max_cases: int | None = 300, workers: int = 6
) -> pd.DataFrame:
    """Extract probe features for a stratified sample of labeled cases."""
    if not cfg.manifest_csv.is_file():
        raise SystemExit(
            f"No manifest at {cfg.manifest_csv}. Run `python -m brats.data.manifest`."
        )
    man = pd.read_csv(cfg.manifest_csv)
    man = man[man.split_source == "train"]

    # Balance the sample across cohorts: an imbalanced probe sample would let a
    # classifier score well by always predicting GLI, which tells us nothing.
    per = (max_cases // len(COHORTS)) if max_cases else None
    sample = pd.concat(
        [
            man[man.cohort == c].head(per) if per else man[man.cohort == c]
            for c in COHORTS
        ]
    )
    tasks = [
        (str(Path(r.path_t1n).parent), r.cohort) for r in sample.itertuples(index=False)
    ]
    log.info("extracting probe features for %d cases", len(tasks))

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(case_features, d, c) for d, c in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                rows.append(r)
            if i % 50 == 0:
                log.info("  %d/%d", i, len(tasks))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def run_probe(
    df: pd.DataFrame, feature_prefix: str | tuple[str, ...], name: str, seed: int = 42
) -> dict:
    """Cross-validated cohort classification from a feature subset.

    Uses grouped stratified CV and reports balanced accuracy against the majority
    baseline. Balanced accuracy, not raw accuracy: with a 12.6:10.1:1 imbalance, raw
    accuracy is dominated by the two large cohorts.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    prefixes = (feature_prefix,) if isinstance(feature_prefix, str) else feature_prefix
    cols = [c for c in df.columns if c.startswith(prefixes)]
    if not cols:
        raise ValueError(f"no columns matching {prefixes}")

    X = df[cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["cohort"].to_numpy()

    clf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1
        ),
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.empty_like(y)
    for tr, te in cv.split(X, y):
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])

    bal = float(balanced_accuracy_score(y, preds))
    # Majority-class balanced accuracy for 3 classes is 1/3 by definition.
    chance = 1.0 / len(np.unique(y))
    return {
        "probe": name,
        "n_features": len(cols),
        "n_cases": len(y),
        "balanced_accuracy": bal,
        "chance": chance,
        "above_chance": bal - chance,
        "confusion": confusion_matrix(y, preds, labels=list(COHORTS)).tolist(),
    }


def interpret(results: list[dict]) -> list[str]:
    """Translate probe numbers into the design decision they imply."""
    out = ["", "-- Interpretation " + "-" * 46]
    for r in results:
        bal = r["balanced_accuracy"]
        if bal > 0.90:
            verdict = (
                "SEVERE confound. The cohorts are near-perfectly separable with no "
                "tumor information at all. A high deep-model accuracy proves nothing "
                "about tumor phenotype."
            )
        elif bal > 0.70:
            verdict = (
                "SUBSTANTIAL confound. Report the classification head as auxiliary "
                "with an explicit caveat, and rely on mask-conditioned pooling."
            )
        elif bal > 0.45:
            verdict = "MODERATE confound. Quantify it; do not dismiss it."
        else:
            verdict = "WEAK confound on these features. Still run the binarization control."
        out.append(f"  {r['probe']}: balanced acc {bal:.3f} (chance {r['chance']:.3f})")
        out.append(f"    -> {verdict}")

    out += [
        "",
        "  Regardless of the numbers above:",
        "    * Default to tumor-attention pooling, not GAP, for the cls head.",
        "    * Keep GAP as an explicit ablation -- it measures the confound's worth.",
        "    * Do NOT fuse age as a covariate: the PED cohort is DEFINED by age, so",
        "      that injects exactly the confound under study.",
        "    * Frame the head as auxiliary/regularizing, never as clinical tumor typing.",
    ]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-cases", type=int, default=300, help="0 = all cases")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--features-csv", default=None, help="reuse a cached feature table")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    cfg = DataConfig.load(args.config) if args.config else DataConfig.load()
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    feat_path = Path(args.features_csv) if args.features_csv else (
        cfg.reports_dir / "confound_features.csv"
    )

    if feat_path.is_file():
        log.info("reusing features from %s", feat_path)
        df = pd.read_csv(feat_path)
    else:
        df = build_feature_table(
            cfg, max_cases=args.max_cases or None, workers=args.workers
        )
        df.to_csv(feat_path, index=False)
        log.info("wrote %s", feat_path)

    results = [
        run_probe(df, "hist_", "intensity-histogram cohort classifier"),
        run_probe(df, "stat_", "intensity-moment cohort classifier"),
        run_probe(df, "geom_", "brain-mask-shape cohort classifier"),
        run_probe(df, ("hist_", "stat_", "geom_"), "all non-tumor features"),
    ]

    lines = ["=" * 64, "Confound pre-screen (no tumor information used)", "=" * 64]
    lines.append(
        pd.DataFrame(results)
        .drop(columns=["confusion"])
        .to_string(index=False)
    )
    lines.append("")
    for r in results:
        lines.append(f"confusion ({r['probe']}), rows=true {list(COHORTS)}:")
        for name, row in zip(COHORTS, r["confusion"], strict=True):
            lines.append(f"  {name}: {row}")
    lines += interpret(results)

    text = "\n".join(lines)
    print(text)
    (cfg.reports_dir / "confound_prescreen.txt").write_text(text, encoding="utf-8")
    pd.DataFrame(results).to_csv(cfg.reports_dir / "confound_prescreen.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
