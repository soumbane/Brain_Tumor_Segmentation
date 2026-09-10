"""Generate the train/val/test split. Run once, commit the CSV, never regenerate.

Rules (from the plan, all enforced here rather than by convention):

* **80/10/10** over the 2350 labeled TrainingData cases.
* **Stratified by cohort**, so all three cohorts appear in all three splits. Without
  this, PED (n=99 against GLI's 1251) could vanish from a split entirely.
* **Grouped by patient**, not by study. If a patient has multiple timepoints they all
  land in one split; otherwise near-duplicate volumes straddle train and test and
  every metric is optimistic.
* **Fixed seed**, deterministic output, written to a committed CSV.

The official ValidationData (405 cases) is *not* part of this split. It has no
``seg.nii.gz`` so segmentation cannot be scored on it -- but its cohort label is
known from the archive it shipped in, which makes it a legitimate and larger
held-out test set for the *classification* task. It is emitted here with
``split="official_val"`` so downstream code can find it without re-deriving it.

Usage::

    .venv\\Scripts\\python.exe -m brats.data.splits
    .venv\\Scripts\\python.exe -m brats.data.splits --check   # verify existing CSV
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from brats.config import REPO_ROOT, DataConfig
from brats.constants import COHORTS

log = logging.getLogger("brats.splits")

SPLITS = ("train", "val", "test")


def assign_splits(
    manifest: pd.DataFrame,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> pd.DataFrame:
    """Assign each labeled case to train/val/test, stratified and grouped.

    Splitting happens at the *patient* level within each cohort. Because patients
    can in principle carry several timepoints, the realized case-level fractions
    can differ slightly from the requested ones; the patient-level fractions are
    what is controlled.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")

    labeled = manifest[manifest.split_source == "train"].copy()
    rows: list[pd.DataFrame] = []

    for cohort in COHORTS:
        sub = labeled[labeled.cohort == cohort]
        if sub.empty:
            log.warning("cohort %s has no labeled cases", cohort)
            continue

        patients = np.sort(sub.patient_id.unique())
        # Seed per cohort so adding a cohort later cannot reshuffle the others.
        rng = np.random.default_rng(abs(hash((seed, cohort))) % (2**32))
        rng.shuffle(patients)

        n = len(patients)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        # Give the remainder to test so the three always sum to n exactly.
        n_test = n - n_train - n_val
        if n_test < 1 <= n:
            # Tiny cohorts (PED) can round to zero test patients; steal from train.
            n_test, n_train = 1, n_train - 1
        if n_val < 1 <= n:
            n_val, n_train = 1, n_train - 1

        assignment = (
            ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        )
        patient_split = dict(zip(patients, assignment, strict=True))

        block = sub[["case_id", "cohort", "patient_id", "timepoint"]].copy()
        block["split"] = block.patient_id.map(patient_split)
        rows.append(block)
        log.info(
            "%s: %d patients / %d cases -> train %d, val %d, test %d (patients)",
            cohort, n, len(sub), n_train, n_val, n_test,
        )

    labeled_out = pd.concat(rows, ignore_index=True)

    # The official ValidationData: no seg labels, but usable for classification.
    official = manifest[manifest.split_source == "val"][
        ["case_id", "cohort", "patient_id", "timepoint"]
    ].copy()
    official["split"] = "official_val"

    out = pd.concat([labeled_out, official], ignore_index=True)
    return out.sort_values(["split", "cohort", "case_id"]).reset_index(drop=True)


def verify(splits: pd.DataFrame) -> list[str]:
    """Return a list of violated invariants (empty means the split is sound)."""
    problems: list[str] = []

    dupes = splits[splits.case_id.duplicated()]
    if len(dupes):
        problems.append(f"{len(dupes)} duplicated case_id rows")

    # No patient may appear in more than one split -- the leakage check.
    leaked = (
        splits.groupby("patient_id")["split"].nunique().loc[lambda s: s > 1]
    )
    if len(leaked):
        problems.append(
            f"{len(leaked)} patients appear in >1 split (leakage): "
            f"{list(leaked.index[:5])}"
        )

    labeled = splits[splits.split.isin(SPLITS)]
    for cohort in COHORTS:
        present = set(labeled[labeled.cohort == cohort].split.unique())
        missing = set(SPLITS) - present
        if missing:
            problems.append(f"cohort {cohort} missing from splits: {sorted(missing)}")

    return problems


def report(splits: pd.DataFrame) -> str:
    lines = ["=" * 64, "Split summary", "=" * 64]
    table = (
        splits.groupby(["cohort", "split"]).size().unstack(fill_value=0)
    )
    order = [s for s in (*SPLITS, "official_val") if s in table.columns]
    lines.append(table[order].to_string())
    lines.append("")
    labeled = splits[splits.split.isin(SPLITS)]
    lines.append(f"labeled total     : {len(labeled)}")
    lines.append(f"official_val total: {int((splits.split == 'official_val').sum())}")
    lines.append("")
    lines.append("Reminders:")
    lines.append("  * test split is opened ONCE, after thresholds are frozen.")
    ped_test = int(((labeled.cohort == 'PED') & (labeled.split == 'test')).sum())
    lines.append(
        f"  * PED test n={ped_test}: report bootstrap CIs, never a bare number; "
        "treat delta < 0.05 Dice as noise."
    )
    lines.append(
        "  * official_val has no seg labels: segmentation is visual-only there, "
        "but classification IS quantitative."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--check", action="store_true", help="verify the committed CSV")
    ap.add_argument("--force", action="store_true", help="overwrite an existing CSV")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    cfg = DataConfig.load(args.config) if args.config else DataConfig.load()
    out_path = REPO_ROOT / cfg.splits["output"]

    if args.check:
        if not out_path.is_file():
            raise SystemExit(f"No split file at {out_path}")
        splits = pd.read_csv(out_path)
        problems = verify(splits)
        print(report(splits))
        if problems:
            print("\nVIOLATIONS:")
            for p in problems:
                print(f"  * {p}")
            return 1
        print("\nSplit invariants OK.")
        return 0

    if out_path.is_file() and not args.force:
        raise SystemExit(
            f"{out_path} already exists. Splits are generated once and committed; "
            "regenerating invalidates every result computed against them. "
            "Pass --force only if you genuinely intend that."
        )

    if not cfg.manifest_csv.is_file():
        raise SystemExit(
            f"No manifest at {cfg.manifest_csv}. Run `python -m brats.data.manifest`."
        )
    manifest = pd.read_csv(cfg.manifest_csv)

    sp = cfg.splits
    splits = assign_splits(
        manifest,
        seed=int(sp["seed"]),
        train_frac=float(sp["train_frac"]),
        val_frac=float(sp["val_frac"]),
        test_frac=float(sp["test_frac"]),
    )

    problems = verify(splits)
    if problems:
        print("VIOLATIONS -- not writing:")
        for p in problems:
            print(f"  * {p}")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out_path, index=False)
    print(report(splits))
    log.info("Wrote %s (seed=%s) -- commit this file", out_path, sp["seed"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
