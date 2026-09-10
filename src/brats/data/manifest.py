"""Build the case manifest and run integrity assertions.

This is the gate the whole project depends on. It answers, from the actual data
rather than from the literature:

* Does every training case have all 5 files, and every validation case 4?
* Do the 4 sequences and the label share one shape and one affine per case?
* **Are label values a subset of {0, 1, 2, 3} with no 4s anywhere?** A 4 means the
  extraction is not BraTS 2023 and every ET result downstream would be garbage.
* How many cases have empty ET or empty ED, per cohort? (Expect many in PED --
  diffuse midline gliomas frequently do not enhance at all.)
* Does any patient appear at more than one timepoint? If so, splits must be
  grouped by patient rather than by study.
* How many MEN cases have tumor touching the brain-mask boundary? (Expect ~90%;
  meningiomas are extra-axial and skull stripping truncates them. These are not
  errors and must not be "fixed".)

Run from the repo root::

    .venv\\Scripts\\python.exe -m brats.data.manifest
    .venv\\Scripts\\python.exe -m brats.data.manifest --workers 12 --quick

Outputs (to ``paths.manifest_dir``): ``manifest.csv``, ``integrity.csv``,
``label_stats.csv``, and a human-readable summary on stdout.

Exit code is non-zero if any *hard* invariant fails, so this can gate a pipeline.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from brats.config import DataConfig
from brats.constants import (
    COHORTS,
    EXPECTED_TRAIN_COUNTS,
    EXPECTED_VAL_COUNTS,
    LABEL_ED,
    LABEL_ET,
    LABEL_NCR,
    NOMINAL_SHAPE,
    REGION_LABELS,
    SEG_SUFFIX,
    SEQUENCES,
    VALID_LABELS,
)

log = logging.getLogger("brats.manifest")

#: e.g. BraTS-GLI-00324-000 -> patient "BraTS-GLI-00324", timepoint "000"
CASE_ID_RE = re.compile(r"^(?P<patient>BraTS-[A-Z]+-\d+)-(?P<timepoint>\d+)$")


@dataclass
class CaseRecord:
    """One row of the manifest plus its integrity findings."""

    case_id: str
    cohort: str
    split_source: str  # "train" | "val"
    patient_id: str
    timepoint: str
    case_dir: str

    path_t1n: str = ""
    path_t1c: str = ""
    path_t2w: str = ""
    path_t2f: str = ""
    path_seg: str = ""

    # geometry
    shape: str = ""
    shape_ok: bool = False
    affine_consistent: bool = False
    shape_consistent: bool = False

    # label facts (train only)
    label_values: str = ""
    has_label_4: bool = False
    labels_valid: bool = False
    vox_ncr: int = 0
    vox_ed: int = 0
    vox_et: int = 0
    vox_wt: int = 0
    vox_tc: int = 0
    empty_et: bool = False
    empty_ed: bool = False
    empty_tc: bool = False
    et_wt_ratio: float = 0.0

    # brain mask / geometry facts
    brain_vox: int = 0
    bbox: str = ""
    tumor_touches_mask_edge: bool = False

    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _parse_case_id(case_id: str) -> tuple[str, str]:
    m = CASE_ID_RE.match(case_id)
    if m:
        return m.group("patient"), m.group("timepoint")
    # Unrecognized layout: treat the whole id as the patient so grouping still
    # errs on the safe side (no accidental cross-split leakage).
    return case_id, ""


def _discover_cases(cfg: DataConfig, cohort: str, split_source: str) -> list[Path]:
    root = cfg.cohort_extract_dir(cohort, split_source)
    if not root.is_dir():
        return []
    return sorted(d for d in root.rglob("*") if d.is_dir() and any(d.glob("*.nii.gz")))


def inspect_case(case_dir_str: str, cohort: str, split_source: str) -> dict:
    """Read one case and compute every integrity fact. Safe to run in a subprocess."""
    case_dir = Path(case_dir_str)
    case_id = case_dir.name
    patient_id, timepoint = _parse_case_id(case_id)
    expect_seg = split_source == "train"

    rec = CaseRecord(
        case_id=case_id,
        cohort=cohort,
        split_source=split_source,
        patient_id=patient_id,
        timepoint=timepoint,
        case_dir=str(case_dir),
    )

    # ---- file presence -------------------------------------------------
    suffixes = list(SEQUENCES) + ([SEG_SUFFIX] if expect_seg else [])
    paths: dict[str, Path] = {}
    for suf in suffixes:
        p = case_dir / f"{case_id}-{suf}.nii.gz"
        if not p.is_file():
            matches = sorted(case_dir.glob(f"*-{suf}.nii.gz"))
            if matches:
                p = matches[0]
            else:
                rec.errors.append(f"missing:{suf}")
                continue
        paths[suf] = p
        setattr(rec, f"path_{suf}", str(p))

    n_expected = len(suffixes)
    if len(paths) != n_expected:
        rec.errors.append(f"file_count:{len(paths)}!={n_expected}")
        return asdict(rec)

    # ---- geometry consistency -----------------------------------------
    shapes, affines = {}, {}
    for suf, p in paths.items():
        try:
            img = nib.load(str(p))
        except Exception as exc:  # noqa: BLE001 - want the reason in the CSV
            rec.errors.append(f"unreadable:{suf}:{type(exc).__name__}")
            return asdict(rec)
        shapes[suf] = tuple(int(v) for v in img.shape)
        affines[suf] = np.asarray(img.affine, dtype=np.float64)

    ref_shape = shapes[SEQUENCES[0]]
    rec.shape = "x".join(str(v) for v in ref_shape)
    rec.shape_consistent = all(s == ref_shape for s in shapes.values())
    if not rec.shape_consistent:
        rec.errors.append("shape_mismatch")

    ref_affine = affines[SEQUENCES[0]]
    rec.affine_consistent = all(
        np.allclose(a, ref_affine, atol=1e-4) for a in affines.values()
    )
    if not rec.affine_consistent:
        rec.errors.append("affine_mismatch")

    # Deviation from 240x240x155 is logged, never silently resampled.
    rec.shape_ok = ref_shape == NOMINAL_SHAPE
    if not rec.shape_ok:
        rec.errors.append(f"nonnominal_shape:{rec.shape}")

    # ---- brain mask (union of nonzero across sequences) ----------------
    brain = np.zeros(ref_shape, dtype=bool)
    for suf in SEQUENCES:
        arr = np.asanyarray(nib.load(str(paths[suf])).dataobj)
        brain |= arr != 0
    rec.brain_vox = int(brain.sum())
    if rec.brain_vox == 0:
        rec.errors.append("empty_brain_mask")
        return asdict(rec)

    idx = np.nonzero(brain)
    rec.bbox = ",".join(
        f"{int(a.min())}:{int(a.max()) + 1}" for a in idx
    )

    # ---- labels (training cases only) ---------------------------------
    if expect_seg:
        seg = np.asanyarray(nib.load(str(paths[SEG_SUFFIX])).dataobj)
        seg = np.rint(seg).astype(np.int16, copy=False)
        values = np.unique(seg)
        rec.label_values = ",".join(str(int(v)) for v in values)

        # THE critical assertion. Label 4 => BraTS 2021 convention => ET channel
        # would silently come out empty everywhere.
        rec.has_label_4 = bool((values == 4).any())
        rec.labels_valid = set(int(v) for v in values).issubset(VALID_LABELS)
        if rec.has_label_4:
            rec.errors.append("LABEL_4_PRESENT")
        elif not rec.labels_valid:
            unexpected = sorted(set(int(v) for v in values) - VALID_LABELS)
            rec.errors.append(f"unexpected_labels:{unexpected}")

        rec.vox_ncr = int((seg == LABEL_NCR).sum())
        rec.vox_ed = int((seg == LABEL_ED).sum())
        rec.vox_et = int((seg == LABEL_ET).sum())

        et_mask = np.isin(seg, list(REGION_LABELS["ET"]))
        tc_mask = np.isin(seg, list(REGION_LABELS["TC"]))
        wt_mask = np.isin(seg, list(REGION_LABELS["WT"]))
        rec.vox_tc = int(tc_mask.sum())
        rec.vox_wt = int(wt_mask.sum())

        rec.empty_et = rec.vox_et == 0
        rec.empty_ed = rec.vox_ed == 0
        rec.empty_tc = rec.vox_tc == 0
        rec.et_wt_ratio = float(rec.vox_et / rec.vox_wt) if rec.vox_wt else 0.0

        # Does tumor abut the edge of the skull-stripped brain? Expected in ~90%
        # of MEN cases: meningiomas are extra-axial and skull stripping cut them.
        # Informational -- these are NOT errors.
        if rec.vox_wt:
            eroded = brain.copy()
            for axis in range(3):
                eroded &= np.roll(brain, 1, axis=axis)
                eroded &= np.roll(brain, -1, axis=axis)
            rec.tumor_touches_mask_edge = bool((wt_mask & ~eroded).any())

        del seg, et_mask, tc_mask, wt_mask

    return asdict(rec)


def build_manifest(cfg: DataConfig, workers: int, limit: int | None = None) -> pd.DataFrame:
    tasks: list[tuple[str, str, str]] = []
    for cohort in COHORTS:
        for src in ("train", "val"):
            dirs = _discover_cases(cfg, cohort, src)
            expected = (
                EXPECTED_TRAIN_COUNTS if src == "train" else EXPECTED_VAL_COUNTS
            )[cohort]
            if len(dirs) != expected:
                log.warning(
                    "%s/%s: found %d case dirs, expected %d",
                    cohort,
                    src,
                    len(dirs),
                    expected,
                )
            if limit:
                dirs = dirs[:limit]
            tasks.extend((str(d), cohort, src) for d in dirs)

    if not tasks:
        raise SystemExit(
            f"No case directories found under {cfg.extract_root}. "
            "Run `python -m brats.data.extract` first."
        )

    log.info("Inspecting %d cases with %d workers", len(tasks), workers)
    rows: list[dict] = []
    if workers <= 1:
        for i, (d, c, s) in enumerate(tasks, 1):
            rows.append(inspect_case(d, c, s))
            if i % 100 == 0:
                log.info("  %d/%d", i, len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(inspect_case, *t): t for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                rows.append(fut.result())
                if i % 100 == 0:
                    log.info("  %d/%d", i, len(tasks))

    df = pd.DataFrame(rows)
    # errors is a list; store as a joined string for CSV round-tripping.
    df["errors"] = df["errors"].apply(lambda e: ";".join(e) if e else "")
    df["ok"] = df["errors"] == ""
    return df.sort_values(["cohort", "split_source", "case_id"]).reset_index(drop=True)


def summarize(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (report_lines, hard_failures)."""
    out: list[str] = []
    hard: list[str] = []
    tr = df[df.split_source == "train"]

    out.append("=" * 72)
    out.append("BraTS 2023 manifest summary")
    out.append("=" * 72)

    out.append("\n-- Case counts ------------------------------------------------")
    counts = df.groupby(["cohort", "split_source"]).size().unstack(fill_value=0)
    out.append(counts.to_string())
    for cohort in COHORTS:
        for src, table in (("train", EXPECTED_TRAIN_COUNTS), ("val", EXPECTED_VAL_COUNTS)):
            got = int(counts.loc[cohort, src]) if cohort in counts.index else 0
            if got != table[cohort]:
                hard.append(f"{cohort}/{src}: {got} cases, expected {table[cohort]}")

    # ---- THE label assertion -----------------------------------------
    out.append("\n-- Label values (the label-3-vs-4 check) ----------------------")
    n4 = int(tr["has_label_4"].sum())
    out.append(f"cases containing label 4 : {n4}   (MUST be 0)")
    if n4:
        hard.append(f"{n4} training cases contain label 4 (BraTS 2021 convention)")
    bad = tr[~tr["labels_valid"]]
    out.append(f"cases with labels outside {{0,1,2,3}} : {len(bad)}")
    if len(bad):
        hard.append(f"{len(bad)} cases have unexpected label values")
    observed = sorted({v for s in tr["label_values"] for v in s.split(",") if v})
    out.append(f"union of observed label values : {observed}")

    # ---- geometry -----------------------------------------------------
    out.append("\n-- Geometry ---------------------------------------------------")
    out.append(f"shape == 240x240x155      : {int(df['shape_ok'].sum())}/{len(df)}")
    out.append(f"shape consistent per case : {int(df['shape_consistent'].sum())}/{len(df)}")
    out.append(f"affine consistent per case: {int(df['affine_consistent'].sum())}/{len(df)}")
    shapes = df["shape"].value_counts()
    if len(shapes) > 1:
        out.append("distinct shapes observed:")
        out.append(shapes.to_string())
    for col, label in (
        ("shape_consistent", "shape"),
        ("affine_consistent", "affine"),
    ):
        n_bad = int((~df[col]).sum())
        if n_bad:
            hard.append(f"{n_bad} cases have inconsistent {label} across files")

    # ---- empty regions ------------------------------------------------
    out.append("\n-- Empty regions per cohort (informational, expect many in PED)")
    emp = tr.groupby("cohort").agg(
        n=("case_id", "size"),
        empty_ET=("empty_et", "sum"),
        empty_ED=("empty_ed", "sum"),
        empty_TC=("empty_tc", "sum"),
    )
    emp["pct_empty_ET"] = (100 * emp.empty_ET / emp.n).round(1)
    out.append(emp.to_string())

    # ---- PED ET/WT ratio: the post-processing gate ---------------------
    ped = tr[(tr.cohort == "PED") & (tr.vox_wt > 0)]
    if len(ped):
        out.append("\n-- PED ET/WT volume ratio (drives the ET gating rule) --------")
        out.append(
            f"n={len(ped)}  median={ped.et_wt_ratio.median():.4f}  "
            f"frac below 0.04 = {(ped.et_wt_ratio < 0.04).mean():.3f}"
        )

    # ---- boundary-abutting tumors -------------------------------------
    out.append("\n-- Tumor touching brain-mask edge (NOT errors) ----------------")
    edge = tr.groupby("cohort").agg(
        n=("case_id", "size"), touching=("tumor_touches_mask_edge", "sum")
    )
    edge["pct"] = (100 * edge.touching / edge.n).round(1)
    out.append(edge.to_string())

    # ---- multi-timepoint patients -------------------------------------
    out.append("\n-- Timepoints per patient (drives grouped splitting) ---------")
    per_patient = tr.groupby("patient_id").size()
    multi = per_patient[per_patient > 1]
    out.append(f"patients with >1 timepoint: {len(multi)} of {len(per_patient)}")
    if len(multi):
        out.append(f"  max timepoints for one patient: {int(multi.max())}")
        out.append("  -> splits MUST be grouped by patient_id (config already does)")

    # ---- integrity errors ---------------------------------------------
    out.append("\n-- Integrity errors ------------------------------------------")
    failed = df[~df["ok"]]
    out.append(f"cases with errors: {len(failed)}/{len(df)}")
    if len(failed):
        tally: dict[str, int] = {}
        for s in failed["errors"]:
            for e in s.split(";"):
                key = e.split(":")[0]
                tally[key] = tally.get(key, 0) + 1
        for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
            out.append(f"  {k}: {v}")

    return out, hard


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--quick", action="store_true", help="inspect only 10 cases per cohort/split"
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )
    cfg = DataConfig.load(args.config) if args.config else DataConfig.load()

    df = build_manifest(cfg, workers=args.workers, limit=10 if args.quick else None)

    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_cols = [
        "case_id", "cohort", "split_source", "patient_id", "timepoint",
        "path_t1n", "path_t1c", "path_t2w", "path_t2f", "path_seg",
        "shape", "bbox", "brain_vox",
    ]
    df[manifest_cols].to_csv(cfg.manifest_csv, index=False)
    df.to_csv(cfg.integrity_csv, index=False)
    label_cols = [
        "case_id", "cohort", "split_source", "label_values", "has_label_4",
        "labels_valid", "vox_ncr", "vox_ed", "vox_et", "vox_tc", "vox_wt",
        "empty_et", "empty_ed", "empty_tc", "et_wt_ratio",
        "tumor_touches_mask_edge",
    ]
    df[df.split_source == "train"][label_cols].to_csv(cfg.label_stats_csv, index=False)

    report, hard = summarize(df)
    text = "\n".join(report)
    print(text)
    (cfg.manifest_dir / "manifest_summary.txt").write_text(text, encoding="utf-8")

    log.info("Wrote %s", cfg.manifest_csv)
    log.info("Wrote %s", cfg.integrity_csv)
    log.info("Wrote %s", cfg.label_stats_csv)

    if args.quick:
        log.warning("--quick was used; counts above are NOT the full dataset")
        return 0
    if hard:
        print("\n" + "!" * 72)
        print("HARD FAILURES -- do not proceed to preprocessing:")
        for h in hard:
            print(f"  * {h}")
        print("!" * 72)
        return 1
    print("\nAll hard invariants passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
