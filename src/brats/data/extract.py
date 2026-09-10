"""Extract the six BraTS 2023 archives to a non-synced local path.

Run from the repo root::

    .venv\\Scripts\\python.exe -m brats.data.extract            # all six
    .venv\\Scripts\\python.exe -m brats.data.extract --cohort PED --split-source val

Idempotent: an archive whose expected case count is already present on disk is
skipped unless ``--force`` is given.

Why not OneDrive: both the archives and this repo live under
"OneDrive - Corewell Health". Extracting ~38 GB of NIfTI there makes OneDrive
attempt to sync every file. The extract target in ``configs/data.yaml`` is
deliberately outside the synced tree, and this script refuses to write inside it.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import zipfile
from pathlib import Path

from brats.config import DataConfig
from brats.constants import COHORTS, EXPECTED_TRAIN_COUNTS, EXPECTED_VAL_COUNTS

log = logging.getLogger("brats.extract")

SPLIT_SOURCES = ("train", "val")


def _refuse_synced_path(path: Path) -> None:
    """Hard-fail if the extraction target is inside a cloud-synced folder."""
    lowered = str(path).lower()
    for marker in ("onedrive", "dropbox", "google drive", "box sync", "sharepoint"):
        if marker in lowered:
            raise SystemExit(
                f"Refusing to extract into a cloud-synced path ({marker!r} in "
                f"{path}).\nChange paths.extract_root in configs/data.yaml to a "
                "local directory such as C:\\data\\brats2023."
            )


def _expected_count(cohort: str, split_source: str) -> int:
    table = EXPECTED_TRAIN_COUNTS if split_source == "train" else EXPECTED_VAL_COUNTS
    return table[cohort]


def _existing_case_dirs(dest: Path) -> list[Path]:
    """Case directories already extracted under ``dest``.

    The archives nest everything one level deep inside a container directory, so
    look for any directory that directly contains ``*.nii.gz`` files.
    """
    if not dest.is_dir():
        return []
    return sorted(
        d for d in dest.rglob("*") if d.is_dir() and any(d.glob("*.nii.gz"))
    )


def _free_gib(path: Path) -> float:
    anchor = path
    while not anchor.exists():
        anchor = anchor.parent
    return shutil.disk_usage(anchor).free / (1024**3)


def extract_one(
    cfg: DataConfig, cohort: str, split_source: str, force: bool = False
) -> int:
    """Extract a single archive. Returns the number of case directories present."""
    archive = cfg.archive_path(cohort, split_source)
    dest = cfg.cohort_extract_dir(cohort, split_source)
    _refuse_synced_path(dest)

    if not archive.is_file():
        raise SystemExit(f"Archive not found: {archive}")

    expected = _expected_count(cohort, split_source)
    present = _existing_case_dirs(dest)
    if len(present) >= expected and not force:
        log.info(
            "%s/%s: %d case dirs already present (expected %d) - skipping",
            cohort,
            split_source,
            len(present),
            expected,
        )
        return len(present)

    size_gib = archive.stat().st_size / (1024**3)
    free_gib = _free_gib(dest)
    # .nii.gz stays compressed, so extracted size is close to archive size.
    # Require 1.5x as a margin for filesystem overhead and partial writes.
    if free_gib < size_gib * 1.5:
        raise SystemExit(
            f"Insufficient free space at {dest}: {free_gib:.1f} GiB free, "
            f"need ~{size_gib * 1.5:.1f} GiB for {archive.name}"
        )

    dest.mkdir(parents=True, exist_ok=True)
    log.info(
        "%s/%s: extracting %s (%.2f GiB) -> %s",
        cohort,
        split_source,
        archive.name,
        size_gib,
        dest,
    )

    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        total = len(members)
        for i, member in enumerate(members, 1):
            # Guard against path traversal in the archive.
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise SystemExit(f"Unsafe path in archive: {member.filename}")
            zf.extract(member, dest)
            if i % 2000 == 0 or i == total:
                log.info("  %s/%s: %d/%d entries", cohort, split_source, i, total)

    found = _existing_case_dirs(dest)
    if len(found) != expected:
        log.warning(
            "%s/%s: extracted %d case dirs but expected %d - the manifest step "
            "will report exactly which cases are malformed",
            cohort,
            split_source,
            len(found),
            expected,
        )
    else:
        log.info("%s/%s: %d case dirs OK", cohort, split_source, len(found))
    return len(found)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None, help="path to data.yaml")
    ap.add_argument("--cohort", choices=COHORTS, action="append", default=None)
    ap.add_argument(
        "--split-source", choices=SPLIT_SOURCES, action="append", default=None
    )
    ap.add_argument("--force", action="store_true", help="re-extract even if present")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )

    cfg = DataConfig.load(args.config) if args.config else DataConfig.load()
    _refuse_synced_path(cfg.extract_root)

    cohorts = args.cohort or list(COHORTS)
    sources = args.split_source or list(SPLIT_SOURCES)

    total = 0
    for cohort in cohorts:
        for src in sources:
            total += extract_one(cfg, cohort, src, force=args.force)

    log.info("Done. %d case directories under %s", total, cfg.extract_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
