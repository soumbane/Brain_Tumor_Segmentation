"""Upload the preprocessed cache and splits to a Snowflake stage.

The uint8 ``.npz`` cache is what gets staged, not the raw NIfTI -- not for size
reasons but because it is the artifact the training code actually reads, and it
removes all decode and normalization work from the training node.
:class:`brats.data.transforms.CachedBratsDataset` reads only ``.npz``, and each
archive already carries ``affine``, ``orig_shape`` and the crop offsets, so
predictions can be written back into the original 240x240x155 grid without the
source NIfTI ever being present.

Sizing (measured). 2755 cases at ~4.3 MB compressed is ~11.8 GB on disk; the same
data occupies ~52.9 GB once decompressed in memory. The ``GPU_NV_M`` node has
93.13 GiB of local disk, so the cache fits with room to spare -- the raw NIfTI
(~40 GB) would fit alongside it if a future step ever needed it on-node.

Stage storage is billed. Check ``scripts/admin_grants.sql`` and attach a budget before
uploading tens of gigabytes.

Usage::

    .venv\\Scripts\\python.exe -m brats.snowflake.stage_data --estimate
    .venv\\Scripts\\python.exe -m brats.snowflake.stage_data --stage @BRATS_MRI.CORE.DATA
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from brats.config import REPO_ROOT, DataConfig

log = logging.getLogger("brats.snowflake.stage_data")

DEFAULT_DATA_STAGE = "@BRATS_MRI.CORE.DATA"
DEFAULT_CKPT_STAGE = "@BRATS_MRI.CORE.CKPT"

CREATE_STAGES_SQL = """
-- Directory tables enabled so the stage can be listed; Snowflake-managed encryption
-- is required for stage volume mounts in SPCS.
CREATE STAGE IF NOT EXISTS BRATS_MRI.CORE.DATA
  DIRECTORY = (ENABLE = TRUE)
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
  COMMENT = 'BraTS 2023 preprocessed uint8 cache';

CREATE STAGE IF NOT EXISTS BRATS_MRI.CORE.CKPT
  DIRECTORY = (ENABLE = TRUE)
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
  COMMENT = 'Per-epoch training checkpoints (resumability guarantee)';

CREATE STAGE IF NOT EXISTS BRATS_MRI.CORE.PAYLOAD
  DIRECTORY = (ENABLE = TRUE)
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
  COMMENT = 'ML Job payload uploads';

CREATE STAGE IF NOT EXISTS BRATS_MRI.CORE.PREDICTIONS
  DIRECTORY = (ENABLE = TRUE)
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
  COMMENT = 'Predictions and evaluation reports';
"""


def estimate(cfg: DataConfig) -> dict[str, float]:
    """Measure the on-disk cache and report against the node's 93.13 GiB limit."""
    files = list(cfg.cache_root.rglob("*.npz"))
    total = sum(f.stat().st_size for f in files)
    n = len(files)
    mean = total / n if n else 0.0
    return {
        "n_cases_cached": n,
        "total_gb": total / 1e9,
        "mean_mb_per_case": mean / 1e6,
        "projected_full_gb": mean * 2755 / 1e9,
        "node_local_disk_gib": 93.13,
    }


def upload(
    cfg: DataConfig,
    stage: str,
    session=None,
    overwrite: bool = False,
    limit: int | None = None,
) -> int:
    """Upload cache files, splits, and the manifest to ``stage``.

    Files are placed under ``<stage>/cache/<cohort>/`` so the training container can
    mount one prefix and resolve paths by cohort.
    """
    from snowflake.snowpark import Session

    session = session or Session.builder.getOrCreate()

    files = sorted(cfg.cache_root.rglob("*.npz"))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(
            f"No .npz files under {cfg.cache_root}. "
            "Run `python -m brats.data.preprocess` first."
        )

    log.info("uploading %d cache files to %s", len(files), stage)
    for i, f in enumerate(files, 1):
        cohort = f.parent.name
        session.file.put(
            str(f),
            f"{stage}/cache/{cohort}/",
            overwrite=overwrite,
            auto_compress=False,  # already compressed; recompressing wastes CPU
        )
        if i % 100 == 0:
            log.info("  %d/%d", i, len(files))

    # Splits and manifest are small but essential: without the committed split CSV the
    # training job would silently re-derive a different split.
    for extra, prefix in (
        (REPO_ROOT / cfg.splits["output"], "splits"),
        (cfg.manifest_csv, "manifest"),
        (cfg.label_stats_csv, "manifest"),
    ):
        if Path(extra).is_file():
            session.file.put(str(extra), f"{stage}/{prefix}/", overwrite=True,
                             auto_compress=False)
            log.info("uploaded %s", Path(extra).name)
        else:
            log.warning("missing, not uploaded: %s", extra)

    return len(files)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--stage", default=DEFAULT_DATA_STAGE)
    ap.add_argument("--estimate", action="store_true", help="size report only, no upload")
    ap.add_argument("--print-sql", action="store_true", help="emit stage DDL and exit")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    if args.print_sql:
        print(CREATE_STAGES_SQL)
        return 0

    cfg = DataConfig.load(args.config) if args.config else DataConfig.load()

    est = estimate(cfg)
    print("=" * 60)
    print("Cache size estimate")
    print("=" * 60)
    for k, v in est.items():
        print(f"  {k:24s}: {v:,.2f}")
    headroom = est["node_local_disk_gib"] - est["projected_full_gb"] / 1.0737
    print(f"  {'headroom_gib':24s}: {headroom:,.2f}")
    if headroom < 10:
        print("\n  WARNING: under 10 GiB headroom on the node. Consider staging only")
        print("  the train+val splits, or fall back to stage-mounted NIfTI decoding.")

    if args.estimate:
        return 0

    n = upload(cfg, args.stage, overwrite=args.overwrite, limit=args.limit)
    log.info("uploaded %d files to %s", n, args.stage)
    return 0


__all__ = ["estimate", "upload", "CREATE_STAGES_SQL", "DEFAULT_DATA_STAGE",
           "DEFAULT_CKPT_STAGE"]


if __name__ == "__main__":
    raise SystemExit(main())
