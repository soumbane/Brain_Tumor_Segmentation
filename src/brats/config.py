"""Shared configuration loading and path helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: Repository root, resolved from this file's location (src/brats/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_CONFIG = REPO_ROOT / "configs" / "data.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file, resolving relative paths against the repo root."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class DataConfig:
    """Typed view over ``configs/data.yaml``."""

    archive_root: Path
    extract_root: Path
    manifest_dir: Path
    cache_root: Path
    reports_dir: Path
    archives: dict[str, dict[str, str]]
    ancillary: dict[str, str]
    preprocess: dict[str, Any]
    splits: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_DATA_CONFIG) -> "DataConfig":
        raw = load_yaml(path)
        paths = raw["paths"]

        def _p(key: str, env_var: str | None = None) -> Path:
            if env_var and os.environ.get(env_var):
                return Path(os.environ[env_var])
            v = Path(paths[key])
            return v if v.is_absolute() else REPO_ROOT / v

        return cls(
            archive_root=_p("archive_root"),
            extract_root=_p("extract_root", "BRATS_EXTRACT_ROOT"),
            manifest_dir=_p("manifest_dir", "BRATS_MANIFEST_DIR"),
            cache_root=_p("cache_root", "BRATS_CACHE_ROOT"),
            reports_dir=_p("reports_dir"),
            archives=raw["archives"],
            ancillary=raw["ancillary"],
            preprocess=raw["preprocess"],
            splits=raw["splits"],
        )

    # -- derived paths ----------------------------------------------------

    def archive_path(self, cohort: str, split_source: str) -> Path:
        """Absolute path to a source ``.zip``."""
        return self.archive_root / self.archives[cohort][split_source]

    def cohort_extract_dir(self, cohort: str, split_source: str) -> Path:
        """Directory a given archive extracts into."""
        return self.extract_root / cohort / split_source

    @property
    def manifest_csv(self) -> Path:
        return self.manifest_dir / "manifest.csv"

    @property
    def integrity_csv(self) -> Path:
        return self.manifest_dir / "integrity.csv"

    @property
    def label_stats_csv(self) -> Path:
        return self.manifest_dir / "label_stats.csv"
