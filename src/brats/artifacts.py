"""Closed, immutable artifact publication for a mounted project stage.

Only completion manifests make a bundle visible to readers. No append, rename,
exclusive-create or filesystem lock semantics are assumed on a stage mount.
The operator must authorize exactly one writer per run. Hashes detect corruption,
not malicious provenance: pickle checkpoints must come from a trusted project run.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

log = logging.getLogger(__name__)


class ArtifactError(RuntimeError):
    """No complete, integrity-verified artifact is available."""


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("run and attempt IDs must be simple nonempty path components")
    return value


def safe_path(root: str | Path, relative: str) -> Path:
    """Reject traversal, absolute paths and symlink escapes, on read and write."""
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative or any(part in {".", ".."} for part in relative.split("/")):
        raise ArtifactError("invalid artifact path")
    name = PurePosixPath(relative)
    if not name.parts or name.is_absolute() or any(p in {".", ".."} for p in name.parts):
        raise ArtifactError("artifact path must be relative to its authorized root")
    root = Path(root).resolve()
    dest = root.joinpath(*name.parts).resolve()
    if not dest.is_relative_to(root) or dest == root:
        raise ArtifactError("artifact path escapes its authorized root")
    return dest


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        json.dump(value, target, sort_keys=True, allow_nan=False)
        target.write("\n")


def initialize_run(run_root: str | Path, *, run_id: str, attempt_id: str,
                   resume: bool, wandb_run_id: str = "") -> dict:
    """Rank-zero collision guard, NOT a distributed lock or multi-writer lease."""
    safe_id(run_id)
    safe_id(attempt_id)
    root = Path(run_root)
    identity_path = safe_path(root, "run.json")
    if resume:
        if not identity_path.is_file():
            raise ArtifactError("explicit resume requires an existing run identity")
        identity = json.loads(identity_path.read_text())
        if identity.get("run_id") != run_id:
            raise ArtifactError("run identity mismatch")
    else:
        if root.exists() and any(root.iterdir()):
            raise ArtifactError("fresh run prefix already exists; choose a new run ID")
        identity = {"schema_version": 1, "run_id": run_id,
                    "wandb_run_id": wandb_run_id or uuid.uuid4().hex}
        _write_json(identity_path, identity)
    attempt_path = safe_path(root, f"attempts/{attempt_id}.json")
    if attempt_path.exists():
        raise ArtifactError("launch-attempt ID already exists; use a fresh attempt ID")
    _write_json(attempt_path, {"attempt_id": attempt_id, "resume": resume})
    return identity


def publish_bundle(run_root: str | Path, files: Mapping[str, str | Path], metadata: dict,
                   *, attempt_id: str, retries: int = 3) -> Path:
    """Copy closed local files sequentially, then write completion metadata LAST.

    Every retry has new object keys, including after an ambiguous failed close.
    Incomplete keys are deliberately retained; remote deletion requires approval.
    """
    safe_id(attempt_id)
    if not files or retries < 1:
        raise ValueError("a bundle needs files and at least one publication attempt")
    root = Path(run_root)
    sources = {name: Path(source) for name, source in files.items()}
    entries = {}
    for name, source in sources.items():
        safe_path(root, name)
        if not source.is_file():
            raise FileNotFoundError(source)
        entries[name] = {"size": source.stat().st_size, "sha256": sha256_file(source)}
    # Fail serialization before uploading any objects.
    canonical_hash(metadata)
    failure = None
    for retry in range(retries):
        token = f"{attempt_id}-{uuid.uuid4().hex}"
        manifest = {"schema_version": 1, "metadata": metadata, "files": {}}
        try:
            for name, source in sources.items():
                relative = f"objects/{token}/{name}"
                dest = safe_path(root, relative)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with source.open("rb") as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                if dest.stat().st_size != entries[name]["size"] or sha256_file(dest) != entries[name]["sha256"]:
                    raise ArtifactError("published object failed integrity verification")
                manifest["files"][name] = {"path": relative, **entries[name]}
            completion = safe_path(root, f"completions/{token}.json")
            _write_json(completion, manifest)
            verified = read_manifest(root, completion)
            if verified["manifest_sha256"] != sha256_file(completion):
                raise ArtifactError("completion changed after publication")
            return completion
        except (OSError, ValueError, ArtifactError) as exc:
            failure = exc
            if retry + 1 < retries:
                time.sleep(0.1 * (retry + 1))
    raise ArtifactError(f"publication failed after {retries} attempts: {failure}") from failure


def read_manifest(run_root: str | Path, manifest_path: str | Path) -> dict:
    root = Path(run_root).resolve()
    path = Path(manifest_path).resolve()
    if not path.is_relative_to(root) or path.parent != safe_path(root, "completions"):
        raise ArtifactError("completion manifest is outside the authorized run prefix")
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
        if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
                or not isinstance(manifest.get("files"), dict) or not manifest["files"]
                or not isinstance(manifest.get("metadata"), dict)):
            raise ArtifactError("unsupported or empty completion manifest")
        for name, entry in manifest["files"].items():
            safe_path(root, name)
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not entry["path"].startswith("objects/"):
                raise ArtifactError("object outside immutable object prefix")
            source = safe_path(root, entry["path"])
            if source.stat().st_size != entry["size"] or sha256_file(source) != entry["sha256"]:
                raise ArtifactError("artifact size or SHA256 mismatch")
        return {**manifest, "manifest_path": str(path),
                "manifest_sha256": hashlib.sha256(raw).hexdigest()}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ArtifactError(f"invalid completion manifest: {path.name}") from exc


def resolve_bundle(run_root: str | Path, *, kind: str = "checkpoint", retries: int = 3,
                   epoch: int | None = None) -> dict:
    """Select newest complete integrity-verified bundle; never test compatibility here.

    Caller must reject incompatible content rather than selecting an older epoch.
    Rank zero alone resolves; distributed peers receive this exact selection.
    """
    root = Path(run_root)
    if retries < 1:
        raise ValueError("retries must be positive")
    candidates = []
    # Read only the small headers to order candidates; verify large checkpoint
    # objects lazily, newest first. Reading every retained epoch costs O(run size).
    for attempt in range(retries):
        for path in sorted(safe_path(root, "completions").glob("*.json")):
            try:
                if not path.resolve().is_relative_to(root.resolve()):
                    raise ArtifactError("manifest symlink escapes run root")
                meta = json.loads(path.read_text())["metadata"]
                candidate_epoch = meta.get("epoch", -1)
                if type(candidate_epoch) is not int:
                    raise ValueError("invalid epoch")
                if meta.get("kind") == kind and (epoch is None or candidate_epoch == epoch):
                    candidates.append((candidate_epoch, str(path)))
            except (ArtifactError, OSError, ValueError, KeyError, TypeError, AttributeError):
                log.warning("ignoring incomplete artifact completion %s", path.name)
        if candidates:
            break
        if attempt + 1 < retries:
            time.sleep(0.1 * (attempt + 1))
    for _, path in sorted(set(candidates), reverse=True):
        for attempt in range(retries):
            try:
                return read_manifest(root, path)
            except ArtifactError:
                if attempt + 1 < retries:
                    time.sleep(0.1 * (attempt + 1))
        log.warning("ignoring corrupt artifact completion %s", Path(path).name)
    raise ArtifactError(f"no verified {kind} bundle exists in the authorized run")


def materialize_bundle(run_root: str | Path, manifest: dict, local_dir: str | Path) -> dict[str, Path]:
    """Load exactly a previously selected manifest, with no independent fallback."""
    verified = read_manifest(run_root, manifest["manifest_path"])
    if verified["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ArtifactError("selected completion manifest changed")
    result = {}
    for name, entry in verified["files"].items():
        dest = safe_path(local_dir, name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with safe_path(run_root, entry["path"]).open("rb") as src, dest.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        if dest.stat().st_size != entry["size"] or sha256_file(dest) != entry["sha256"]:
            raise ArtifactError("materialized artifact failed integrity verification")
        result[name] = dest
    return result
