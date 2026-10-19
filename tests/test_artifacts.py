"""Immutable stage publication, interruptions, selection and trust-boundary tests."""
import json
from pathlib import Path

import pytest

from brats import artifacts


def _publish(tmp_path, epoch=0):
    source = tmp_path / f"source-{epoch}.pt"
    source.write_bytes(f"checkpoint epoch {epoch}".encode())
    root = tmp_path / "stage" / "run"
    path = artifacts.publish_bundle(root, {"checkpoint.pt": source},
                                    {"kind": "checkpoint", "epoch": epoch}, attempt_id="attempt")
    return root, path


def test_bundle_roundtrip_and_exact_materialization(tmp_path):
    root, path = _publish(tmp_path)
    selected = artifacts.resolve_bundle(root)
    assert selected["manifest_path"] == str(path)
    local = artifacts.materialize_bundle(root, selected, tmp_path / "local")
    assert local["checkpoint.pt"].read_bytes() == b"checkpoint epoch 0"


def test_incomplete_and_corrupt_newest_fall_back(tmp_path):
    root, first = _publish(tmp_path)
    _, latest = _publish(tmp_path, 1)
    selected = artifacts.read_manifest(root, latest)
    artifacts.safe_path(root, selected["files"]["checkpoint.pt"]["path"]).write_bytes(b"broken")
    (root / "completions" / "partial.json").write_text('{"schema_version":')
    assert artifacts.resolve_bundle(root)["manifest_path"] == str(first)


def test_publication_retry_uses_new_keys_and_completion_is_last(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"data")
    original = artifacts._write_json
    seen = []
    def interrupted(path, value):
        seen.append(path)
        for entry in value["files"].values():
            assert artifacts.safe_path(tmp_path / "stage", entry["path"]).read_bytes() == b"data"
        if len(seen) == 1:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{")
            raise OSError("ambiguous interrupted completion close")
        original(path, value)
    monkeypatch.setattr(artifacts, "_write_json", interrupted)
    complete = artifacts.publish_bundle(tmp_path / "stage", {"checkpoint.pt": source},
                                        {"kind": "checkpoint", "epoch": 2}, attempt_id="launch", retries=2)
    assert len(seen) == 2 and seen[0] != seen[1] == complete
    assert len(list((tmp_path / "stage" / "objects").iterdir())) == 2
    assert artifacts.resolve_bundle(tmp_path / "stage")["metadata"]["epoch"] == 2


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../../bad", "a\\b", ""])
def test_reject_unsafe_artifact_paths(tmp_path, name):
    with pytest.raises(artifacts.ArtifactError):
        artifacts.safe_path(tmp_path, name)


def test_symlink_escape_and_tampered_manifest_rejected(tmp_path):
    root, path = _publish(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(artifacts.ArtifactError):
        artifacts.safe_path(root, "escape/stolen")
    manifest = json.loads(path.read_text())
    manifest["files"]["checkpoint.pt"]["path"] = "../source-0.pt"
    path.write_text(json.dumps(manifest))
    with pytest.raises(artifacts.ArtifactError):
        artifacts.read_manifest(root, path)


def test_materialization_cannot_fall_back_after_selection(tmp_path):
    root, _ = _publish(tmp_path)
    _, second = _publish(tmp_path, 1)
    selected = artifacts.resolve_bundle(root)
    second.write_text("{}")
    with pytest.raises(artifacts.ArtifactError):
        artifacts.materialize_bundle(root, selected, tmp_path / "local")


def test_fresh_collision_resume_identity_and_attempt_uniqueness(tmp_path):
    root = tmp_path / "run"
    first = artifacts.initialize_run(root, run_id="run", attempt_id="a", resume=False)
    with pytest.raises(artifacts.ArtifactError, match="already exists"):
        artifacts.initialize_run(root, run_id="run", attempt_id="b", resume=False)
    second = artifacts.initialize_run(root, run_id="run", attempt_id="b", resume=True)
    assert second == first
    with pytest.raises(artifacts.ArtifactError, match="attempt"):
        artifacts.initialize_run(root, run_id="run", attempt_id="b", resume=True)
    with pytest.raises(artifacts.ArtifactError, match="existing"):
        artifacts.initialize_run(tmp_path / "missing", run_id="run", attempt_id="a", resume=True)


def test_empty_resume_bundle_fails(tmp_path):
    with pytest.raises(artifacts.ArtifactError, match="no verified"):
        artifacts.resolve_bundle(tmp_path, retries=1)
