"""CPU-only import/model and source provenance check for the built image."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys


def source_identity(root: Path) -> dict:
    files = {}
    for directory in ("src", "configs", "splits", "scripts", "spcs"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}:
                files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"source_tree_sha256": digest, "files": files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--model-smoke", action="store_true")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = source_identity(root)
    if args.source_only:
        print(result["source_tree_sha256"])
        return 0
    import torch
    import monai
    import brats.train
    import brats.evaluate
    import brats.qc
    import brats.tracking
    assert (3, 11) <= sys.version_info[:2] < (3, 13), sys.version
    assert platform.machine() == "x86_64", platform.machine()
    assert torch.__version__.split("+")[0] == "2.13.0", torch.__version__
    assert torch.version.cuda == "12.6", torch.version.cuda
    assert monai.__version__ == "1.6.1", monai.__version__
    result.update(python=sys.version, platform=platform.platform(), cuda=torch.version.cuda,
                  dependencies={name: importlib.metadata.version(name) for name in
                                ("torch", "monai", "wandb", "numpy", "psutil")})
    # Settings schema validation never initializes an external W&B run.
    import wandb
    settings = wandb.Settings(console="off", disable_code=True, disable_git=True,
                              save_code=False, x_disable_stats=True, x_disable_meta=True,
                              x_disable_machine_info=True, x_disable_viewer=True,
                              x_save_requirements=False, disable_job_creation=True,
                              host="spcs", program="brats.train", git_remote_url="", git_commit="",
                              x_files_dir="/work/private-sdk-files",
                              ignore_globs=["*", "**/*"], init_timeout=60,
                              x_service_wait=30, x_file_stream_timeout_seconds=30,
                              x_file_stream_retry_max=2, x_file_transfer_retry_max=2,
                              x_file_transfer_timeout_seconds=30)
    assert settings.console == "off"
    if args.model_smoke:
        from brats.models.multitask import MultiTaskConfig, build_model
        torch.set_num_threads(1)
        model = build_model(MultiTaskConfig(init_filters=4, blocks_down=(1, 1, 1, 1), dsdepth=2, cls_hidden=8))
        outputs = model(torch.randn(2, 4, 32, 32, 32))
        loss = sum(tensor.square().mean() for tensor in outputs["seg"]) + outputs["cls"].square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        result["cpu_model_forward_backward"] = "passed"
    if args.provenance:
        args.provenance.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
