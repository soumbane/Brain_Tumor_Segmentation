"""Render a local SPCS smoke/full-training spec. Never connect, upload or submit.

Only a Snowflake SECRET object name is accepted, never an API key. Remote secret
access, audit contents, EAI egress and job privileges still require verification.
The generated file is launch preparation, not authorization to consume GPU time.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path, PurePosixPath
import re
import uuid

import yaml

TEMPLATE = Path(__file__).with_name("service_spec.yaml")
POOL = "CARE_ML_RSRCH_GPU_M"
EAI = "CARE_ML_RSRCH_INTERNET_EAI"
PROJECT_SCHEMA = "CARE_ML_RESEARCH_WKSP.BRAIN_TUMOR_SEG"
ENTITY = "banerjee-s-corewell"
TRAIN_CONFIG = "/app/configs/segresnet_spcs.yaml"
LAUNCHER = [
    "python", "/app/spcs/supervise.py", "--grace", "120", "--",
    "python", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
    "--nproc_per_node=4",
]


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"{label} must be a simple nonempty path component")
    if "REQUIRED_" in value:
        raise ValueError(f"{label} still contains a placeholder")
    return value


def _audit_path(value: str) -> str:
    if not isinstance(value, str) or "\\" in value or any(ord(char) < 32 for char in value):
        raise ValueError("cache audit path must be a mounted OUTPUT_STAGE JSON path")
    path = PurePosixPath(value)
    if (not path.is_relative_to(PurePosixPath("/mnt/output")) or path.suffix != ".json"
            or any(part in {"", ".", ".."} for part in value.split("/")[1:])
            or "REQUIRED_" in value):
        raise ValueError("cache audit path must be a JSON file below /mnt/output without traversal")
    return str(path)


def prepare_spec(*, mode: str, secret_name: str, cache_audit_path: str,
                 split_sha256: str, runtime_seconds: int, run_id: str | None = None,
                 attempt_id: str | None = None, resume: bool = False,
                 smoke_steps: int | None = None, smoke_lambda_cls: float | None = None,
                 interrupt_after_publication: bool = False) -> dict:
    """Return the spec only; no remote check or execution is performed."""
    if mode not in {"smoke", "train"}:
        raise ValueError("choose smoke or train explicitly")
    if (not isinstance(secret_name, str)
            or not re.fullmatch(re.escape(PROJECT_SCHEMA) + r"\.[A-Z_][A-Z0-9_]*", secret_name)
            or "REQUIRED_" in secret_name):
        raise ValueError(f"secret-name must be an existing unquoted SECRET name in {PROJECT_SCHEMA}")
    if not isinstance(split_sha256, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", split_sha256):
        raise ValueError("split-sha256 must be the exact frozen split SHA256 (64 hex characters)")
    if type(runtime_seconds) is not int or runtime_seconds <= 120:
        raise ValueError("runtime-seconds must be an explicit integer greater than the 120-second grace")
    if resume and not run_id:
        raise ValueError("resume requires the existing run-id and a new attempt-id")
    if mode == "train" and (smoke_steps is not None or smoke_lambda_cls is not None or interrupt_after_publication):
        raise ValueError("smoke-only options cannot be used for full training")
    steps = 2 if smoke_steps is None else smoke_steps
    if type(steps) is not int or steps < 2:
        raise ValueError("smoke-steps must be at least two successful updates per attention phase")
    coefficient = 0.1 if smoke_lambda_cls is None else smoke_lambda_cls
    if coefficient not in (0.0, 0.1):
        raise ValueError("smoke-lambda-cls must be 0.1 or 0.0 for the matched smoke gates")
    run_id = _identifier(run_id or f"{mode}-{uuid.uuid4().hex}", "run-id")
    attempt_id = _identifier(attempt_id or uuid.uuid4().hex, "attempt-id")
    if (mode == "smoke") != run_id.startswith("smoke-"):
        raise ValueError("smoke runs require smoke-* IDs; full training must use a separate run prefix")

    document = copy.deepcopy(yaml.safe_load(TEMPLATE.read_text(encoding="utf-8")))
    container = document["spec"]["containers"][0]
    if container["command"] != LAUNCHER + ["-m", "brats.train", "--config", TRAIN_CONFIG]:
        raise ValueError("training template command changed; review the preparation contract")
    if container["env"]["BRATS_WANDB_ENTITY"] != ENTITY:
        raise ValueError("template must use the confirmed W&B team")
    if container["resources"]["requests"]["nvidia.com/gpu"] != 4 or container["resources"]["limits"]["nvidia.com/gpu"] != 4:
        raise ValueError("training and smoke specs must reserve four GPUs")
    container["env"].update({
        "BRATS_RUN_ID": run_id,
        "BRATS_ATTEMPT_ID": attempt_id,
        "BRATS_CACHE_AUDIT_PATH": _audit_path(cache_audit_path),
        "BRATS_EXPECTED_SPLIT_SHA256": split_sha256.lower(),
        "BRATS_MAX_RUNTIME_SECONDS": str(runtime_seconds),
    })
    container["secrets"][0]["snowflakeSecret"]["objectName"] = secret_name
    if mode == "smoke":
        container["command"] = LAUNCHER + [
            "/app/scripts/gpu_smoke_test.py", "--config", TRAIN_CONFIG,
            "--run-id", run_id, "--attempt-id", attempt_id,
            "--steps", str(steps), "--lambda-cls", str(coefficient),
            "--acknowledge-paid-gpu",
        ]
        if interrupt_after_publication:
            container["command"].append("--interrupt-after-publication")
    if resume:
        container["command"].append("--resume")
    rendered = yaml.safe_dump(document, sort_keys=False)
    if "REQUIRED_" in rendered or "{{" in rendered or "}}" in rendered:
        raise ValueError("unresolved placeholder in prepared specification")
    if "WANDB_API_KEY" in container["env"] or "WANDB_BASE_URL" in container["env"]:
        raise ValueError("do not embed credentials or infer an API endpoint from the Forge browser URL")
    return document


def write_spec(document: dict, output: Path, *, mode: str) -> None:
    """Exclusive local write. Existing files, including symlinks, are never replaced."""
    output = Path(output)
    if output.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("output must be a local .yaml or .yml file")
    if output.resolve().is_relative_to(Path("/mnt/data")) or output.resolve().is_relative_to(Path("/mnt/output")):
        raise ValueError("render into local storage, not a dataset/output stage mount")
    payload = yaml.safe_dump(document, sort_keys=False)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(
            f"# Prepared {mode} specification; NOT SUBMITTED.\n"
            f"# Target compute pool: {POOL}\n"
            f"# Required job property (not a YAML field): EXTERNAL_ACCESS_INTEGRATIONS = ({EAI})\n"
            "# Verify remote secret/audit/access and obtain launch approval before submission.\n"
        )
        stream.write(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", required=True, choices=["smoke", "train"])
    parser.add_argument("--secret-name", required=True, help="verified Snowflake SECRET FQN, NOT its value")
    parser.add_argument("--cache-audit-path", required=True, help="passed audit JSON on the /mnt/output stage mount")
    parser.add_argument("--split-sha256", required=True, help="SHA256 of the exact frozen split CSV bytes")
    parser.add_argument("--runtime-seconds", required=True, type=int, help="explicit run limit; preparation does not approve spend")
    parser.add_argument("--run-id", help="fresh ID generated if omitted; mandatory for resume")
    parser.add_argument("--attempt-id", help="new ID generated if omitted; never reuse a launch attempt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--smoke-lambda-cls", type=float, choices=[0.0, 0.1])
    parser.add_argument("--interrupt-after-publication", action="store_true", help="smoke only: deliberate exit 75 after checkpoint publication")
    parser.add_argument("--output", type=Path, required=True, help="new local YAML file in an existing directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        document = prepare_spec(**{key: value for key, value in vars(args).items() if key != "output"})
        write_spec(document, args.output, mode=args.mode)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    container = document["spec"]["containers"][0]
    print(json.dumps({
        "status": "prepared_not_submitted", "mode": args.mode, "specification": str(args.output),
        "compute_pool": POOL, "external_access_integration": EAI,
        "image": container["image"], "run_id": container["env"]["BRATS_RUN_ID"],
        "attempt_id": container["env"]["BRATS_ATTEMPT_ID"],
        "remote_prerequisites_verified": False,
        "remaining_checks": ["SECRET existence/read privilege", "passed staged cache audit and exact split",
                             "EAI/API endpoint and role access", "job cancellation and external startup deadline",
                             "smoke acceptance before full training", "explicit launch approval"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
