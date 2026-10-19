"""Offline operator checks. No authentication, SQL, cancellation or pool changes.

Use observed node time (including startup/idle/tail), not training time alone.
A receipt records only this operator's exact job and separately approved deadlines.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re


def estimate(node_hours: float, credits_per_node_hour: float | None = None) -> dict:
    if not math.isfinite(node_hours) or node_hours < 0:
        raise ValueError("node_hours must be finite and nonnegative")
    if credits_per_node_hour is not None and (not math.isfinite(credits_per_node_hour) or credits_per_node_hour <= 0):
        raise ValueError("rate must be evidenced, positive and finite")
    return {"kind": "estimate_not_billing", "observed_node_hours": node_hours,
            "credits_per_node_hour": credits_per_node_hour,
            "estimated_credits": None if credits_per_node_hour is None else node_hours * credits_per_node_hour,
            "limitations": "include startup, idle and tail; shared-pool attribution is not exact; SQL warehouse cost excluded"}


def deadline_action(receipt: dict, observation: dict, now: datetime) -> dict:
    job = receipt["job"]
    if not re.fullmatch(r"CARE_ML_RESEARCH_WKSP\.BRAIN_TUMOR_SEG\.BRATS_[A-Z0-9_]+", job):
        raise ValueError("receipt must name one fully qualified project BRATS job")
    for key in ("job", "created_on", "owner"):
        if observation.get(key) != receipt.get(key) or not receipt.get(key):
            raise ValueError(f"job identity mismatch: {key}; do not cancel")
    if receipt["owner"] != "SFK_CARE_ML_RSRCH_ADM":
        raise ValueError("unexpected job owner")
    if receipt.get("cancellation_approved") is not True:
        raise ValueError("own-job cancellation approval must be recorded")
    def timestamp(key):
        value = datetime.fromisoformat(receipt[key].replace("Z", "+00:00"))
        if value.tzinfo is None:
            raise ValueError("deadlines must include timezone")
        return value
    if now.tzinfo is None:
        raise ValueError("current time must include timezone")
    start_deadline = timestamp("startup_deadline")
    hard_deadline = timestamp("hard_deadline")
    if hard_deadline <= start_deadline:
        raise ValueError("hard deadline must follow startup deadline")
    state = observation["state"].upper()
    if state in {"DONE", "FAILED", "CANCELLED", "DELETED"}:
        action = "terminal_verify_no_running_containers"
    elif now >= hard_deadline or (state != "RUNNING" and now >= start_deadline):
        action = "cancel_exact_job_then_poll_terminal"
    else:
        action = "observe_again_before_deadline"
    return {"job": job, "action": action, "pool_suspend_allowed": False,
            "note": "This tool makes no remote changes. Unknown/unreadable status never proves terminal."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    costs = sub.add_parser("estimate")
    costs.add_argument("--node-hours", type=float, required=True)
    costs.add_argument("--credits-per-node-hour", type=float)
    deadline = sub.add_parser("deadline")
    deadline.add_argument("--receipt", type=Path, required=True)
    deadline.add_argument("--observation", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "estimate":
        result = estimate(args.node_hours, args.credits_per_node_hour)
    else:
        result = deadline_action(json.loads(args.receipt.read_text()),
                                 json.loads(args.observation.read_text()), datetime.now(timezone.utc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
