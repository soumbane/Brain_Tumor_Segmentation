import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "spcs/monitor.py"
spec = importlib.util.spec_from_file_location("brats_operator_monitor", path)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_no_guessed_credit_rate():
    assert monitor.estimate(2)["estimated_credits"] is None
    assert monitor.estimate(2, 4)["estimated_credits"] == 8
    for rate in (0, -1, float("inf")):
        with pytest.raises(ValueError):
            monitor.estimate(1, rate)


def test_own_job_deadline_cannot_target_another_run():
    receipt = {"job": "CARE_ML_RESEARCH_WKSP.BRAIN_TUMOR_SEG.BRATS_SMOKE_ABC",
               "owner": "SFK_CARE_ML_RSRCH_ADM", "created_on": "2026-10-15T00:00:00Z",
               "startup_deadline": "2026-10-15T00:10:00Z", "hard_deadline": "2026-10-15T00:30:00Z",
               "cancellation_approved": True}
    observation = {**receipt, "state": "PENDING"}
    now = datetime(2026, 10, 15, 0, 11, tzinfo=timezone.utc)
    result = monitor.deadline_action(receipt, observation, now)
    assert result["action"] == "cancel_exact_job_then_poll_terminal"
    assert result["pool_suspend_allowed"] is False
    observation["created_on"] = "2026-10-14T00:00:00Z"
    with pytest.raises(ValueError, match="identity mismatch"):
        monitor.deadline_action(receipt, observation, now)


def test_running_and_terminal_states():
    receipt = {"job": "CARE_ML_RESEARCH_WKSP.BRAIN_TUMOR_SEG.BRATS_A",
               "owner": "SFK_CARE_ML_RSRCH_ADM", "created_on": "2026-10-15T00:00:00Z",
               "startup_deadline": "2026-10-15T00:10:00Z", "hard_deadline": "2026-10-15T00:30:00Z",
               "cancellation_approved": True}
    now = datetime(2026, 10, 15, 0, 11, tzinfo=timezone.utc)
    assert monitor.deadline_action(receipt, {**receipt, "state": "RUNNING"}, now)["action"] == "observe_again_before_deadline"
    assert monitor.deadline_action(receipt, {**receipt, "state": "FAILED"}, now)["action"].startswith("terminal_verify")
    with pytest.raises(ValueError, match="approval"):
        monitor.deadline_action({**receipt, "cancellation_approved": False}, {**receipt, "state": "RUNNING"}, now)
