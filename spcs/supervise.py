"""Linux whole-job deadline, including torch elastic sessions and loader children.

The supervisor must be a dedicated process: its children belong only to this job.
PR_SET_CHILD_SUBREAPER keeps double-forked/orphaned ranks in the owned tree even
when the torchrun launcher exits. Process creation time protects against PID reuse.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import signal
import subprocess
import sys
import time

import psutil


def _subreaper() -> None:
    if sys.platform != "linux":
        raise RuntimeError("whole-job supervision requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "could not establish child subreaper")


def supervise(command: list[str], *, seconds: float, grace: float = 120,
              poll: float = 0.1) -> int:
    if not command or not 0 < seconds < float("inf") or not 0 <= grace <= 300:
        raise ValueError("command, finite positive deadline and grace <=300s required")
    _subreaper()
    parent = psutil.Process()
    owned: dict[int, psutil.Process] = {}
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}

    def discover():
        for process in parent.children(recursive=True):
            try:
                process.create_time()  # cache identity before any signaling
                previous = owned.get(process.pid)
                if previous is None or previous.create_time() != process.create_time():
                    owned[process.pid] = process
            except psutil.NoSuchProcess:
                pass

    def living():
        result = []
        for process in owned.values():
            try:
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    result.append(process)
            except psutil.NoSuchProcess:
                pass
        return result

    def terminate(signum):
        discover()
        for process in list(owned.values()):
            try:
                process.send_signal(signum)  # psutil checks creation identity
            except psutil.NoSuchProcess:
                pass

    launcher = None
    timed_out = False
    try:
        launcher = subprocess.Popen(command, start_new_session=True)
        deadline = time.monotonic() + seconds
        while True:
            discover()
            status = launcher.poll()
            timed_out = time.monotonic() >= deadline
            if status is not None or timed_out or stopping:
                break
            time.sleep(poll)
        terminate(signal.SIGTERM)
        grace_deadline = time.monotonic() + grace
        while living() and time.monotonic() < grace_deadline:
            discover()
            time.sleep(poll)
        # Re-discover adopted grandchildren, including children created during grace.
        hard_deadline = time.monotonic() + 5
        while time.monotonic() < hard_deadline:
            terminate(signal.SIGKILL)
            launcher.poll()
            while True:
                try:
                    child, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if child == 0:
                    break
            if not living() and not parent.children(recursive=True):
                break
            time.sleep(poll)
        if living():
            raise RuntimeError("whole-job termination could not be confirmed")
        return 124 if timed_out else 143 if stopping else int(status or 0)
    finally:
        if launcher is not None:
            terminate(signal.SIGKILL)
            launcher.wait(timeout=5)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=os.environ.get("BRATS_MAX_RUNTIME_SECONDS"))
    parser.add_argument("--grace", type=float, default=120)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.seconds is None:
        parser.error("set an explicitly approved BRATS_MAX_RUNTIME_SECONDS or --seconds")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return supervise(command, seconds=args.seconds, grace=args.grace)


if __name__ == "__main__":
    raise SystemExit(main())
