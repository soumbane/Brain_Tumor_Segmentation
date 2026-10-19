"""Execute Linux whole-job cleanup against independent sessions and descendants."""
import importlib.util
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

psutil = pytest.importorskip("psutil")
ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "spcs" / "supervise.py"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper")


def alive(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.parametrize("launcher_exits", [False, True])
def test_deadline_kills_stubborn_rank_session_and_loader(tmp_path, launcher_exits):
    pids = tmp_path / "pids"
    # A rank uses setsid like torch elastic subprocess workers. Its loader ignores
    # SIGTERM too. Exiting launcher exercises adopted orphan discovery.
    worker = tmp_path / "worker.py"
    worker.write_text("import os,signal,subprocess,sys,time\n"
                      "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                      "with open(sys.argv[1], 'a') as f: f.write(str(os.getpid())+'\\n'); f.flush()\n"
                      "if len(sys.argv)==2: subprocess.Popen([sys.executable,__file__,sys.argv[1],'loader'],start_new_session=True)\n"
                      "time.sleep(60)\n")
    launcher = tmp_path / "launcher.py"
    launcher.write_text("import subprocess,sys,time,signal\n"
                        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                        f"subprocess.Popen([sys.executable,{str(worker)!r},{str(pids)!r}],start_new_session=True)\n"
                        + ("time.sleep(0.4)\n" if launcher_exits else "time.sleep(60)\n"))
    started = time.monotonic()
    result = subprocess.run([sys.executable, str(SUPERVISOR), "--seconds", "1.5", "--grace", "0.2", "--",
                             sys.executable, str(launcher)], capture_output=True, text=True, timeout=15)
    assert result.returncode == (0 if launcher_exits else 124), result.stderr
    assert time.monotonic() - started < 10
    child_ids = [int(value) for value in pids.read_text().splitlines()]
    assert len(child_ids) == 2
    assert not any(alive(pid) for pid in child_ids)


def test_real_torch_elastic_sessions_and_descendants_are_stopped(tmp_path):
    pytest.importorskip("torch")
    pids = tmp_path / "elastic-pids"
    rank_script = tmp_path / "rank.py"
    rank_script.write_text(
        "import os,signal,subprocess,sys,time\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        "with open(sys.argv[1],'a') as f: f.write(str(os.getpid())+'\\n'); f.flush()\n"
        "if len(sys.argv)==2: subprocess.Popen([sys.executable,__file__,sys.argv[1],'loader'])\n"
        "time.sleep(60)\n"
    )
    process = subprocess.Popen([
        sys.executable, str(SUPERVISOR), "--seconds", "30", "--grace", "0.2", "--",
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        "--nproc_per_node=2", str(rank_script), str(pids),
    ])
    ready_deadline = time.monotonic() + 20
    try:
        while time.monotonic() < ready_deadline:
            if pids.exists() and len(pids.read_text().splitlines()) == 4:
                break
            if process.poll() is not None:
                pytest.fail("torch elastic exited before creating both ranks/loaders")
            time.sleep(0.1)
        else:
            pytest.fail("torch elastic did not initialize within 20s")
        child_ids = [int(value) for value in pids.read_text().splitlines()]
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) == 143
        assert not any(alive(pid) for pid in child_ids)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=10)


def test_external_stop_cleanup_and_missing_deadline(tmp_path):
    result = subprocess.run([sys.executable, str(SUPERVISOR), "--", sys.executable, "-c", "pass"],
                            capture_output=True, timeout=10)
    assert result.returncode != 0
    process = subprocess.Popen([sys.executable, str(SUPERVISOR), "--seconds", "30", "--grace", "0.2", "--",
                                sys.executable, "-c", "import time;time.sleep(60)"])
    time.sleep(0.5)
    children = psutil.Process(process.pid).children(recursive=True)
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=10) == 143
    assert not any(alive(child.pid) for child in children)
