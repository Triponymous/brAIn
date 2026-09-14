"""Tests for the daemon control server — lifecycle with injected spawn/killer."""
import os

from fastapi.testclient import TestClient

import server.control as control


def test_daemon_lifecycle(monkeypatch, tmp_path):
    monkeypatch.setattr(control, "_PIDFILE", tmp_path / "braind.pid")
    killed: list[int] = []
    app = control.build_control_app(
        spawn=lambda mock: os.getpid(),                 # a real, running pid → probe succeeds
        killer=lambda pid, sig: killed.append(pid),     # don't actually kill anything
    )
    c = TestClient(app)

    assert c.get("/daemon/status").json()["running"] is False

    started = c.post("/daemon/start", json={"mock": True}).json()
    assert started["running"] is True and started["pid"] == os.getpid()
    assert c.get("/daemon/status").json()["running"] is True

    # starting again is idempotent
    assert c.post("/daemon/start", json={}).json().get("already") is True

    stopped = c.post("/daemon/stop").json()
    assert stopped["running"] is False
    assert killed == [os.getpid()]
    assert c.get("/daemon/status").json()["running"] is False


def test_serves_console(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "_PIDFILE", tmp_path / "braind.pid")
    c = TestClient(control.build_control_app(spawn=lambda m: 0, killer=lambda p, s: None))
    r = c.get("/")
    assert r.status_code == 200
    assert "Felt-State" in r.text


# ── Supervision ──────────────────────────────────────────────────────────────
# The pidfile is the DESIRED state: present = the daemon should be running.

def _dead_pid() -> int:
    """A pid that certainly is not alive (probe raises ProcessLookupError)."""
    p = os.fork()
    if p == 0:
        os._exit(0)
    os.waitpid(p, 0)
    return p


def test_respawn_if_down_restarts_a_crashed_daemon(monkeypatch, tmp_path):
    pidfile = tmp_path / "braind.pid"
    monkeypatch.setattr(control, "_PIDFILE", pidfile)
    pidfile.write_text(str(_dead_pid()))  # wanted, but gone
    spawned: list[bool] = []

    pid = control.respawn_if_down(lambda mock: spawned.append(mock) or os.getpid(), mock=True)

    assert pid == os.getpid()
    assert spawned == [True]
    assert pidfile.read_text() == str(os.getpid())


def test_respawn_if_down_leaves_a_live_daemon_alone(monkeypatch, tmp_path):
    pidfile = tmp_path / "braind.pid"
    monkeypatch.setattr(control, "_PIDFILE", pidfile)
    pidfile.write_text(str(os.getpid()))
    spawned: list[bool] = []

    assert control.respawn_if_down(lambda mock: spawned.append(mock) or 1, mock=False) is None
    assert spawned == []


def test_respawn_if_down_respects_stop(monkeypatch, tmp_path):
    """After Stop there is no pidfile, so a supervisor must not bring it back."""
    monkeypatch.setattr(control, "_PIDFILE", tmp_path / "braind.pid")
    killed: list[int] = []
    app = control.build_control_app(spawn=lambda mock: os.getpid(),
                                    killer=lambda pid, sig: killed.append(pid))
    c = TestClient(app)
    c.post("/daemon/start", json={})
    c.post("/daemon/stop")
    spawned: list[bool] = []

    assert control.respawn_if_down(lambda mock: spawned.append(mock) or 1, mock=False) is None
    assert spawned == []
    assert killed == [os.getpid()]


def test_autostart_brings_the_daemon_up_on_launch(monkeypatch, tmp_path):
    pidfile = tmp_path / "braind.pid"
    monkeypatch.setattr(control, "_PIDFILE", pidfile)
    spawned: list[bool] = []
    app = control.build_control_app(spawn=lambda mock: spawned.append(mock) or os.getpid(),
                                    killer=lambda pid, sig: None, autostart=True, mock=True)

    with TestClient(app) as c:  # the context manager runs the lifespan
        assert spawned == [True]
        assert c.get("/daemon/status").json() == {"running": True, "pid": os.getpid()}


def test_autostart_does_not_double_start(monkeypatch, tmp_path):
    pidfile = tmp_path / "braind.pid"
    monkeypatch.setattr(control, "_PIDFILE", pidfile)
    pidfile.write_text(str(os.getpid()))  # already alive from a previous control server
    spawned: list[bool] = []
    app = control.build_control_app(spawn=lambda mock: spawned.append(mock) or 1,
                                    killer=lambda pid, sig: None, autostart=True)

    with TestClient(app):
        assert spawned == []


def test_without_autostart_nothing_is_spawned_on_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(control, "_PIDFILE", tmp_path / "braind.pid")
    spawned: list[bool] = []
    app = control.build_control_app(spawn=lambda mock: spawned.append(mock) or 1,
                                    killer=lambda pid, sig: None)

    with TestClient(app):
        assert spawned == []


def test_parser_flags():
    args = control.build_parser().parse_args(["--autostart", "--mock", "--port", "8901"])
    assert args.autostart and args.mock and args.port == 8901
    assert control.build_parser().parse_args([]).autostart is False


# ── launchd install script ───────────────────────────────────────────────────

def test_install_launchd_renders_a_valid_plist_for_this_checkout():
    import plistlib
    import subprocess
    from pathlib import Path

    root = Path(control.__file__).resolve().parents[1]
    out = subprocess.run(["bash", "scripts/install_launchd.sh", "--render-only"],
                         cwd=root, capture_output=True, text=True, check=True).stdout

    assert "__PROJ_DIR__" not in out and "__PYTHON__" not in out
    plist = plistlib.loads(out.encode())
    assert plist["Label"] == "com.brain.control"
    assert plist["WorkingDirectory"] == str(root)
    assert plist["ProgramArguments"] == [str(root / ".venv/bin/python"), "-m", "server.control", "--autostart"]
    assert plist["KeepAlive"] is True
    assert plist["StandardOutPath"] == str(root / "logs/brain.out.log")


def test_running_pid_reaps_a_crashed_child(monkeypatch, tmp_path):
    """A crashed daemon is a zombie child: kill-0 says alive, so the probe must reap."""
    pidfile = tmp_path / "braind.pid"
    monkeypatch.setattr(control, "_PIDFILE", pidfile)
    child = os.fork()
    if child == 0:
        os._exit(0)
    os.waitid(os.P_PID, child, os.WEXITED | os.WNOWAIT)  # exited, deliberately NOT reaped
    pidfile.write_text(str(child))

    assert control._running_pid() is None
    import pytest
    with pytest.raises(ChildProcessError):  # and it has been reaped
        os.waitpid(child, os.WNOHANG)
