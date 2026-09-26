"""Tests for the daemon control server — lifecycle with injected spawn/killer."""
import os
import signal
import subprocess
import sys
import time

import pytest

from fastapi.testclient import TestClient

import server.control as control

LOCAL = "http://127.0.0.1:8900"  # the control server binds loopback only


def test_daemon_lifecycle(monkeypatch, tmp_path):
    monkeypatch.setattr(control, "_PIDFILE", tmp_path / "braind.pid")
    killed: list[int] = []
    app = control.build_control_app(
        spawn=lambda mock: os.getpid(),                 # a real, running pid → probe succeeds
        killer=lambda pid, sig: killed.append(pid),     # don't actually kill anything
        stop_wait=0,                                    # ...so it never exits
    )
    c = TestClient(app, base_url=LOCAL)

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


def test_serves_the_dashboard(tmp_path, monkeypatch):
    """The dashboard is the one UI; the control server hands it out next to /daemon/*."""
    monkeypatch.setattr(control, "_PIDFILE", tmp_path / "braind.pid")
    c = TestClient(control.build_control_app(spawn=lambda m: 0, killer=lambda p, s: None), base_url=LOCAL)
    r = c.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/observatory.html"
    page = c.get("/observatory.html")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert c.get("/live-data.js").status_code == 200
    assert c.get("/daemon/status").json() == {"running": False, "pid": None}   # routes win over files


def test_stop_answers_once_the_daemon_has_exited(monkeypatch, tmp_path):
    """Until it exits the daemon saves and holds its checkpoint; a Start meanwhile would fail."""
    monkeypatch.setattr(control, "_PIDFILE", tmp_path / "braind.pid")
    saving = ("import signal, sys, time\n"
              "signal.signal(signal.SIGTERM, lambda *_: (time.sleep(0.5), sys.exit(0)))\n"
              "print('ready', flush=True)\n"
              "time.sleep(60)\n")
    children = []

    def spawn(mock):
        child = subprocess.Popen([sys.executable, "-c", saving], stdout=subprocess.PIPE, text=True)
        children.append(child)
        child.stdout.readline()  # the SIGTERM handler is in place
        return child.pid

    c = TestClient(control.build_control_app(spawn=spawn), base_url=LOCAL)
    try:
        pid = c.post("/daemon/start", json={}).json()["pid"]
        began = time.monotonic()
        assert c.post("/daemon/stop").json() == {"running": False}
        assert time.monotonic() - began >= 0.4
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)  # exited and reaped before the answer
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.stdout.close()


def test_start_keeps_the_sensor_mode_unless_asked(monkeypatch, tmp_path):
    """The dashboard sends no mode: Stop then Start must not swap mock for real sensors."""
    monkeypatch.setattr(control, "_PIDFILE", tmp_path / "braind.pid")
    spawned: list[bool] = []
    c = TestClient(control.build_control_app(spawn=lambda mock: spawned.append(mock) or os.getpid(),
                                             killer=lambda pid, sig: None, mock=True, stop_wait=0),
                   base_url=LOCAL)
    for body in ({}, {"mock": False}, {}):
        c.post("/daemon/start", json=body)
        c.post("/daemon/stop")
    assert spawned == [True, False, False]


def test_the_dashboard_preview_may_control_the_daemon(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "_PIDFILE", tmp_path / "braind.pid")
    c = TestClient(control.build_control_app(spawn=lambda m: 0, killer=lambda p, s: None), base_url=LOCAL)
    r = c.get("/daemon/status", headers={"Origin": "http://127.0.0.1:4178"})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "http://127.0.0.1:4178"
    assert c.post("/daemon/stop", headers={"Origin": "http://localhost:4178"}).status_code == 200


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
                                    killer=lambda pid, sig: killed.append(pid), stop_wait=0)
    c = TestClient(app, base_url=LOCAL)
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

    with TestClient(app, base_url=LOCAL) as c:  # the context manager runs the lifespan
        assert spawned == [True]
        assert c.get("/daemon/status").json() == {"running": True, "pid": os.getpid()}


def test_autostart_does_not_double_start(monkeypatch, tmp_path):
    pidfile = tmp_path / "braind.pid"
    monkeypatch.setattr(control, "_PIDFILE", pidfile)
    pidfile.write_text(str(os.getpid()))  # already alive from a previous control server
    spawned: list[bool] = []
    app = control.build_control_app(spawn=lambda mock: spawned.append(mock) or 1,
                                    killer=lambda pid, sig: None, autostart=True)

    with TestClient(app, base_url=LOCAL):
        assert spawned == []


def test_without_autostart_nothing_is_spawned_on_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(control, "_PIDFILE", tmp_path / "braind.pid")
    spawned: list[bool] = []
    app = control.build_control_app(spawn=lambda mock: spawned.append(mock) or 1,
                                    killer=lambda pid, sig: None)

    with TestClient(app, base_url=LOCAL):
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
    pidfile.write_text(str(child))
    try:
        # Let the actual probe reap the child; waitid/WNOWAIT is unavailable on macOS.
        deadline = time.monotonic() + 5
        while control._running_pid() is not None:
            assert time.monotonic() < deadline, "exited child was not reaped"
            time.sleep(0.01)
        with pytest.raises(ChildProcessError):
            os.waitpid(child, os.WNOHANG)
    finally:
        try:
            remaining, _ = os.waitpid(child, os.WNOHANG)
            if remaining == 0:
                os.kill(child, signal.SIGKILL)
                os.waitpid(child, 0)
        except ChildProcessError:
            pass


def test_only_the_dashboard_may_control_the_daemon(monkeypatch, tmp_path):
    monkeypatch.setattr(control, "_PIDFILE", tmp_path / "braind.pid")
    killed = []
    app = control.build_control_app(spawn=lambda m: 0, killer=lambda p, s: killed.append(p))
    c = TestClient(app, base_url=LOCAL)
    assert c.post("/daemon/stop", headers={"Origin": "https://evil.example"}).status_code == 403
    assert c.post("/daemon/start", headers={"Origin": "https://evil.example"}, json={}).status_code == 403
    assert c.post("/daemon/stop", headers={"Origin": LOCAL}).status_code == 200   # the dashboard served here
    rebound = TestClient(app, base_url="http://evil.example:8900")                  # DNS rebinding
    assert rebound.post("/daemon/stop", headers={"Origin": "http://evil.example:8900"}).status_code == 400
