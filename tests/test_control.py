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
