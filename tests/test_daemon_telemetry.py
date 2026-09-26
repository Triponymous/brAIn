"""The persistent daemon in the dashboard's telemetry contract: observed values only, recorded only while watched."""
import time

import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from adapters.mac_desktop.adapter import SOURCES, MacDesktopAdapter
from brain.core import Brain
from bridge.felt_state import FeltState
from bridge.state_detector import StateDetector
from server.consent import ConsentStore
from server.daemon_telemetry import DaemonTelemetry, build_daemon_telemetry_router, sensor_frame
from server.feel import build_feel_router


def _adapter(**shared):
    return MacDesktopAdapter(mock_mode=True, enabled={key: True for key in shared})


def test_sensors_report_what_was_observed_and_nothing_else():
    adapter = _adapter(idle=True, active_app=True, mic=True)
    adapter.bus.write("idle", {"seconds": 4.5})
    adapter.bus.write("active_app", {"name": "Visual Studio Code", "background_apps": ["Mail"]})
    sensors = sensor_frame(adapter)
    assert sensors["keystroke_rate"]["status"] == "disabled" and sensors["keystroke_rate"]["value"] is None
    assert sensors["idle"]["status"] == "available" and sensors["idle"]["value"] == 4.5
    assert abs(sensors["idle"]["observed_at"] - time.time()) < 2
    assert sensors["active_app"]["value"] == "development"          # a category, never the app name
    assert "Visual Studio Code" not in str(sensors) and "Mail" not in str(sensors)
    assert sensors["microphone"]["status"] == "unavailable"         # shared, but nothing heard yet: not zero
    assert sensors["wearable"]["status"] == "not_connected"


def _telemetry(tmp_path, adapter):
    brain = Brain()
    store = ConsentStore(tmp_path / "consent.json")
    return brain, DaemonTelemetry(brain, adapter, checkpoint=tmp_path / "brain.sqlite", loaded=False, consent=store)


def test_frames_are_recorded_only_while_a_dashboard_watches(tmp_path):
    adapter = _adapter(idle=True)
    brain, telemetry = _telemetry(tmp_path, adapter)
    for _ in range(10):
        telemetry.record(brain, brain.tick(torch.zeros(200)))
    assert len(telemetry.buffer.frames) == 0                         # nobody polled: nothing kept
    first = telemetry.read()
    assert first["schema"] == "brain.telemetry.v1" and first["frames"] == []
    assert first["architecture"] == {"s": 200, "e": 500, "c": 200, "w": 100}
    assert first["source"]["runner"] == "braind" and first["source"]["input_kind"] == "test_fixture"
    for _ in range(10):
        telemetry.record(brain, brain.tick(torch.zeros(200)))
    frames = telemetry.read(session=first["session_id"])["frames"]
    assert [f["tick"] % 5 for f in frames] == [0, 0]                   # every 5th tick
    assert frames[0]["sensors"]["idle"]["status"] in ("available", "unavailable")
    assert frames[0]["capture_revision"] == 0


def test_a_new_viewing_starts_fresh_and_paused_says_so(tmp_path):
    adapter = _adapter(idle=True)
    brain, telemetry = _telemetry(tmp_path, adapter)
    telemetry.read()
    for _ in range(10):
        telemetry.record(brain, brain.tick(torch.zeros(200)))
    telemetry._wanted_until = 0                                        # the dashboard went away
    assert telemetry.read()["frames"] == []                            # old frames are not replayed as recent
    adapter.apply({})
    assert telemetry.read()["status"] == "paused"


def test_real_mode_is_labelled_as_desktop_sensors(tmp_path):
    adapter = MacDesktopAdapter(mock_mode=False)
    _, telemetry = _telemetry(tmp_path, adapter)
    source = telemetry.read()["source"]
    assert source["input_kind"] == "desktop_sensors" and source["checkpoint"] == "brain.sqlite"
    assert len(source["code_sha256"]) == 64


def test_router_serves_the_contract(tmp_path):
    _, telemetry = _telemetry(tmp_path, _adapter(idle=True))
    app = FastAPI()
    app.include_router(build_daemon_telemetry_router(telemetry))
    body = TestClient(app).get("/api/telemetry", params={"after": 0}).json()
    assert body["schema"] == "brain.telemetry.v1" and body["status"] == "waiting"


def test_feel_while_paused_recognizes_nothing_but_shows_an_open_ask():
    brain = Brain(num_sensory=8, num_concept=4, num_wm=4)
    brain.felt_state = FeltState()
    brain.felt_state.label("flow", [0.03] * 6)
    brain._adapter = MacDesktopAdapter(mock_mode=True)                 # nothing shared
    brain._pending_ask = {"signature": [0.02] * 6, "cluster": -1, "at": time.time()}
    app = FastAPI()
    app.include_router(build_feel_router(brain, StateDetector()))
    body = TestClient(app).get("/api/feel").json()
    assert body["paused"] is True and body["recognized"] is None and body["signature"] is None
    assert body["known_labels"] == ["flow"] and body["pending_ask"]["at"] == brain._pending_ask["at"]
