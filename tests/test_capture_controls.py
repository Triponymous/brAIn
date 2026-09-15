"""Capture policy tests use instrumented fakes, never personal OS input."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import httpx
import pytest
import torch
from fastapi import HTTPException

from adapters.mac_desktop.metadata import INPUTS, DesktopMetadata
from brain.core import Brain
from server.capture import CaptureControl, CaptureUpdate
from server.telemetry import TelemetryBuffer, build_telemetry_app


def fake_collector():
    calls = []
    counter = {"value": 10}
    def count(_, event):
        calls.append(event)
        return counter["value"]
    def permission():
        calls.append("permission")
        return True
    def idle(*_):
        calls.append("idle")
        return 2.0
    def app():
        calls.append("app")
        return SimpleNamespace(localizedName=lambda: "Test editor")
    collector = DesktopMetadata.__new__(DesktopMetadata)
    collector.cg = SimpleNamespace(CGPreflightListenEventAccess=permission,
        CGEventSourceCounterForEventType=count, CGEventSourceSecondsSinceLastEventType=idle,
        kCGEventSourceStateHIDSystemState=1)
    collector.workspace = SimpleNamespace(frontmostApplication=app)
    collector.keys, collector.mouse = [1], [2, 3]
    collector.reset()
    return collector, calls, counter


def update(control, enabled, revision=None, session=None):
    return control.update(CaptureUpdate(session_id=session or control.session_id,
        revision=control.revision if revision is None else revision, enabled=enabled))


def test_all_off_never_calls_any_os_input_api():
    collector, calls, _ = fake_collector()
    sensors = collector.sample(dict.fromkeys(INPUTS, False))
    assert not calls
    assert all(sensors[key]["status"] == "disabled" for key in INPUTS)
    assert all(sensors[key]["value"] is None for key in INPUTS)


@pytest.mark.parametrize("enabled,expected", [
    ("keystroke_rate", ["permission", 1]), ("mouse_rate", ["permission", 2, 3]),
    ("idle", ["permission", "idle"]), ("active_app", ["app"]),
])
def test_each_input_queries_only_its_own_apis(enabled, expected):
    collector, calls, _ = fake_collector()
    policy = {key: key == enabled for key in INPUTS}
    sensors = collector.sample(policy)
    assert calls == expected
    assert all(sensors[key]["status"] == "disabled" for key in INPUTS if key != enabled)


def test_reenable_does_not_include_events_from_the_off_period(monkeypatch):
    collector, _, counter = fake_collector()
    now = [1.0]
    monkeypatch.setattr("adapters.mac_desktop.metadata.time.monotonic", lambda: now[0])
    on = {key: key == "keystroke_rate" for key in INPUTS}
    assert collector.sample(on)["keystroke_rate"]["status"] == "unavailable"
    now[0], counter["value"] = 2.0, 14
    assert collector.sample(on)["keystroke_rate"]["value"] == 4
    collector.sample(dict.fromkeys(INPUTS, False))
    now[0], counter["value"] = 3.0, 900
    assert collector.sample(on)["keystroke_rate"]["status"] == "unavailable"
    now[0], counter["value"] = 4.0, 903
    assert collector.sample(on)["keystroke_rate"]["value"] == 3


def test_every_new_session_starts_off_and_no_automatic_permission_request():
    collector, calls, _ = fake_collector()
    for name in ("one", "two"):
        control = CaptureControl(name, armed=True)
        control.sample(collector, 10)
        assert not any(control.enabled.values())
        assert control.vector is None
        assert not calls


def test_off_discards_cached_input_and_stops_real_model_steps():
    torch.manual_seed(11)
    brain = Brain()
    buffer = TelemetryBuffer(brain, input_kind="desktop_metadata")
    control = CaptureControl(buffer.session_id, armed=True)
    collector, calls, _ = fake_collector()
    update(control, {"active_app": True})
    control.sample(collector, 1)
    assert control.step(brain, buffer)
    assert buffer.read()["frames"][-1]["capture_revision"] == 1
    weights = brain.synapses["sensory_concept"].weights.clone()
    update(control, {"active_app": False})
    calls.clear()
    control.sample(collector, 2)
    assert not control.step(brain, buffer)
    assert not calls and control.vector is None and control.sensors is None
    assert brain.tick_count == 1
    assert torch.equal(weights, brain.synapses["sensory_concept"].weights)
    assert len(buffer.frames) == 1  # Stop is not deletion of historical samples.


def test_stop_confirmation_waits_for_inflight_step():
    control = CaptureControl("thread-test", armed=True)
    collector, _, _ = fake_collector()
    update(control, {"active_app": True})
    control.sample(collector, 1)
    entered, release, stop_started = Event(), Event(), Event()
    def tick(_):
        entered.set()
        assert release.wait(2)
        return {}
    def stop():
        stop_started.set()
        return update(control, dict.fromkeys(INPUTS, False), revision=0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        step = pool.submit(control.step, SimpleNamespace(tick=tick), SimpleNamespace(record=lambda *a, **kw: None))
        assert entered.wait(1)
        stopped = pool.submit(stop)
        assert stop_started.wait(1)
        assert not stopped.done()
        release.set()
        assert step.result(timeout=1)
        assert not any(v["enabled"] for v in stopped.result(timeout=1)["inputs"].values())
        assert not control.step(None, None)


def test_stale_stop_wins_and_prevents_late_enable():
    control = CaptureControl("same", armed=True)
    update(control, {"active_app": True}, revision=0)
    update(control, dict.fromkeys(INPUTS, False), revision=0)
    with pytest.raises(HTTPException) as error:
        update(control, {"active_app": True}, revision=0)
    assert error.value.status_code == 409
    assert not any(control.enabled.values())
    with pytest.raises(HTTPException):
        update(control, {"active_app": False}, session="old-session")


def test_configuration_change_resets_baseline_even_if_no_sample_while_off():
    control = CaptureControl("reset", armed=True)
    collector, _, _ = fake_collector()
    update(control, {"keystroke_rate": True})
    control.sample(collector, 1)
    update(control, {"keystroke_rate": False})
    update(control, {"keystroke_rate": True})
    control.sample(collector, 2)
    assert control.status["keystroke_rate"] == "unavailable"


def test_failed_or_unarmed_runner_cannot_be_enabled():
    for control in (CaptureControl("unarmed"), CaptureControl("failed", armed=True)):
        if control.armed:
            control.fail()
        with pytest.raises(HTTPException) as error:
            update(control, {"active_app": True})
        assert error.value.status_code == 403
        update(control, dict.fromkeys(INPUTS, False))
        assert control.vector is None


@pytest.mark.asyncio
async def test_capture_http_validation_origins_and_paused_telemetry():
    brain = Brain()
    buffer = TelemetryBuffer(brain, input_kind="desktop_metadata")
    control = CaptureControl(buffer.session_id, armed=True)
    app = build_telemetry_app(buffer, control)
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 8888))
    headers = {"Origin": "http://127.0.0.1:4178", "X-Brain-Control": "capture-v1"}
    body = {"session_id": buffer.session_id, "revision": 0, "enabled": {"active_app": True}}
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8001") as client:
        assert (await client.get("/api/capture")).json()["revision"] == 0
        assert (await client.get("/api/telemetry")).json()["status"] == "paused"
        for bad_headers in ({}, {"Origin": headers["Origin"]}, {**headers, "Origin": "null"},
                            {**headers, "Origin": "https://example.org"}, {**headers, "X-Brain-Control": "bad"}):
            assert (await client.post("/api/capture", headers=bad_headers, json=body)).status_code == 403
        for enabled in ({"microphone": True}, {"active_app": 1}, {}, {"active_app": "false"}):
            assert (await client.post("/api/capture", headers=headers, json={**body, "enabled": enabled})).status_code == 422
        assert (await client.post("/api/capture", headers=headers, json={**body, "revision": True})).status_code == 422
        assert (await client.post("/api/capture", headers=headers, json={**body, "extra": 1})).status_code == 422
        assert (await client.post("/api/capture", headers=headers, json={**body, "session_id": "old"})).status_code == 409
        assert (await client.post("/api/capture", headers={**headers, "Content-Type": "application/json"}, content=b"x" * 2049)).status_code == 413
        response = await client.post("/api/capture", headers=headers, json=body)
        assert response.status_code == 200 and response.json()["inputs"]["active_app"]["enabled"] is True
        assert response.headers["access-control-allow-origin"] == headers["Origin"]
        assert response.headers["cache-control"] == "no-store"
        collector, _, _ = fake_collector()
        control.sample(collector, 1)
        control.step(brain, buffer)
        stop = {**body, "enabled": dict.fromkeys(INPUTS, False)}
        assert (await client.post("/api/capture", headers=headers, json=stop)).status_code == 200
        data = (await client.get("/api/telemetry")).json()
        assert data["status"] == "paused" and data["frames"][-1]["tick"] == 1
        assert data["frames"][-1]["capture_revision"] == 1
        assert data["capture"]["revision"] == 2
        preflight = await client.options("/api/capture", headers={"Origin": headers["Origin"],
            "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "x-brain-control,content-type"})
        assert preflight.status_code == 200
        assert (await client.post("/api/capture", headers={**headers, "Host": "evil.example"}, json=stop)).status_code == 400
    remote = httpx.ASGITransport(app=app, client=("192.168.1.50", 8888))
    async with httpx.AsyncClient(transport=remote, base_url="http://127.0.0.1:8001") as client:
        assert (await client.post("/api/capture", headers=headers, json=body)).status_code == 403
