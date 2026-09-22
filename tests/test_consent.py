"""Per-source consent in the persistent daemon: nothing is acquired unless it is
shared, a restart never widens the choice, and switching a source off stops its
acquisition. Mock sensors and fakes only; no real input, audio or app data."""
import asyncio
import json
import sys
import time
import types
from typing import get_args

import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from adapters.base import SensorBus
from adapters.mac_desktop.adapter import SOURCES, MacDesktopAdapter
from adapters.mac_desktop.sensor_app import ActiveAppSensor
from adapters.mac_desktop.sensor_idle import IdleSensor
from adapters.mac_desktop.sensor_mic import MicSensor
from brain.core import Brain
from bridge.exporter import BrainStateExporter
from bridge.proactive import ProactiveEngine
from server.consent import ConsentStore, ConsentUpdate, SourceName, build_consent_router
from server.main import brain_tick_loop
from server.voice import build_voice_router
from server.ws import WSPusher

ALL_ON = dict.fromkeys(SOURCES, True)


def _update(store, revision, **enabled):
    store.update(ConsentUpdate(revision=revision, enabled=enabled), now=1000.0)


async def _until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.02)


async def _cancel(task):
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


# ── the stored choice ────────────────────────────────────────────────

def test_the_api_names_exactly_the_adapter_sources():
    assert set(get_args(SourceName)) == set(SOURCES)


def test_first_start_shares_nothing(tmp_path):
    store = ConsentStore(tmp_path / "consent.json")
    assert store.revision == 0 and not any(store.enabled().values())
    assert not (tmp_path / "consent.json").exists()           # nothing is written until the user chooses


def test_the_choice_survives_a_restart(tmp_path):
    path = tmp_path / "consent.json"
    _update(ConsentStore(path), 0, mic=True, idle=True)
    again = ConsentStore(path)
    assert again.enabled() == {**dict.fromkeys(SOURCES, False), "mic": True, "idle": True}
    assert again.revision == 1 and again.read()["sources"]["mic"]["changed_at"] == 1000.0


@pytest.mark.parametrize("content", [
    "not json",
    json.dumps({"schema": "brain.consent.v0", "revision": 3,
                "sources": {"mic": {"enabled": True, "changed_at": 1}}}),       # unknown format
    json.dumps({"schema": "brain.consent.v1", "revision": 3,
                "sources": {"mic": {"enabled": "yes", "changed_at": 1}}}),      # only a literal true counts
    json.dumps({"schema": "brain.consent.v1", "revision": 3, "sources": ["mic"]}),
    json.dumps({"schema": "brain.consent.v1",
                "sources": {"mic": {"enabled": True, "changed_at": 1}}}),       # no revision
    '{"schema": "brain.consent.v1", "revision": 1e400, "sources": {}}',       # int(inf) overflows
    "[" * 100_000,                                                             # too deep for the parser
], ids=["not-json", "foreign-schema", "string-true", "sources-list", "no-revision", "overflow", "deep-nesting"])
def test_unreadable_consent_means_nothing_is_shared(tmp_path, content):
    path = tmp_path / "consent.json"
    path.write_text(content)
    assert not any(ConsentStore(path).enabled().values())


def test_a_source_added_later_starts_off(tmp_path):
    """A file written before a source existed never enables it; unknown names are ignored."""
    path = tmp_path / "consent.json"
    path.write_text(json.dumps({"schema": "brain.consent.v1", "revision": 2, "sources": {
        "idle": {"enabled": True, "changed_at": 5.0}, "screen": {"enabled": True, "changed_at": 5.0}}}))
    store = ConsentStore(path)
    assert store.enabled() == {**dict.fromkeys(SOURCES, False), "idle": True} and store.revision == 2


def test_switching_on_needs_the_current_revision(tmp_path):
    store = ConsentStore(tmp_path / "consent.json")
    _update(store, 0, idle=True)                                  # revision 0 -> 1
    with pytest.raises(HTTPException) as stale:
        _update(store, 0, mic=True)                               # a second tab still showing revision 0
    assert stale.value.status_code == 409 and not store.enabled()["mic"]
    with pytest.raises(HTTPException):
        _update(store, 7, idle=False)                              # a revision from the future is never valid


def test_a_stop_wins_over_an_outdated_view(tmp_path):
    store = ConsentStore(tmp_path / "consent.json")
    _update(store, 0, idle=True, mic=True)
    _update(store, 1, idle=False)
    _update(store, 0, mic=False)                                  # stale, but only switches off
    assert not any(store.enabled().values()) and store.revision == 3


def test_an_unsaved_stop_still_holds_and_nothing_widens(tmp_path, monkeypatch):
    store = ConsentStore(tmp_path / "consent.json")
    _update(store, 0, mic=True)

    def full_disk(*_):
        raise OSError("No space left on device")

    monkeypatch.setattr(store, "_save", full_disk)
    with pytest.raises(OSError):
        _update(store, 1, mic=False)
    assert store.enabled()["mic"] is False                        # stopped for this run
    with pytest.raises(OSError):
        _update(store, 1, idle=True)
    assert store.enabled()["idle"] is False                       # never widened without being saved


# ── acquisition follows consent ──────────────────────────────────────

async def test_without_consent_only_the_clock_runs():
    adapter = MacDesktopAdapter(mock_mode=True)
    task = asyncio.create_task(adapter.run())
    await _until(lambda: "time_tonic" in adapter.bus.snapshot())
    await asyncio.sleep(0.2)
    assert set(adapter.bus.snapshot()) == {"time_tonic"}
    assert not adapter.acquiring and set(adapter.status().values()) == {"disabled"}
    await _cancel(task)


async def test_switching_a_source_off_stops_it_and_forgets_its_value():
    adapter = MacDesktopAdapter(mock_mode=True, enabled=ALL_ON)
    task = asyncio.create_task(adapter.run())
    await _until(lambda: {"mic", "active_app", "idle"} <= set(adapter.bus.snapshot()))
    assert adapter.status()["mic"] == "available"
    mic_task = adapter._tasks["mic"]

    adapter.apply({**ALL_ON, "mic": False})
    await asyncio.sleep(0.2)                                      # ten microphone periods
    assert mic_task.done()                                        # acquisition stopped, not merely hidden
    assert "mic" not in adapter.bus.snapshot()
    assert adapter.status()["mic"] == "disabled" and adapter.acquiring
    assert adapter.encode()[112:148].sum() == 0                   # the brain hears nothing, not silence

    adapter.apply(ALL_ON)
    await _until(lambda: "mic" in adapter.bus.snapshot())
    assert adapter.status()["mic"] == "available"
    await _cancel(task)
    adapter.stop()


class _FakeStream:
    def __init__(self, log):
        self.log = log
        log.append("open")

    def start(self):
        self.log.append("start")

    def stop(self):
        self.log.append("stop")

    def close(self):
        self.log.append("close")


def test_the_microphone_opens_only_while_shared(monkeypatch):
    log = []
    monkeypatch.setitem(sys.modules, "sounddevice",
                        types.SimpleNamespace(InputStream=lambda **_: _FakeStream(log)))
    monkeypatch.setattr(sys, "platform", "darwin")               # the real path, with a fake device
    adapter = MacDesktopAdapter(mock_mode=False)
    assert log == []                                              # constructing touched no microphone
    adapter.apply({"mic": True})
    assert log == ["open", "start"]
    adapter.apply({"mic": False})
    assert log == ["open", "start", "stop", "close"]


async def test_a_microphone_that_fails_to_open_is_missing_not_simulated(monkeypatch):
    def no_device(**_):
        raise RuntimeError("no input device")

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(InputStream=no_device))
    monkeypatch.setattr(sys, "platform", "darwin")
    mic = MicSensor(mock_mode=False)
    mic.start()
    assert mic.mock_mode is False                                 # it used to switch to simulated room noise
    assert await mic.sample() is None
    bus = SensorBus()
    task = asyncio.create_task(mic.run(bus))
    await asyncio.sleep(0.1)
    await _cancel(task)
    assert "mic" not in bus.snapshot()


def test_stopping_the_microphone_clears_its_buffer():
    mic = MicSensor(mock_mode=True)
    mic._inject_audio(np.ones(1024, dtype=np.float32))
    mic.stop()
    assert not mic._latest_audio.any()


async def test_idle_and_app_without_their_macos_apis_report_nothing(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")               # Quartz/AppKit are not importable here
    assert await IdleSensor(mock_mode=False).sample() is None     # not "the user is active"
    assert await ActiveAppSensor(mock_mode=False).sample() is None  # not a rotating list of invented apps


# ── the organism waits while nothing is shared ───────────────────────

async def test_the_brain_does_not_step_while_nothing_is_shared():
    brain = Brain()
    adapter = MacDesktopAdapter(mock_mode=True)
    loop = asyncio.create_task(brain_tick_loop(brain, adapter, hz=200.0))
    await asyncio.sleep(0.3)
    assert brain.tick_count == 0
    adapter.apply({"idle": True})
    await _until(lambda: brain.tick_count > 0)
    await _cancel(loop)


async def test_proactive_engine_waits_while_nothing_is_shared():
    """A paused brain's state is frozen; the watcher would take it for a held state and ask."""
    brain = Brain(num_sensory=8, num_concept=4, num_wm=4)
    brain._adapter = MacDesktopAdapter(mock_mode=True)
    engine = ProactiveEngine(brain, BrainStateExporter(brain), WSPusher())
    checks = []
    engine._check_felt_state = checks.append
    engine._should_speak = lambda: False
    task = asyncio.create_task(engine.run(check_interval=0.01))
    await asyncio.sleep(0.1)
    assert checks == []
    brain._adapter.apply({"idle": True})
    await _until(lambda: checks)
    await _cancel(task)


def test_recording_is_refused_unless_the_microphone_is_shared(monkeypatch):
    recorded = []
    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(
        rec=lambda frames, **_: recorded.append(frames) or np.zeros((frames, 1), dtype=np.float32),
        wait=lambda: None))
    shared = {"mic": False}
    app = FastAPI()
    app.include_router(build_voice_router(
        tts_engine=None, stt_engine=types.SimpleNamespace(transcribe=lambda audio, sample_rate: "hallo"),
        chat_fn=None, mic_shared=lambda: shared["mic"]))
    c = TestClient(app)
    assert c.post("/api/stt/record", json={"duration": 1}).status_code == 403
    assert c.post("/api/voice-chat", json={"duration": 1}).status_code == 403
    assert recorded == []                                         # the device was never touched
    shared["mic"] = True
    assert c.post("/api/stt/record", json={"duration": 1}).json() == {"text": "hallo"}
    assert c.post("/api/stt/record", json={"duration": 3600}).status_code == 422  # bounded even when shared
    assert recorded == [16000]


# ── the HTTP interface the console uses ──────────────────────────────

def _consent_app(tmp_path):
    store = ConsentStore(tmp_path / "consent.json")
    adapter = MacDesktopAdapter(mock_mode=True)
    app = FastAPI()
    app.include_router(build_consent_router(store, adapter))
    return TestClient(app), store, adapter


def test_consent_api_round_trip(tmp_path):
    c, store, adapter = _consent_app(tmp_path)
    state = c.get("/api/consent").json()
    assert state["paused"] is True and state["revision"] == 0
    assert {e["status"] for e in state["sources"].values()} == {"disabled"}

    state = c.post("/api/consent", json={"revision": 0, "enabled": {"idle": True}}).json()
    assert state["paused"] is False and state["sources"]["idle"]["enabled"] is True
    assert state["sources"]["idle"]["status"] == "waiting"        # shared, nothing observed yet
    assert adapter.enabled["idle"] is True
    assert ConsentStore(tmp_path / "consent.json").enabled()["idle"] is True   # saved

    assert c.post("/api/consent", json={"revision": 0, "enabled": {"mic": True}}).status_code == 409
    assert c.post("/api/consent", json={"revision": 0, "enabled": {"idle": False}}).status_code == 200
    assert adapter.acquiring is False


@pytest.mark.parametrize("body", [
    {"revision": 0, "enabled": {"screen": True}},                 # not a source
    {"revision": 0, "enabled": {"mic": "true"}},                   # a string is not consent
    {"revision": 0, "enabled": {}},
    {"revision": -1, "enabled": {"mic": False}},
    {"revision": 0, "enabled": {"mic": True}, "extra": 1},
])
def test_consent_api_rejects_malformed_changes(tmp_path, body):
    c, store, adapter = _consent_app(tmp_path)
    assert c.post("/api/consent", json=body).status_code == 422
    assert store.revision == 0 and not adapter.acquiring


def test_consent_api_limits_the_payload(tmp_path):
    c, _, _ = _consent_app(tmp_path)
    r = c.post("/api/consent", content=b" " * 5000, headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_consent_api_reports_an_unsaved_stop(tmp_path, monkeypatch):
    c, store, adapter = _consent_app(tmp_path)
    c.post("/api/consent", json={"revision": 0, "enabled": {"mic": True, "idle": True}})

    def read_only(*_):
        raise OSError("Read-only file system")

    monkeypatch.setattr(store, "_save", read_only)
    r = c.post("/api/consent", json={"revision": 1, "enabled": {"mic": False}})
    assert r.status_code == 500 and "restart would bring back" in r.json()["detail"]
    assert adapter.enabled["mic"] is False and adapter.enabled["idle"] is True


# ── review follow-ups: what must not happen at the edges ─────────────

class _Listener:
    def __init__(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def stop(self):
        self.alive = False


def test_a_listener_that_cannot_observe_is_missing_not_silence(monkeypatch):
    """Without Input Monitoring macOS refuses the event tap and pynput's thread ends quietly."""
    adapter = MacDesktopAdapter(mock_mode=False)
    listener = _Listener()
    monkeypatch.setattr(adapter, "_start_listener", lambda name: adapter._listeners.__setitem__(name, listener))
    adapter.apply({"keystroke_rate": True})
    adapter._on_key()
    adapter.encode()
    assert adapter.bus.snapshot()["keystroke_rate"]["count"] == 1
    assert adapter.status()["keystroke_rate"] == "available"
    listener.alive = False
    vec = adapter.encode()
    assert "keystroke_rate" not in adapter.bus.snapshot()
    assert adapter.status()["keystroke_rate"] == "waiting"
    assert vec[60:80].sum() == 0 and vec[161] == 0          # neither a rate nor "not doing much"


async def test_after_a_pause_the_trend_and_signature_start_afresh():
    from bridge.felt_state import FeltState
    from bridge.state_detector import StateDetector
    from server.main import push_loop
    brain = Brain(num_sensory=8, num_concept=4, num_wm=4)
    brain.felt_state = FeltState()
    brain._last_signature = [0.1] * 6                        # from before the pause
    adapter = MacDesktopAdapter(mock_mode=True)              # nothing shared
    detector = StateDetector()
    resets = []
    reset = detector.reset
    detector.reset = lambda: (resets.append(1), reset())
    task = asyncio.create_task(push_loop(brain, WSPusher(rate_hz=100.0), adapter=adapter, detector=detector))
    await asyncio.sleep(0.2)
    assert brain._last_signature is None and len(detector._history) == 0   # a frozen state is no moment
    adapter.apply({"idle": True})
    await _until(lambda: brain._last_signature is not None)
    assert resets == [1]
    await _cancel(task)


async def test_a_pause_breaks_an_unknown_stretch():
    brain = Brain(num_sensory=8, num_concept=4, num_wm=4)
    brain._adapter = MacDesktopAdapter(mock_mode=True)      # nothing shared
    engine = ProactiveEngine(brain, BrainStateExporter(brain), WSPusher())
    engine._felt_watcher.observe(None, now=0.0)              # unknown since t=0
    task = asyncio.create_task(engine.run(check_interval=0.01))
    await asyncio.sleep(0.05)
    await _cancel(task)
    assert engine._felt_watcher.should_ask(now=3600.0) is False   # would ask at once after the pause


def test_live_labeling_needs_a_shared_source_but_a_pending_ask_can_be_answered():
    from bridge.felt_state import FeltState
    from bridge.state_detector import StateDetector
    from server.feel import build_feel_router
    brain = Brain(num_sensory=8, num_concept=4, num_wm=4)
    brain.felt_state = FeltState()
    brain._adapter = MacDesktopAdapter(mock_mode=True)      # paused
    app = FastAPI()
    app.include_router(build_feel_router(brain, StateDetector()))
    c = TestClient(app)
    assert c.post("/api/feel", json={"label": "flow"}).status_code == 409
    brain._pending_ask = {"signature": [0.02] * 6, "cluster": -1, "at": time.time()}
    assert c.post("/api/feel", json={"label": "flow"}).status_code == 200   # that moment was observed
    assert brain.felt_state.known_labels() == ["flow"]


async def test_without_the_idle_source_sleep_is_left_alone():
    brain = Brain()
    brain.enter_sleep()
    adapter = MacDesktopAdapter(mock_mode=True, enabled={"keystroke_rate": True})
    loop = asyncio.create_task(brain_tick_loop(brain, adapter, hz=200.0))
    await _until(lambda: brain.tick_count > 20)
    assert brain.sleep_mode is True          # a missing idle timer used to read as 0 s and wake it
    await _cancel(loop)


def test_a_recording_is_discarded_if_the_microphone_is_switched_off_meanwhile(monkeypatch):
    shared = {"mic": True}

    def wait():
        shared["mic"] = False                # switched off while the worker thread records

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(
        rec=lambda frames, **_: np.zeros((frames, 1), dtype=np.float32), wait=wait))
    transcribed = []
    app = FastAPI()
    app.include_router(build_voice_router(
        tts_engine=None, chat_fn=None, mic_shared=lambda: shared["mic"],
        stt_engine=types.SimpleNamespace(transcribe=lambda audio, sample_rate: transcribed.append(1) or "x")))
    assert TestClient(app).post("/api/stt/record", json={"duration": 1}).status_code == 403
    assert transcribed == []
