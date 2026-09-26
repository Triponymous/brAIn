"""The persistent daemon in the dashboard's telemetry contract (brain.telemetry.v1).

The same frames the opt-in observer serves, so the dashboard's Live view reads
either runtime with one renderer. Frames are recorded only while a dashboard
is polling: with no one watching, the daemon keeps nothing extra in memory.
Sensors carry what the adapter actually observed; a source that is not shared
or has not reported is "disabled" or "unavailable", never a zero. The app is
reported as a broad category, as in the observer, never by name.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

from adapters.mac_desktop.metadata import app_category
from server.telemetry import TelemetryBuffer, code_sha256

_WANTED_S = 5.0      # a poll keeps recording on for this long
_EVERY = 5           # record every 5th tick: 20 Hz at the daemon's 100 Hz, 200 frames = 10 s


def sensor_frame(adapter: Any, now: float | None = None, mono: float | None = None) -> dict[str, dict]:
    """The six contract sensors from the adapter, with the time each was observed."""
    now = time.time() if now is None else now
    mono = time.monotonic() if mono is None else mono
    snap, written, status = adapter.bus.snapshot(), adapter.bus.written_at(), adapter.status()

    def item(state, value=None, unit=None, key=None, window=None):
        return {"status": state, "value": value, "unit": unit,
                "observed_at": now - (mono - written[key]) if state == "available" else None,
                "window_s": window, "source": "braind"}

    def missing(source):
        return item("disabled" if status[source] == "disabled" else "unavailable")

    def live(source):
        return status[source] == "available" and source in snap and source in written

    out = {}
    for key in ("keystroke_rate", "mouse_rate"):
        out[key] = item("available", float(snap[key]["count"]), "events/s", key, 1.0) if live(key) else missing(key)
    out["idle"] = item("available", float(snap["idle"]["seconds"]), "s", "idle") if live("idle") else missing("idle")
    out["active_app"] = (item("available", app_category(str(snap["active_app"]["name"])), "category", "active_app")
                         if live("active_app") else missing("active_app"))
    out["microphone"] = item("available", float(snap["mic"]["rms"]), "rms", "mic") if live("mic") else missing("mic")
    out["wearable"] = item("not_connected")
    return out


class DaemonTelemetry:
    def __init__(self, brain: Any, adapter: Any, *, checkpoint: Path, loaded: bool, consent: Any) -> None:
        self.adapter, self.consent = adapter, consent
        self.buffer = TelemetryBuffer(brain, input_kind="test_fixture" if adapter.mock_mode else "desktop_sensors",
                                      seed=None)
        root = Path(__file__).resolve().parent.parent
        self.buffer.source.update({
            "runner": "braind", "initialization": "checkpoint" if loaded else "fresh",
            "checkpoint": checkpoint.name, "encoder": "mac-desktop-v1",
            "storage": "checkpoint saved every 60 s; episodes, experience and consent beside it",
            "llm": "optional; routed per config.json",
            "microphone": "loudness and 32 frequency bands while shared; never stored",
            "code_sha256": code_sha256(root, sorted((root / "brain").glob("*.py")) + [
                root / p for p in ("adapters/mac_desktop/adapter.py", "adapters/mac_desktop/encoding.py",
                                   "server/braind.py", "server/main.py", "server/daemon_telemetry.py")]),
        })
        self._wanted_until = 0.0

    def wanted(self, mono: float | None = None) -> bool:
        return (time.monotonic() if mono is None else mono) < self._wanted_until

    def record(self, brain: Any, outputs: dict) -> None:
        """Called from the tick thread after a step."""
        if brain.tick_count % _EVERY or not self.wanted():
            return
        self.buffer.record(brain, outputs, sensor_frame(self.adapter), capture_revision=self.consent.revision)

    def read(self, after: int = 0, session: str | None = None) -> dict:
        mono = time.monotonic()
        if not self.wanted(mono):
            with self.buffer.lock:  # a new viewing starts fresh, not with frames from an old one
                self.buffer.frames.clear()
                self.buffer.state = "waiting"
        self._wanted_until = mono + _WANTED_S
        data = self.buffer.read(after, session)
        if not self.adapter.acquiring:
            data["status"] = "paused"
        return data


def build_daemon_telemetry_router(telemetry: DaemonTelemetry) -> APIRouter:
    router = APIRouter()

    @router.get("/api/telemetry")
    async def read_telemetry(after: int = Query(0, ge=0), session: str | None = Query(None, max_length=64)):
        return telemetry.read(after, session)

    return router
