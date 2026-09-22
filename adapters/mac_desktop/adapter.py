"""MacDesktopAdapter — composes the 6 Mac desktop sensors and encodes them
into a 200-dim sensory input vector for the Brain.

Nothing is acquired without consent. Each source in SOURCES runs only while
it is enabled; apply() starts and stops acquisition itself (input listeners,
the microphone stream, polling tasks), and a stopped source is removed from
the bus so every reader sees it as missing. The clock (time_tonic) observes
no one and always runs.

Usage:
    adapter = MacDesktopAdapter(mock_mode=False, enabled={"idle": True})
    asyncio.create_task(adapter.run())
    while True:
        if adapter.acquiring:
            brain.tick(adapter.encode())
        await asyncio.sleep(0.01)
"""
from __future__ import annotations
import asyncio
import threading
import time
from collections import deque
from typing import Any, Mapping

import torch

from adapters.base import Sensor, SensorBus
from adapters.mac_desktop.sensor_app import ActiveAppSensor
from adapters.mac_desktop.sensor_keymouse import KeystrokeRateSensor, MouseRateSensor
from adapters.mac_desktop.sensor_idle import IdleSensor
from adapters.mac_desktop.sensor_mic import MicSensor
from adapters.mac_desktop.sensor_time import TimeTonicSensor
from adapters.mac_desktop.encoding import encode_snapshot, SENSORY_DIM

# The sources a user can share. The first four carry the same names as the
# observer's inputs (adapters/mac_desktop/metadata.py).
SOURCES = ("keystroke_rate", "mouse_rate", "idle", "active_app", "mic")
_INPUT_LISTENERS = ("keystroke_rate", "mouse_rate")
_FRESH_S = 5.0  # a source whose last sample is older than this reads as "waiting"


def _rate_and_rhythm(times: list[float], now: float, window: float = 1.0) -> dict:
    """Events in the last `window` seconds + typing rhythm, from event timestamps.

    count       — events in [now-window, now]. This is the PER-SECOND rate the
                  encoder is scaled for (~6-20 keys/s typing, rate bins max at 50),
                  NOT the per-tick count (which is ~0-1 at a 100Hz tick loop).
    variability — coefficient of variation of inter-event gaps: steady typing -> ~0,
                  erratic/bursty -> high. Drives the 'erratic typing' = stress neuron
                  (76), which was dead before because variability was hard-coded to 0.
    burst       — the recent half of the window is much denser than the whole.
    """
    recent = [t for t in times if t > now - window]
    count = len(recent)
    if count < 3:
        return {"count": count, "variability": 0.0, "burst": 0.0}
    gaps = [recent[i + 1] - recent[i] for i in range(count - 1)]
    mean = sum(gaps) / len(gaps)
    if mean <= 0:
        return {"count": count, "variability": 0.0, "burst": 0.0}
    std = (sum((g - mean) ** 2 for g in gaps) / len(gaps)) ** 0.5
    variability = round(min(2.0, std / mean), 3)
    recent_half = sum(1 for t in recent if t > now - window / 2)
    burst = 1.0 if count > 5 and recent_half > count * 0.7 else 0.0
    return {"count": count, "variability": variability, "burst": burst}


class MacDesktopAdapter:
    def __init__(self, mock_mode: bool = False, enabled: Mapping[str, bool] | None = None) -> None:
        self.mock_mode = mock_mode
        self.bus = SensorBus()
        # KeystrokeRate and MouseRate sensors are always mock: in real mode the
        # adapter counts input itself with pynput listeners (see _start), because
        # the sensor-level pynput fallback starts in a worker thread where macOS
        # doesn't deliver events (Cocoa RunLoop issue).
        self.sensors: list[Sensor] = [
            ActiveAppSensor(mock_mode=mock_mode),
            KeystrokeRateSensor(mock_mode=True),
            MouseRateSensor(mock_mode=True),
            IdleSensor(mock_mode=mock_mode),
            MicSensor(mock_mode=mock_mode),
            TimeTonicSensor(mock_mode=mock_mode),
        ]
        self._by_name = {s.name: s for s in self.sensors}
        self._tasks: dict[str, asyncio.Task] = {}
        self._running = False
        self._lock = threading.Lock()  # enabled flags + input timestamps, shared with the tick thread

        # Event TIMESTAMPS (not a per-tick counter) → rolling per-second rate in encode().
        self._key_times: deque[float] = deque(maxlen=2000)
        self._mouse_times: deque[float] = deque(maxlen=2000)
        self._listeners: dict[str, Any] = {}
        self._last_keys = 0.0  # last per-second keyboard rate (chat endpoint)
        self._last_mouse = 0.0  # last per-second mouse rate (chat endpoint)

        self.enabled = dict.fromkeys(SOURCES, False)
        self.apply(enabled or {})

    @property
    def acquiring(self) -> bool:
        """True while at least one source is shared. With none, there is nothing to learn from."""
        return any(self.enabled.values())

    def apply(self, enabled: Mapping[str, bool]) -> None:
        """Start and stop acquisition so that exactly the enabled sources run.

        Call it from the thread that runs the event loop (an async route
        handler): the input listeners must start on the main thread.
        """
        for name in SOURCES:
            on = bool(enabled.get(name, False))
            if on != self.enabled[name]:
                (self._start if on else self._stop)(name)

    def status(self) -> dict[str, str]:
        """Per source: disabled, waiting (shared, nothing observed lately) or available."""
        written, now = self.bus.written_at(), time.monotonic()
        return {name: "disabled" if not self.enabled[name]
                else "available" if now - written.get(name, float("-inf")) < _FRESH_S
                else "waiting"
                for name in SOURCES}

    def _start(self, name: str) -> None:
        with self._lock:
            self.enabled[name] = True
        if name in _INPUT_LISTENERS and not self.mock_mode:
            self._start_listener(name)  # encode() turns its timestamps into the bus value
            return
        if name == "mic":
            self._by_name["mic"].start()
        if self._running:
            self._spawn(name)

    def _stop(self, name: str) -> None:
        with self._lock:
            self.enabled[name] = False
            self.bus.discard(name)
            if name == "keystroke_rate":
                self._key_times.clear()
            elif name == "mouse_rate":
                self._mouse_times.clear()
        listener = self._listeners.pop(name, None)
        if listener is not None:
            listener.stop()
        task = self._tasks.pop(name, None)
        if task is not None:
            task.cancel()  # suspended at its sleep, so it cannot write again
        if name == "mic":
            self._by_name["mic"].stop()

    def _start_listener(self, name: str) -> None:
        try:
            if name == "keystroke_rate":
                from pynput.keyboard import Listener as KListener
                listener = KListener(on_press=self._on_key)
            else:
                from pynput.mouse import Listener as MListener
                listener = MListener(on_move=self._on_mouse, on_click=self._on_mouse, on_scroll=self._on_mouse)
            listener.daemon = True
            listener.start()
            self._listeners[name] = listener
            print(f"[adapter] pynput {name} listener started")
        except Exception as e:
            print(f"[adapter] pynput {name} listener failed: {e}")

    def _on_key(self, *args) -> None:
        with self._lock:
            self._key_times.append(time.monotonic())

    def _on_mouse(self, *args) -> None:
        with self._lock:
            self._mouse_times.append(time.monotonic())

    def _spawn(self, name: str) -> None:
        self._tasks[name] = asyncio.get_running_loop().create_task(self._by_name[name].run(self.bus))

    async def run(self) -> None:
        """Run the enabled sources (and the clock) until cancelled."""
        self._running = True
        for name in ("time_tonic", *SOURCES):
            if name == "time_tonic" or (self.enabled[name] and (
                    self.mock_mode or name not in _INPUT_LISTENERS)):
                self._spawn(name)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            for t in self._tasks.values():
                t.cancel()
            self._tasks.clear()

    def stop(self) -> None:
        """Stop every source's acquisition (streams, listeners, tasks)."""
        for name in SOURCES:
            if self.enabled[name]:
                self._stop(name)
        for t in self._tasks.values():
            t.cancel()

    def encode(self) -> torch.Tensor:
        """Encode the current bus snapshot into the 200-dim sensory vector.

        Keyboard/mouse are aggregated over a rolling 1-SECOND window from event
        timestamps — NOT the per-tick count. At a 100Hz tick loop the per-tick
        count is ~0-1 even while typing fast, which the encoder reads as silence;
        the per-second rate (~6-20) lands in the encoder's scale and fires the
        steady/erratic neurons. Written every call, so it is authoritative over
        the vestigial 5Hz mock keystroke/mouse sensors (which write count=0).
        """
        if not self.mock_mode:
            now = time.monotonic()
            cutoff = now - 2.0
            # Under the lock that _stop() takes, so a source switched off
            # mid-tick is not written back onto the bus.
            with self._lock:
                for name, times in (("keystroke_rate", self._key_times), ("mouse_rate", self._mouse_times)):
                    if not self.enabled[name] or name not in self._listeners:
                        continue  # not shared, or no listener: nothing was counted
                    while times and times[0] < cutoff:
                        times.popleft()
                    rate = _rate_and_rhythm(list(times), now, window=1.0)
                    self.bus.write(name, rate)
                    if name == "keystroke_rate":
                        self._last_keys = rate["count"]
                    else:
                        self._last_mouse = rate["count"]
        return encode_snapshot(self.bus.snapshot())
