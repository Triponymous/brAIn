"""MacDesktopAdapter — composes the 6 Mac desktop sensors and encodes them
into a 200-dim sensory input vector for the Brain.

Usage:
    adapter = MacDesktopAdapter(mock_mode=False)
    asyncio.create_task(adapter.run())
    while True:
        sensory_input = adapter.encode()
        brain.tick(sensory_input)
        await asyncio.sleep(0.01)
"""
from __future__ import annotations
import asyncio
import time
from collections import deque
from typing import Any

import torch

from adapters.base import Sensor, SensorBus
from adapters.mac_desktop.sensor_app import ActiveAppSensor
from adapters.mac_desktop.sensor_keymouse import KeystrokeRateSensor, MouseRateSensor
from adapters.mac_desktop.sensor_idle import IdleSensor
from adapters.mac_desktop.sensor_mic import MicSensor
from adapters.mac_desktop.sensor_time import TimeTonicSensor
from adapters.mac_desktop.encoding import encode_snapshot, SENSORY_DIM


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
    def __init__(self, mock_mode: bool = False) -> None:
        self.mock_mode = mock_mode
        self.bus = SensorBus()
        # KeystrokeRate and MouseRate sensors: set to mock_mode=True
        # because we handle input via main-thread pynput listeners below.
        # The sensor-level pynput fallback starts in a worker thread where
        # macOS doesn't deliver events (Cocoa RunLoop issue).
        self.sensors: list[Sensor] = [
            ActiveAppSensor(mock_mode=mock_mode),
            KeystrokeRateSensor(mock_mode=True),  # we inject counts from adapter
            MouseRateSensor(mock_mode=True),       # we inject counts from adapter
            IdleSensor(mock_mode=mock_mode),
            MicSensor(mock_mode=mock_mode),
            TimeTonicSensor(mock_mode=mock_mode),
        ]
        self._tasks: list[asyncio.Task] = []

        # Event TIMESTAMPS (not a per-tick counter) → rolling per-second rate in encode().
        self._key_times: deque[float] = deque(maxlen=2000)
        self._mouse_times: deque[float] = deque(maxlen=2000)
        self._pynput_listeners = []
        self._last_keys = 0.0  # last per-second keyboard rate (chat endpoint)
        self._last_mouse = 0.0  # last per-second mouse rate (chat endpoint)
        if not mock_mode:
            self._start_main_thread_listeners()

    def _start_main_thread_listeners(self) -> None:
        """Start pynput listeners for keyboard+mouse in the current thread."""
        import threading
        self._pynput_lock = threading.Lock()
        try:
            from pynput.keyboard import Listener as KListener
            kl = KListener(on_press=self._on_key)
            kl.daemon = True
            kl.start()
            self._pynput_listeners.append(kl)
            print("[adapter] pynput keyboard listener started (main thread)")
        except Exception as e:
            print(f"[adapter] pynput keyboard failed: {e}")
        try:
            from pynput.mouse import Listener as MListener
            ml = MListener(on_move=self._on_mouse, on_click=self._on_mouse, on_scroll=self._on_mouse)
            ml.daemon = True
            ml.start()
            self._pynput_listeners.append(ml)
            print("[adapter] pynput mouse listener started (main thread)")
        except Exception as e:
            print(f"[adapter] pynput mouse failed: {e}")

    def _on_key(self, *args) -> None:
        with self._pynput_lock:
            self._key_times.append(time.monotonic())

    def _on_mouse(self, *args) -> None:
        with self._pynput_lock:
            self._mouse_times.append(time.monotonic())

    async def run(self) -> None:
        """Start all sensor coroutines. Returns when all are cancelled."""
        self._tasks = [asyncio.create_task(s.run(self.bus)) for s in self.sensors]
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        """Stop background streams (mic, key/mouse listeners) and cancel tasks."""
        for s in self.sensors:
            stop_fn = getattr(s, "stop", None)
            if callable(stop_fn):
                stop_fn()
        for t in self._tasks:
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
        if hasattr(self, '_pynput_lock'):
            now = time.monotonic()
            cutoff = now - 2.0
            with self._pynput_lock:
                while self._key_times and self._key_times[0] < cutoff:
                    self._key_times.popleft()
                while self._mouse_times and self._mouse_times[0] < cutoff:
                    self._mouse_times.popleft()
                keys = list(self._key_times)
                mice = list(self._mouse_times)
            kr = _rate_and_rhythm(keys, now, window=1.0)
            mr = _rate_and_rhythm(mice, now, window=1.0)
            self.bus.write("keystroke_rate", kr)
            self.bus.write("mouse_rate", mr)
            self._last_keys = kr["count"]
            self._last_mouse = mr["count"]
        return encode_snapshot(self.bus.snapshot())
