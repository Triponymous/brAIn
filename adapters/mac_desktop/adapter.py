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
from typing import Any

import torch

from adapters.base import Sensor, SensorBus
from adapters.mac_desktop.sensor_app import ActiveAppSensor
from adapters.mac_desktop.sensor_keymouse import KeystrokeRateSensor, MouseRateSensor
from adapters.mac_desktop.sensor_idle import IdleSensor
from adapters.mac_desktop.sensor_mic import MicSensor
from adapters.mac_desktop.sensor_time import TimeTonicSensor
from adapters.mac_desktop.encoding import encode_snapshot, SENSORY_DIM


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

        self._pynput_key_count = 0
        self._pynput_mouse_count = 0
        self._pynput_listeners = []
        self._last_keys = 0.0  # smoothed keyboard count for chat endpoint
        self._last_mouse = 0.0  # smoothed mouse count for chat endpoint
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
            self._pynput_key_count += 1

    def _on_mouse(self, *args) -> None:
        with self._pynput_lock:
            self._pynput_mouse_count += 1

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
        """Encode the current bus snapshot into the 200-dim sensory vector."""
        # Inject main-thread pynput counts into the bus before encoding
        if hasattr(self, '_pynput_lock'):
            with self._pynput_lock:
                key_count = self._pynput_key_count
                mouse_count = self._pynput_mouse_count
                self._pynput_key_count = 0
                self._pynput_mouse_count = 0
            if key_count > 0:
                self.bus.write("keystroke_rate", {"count": key_count, "variability": 0.0, "burst": 0.0})
            if mouse_count > 0:
                self.bus.write("mouse_rate", {"count": mouse_count, "variability": 0.0, "burst": 0.0})
            # Keep a smoothed copy for the chat endpoint to read
            self._last_keys = max(key_count, getattr(self, '_last_keys', 0) * 0.8)
            self._last_mouse = max(mouse_count, getattr(self, '_last_mouse', 0) * 0.8)
        return encode_snapshot(self.bus.snapshot())
