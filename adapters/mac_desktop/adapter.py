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
        self.sensors: list[Sensor] = [
            ActiveAppSensor(mock_mode=mock_mode),
            KeystrokeRateSensor(mock_mode=mock_mode),
            MouseRateSensor(mock_mode=mock_mode),
            IdleSensor(mock_mode=mock_mode),
            MicSensor(mock_mode=mock_mode),
            TimeTonicSensor(mock_mode=mock_mode),
        ]
        self._tasks: list[asyncio.Task] = []

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
        return encode_snapshot(self.bus.snapshot())
