"""SensorBus and Sensor base class.

Architecture:
- A `SensorBus` is a thread-safe key/value store. Each sensor writes its
  latest encoded value under its name. Readers (the brain tick loop) call
  `.snapshot()` to get the current state.
- A `Sensor` is an asyncio coroutine with a name and a target rate. Its
  `.sample()` method is called periodically and the return value is written
  to the bus under its name. The default `.run(bus)` loop handles timing.

Sensors are independent: each runs at its own rate, the bus holds the
last value (sample-and-hold). The brain tick reads the snapshot — sensors
that haven't updated yet still have their previous value, and a sensor that
has never sampled (or whose source was switched off) is absent.

A sample of `None` means "nothing observed" (device unavailable, source not
shared) and is not written: missing data stays missing instead of turning
into a zero the brain would learn from.

The `mock_mode` constructor flag is the convention for test-friendly
sensors: in mock mode, the sensor returns a deterministic synthetic value
without touching the OS.
"""
from __future__ import annotations
import asyncio
import copy
import threading
import time
from typing import Any


class SensorBus:
    """Sample-and-hold store shared by the event loop and the tick thread.

    Locked because sources are switched on and off at runtime: a key added or
    removed while the tick thread deep-copies the dict would raise
    "dictionary changed size during iteration" and kill the tick thread.
    """

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._written_at: dict[str, float] = {}
        self._lock = threading.Lock()

    def write(self, name: str, value: Any) -> None:
        with self._lock:
            self._values[name] = value
            self._written_at[name] = time.monotonic()

    def discard(self, name: str) -> None:
        """Forget a source entirely, so readers see it as missing, not as its last value."""
        with self._lock:
            self._values.pop(name, None)
            self._written_at.pop(name, None)

    def written_at(self) -> dict[str, float]:
        """Monotonic time of each source's last write."""
        with self._lock:
            return dict(self._written_at)

    def snapshot(self) -> dict[str, Any]:
        """Return a deep copy of the current bus state."""
        with self._lock:
            return copy.deepcopy(self._values)


class Sensor:
    """Base class for sensors. Subclasses set name + rate_hz and override sample()."""

    name: str = "base"
    rate_hz: float = 1.0

    def __init__(self, mock_mode: bool = False) -> None:
        self.mock_mode = mock_mode

    async def sample(self) -> Any:
        """Override in subclass. Return one encoded sample."""
        raise NotImplementedError

    async def run(self, bus: SensorBus) -> None:
        """Run loop: sample at rate_hz, write to bus, sleep, repeat.

        Cancellation-safe. Exceptions in sample() are logged and the loop
        continues — a single broken sample shouldn't kill the sensor.
        """
        period = 1.0 / self.rate_hz
        while True:
            try:
                value = await self.sample()
                if value is not None:
                    bus.write(self.name, value)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Log and continue. We'll plug a real logger in later.
                print(f"[sensor:{self.name}] sample failed: {e}")
            await asyncio.sleep(period)
