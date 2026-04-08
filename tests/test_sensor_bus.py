"""Tests for the SensorBus and Sensor base class.

The SensorBus is a thread-safe sample-and-hold store: each sensor writes
its latest encoded value to a named slot, and the brain tick reads the
current snapshot. Sensors run at independent rates; readers always get
the most recent value.
"""
import asyncio
import pytest
from adapters.base import SensorBus, Sensor


def test_bus_initial_empty():
    bus = SensorBus()
    assert bus.snapshot() == {}


def test_bus_write_and_read():
    bus = SensorBus()
    bus.write("active_app", {"name": "VSCode", "since": 1.0})
    snap = bus.snapshot()
    assert snap == {"active_app": {"name": "VSCode", "since": 1.0}}


def test_bus_multiple_writes_overwrites():
    bus = SensorBus()
    bus.write("idle", 0)
    bus.write("idle", 5)
    bus.write("idle", 12)
    assert bus.snapshot()["idle"] == 12


def test_bus_snapshot_is_a_copy():
    """Mutating the snapshot must not affect the bus."""
    bus = SensorBus()
    bus.write("k", [1, 2, 3])
    snap = bus.snapshot()
    snap["k"] = [9, 9, 9]
    assert bus.snapshot()["k"] == [1, 2, 3]


class _DummySensor(Sensor):
    name = "dummy"
    rate_hz = 10.0

    def __init__(self) -> None:
        super().__init__()
        self.tick_count = 0

    async def sample(self) -> int:
        self.tick_count += 1
        return self.tick_count


def test_sensor_subclass_has_name_and_rate():
    s = _DummySensor()
    assert s.name == "dummy"
    assert s.rate_hz == 10.0


@pytest.mark.asyncio
async def test_sensor_run_writes_to_bus():
    bus = SensorBus()
    s = _DummySensor()
    # Run sensor for ~150ms, should produce ~1-2 samples at 10Hz
    task = asyncio.create_task(s.run(bus))
    await asyncio.sleep(0.15)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert "dummy" in bus.snapshot()
    assert bus.snapshot()["dummy"] >= 1
