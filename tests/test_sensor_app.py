"""Tests for ActiveAppSensor — uses pyobjc NSWorkspace to read frontmost app.

In tests we use mock_mode=True which returns a deterministic fake app name.
"""
import pytest
from adapters.mac_desktop.sensor_app import ActiveAppSensor


def test_construction_defaults():
    s = ActiveAppSensor()
    assert s.name == "active_app"
    assert s.rate_hz == 1.0
    assert s.mock_mode is False


def test_construction_mock_mode():
    s = ActiveAppSensor(mock_mode=True)
    assert s.mock_mode is True


@pytest.mark.asyncio
async def test_mock_mode_returns_synthetic_apps():
    s = ActiveAppSensor(mock_mode=True)
    samples = []
    for _ in range(5):
        samples.append(await s.sample())
    for sample in samples:
        assert "name" in sample
        assert isinstance(sample["name"], str)
        # New fields present
        assert "switched" in sample
        assert "switch_rate" in sample
        assert isinstance(sample["switch_rate"], float)
    names = {s["name"] for s in samples}
    assert len(names) >= 1


@pytest.mark.asyncio
async def test_switch_rate_increases_on_switch():
    """Rapid app switches should increase switch_rate."""
    s = ActiveAppSensor(mock_mode=True)
    # Mock rotates through 5 apps, so each call is a switch
    samples = []
    for _ in range(10):
        samples.append(await s.sample())
    # After 10 switches in rapid succession, switch_rate should be > 0
    last = samples[-1]
    assert last["switch_rate"] > 0
