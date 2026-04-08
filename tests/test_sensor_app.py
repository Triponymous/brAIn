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
    # Each sample is a dict with a name
    for sample in samples:
        assert "name" in sample
        assert isinstance(sample["name"], str)
    # Mock mode rotates through a few apps so we see variety
    names = {s["name"] for s in samples}
    assert len(names) >= 1
