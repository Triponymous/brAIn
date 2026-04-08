"""Tests for IdleSensor — seconds since last input event."""
import pytest
from adapters.mac_desktop.sensor_idle import IdleSensor


def test_construction():
    s = IdleSensor()
    assert s.name == "idle"
    assert s.rate_hz == 1.0


@pytest.mark.asyncio
async def test_mock_mode_returns_zero():
    s = IdleSensor(mock_mode=True)
    sample = await s.sample()
    assert "seconds" in sample
    assert sample["seconds"] >= 0.0


@pytest.mark.asyncio
async def test_mock_mode_inject_idle():
    s = IdleSensor(mock_mode=True)
    s._mock_idle_seconds = 42.5
    sample = await s.sample()
    assert sample["seconds"] == 42.5
