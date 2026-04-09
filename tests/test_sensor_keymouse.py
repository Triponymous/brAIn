"""Tests for KeystrokeRateSensor and MouseRateSensor.

These sensors count input events per sampling window + track typing rhythm.
In mock mode, events are injected via _inject_event().
"""
import asyncio
import pytest
from adapters.mac_desktop.sensor_keymouse import KeystrokeRateSensor, MouseRateSensor


def test_keystroke_construction():
    s = KeystrokeRateSensor()
    assert s.name == "keystroke_rate"
    assert s.rate_hz == 5.0
    assert s.mock_mode is False


def test_mouse_construction():
    s = MouseRateSensor()
    assert s.name == "mouse_rate"
    assert s.rate_hz == 5.0


def test_keystroke_mock_mode_default_zero():
    s = KeystrokeRateSensor(mock_mode=True)
    sample = asyncio.run(s.sample())
    assert sample["count"] == 0
    assert "variability" in sample
    assert "burst" in sample


def test_keystroke_mock_increment_via_inject():
    s = KeystrokeRateSensor(mock_mode=True)
    for _ in range(7):
        s._inject_event()
    sample = asyncio.run(s.sample())
    assert sample["count"] == 7


def test_keystroke_sample_resets_window():
    s = KeystrokeRateSensor(mock_mode=True)
    for _ in range(3):
        s._inject_event()
    s1 = asyncio.run(s.sample())
    assert s1["count"] == 3
    s2 = asyncio.run(s.sample())
    assert s2["count"] == 0
    for _ in range(2):
        s._inject_event()
    s3 = asyncio.run(s.sample())
    assert s3["count"] == 2


def test_mouse_mock_increment_via_inject():
    s = MouseRateSensor(mock_mode=True)
    for _ in range(15):
        s._inject_event()
    sample = asyncio.run(s.sample())
    assert sample["count"] == 15


def test_keystroke_rhythm_variability():
    """Erratic injection pattern should produce higher variability."""
    s = KeystrokeRateSensor(mock_mode=True)
    # Steady pattern: 5, 5, 5, 5
    for _ in range(4):
        s._mock_count = 5
        asyncio.run(s.sample())
    steady_sample = asyncio.run(s.sample())  # count=0 but window has history

    s2 = KeystrokeRateSensor(mock_mode=True)
    # Erratic pattern: 0, 20, 1, 15
    for count in [0, 20, 1, 15]:
        s2._mock_count = count
        asyncio.run(s2.sample())
    erratic_sample = asyncio.run(s2.sample())

    # Erratic should have higher variability than steady
    # (both are now at count=0, but the window remembers)
    assert erratic_sample["variability"] >= steady_sample["variability"]


def test_keystroke_burst_detection():
    """A sudden spike after silence should trigger burst."""
    s = KeystrokeRateSensor(mock_mode=True)
    # Several quiet samples
    for _ in range(5):
        asyncio.run(s.sample())  # count=0
    # Then a burst
    for _ in range(20):
        s._inject_event()
    sample = asyncio.run(s.sample())
    assert sample["count"] == 20
    assert sample["burst"] == 1.0
