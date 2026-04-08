"""Tests for KeystrokeRateSensor and MouseRateSensor.

These sensors count input events per sampling window. NEVER read content.
In mock mode, the sensor's internal counter can be incremented manually.
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


def test_keystroke_mock_increment_via_inject():
    """In mock mode, calling _inject_event() bumps the counter."""
    s = KeystrokeRateSensor(mock_mode=True)
    for _ in range(7):
        s._inject_event()
    sample = asyncio.run(s.sample())
    assert sample["count"] == 7


def test_keystroke_sample_resets_window():
    """After a sample(), the next sample should report only NEW events."""
    s = KeystrokeRateSensor(mock_mode=True)
    for _ in range(3):
        s._inject_event()
    s1 = asyncio.run(s.sample())
    assert s1["count"] == 3
    # No new events
    s2 = asyncio.run(s.sample())
    assert s2["count"] == 0
    # Two more events
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
