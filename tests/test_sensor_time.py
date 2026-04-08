"""Tests for TimeTonicSensor — slow oscillators encoding day-of-day and day-of-week."""
import math
from datetime import datetime
import pytest
from adapters.mac_desktop.sensor_time import TimeTonicSensor


def test_time_tonic_construction():
    s = TimeTonicSensor()
    assert s.name == "time_tonic"
    assert s.rate_hz == 1.0


@pytest.mark.asyncio
async def test_time_tonic_returns_8_phase_values():
    s = TimeTonicSensor()
    sample = await s.sample()
    # Returns dict with two keys, each a list of 4 floats
    assert "day_phase" in sample
    assert "week_phase" in sample
    assert len(sample["day_phase"]) == 4
    assert len(sample["week_phase"]) == 4
    for v in sample["day_phase"] + sample["week_phase"]:
        assert -1.0 <= v <= 1.0


def test_time_tonic_known_value_at_midnight():
    """At midnight Monday, day phase 0 should be sin(0)=0 and the cosine
    components should be cos(0)=1 (for at least one of the 4 phases).
    Verifies the encoding is deterministic given a fixed timestamp."""
    s = TimeTonicSensor()
    # Monday 2026-01-05 00:00:00 (a known Monday)
    fake = datetime(2026, 1, 5, 0, 0, 0)
    sample = s._encode(fake)
    # day_phase should have at least one near-1 (cos(0)) and one near-0 (sin(0))
    assert max(sample["day_phase"]) > 0.99
    assert min(abs(v) for v in sample["day_phase"]) < 0.01
    # week_phase Monday 00:00 → angle 0, similar
    assert max(sample["week_phase"]) > 0.99


def test_time_tonic_phases_evolve_smoothly():
    """One hour should change day_phase moderately, week_phase very little."""
    s = TimeTonicSensor()
    t1 = datetime(2026, 1, 5, 12, 0, 0)
    t2 = datetime(2026, 1, 5, 13, 0, 0)
    s1 = s._encode(t1)
    s2 = s._encode(t2)
    # day_phase difference noticeable
    day_diff = sum(abs(a - b) for a, b in zip(s1["day_phase"], s2["day_phase"]))
    week_diff = sum(abs(a - b) for a, b in zip(s1["week_phase"], s2["week_phase"]))
    assert day_diff > week_diff
