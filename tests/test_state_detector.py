"""StateDetector tests — flow, stress, meeting, break detection.

Verifies that the StateDetector correctly identifies sustained user states
from rolling sensor snapshots and neuromodulator levels.
"""
import pytest
from unittest.mock import MagicMock
from bridge.state_detector import StateDetector


def _mock_brain(app="Claude", keys=20, mouse=5, mic=0.001, idle=0,
                switch_rate=0, ne=0.01, sht=0.03):
    brain = MagicMock()
    brain._last_sensor_display = {
        "app": app, "keys": keys, "mouse": mouse,
        "mic_rms": mic, "idle": idle, "switch_rate": switch_rate,
    }
    brain.modulators.snapshot.return_value = {
        "DA": 0.02, "NE": ne, "ACh": 0.03, "5HT": sht,
    }
    brain.tick_count = 100000
    return brain


def test_empty_detect():
    """detect() before any update() returns safe defaults."""
    sd = StateDetector()
    state = sd.detect()
    assert state["flow"] is False
    assert state["stress"] is False
    assert state["meeting"] is False
    assert state["needs_break"] is False
    assert state["flow_app"] is None
    assert state["active_minutes"] == 0


def test_flow_detection():
    """Same app + steady typing + low switch rate for 30+ min = flow."""
    sd = StateDetector()
    brain = _mock_brain(app="Claude", keys=15, switch_rate=0.2)
    # Simulate 30 minutes of same state (1800 snapshots at 1/sec)
    for _ in range(1800):
        sd.update(brain)
    state = sd.detect()
    assert state["flow"] is True
    assert state["flow_duration_min"] >= 30
    assert state["flow_app"] == "Claude"


def test_no_flow_with_switching():
    """High app switching breaks flow."""
    sd = StateDetector()
    brain = _mock_brain(app="Claude", keys=15, switch_rate=5.0)
    for _ in range(1800):
        sd.update(brain)
    state = sd.detect()
    assert state["flow"] is False


def test_no_flow_without_typing():
    """No keyboard activity means no flow, even with same app for 30+ min."""
    sd = StateDetector()
    brain = _mock_brain(app="Claude", keys=0, switch_rate=0.2)
    for _ in range(1800):
        sd.update(brain)
    state = sd.detect()
    assert state["flow"] is False


def test_flow_breaks_on_app_change():
    """Switching apps resets the flow timer."""
    sd = StateDetector()
    brain_a = _mock_brain(app="Claude", keys=15, switch_rate=0.2)
    for _ in range(1500):  # 25 min on Claude
        sd.update(brain_a)
    brain_b = _mock_brain(app="Safari", keys=15, switch_rate=0.2)
    sd.update(brain_b)  # switch
    brain_a2 = _mock_brain(app="Claude", keys=15, switch_rate=0.2)
    for _ in range(300):  # 5 min back on Claude
        sd.update(brain_a2)
    state = sd.detect()
    assert state["flow"] is False  # only 5 min on current app


def test_stress_detection():
    """High NE + low 5HT + high switch rate for >2 min = stress."""
    sd = StateDetector()
    brain = _mock_brain(keys=30, switch_rate=6.0, ne=0.15, sht=0.005)
    for _ in range(120):
        sd.update(brain)
    state = sd.detect()
    assert state["stress"] is True


def test_no_stress_with_normal_modulators():
    """Normal modulator levels should not trigger stress."""
    sd = StateDetector()
    brain = _mock_brain(keys=30, switch_rate=6.0, ne=0.02, sht=0.03)
    for _ in range(120):
        sd.update(brain)
    state = sd.detect()
    assert state["stress"] is False


def test_meeting_detection():
    """Zoom + mic active + no typing = meeting."""
    sd = StateDetector()
    brain = _mock_brain(app="Zoom", keys=0, mic=0.02, idle=0)
    for _ in range(60):
        sd.update(brain)
    state = sd.detect()
    assert state["meeting"] is True
    assert state["meeting_duration_min"] > 0


def test_meeting_teams():
    """Microsoft Teams should also be detected as meeting app."""
    sd = StateDetector()
    brain = _mock_brain(app="Microsoft Teams", keys=0, mic=0.015, idle=0)
    for _ in range(60):
        sd.update(brain)
    state = sd.detect()
    assert state["meeting"] is True


def test_no_meeting_without_mic():
    """Meeting app without mic activity is not a meeting."""
    sd = StateDetector()
    brain = _mock_brain(app="Zoom", keys=0, mic=0.001, idle=0)
    for _ in range(60):
        sd.update(brain)
    state = sd.detect()
    assert state["meeting"] is False


def test_break_needed():
    """Active work for 90+ min without break = needs_break."""
    sd = StateDetector()
    brain = _mock_brain(keys=15, idle=0)
    for _ in range(5400):  # 90 min
        sd.update(brain)
    state = sd.detect()
    assert state["needs_break"] is True
    assert state["active_minutes"] >= 90


def test_break_reset_after_idle():
    """Idle period > 5 min resets the break timer."""
    sd = StateDetector()
    brain_active = _mock_brain(keys=15, idle=0)
    brain_idle = _mock_brain(keys=0, idle=400)
    for _ in range(5400):
        sd.update(brain_active)
    for _ in range(60):
        sd.update(brain_idle)
    state = sd.detect()
    assert state["needs_break"] is False


def test_no_break_before_threshold():
    """Less than 90 minutes active should not trigger break."""
    sd = StateDetector()
    brain = _mock_brain(keys=15, idle=0)
    for _ in range(5000):  # ~83 min
        sd.update(brain)
    state = sd.detect()
    assert state["needs_break"] is False
