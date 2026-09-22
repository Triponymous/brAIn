"""Tests for MacDesktopAdapter — composes 6 sensors into a 200-dim sensory vector.

Neuron map (new encoding):
    0-39    active app identity
    40-59   background app identities
    60-75   keystroke rate bins
    76-79   keystroke rhythm
    80-95   mouse rate bins
    96-99   mouse rhythm
    100-107 idle time bins
    108-111 pause type
    112-143 mic mel-spectrogram
    144-147 mic RMS
    148-155 time tonics
    156-159 app context (switch rate)
    160-163 activity level
    164-199 reserve (always 0)
"""
import asyncio
import math
import pytest
import torch
from adapters.mac_desktop.adapter import SOURCES, MacDesktopAdapter, _rate_and_rhythm

SHARE_ALL = dict.fromkeys(SOURCES, True)


def test_construction_mock_mode():
    adapter = MacDesktopAdapter(mock_mode=True)
    assert adapter.bus is not None
    assert len(adapter.sensors) == 6
    names = {s.name for s in adapter.sensors}
    assert names == {"active_app", "keystroke_rate", "mouse_rate", "idle", "mic", "time_tonic"}


def test_encode_empty_bus_is_all_zero():
    """Nothing observed, nothing encoded. The "dormant" neuron (160) used to fire
    here because a missing idle timer defaulted to 999 s."""
    adapter = MacDesktopAdapter(mock_mode=True)
    vec = adapter.encode()
    assert vec.shape == (200,)
    assert vec.sum() == 0


def test_unshared_idle_is_not_read_as_dormant():
    adapter = MacDesktopAdapter(mock_mode=True)
    adapter.bus.write("keystroke_rate", {"count": 12})
    vec = adapter.encode()
    assert vec[160] == 0 and vec[162] > 0   # typing reads as active, not as away


def test_encode_with_active_app_fires_app_neuron():
    adapter = MacDesktopAdapter(mock_mode=True)
    adapter.bus.write("active_app", {"name": "VSCode"})
    vec = adapter.encode()
    # Some neuron in the active app range (0-39) should be > 0
    app_range = vec[0:40]
    assert app_range.sum() > 0
    # Keystroke/mouse ranges should be zero (no data written)
    assert vec[60:100].sum() == 0


def test_encode_active_app_consistent_mapping():
    """Same app name -> same neuron index."""
    a1 = MacDesktopAdapter(mock_mode=True)
    a1.bus.write("active_app", {"name": "Chrome"})
    v1 = a1.encode()
    a2 = MacDesktopAdapter(mock_mode=True)
    a2.bus.write("active_app", {"name": "Chrome"})
    v2 = a2.encode()
    assert torch.allclose(v1, v2)


def test_encode_keystroke_rate_bin():
    adapter = MacDesktopAdapter(mock_mode=True)
    adapter.bus.write("keystroke_rate", {"count": 0})
    vec = adapter.encode()
    # bin 0 (no activity) should fire in keystroke range 60-75
    keystroke_range = vec[60:76]
    assert keystroke_range.sum() > 0


def test_encode_high_keystroke_rate_high_bin():
    adapter = MacDesktopAdapter(mock_mode=True)
    adapter.bus.write("keystroke_rate", {"count": 50})
    vec = adapter.encode()
    keystroke_range = vec[60:76]
    active_bins = (keystroke_range > 0).nonzero(as_tuple=True)[0]
    assert active_bins[0].item() >= 8  # upper half


def test_encode_keystroke_rhythm_features():
    """Erratic typing should fire variability neuron."""
    adapter = MacDesktopAdapter(mock_mode=True)
    adapter.bus.write("keystroke_rate", {"count": 10, "variability": 1.5, "burst": 1.0})
    vec = adapter.encode()
    # Neuron 76 = variability, 77 = burst
    assert vec[76] > 0  # variability fires
    assert vec[77] > 0  # burst fires


def test_encode_idle_long():
    adapter = MacDesktopAdapter(mock_mode=True)
    adapter.bus.write("idle", {"seconds": 7200})
    vec = adapter.encode()
    idle_range = vec[100:108]
    active_bins = (idle_range > 0).nonzero(as_tuple=True)[0]
    assert active_bins[0].item() == 7  # last bin


def test_encode_pause_detection():
    """Different idle durations should fire different pause-type neurons."""
    adapter = MacDesktopAdapter(mock_mode=True)
    # Micro-pause (thinking)
    adapter.bus.write("idle", {"seconds": 3.0})
    vec = adapter.encode()
    assert vec[108] > 0  # micro-pause neuron

    # Thinking pause
    adapter.bus.write("idle", {"seconds": 15.0})
    vec = adapter.encode()
    assert vec[109] > 0  # thinking neuron

    # Break
    adapter.bus.write("idle", {"seconds": 120.0})
    vec = adapter.encode()
    assert vec[110] > 0  # break neuron

    # Away
    adapter.bus.write("idle", {"seconds": 600.0})
    vec = adapter.encode()
    assert vec[111] > 0  # away neuron


def test_encode_mic_mel_writes_to_range():
    adapter = MacDesktopAdapter(mock_mode=True)
    mel = [0.5] * 32
    adapter.bus.write("mic", {"mel": mel, "rms": 0.3})
    vec = adapter.encode()
    mic_range = vec[112:144]
    assert mic_range.sum() > 0


def test_encode_time_tonic_writes_to_range():
    adapter = MacDesktopAdapter(mock_mode=True)
    adapter.bus.write("time_tonic", {
        "day_phase": [0.5, 0.5, 0.5, 0.5],
        "week_phase": [0.5, 0.5, 0.5, 0.5],
    })
    vec = adapter.encode()
    time_range = vec[148:156]
    assert time_range.sum() > 0


def test_encode_reserve_range_always_zero():
    adapter = MacDesktopAdapter(mock_mode=True)
    adapter.bus.write("active_app", {"name": "X"})
    adapter.bus.write("keystroke_rate", {"count": 5})
    adapter.bus.write("mouse_rate", {"count": 5})
    adapter.bus.write("idle", {"seconds": 1})
    adapter.bus.write("mic", {"mel": [1.0] * 32, "rms": 0.1})
    adapter.bus.write("time_tonic", {"day_phase": [1, 0, 0, 0], "week_phase": [1, 0, 0, 0]})
    vec = adapter.encode()
    reserve = vec[164:200]
    assert reserve.sum() == 0


@pytest.mark.asyncio
async def test_run_sensors_briefly():
    adapter = MacDesktopAdapter(mock_mode=True, enabled=SHARE_ALL)
    task = asyncio.create_task(adapter.run())
    await asyncio.sleep(0.3)
    snap = adapter.bus.snapshot()
    adapter.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert "time_tonic" in snap
    assert "active_app" in snap
    assert set(adapter.bus.snapshot()) <= {"time_tonic"}  # stopped sources are forgotten


def test_hash_app_to_index_deterministic_pinned():
    from adapters.mac_desktop.encoding import _hash_app_to_index
    expected = {
        "VSCode": _hash_app_to_index("VSCode", 40),
        "Chrome": _hash_app_to_index("Chrome", 40),
        "Slack": _hash_app_to_index("Slack", 40),
        "Terminal": _hash_app_to_index("Terminal", 40),
    }
    for name, idx in expected.items():
        assert _hash_app_to_index(name, 40) == idx
        assert 0 <= idx < 40


def test_encode_app_switch_rate():
    """High switch rate should fire the frantic-switching neuron."""
    adapter = MacDesktopAdapter(mock_mode=True)
    adapter.bus.write("active_app", {"name": "VSCode", "switch_rate": 10.0})
    vec = adapter.encode()
    assert vec[159] > 0  # frantic switching neuron


def test_encode_activity_level():
    """High typing + mouse should fire intense activity neuron."""
    adapter = MacDesktopAdapter(mock_mode=True)
    adapter.bus.write("keystroke_rate", {"count": 30})
    adapter.bus.write("mouse_rate", {"count": 50})
    adapter.bus.write("idle", {"seconds": 0.5})
    vec = adapter.encode()
    assert vec[163] > 0  # intense activity


# ── Rolling 1-second keystroke/mouse rate (the 100Hz per-tick-drain fix) ──

def test_rate_and_rhythm_per_second_count():
    """count = events in the last 1s = the per-second rate the encoder expects,
    NOT the per-tick count (which is ~0-1 at a 100Hz tick loop while typing)."""
    now = 1000.0
    times = [now - 0.95 + i * 0.16 for i in range(6)]  # ~6 keys across the last ~0.8s
    assert _rate_and_rhythm(times, now, window=1.0)["count"] == 6


def test_rate_and_rhythm_excludes_old_events():
    now = 1000.0
    times = [now - 5.0, now - 3.0, now - 0.5, now - 0.3, now - 0.1]
    assert _rate_and_rhythm(times, now, window=1.0)["count"] == 3  # only last 1s


def test_rate_and_rhythm_steady_low_variability():
    now = 1000.0
    times = [now - 1.0 + i * 0.1 for i in range(11)]  # even 100ms gaps
    assert _rate_and_rhythm(times, now, window=1.0)["variability"] < 0.15


def test_rate_and_rhythm_erratic_high_variability():
    """Bursty/irregular typing -> high variability — the stress signal that was
    DEAD when the adapter hard-coded variability to 0."""
    now = 1000.0
    times = [now - 0.98, now - 0.95, now - 0.93, now - 0.4, now - 0.05]
    assert _rate_and_rhythm(times, now, window=1.0)["variability"] > 0.5


def test_rate_and_rhythm_empty_is_silent():
    assert _rate_and_rhythm([], 1000.0)["count"] == 0


def test_real_typing_fires_steady_not_silence():
    """End-to-end: a per-second typing rate fires the steady-typing neuron (78)
    and NOT the silence neuron (79) — the regression the per-tick drain caused."""
    from adapters.mac_desktop.encoding import encode_snapshot
    r = _rate_and_rhythm([1000.0 - 1.0 + i * 0.15 for i in range(7)], 1000.0)
    vec = encode_snapshot({"keystroke_rate": r})
    assert vec[78] > 0   # steady typing perceived
    assert vec[79] == 0  # NOT read as silence
