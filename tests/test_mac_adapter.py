"""Tests for MacDesktopAdapter — composes 6 sensors into a 200-dim sensory vector."""
import asyncio
import math
import pytest
import torch
from adapters.mac_desktop.adapter import MacDesktopAdapter


def test_construction_mock_mode():
    adapter = MacDesktopAdapter(mock_mode=True)
    assert adapter.bus is not None
    assert len(adapter.sensors) == 6
    # Sensor names
    names = {s.name for s in adapter.sensors}
    assert names == {"active_app", "keystroke_rate", "mouse_rate", "idle", "mic", "time_tonic"}


def test_encode_empty_bus_returns_zero_vector():
    adapter = MacDesktopAdapter(mock_mode=True)
    vec = adapter.encode()
    assert vec.shape == (200,)
    assert torch.all(vec == 0)


def test_encode_with_active_app_fires_app_neuron():
    adapter = MacDesktopAdapter(mock_mode=True)
    adapter.bus.write("active_app", {"name": "VSCode"})
    vec = adapter.encode()
    # Some neuron in the active_app range (0-63) should be > 0
    app_range = vec[0:64]
    assert app_range.sum() > 0
    # Other ranges still zero
    assert vec[64:].sum() == 0


def test_encode_active_app_consistent_mapping():
    """Same app name → same neuron index."""
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
    # bin 0 (no activity) should be active in keystroke range 64-79
    keystroke_range = vec[64:80]
    assert keystroke_range.sum() > 0


def test_encode_high_keystroke_rate_high_bin():
    adapter = MacDesktopAdapter(mock_mode=True)
    adapter.bus.write("keystroke_rate", {"count": 50})  # very high count → top bin
    vec = adapter.encode()
    keystroke_range = vec[64:80]
    # The active bin should be in the upper half
    active_bins = (keystroke_range > 0).nonzero(as_tuple=True)[0]
    assert active_bins[0].item() >= 8


def test_encode_idle_long():
    adapter = MacDesktopAdapter(mock_mode=True)
    adapter.bus.write("idle", {"seconds": 7200})  # 2 hours → highest bin
    vec = adapter.encode()
    idle_range = vec[96:104]
    active_bins = (idle_range > 0).nonzero(as_tuple=True)[0]
    assert active_bins[0].item() == 7  # last bin


def test_encode_mic_mel_writes_to_range():
    adapter = MacDesktopAdapter(mock_mode=True)
    mel = [0.5] * 32
    adapter.bus.write("mic", {"mel": mel, "rms": 0.3})
    vec = adapter.encode()
    mic_range = vec[104:136]
    assert mic_range.sum() > 0


def test_encode_time_tonic_writes_to_range():
    adapter = MacDesktopAdapter(mock_mode=True)
    adapter.bus.write("time_tonic", {
        "day_phase": [0.5, 0.5, 0.5, 0.5],
        "week_phase": [0.5, 0.5, 0.5, 0.5],
    })
    vec = adapter.encode()
    time_range = vec[144:152]
    assert time_range.sum() > 0


def test_encode_reserve_range_always_zero():
    adapter = MacDesktopAdapter(mock_mode=True)
    # Write to all sensors
    adapter.bus.write("active_app", {"name": "X"})
    adapter.bus.write("keystroke_rate", {"count": 5})
    adapter.bus.write("mouse_rate", {"count": 5})
    adapter.bus.write("idle", {"seconds": 1})
    adapter.bus.write("mic", {"mel": [1.0] * 32, "rms": 0.1})
    adapter.bus.write("time_tonic", {"day_phase": [1, 0, 0, 0], "week_phase": [1, 0, 0, 0]})
    vec = adapter.encode()
    reserve = vec[152:200]
    assert reserve.sum() == 0


@pytest.mark.asyncio
async def test_run_sensors_briefly():
    """Start the adapter, let sensors run for ~300ms in mock mode, stop."""
    adapter = MacDesktopAdapter(mock_mode=True)
    task = asyncio.create_task(adapter.run())
    await asyncio.sleep(0.3)
    adapter.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    snap = adapter.bus.snapshot()
    # At least the time tonic and active app should have produced samples
    assert "time_tonic" in snap
    assert "active_app" in snap
