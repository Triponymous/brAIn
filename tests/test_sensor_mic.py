"""Tests for MicSensor — produces 32 mel-band intensities + 1 RMS loudness.

In mock mode, we feed the sensor synthetic audio frames directly via
inject_audio() and verify the encoder produces sensible output without
actually opening a microphone.
"""
import math
import numpy as np
import pytest
from adapters.mac_desktop.sensor_mic import MicSensor


def test_construction():
    s = MicSensor()
    assert s.name == "mic"
    assert s.rate_hz == 50.0
    assert s.num_mel_bands == 32


@pytest.mark.asyncio
async def test_mock_mode_silence_returns_low_values():
    s = MicSensor(mock_mode=True)
    silence = np.zeros(1024, dtype=np.float32)
    s._inject_audio(silence)
    sample = await s.sample()
    assert "mel" in sample
    assert "rms" in sample
    assert len(sample["mel"]) == 32
    assert sample["rms"] < 0.001


@pytest.mark.asyncio
async def test_mock_mode_loud_signal_high_rms():
    s = MicSensor(mock_mode=True)
    # Loud sinusoid at 1kHz, 1 second at 16kHz
    sr = 16000
    t = np.arange(0, 1.0, 1.0 / sr, dtype=np.float32)
    loud = 0.5 * np.sin(2 * math.pi * 1000 * t)
    s._inject_audio(loud)
    sample = await s.sample()
    assert sample["rms"] > 0.1


@pytest.mark.asyncio
async def test_mel_bands_react_to_frequency():
    """A high-frequency tone should activate higher mel bands more than low ones."""
    s = MicSensor(mock_mode=True)
    sr = 16000
    t = np.arange(0, 0.1, 1.0 / sr, dtype=np.float32)
    high_tone = 0.5 * np.sin(2 * math.pi * 4000 * t).astype(np.float32)
    s._inject_audio(high_tone)
    sample = await s.sample()
    mel = sample["mel"]
    # Sum of upper half should exceed sum of lower half for a high tone
    lower_half = sum(mel[:16])
    upper_half = sum(mel[16:])
    assert upper_half > lower_half, f"High tone should activate upper bands more, got lower={lower_half}, upper={upper_half}"
