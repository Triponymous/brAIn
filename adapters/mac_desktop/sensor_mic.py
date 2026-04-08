"""MicSensor — 32-band mel-spectrogram + RMS loudness from microphone.

Continuous audio capture via sounddevice in real mode. The sample() method
returns the latest mel-band activation snapshot. In mock mode, audio is
fed via _inject_audio() instead of opening the device.

Privacy: only spectral features and RMS are extracted. The raw audio buffer
is overwritten each call. Nothing is written to disk.
"""
from __future__ import annotations
import sys
import threading
from typing import Any

import numpy as np

from adapters.base import Sensor


_SAMPLE_RATE = 16000
_FRAME_SIZE = 1024  # FFT window size
_NUM_MEL_BANDS = 32


def _mel_filterbank(num_bands: int, fft_size: int, sample_rate: int) -> np.ndarray:
    """Build a triangular mel filterbank: shape (num_bands, fft_size//2 + 1)."""
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)
    low_mel = hz_to_mel(0.0)
    high_mel = hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(low_mel, high_mel, num_bands + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    n_bins = fft_size // 2 + 1
    bin_points = np.clip(bin_points, 0, n_bins - 1)
    fb = np.zeros((num_bands, n_bins), dtype=np.float32)
    for m in range(1, num_bands + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        if center == left:
            center = left + 1
        if right == center:
            right = center + 1
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / (right - center)
    return fb


class MicSensor(Sensor):
    name = "mic"
    rate_hz = 50.0
    num_mel_bands = _NUM_MEL_BANDS

    def __init__(self, mock_mode: bool = False) -> None:
        super().__init__(mock_mode=mock_mode)
        self._lock = threading.Lock()
        self._latest_audio: np.ndarray = np.zeros(_FRAME_SIZE, dtype=np.float32)
        self._filterbank = _mel_filterbank(_NUM_MEL_BANDS, _FRAME_SIZE, _SAMPLE_RATE)
        self._stream = None
        if not mock_mode and sys.platform == "darwin":
            try:
                import sounddevice as sd  # type: ignore
                self._stream = sd.InputStream(
                    samplerate=_SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    blocksize=_FRAME_SIZE,
                    callback=self._sd_callback,
                )
                self._stream.start()
            except Exception as e:
                print(f"[MicSensor] failed to open mic: {e}")
                self.mock_mode = True

    def _sd_callback(self, indata, frames, time_info, status) -> None:
        with self._lock:
            self._latest_audio = indata[:, 0].copy()

    def _inject_audio(self, audio: np.ndarray) -> None:
        """Test hook: feed a buffer as if it came from the mic."""
        with self._lock:
            self._latest_audio = audio.astype(np.float32)

    def _encode(self, audio: np.ndarray) -> dict[str, Any]:
        # Trim/pad to FRAME_SIZE
        if len(audio) >= _FRAME_SIZE:
            frame = audio[:_FRAME_SIZE]
        else:
            frame = np.zeros(_FRAME_SIZE, dtype=np.float32)
            frame[: len(audio)] = audio
        # Window
        window = np.hanning(_FRAME_SIZE).astype(np.float32)
        windowed = frame * window
        # FFT magnitude
        spectrum = np.abs(np.fft.rfft(windowed))
        # Mel filterbank
        mel = self._filterbank @ spectrum
        # Log compression to keep dynamic range tame
        mel = np.log1p(mel)
        # RMS loudness
        rms = float(np.sqrt(np.mean(audio**2)))
        return {"mel": mel.tolist(), "rms": rms}

    async def sample(self) -> dict[str, Any]:
        with self._lock:
            audio = self._latest_audio.copy()
        return self._encode(audio)

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
