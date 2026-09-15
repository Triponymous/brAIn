"""Offline Mel/RMS features plus an explicitly supplied, frozen Silero ONNX model."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import numpy as np

from adapters.mac_desktop.sensor_mic import MicSensor
from benchmark.audio_corpus import SAMPLE_RATE


class LocalVad:
    def __init__(self, model: Path, expected_sha256: str):
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            raise ValueError("Supply the trusted model's 64-character SHA-256")
        try:
            if model.stat().st_size > 8_000_000:
                raise ValueError("VAD model exceeds 8 MB")
            payload = model.read_bytes()
        except OSError:
            raise ValueError("Cannot read the local VAD model") from None
        self.sha256 = hashlib.sha256(payload).hexdigest()
        if self.sha256 != expected_sha256.lower():
            raise ValueError("VAD model SHA-256 mismatch")
        try:
            import onnxruntime as ort
        except ImportError:
            raise ValueError("Install the optional audio-benchmark dependencies first") from None
        self.runtime_version = ort.__version__
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.log_severity_level = 4
        try:
            self.session = ort.InferenceSession(
                payload, sess_options=options, providers=["CPUExecutionProvider"],
            )
            if ({item.name for item in self.session.get_inputs()} != {"input", "state", "sr"}
                    or len(self.session.get_outputs()) != 2):
                raise ValueError("Unexpected model interface")
        except Exception:
            raise ValueError("Model must implement the Silero 16 kHz ONNX interface") from None

    def probabilities(self, audio: np.ndarray) -> np.ndarray:
        # Recurrent state and acoustic context must never cross recording boundaries.
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, 64), dtype=np.float32)
        scores = []
        for frame in audio.reshape(-1, 512):
            window = np.concatenate((context, frame[None, :]), axis=1)
            try:
                output, state = self.session.run(None, {
                    "input": window, "state": state, "sr": np.array(SAMPLE_RATE, dtype=np.int64),
                })
            except Exception:
                raise ValueError("VAD inference failed; no substitute scores were generated") from None
            if (output.shape != (1, 1) or state.shape != (2, 1, 128)
                    or not np.isfinite(state).all() or not np.isfinite(output).all()
                    or not 0 <= output.item() <= 1):
                raise ValueError("VAD produced an invalid score or recurrent state")
            scores.append(output.item())
            context = window[:, -64:]
        return np.asarray(scores)


def mel_features(audio: np.ndarray) -> np.ndarray:
    # The existing sensor's deterministic encoder is reused without opening a device
    # or calling sample(), whose legacy mock path can synthesize noise.
    encoder = MicSensor(mock_mode=True)
    frames = []
    for frame in audio.reshape(-1, 1024):
        encoded = encoder._encode(frame)
        frames.append([*encoded["mel"], encoded["rms"]])
    values = np.asarray(frames)
    return np.concatenate((values.mean(axis=0), values.std(axis=0)))


def vad_features(scores: np.ndarray) -> np.ndarray:
    active = scores >= 0.5
    seconds = len(scores) * 512 / SAMPLE_RATE
    transitions = np.count_nonzero(active[1:] != active[:-1]) / seconds
    return np.array([scores.mean(), scores.std(), active.mean(), transitions])
