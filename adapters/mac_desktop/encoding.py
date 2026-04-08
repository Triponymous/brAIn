"""Encoding sensor bus snapshots into the 200-dim sensory neuron vector.

Neuron map (per Phase 3+ design doc):
    0-63    active app (one-hot, hash-mapped, first-64 + "other" bucket)
    64-79   keystroke rate (16 log bins, 0..>=10/s)
    80-95   mouse rate (16 log bins, 0..>=1000/s)
    96-103  idle time (8 log bins, <1s..>1h)
    104-135 mic mel-spectrogram (32 bands, scaled)
    136-143 mic RMS loudness (8 log bins)
    144-147 time tonic day_phase (4 cosine/sine basis values, scaled to spike)
    148-151 time tonic week_phase (4 cosine/sine basis)
    152-199 reserve (always 0 for now)

Each "active" neuron in a one-hot or bin slot fires with current 3.0
(strong drive, well above LIF threshold of 1.0). Mel-spectrogram and
tonics scale their values to a similar range.
"""
from __future__ import annotations
import math
from typing import Any

import torch


SENSORY_DIM = 200

_DRIVE_STRENGTH = 3.0


def _hash_app_to_index(name: str, num_slots: int = 63) -> int:
    """Stable hash from app name to index in [0, num_slots). Slot num_slots = "other"."""
    return abs(hash(name)) % num_slots


def _log_bin(value: float, max_value: float, num_bins: int) -> int:
    """Map value to log-scaled bin in [0, num_bins-1]."""
    if value <= 0:
        return 0
    if value >= max_value:
        return num_bins - 1
    log_v = math.log1p(value)
    log_max = math.log1p(max_value)
    frac = log_v / log_max
    return min(num_bins - 1, int(frac * num_bins))


def encode_snapshot(snap: dict[str, Any]) -> torch.Tensor:
    """Encode a SensorBus snapshot into a 200-dim sensory input tensor."""
    vec = torch.zeros(SENSORY_DIM, dtype=torch.float32)

    # Active app: 0-63
    app = snap.get("active_app")
    if app and "name" in app:
        idx = _hash_app_to_index(app["name"])
        vec[idx] = _DRIVE_STRENGTH

    # Keystroke rate: 64-79 (16 bins)
    ks = snap.get("keystroke_rate")
    if ks is not None and "count" in ks:
        # 5 Hz windows → max ~50 events/window for "intensive typing"
        bin_idx = _log_bin(ks["count"], max_value=50.0, num_bins=16)
        vec[64 + bin_idx] = _DRIVE_STRENGTH

    # Mouse rate: 80-95 (16 bins)
    ms = snap.get("mouse_rate")
    if ms is not None and "count" in ms:
        bin_idx = _log_bin(ms["count"], max_value=200.0, num_bins=16)
        vec[80 + bin_idx] = _DRIVE_STRENGTH

    # Idle time: 96-103 (8 bins, log)
    idle = snap.get("idle")
    if idle is not None and "seconds" in idle:
        bin_idx = _log_bin(idle["seconds"], max_value=3600.0, num_bins=8)
        vec[96 + bin_idx] = _DRIVE_STRENGTH

    # Mic mel: 104-135 (32 bands), Mic RMS: 136-143 (8 bins)
    mic = snap.get("mic")
    if mic is not None:
        if "mel" in mic and len(mic["mel"]) == 32:
            mel = torch.tensor(mic["mel"], dtype=torch.float32)
            # Scale: log-mel values are typically 0-5; scale to drive 0-_DRIVE_STRENGTH
            mel_scaled = (mel / 5.0).clamp(0.0, 1.0) * _DRIVE_STRENGTH
            vec[104:136] = mel_scaled
        if "rms" in mic:
            bin_idx = _log_bin(mic["rms"], max_value=1.0, num_bins=8)
            vec[136 + bin_idx] = _DRIVE_STRENGTH

    # Time tonics: 144-151
    tt = snap.get("time_tonic")
    if tt is not None:
        if "day_phase" in tt and len(tt["day_phase"]) == 4:
            day = torch.tensor(tt["day_phase"], dtype=torch.float32)
            # Tonic values are -1..1; map to 0..2*drive (a baseline tonic)
            vec[144:148] = (day + 1.0) * _DRIVE_STRENGTH * 0.5
        if "week_phase" in tt and len(tt["week_phase"]) == 4:
            week = torch.tensor(tt["week_phase"], dtype=torch.float32)
            vec[148:152] = (week + 1.0) * _DRIVE_STRENGTH * 0.5

    # Reserve 152-199 stays zero
    return vec
