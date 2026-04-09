"""Encoding sensor bus snapshots into the 200-dim sensory neuron vector.

Designed for an AI pet that learns desktop activity patterns. The encoding
prioritizes features that distinguish BEHAVIORAL STATES (focused coding,
hektisches Multitasking, ruhiges Lesen, Pause, Telefonierthat) over raw
input counts.

Neuron map:
    0-39    active app identity (one-hot, hash-mapped, 40 slots)
    40-59   background app identities (hash-mapped, 20 slots, weaker drive)
    60-75   keystroke rate (16 log bins)
    76-79   keystroke rhythm (4 neurons: variability, burst, steady, silence)
    80-95   mouse rate (16 log bins)
    96-99   mouse rhythm (4 neurons: variability, burst, steady, silence)
    100-107 idle time (8 log bins)
    108-111 pause type (4 neurons: micro-pause, thinking, break, away)
    112-143 mic mel-spectrogram (32 bands)
    144-147 mic RMS loudness (4 log bins, simplified)
    148-151 time tonic day_phase (4 cos/sin basis)
    152-155 time tonic week_phase (4 cos/sin basis)
    156-159 app context (switch rate bins: calm/normal/busy/frantic)
    160-163 activity level (4 neurons: dormant, idle, active, intense)
    164-199 reserve (always 0)

Drive strengths:
    Foreground app: 8.0 (strong, reliable signal)
    Background apps: 3.0 (present but not focused)
    All other features: 8.0 for the active bin
    Rhythm/context features: 6.0 (important but secondary)
"""
from __future__ import annotations
import hashlib
import math
from typing import Any

import torch


SENSORY_DIM = 200

_DRIVE = 8.0       # strong primary drive
_DRIVE_BG = 3.0    # background app drive
_DRIVE_CTX = 6.0   # context/rhythm feature drive


def _hash_app_to_index(name: str, num_slots: int) -> int:
    """Deterministic hash from app name to index in [0, num_slots).

    Uses md5 (not Python hash) because Python randomizes hash per process.
    """
    digest = hashlib.md5(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % num_slots


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

    # ── APP IDENTITY: 0-39 (foreground), 40-59 (background) ──
    app = snap.get("active_app")
    if app and "name" in app:
        idx = _hash_app_to_index(app["name"], num_slots=40)
        vec[idx] = _DRIVE

        # Background apps: each gets one neuron in slots 40-59
        for bg_name in app.get("background_apps", [])[:10]:
            bg_idx = _hash_app_to_index(bg_name, num_slots=20)
            vec[40 + bg_idx] = _DRIVE_BG

    # ── KEYSTROKE: 60-79 ──
    ks = snap.get("keystroke_rate")
    if ks is not None and "count" in ks:
        count = ks["count"]
        # Rate bins: 60-75
        bin_idx = _log_bin(count, max_value=50.0, num_bins=16)
        vec[60 + bin_idx] = _DRIVE

        # Rhythm features: 76-79
        variability = ks.get("variability", 0.0)
        burst = ks.get("burst", 0.0)
        # 76: high variability = erratic/stressed typing
        vec[76] = min(_DRIVE_CTX, variability * 3.0)
        # 77: burst detected = sudden activity spike
        vec[77] = _DRIVE_CTX if burst > 0.5 else 0.0
        # 78: steady typing = low variability + non-zero count
        if count > 2 and variability < 0.5:
            vec[78] = _DRIVE_CTX
        # 79: silence (no typing at all)
        if count == 0:
            vec[79] = _DRIVE_CTX

    # ── MOUSE: 80-99 ──
    ms = snap.get("mouse_rate")
    if ms is not None and "count" in ms:
        count = ms["count"]
        # Rate bins: 80-95
        bin_idx = _log_bin(count, max_value=200.0, num_bins=16)
        vec[80 + bin_idx] = _DRIVE

        # Rhythm features: 96-99
        variability = ms.get("variability", 0.0)
        burst = ms.get("burst", 0.0)
        vec[96] = min(_DRIVE_CTX, variability * 3.0)
        vec[97] = _DRIVE_CTX if burst > 0.5 else 0.0
        if count > 5 and variability < 0.5:
            vec[98] = _DRIVE_CTX
        if count == 0:
            vec[99] = _DRIVE_CTX

    # ── IDLE TIME: 100-111 ──
    idle = snap.get("idle")
    if idle is not None and "seconds" in idle:
        secs = idle["seconds"]
        # Coarse bins: 100-107
        bin_idx = _log_bin(secs, max_value=3600.0, num_bins=8)
        vec[100 + bin_idx] = _DRIVE

        # Pause type neurons: 108-111
        # These let the SNN distinguish "micro-pause while thinking"
        # from "left for coffee" from "gone for the day"
        if 1.0 < secs <= 5.0:
            vec[108] = _DRIVE_CTX   # micro-pause (thinking)
        elif 5.0 < secs <= 30.0:
            vec[109] = _DRIVE_CTX   # thinking pause
        elif 30.0 < secs <= 300.0:
            vec[110] = _DRIVE_CTX   # break (coffee, etc.)
        elif secs > 300.0:
            vec[111] = _DRIVE_CTX   # away (gone)

    # ── MIC: 112-147 ──
    mic = snap.get("mic")
    if mic is not None:
        # Mel-spectrogram: 112-143
        if "mel" in mic and len(mic["mel"]) == 32:
            mel = torch.tensor(mic["mel"], dtype=torch.float32)
            mel_scaled = (mel / 5.0).clamp(0.0, 1.0) * _DRIVE
            vec[112:144] = mel_scaled
        # RMS loudness: 144-147 (4 bins, simplified)
        if "rms" in mic:
            bin_idx = _log_bin(mic["rms"], max_value=1.0, num_bins=4)
            vec[144 + bin_idx] = _DRIVE

    # ── TIME TONICS: 148-155 ──
    tt = snap.get("time_tonic")
    if tt is not None:
        if "day_phase" in tt and len(tt["day_phase"]) == 4:
            day = torch.tensor(tt["day_phase"], dtype=torch.float32)
            vec[148:152] = (day + 1.0) * _DRIVE * 0.5
        if "week_phase" in tt and len(tt["week_phase"]) == 4:
            week = torch.tensor(tt["week_phase"], dtype=torch.float32)
            vec[152:156] = (week + 1.0) * _DRIVE * 0.5

    # ── APP CONTEXT: 156-159 ──
    if app:
        switch_rate = app.get("switch_rate", 0.0)
        # 4 bins: calm (<1/min), normal (1-3), busy (3-8), frantic (>8)
        if switch_rate < 1:
            vec[156] = _DRIVE_CTX   # calm/focused
        elif switch_rate < 3:
            vec[157] = _DRIVE_CTX   # normal multitasking
        elif switch_rate < 8:
            vec[158] = _DRIVE_CTX   # busy
        else:
            vec[159] = _DRIVE_CTX   # frantic switching

    # ── COMPOSITE ACTIVITY LEVEL: 160-163 ──
    # Derived from keyboard + mouse + idle to give a clean "how active is the user" signal
    ks_count = snap.get("keystroke_rate", {}).get("count", 0) if snap.get("keystroke_rate") else 0
    ms_count = snap.get("mouse_rate", {}).get("count", 0) if snap.get("mouse_rate") else 0
    idle_secs = snap.get("idle", {}).get("seconds", 999) if snap.get("idle") else 999
    total_activity = ks_count + ms_count * 0.1

    if idle_secs > 300:
        vec[160] = _DRIVE_CTX   # dormant
    elif total_activity < 1:
        vec[161] = _DRIVE_CTX   # idle (present but not doing much)
    elif total_activity < 15:
        vec[162] = _DRIVE_CTX   # active (normal work)
    else:
        vec[163] = _DRIVE_CTX   # intense (heavy typing/mousing)

    # Reserve 164-199 stays zero
    return vec
