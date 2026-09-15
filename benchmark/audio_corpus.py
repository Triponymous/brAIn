"""Explicit, local WAV inputs for the audio feature experiment. No capture."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import wave

import numpy as np


LABELS = ("meeting", "podcast", "instrumental_music", "vocal_music", "ambient")
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Clip:
    path: Path
    label: int
    split: str
    group: str
    pcm_sha256: str


def read_wav(path: Path) -> tuple[np.ndarray, str]:
    try:
        with wave.open(str(path), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(),
                    wav.getcomptype()) != (1, 2, SAMPLE_RATE, "NONE"):
                raise ValueError("WAV must be mono, 16-bit PCM, 16000 Hz")
            frames = wav.getnframes()
            if not 2 * SAMPLE_RATE <= frames <= 30 * SAMPLE_RATE:
                raise ValueError("Each WAV must contain 2 to 30 seconds")
            pcm = wav.readframes(frames)
            if len(pcm) != frames * 2:
                raise ValueError("Truncated WAV payload")
    except (OSError, EOFError, wave.Error):
        raise ValueError("Cannot read a valid local WAV") from None
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    return audio, hashlib.sha256(pcm).hexdigest()


def _clip(row: object, root: Path) -> Clip:
    fields = {"path", "label", "split", "group", "license"}
    if not isinstance(row, dict) or set(row) != fields:
        raise ValueError("Expected path, label, split, group and license")
    if any(not isinstance(value, str) or not value.strip() for value in row.values()):
        raise ValueError("All clip fields must be nonempty strings")
    if row["label"] not in LABELS or row["split"] not in {"train", "test"}:
        raise ValueError("Unknown label or split")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", row["group"]):
        raise ValueError("Group must be an opaque alphanumeric identifier")
    path = Path(row["path"])
    if path.is_absolute():
        raise ValueError("Use paths relative to the manifest directory")
    path = (root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("WAV must exist inside the manifest directory")
    _, digest = read_wav(path)
    return Clip(path, LABELS.index(row["label"]), row["split"], row["group"], digest)


def load_corpus(manifest: Path) -> tuple[str, list[Clip]]:
    try:
        raw = manifest.read_bytes()
        if len(raw) > 1_000_000:
            raise ValueError("Manifest exceeds 1 MB")
        data = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("Cannot read a valid JSON manifest") from None
    if not isinstance(data, dict) or set(data) != {"schema", "kind", "clips"}:
        raise ValueError("Expected schema, kind and clips")
    if data["schema"] != "brain.audio-corpus.v1" or data["kind"] not in ("recorded", "synthetic"):
        raise ValueError("Unknown corpus schema or kind")
    rows = data["clips"]
    if not isinstance(rows, list) or not 20 <= len(rows) <= 1000:
        raise ValueError("Provide 20 to 1000 clips; at least 2 per class in each split")
    clips, groups, digests = [], set(), set()
    root = manifest.resolve().parent
    for index, row in enumerate(rows, 1):
        try:
            clip = _clip(row, root)
            if clip.group in groups:
                raise ValueError("Only one clip per independent source group is allowed")
            if clip.pcm_sha256 in digests:
                raise ValueError("Duplicate PCM content; renamed files are not independent")
        except ValueError as exc:
            raise ValueError(f"Clip {index}: {exc}") from None
        groups.add(clip.group)
        digests.add(clip.pcm_sha256)
        clips.append(clip)
    counts = Counter((clip.split, clip.label) for clip in clips)
    if any(counts[split, label] < 2 for split in ("train", "test") for label in range(len(LABELS))):
        raise ValueError("Each class needs at least 2 independent clips in train AND test")
    return data["kind"], clips
