"""STT engine — wraps pywhispercpp for local speech-to-text.

Uses the Whisper small model for German transcription. The model is loaded
lazily on first transcribe() call to avoid blocking daemon startup.
"""
from __future__ import annotations
import numpy as np
from pywhispercpp.model import Model


class STTEngine:
    def __init__(self, model_name: str = "small", language: str = "de") -> None:
        self.model_name = model_name
        self.language = language
        self._model: Model | None = None

    def _ensure_model(self) -> Model:
        if self._model is None:
            self._model = Model(self.model_name)
        return self._model

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a float32 numpy audio array to text."""
        model = self._ensure_model()
        segments = model.transcribe(audio, language=self.language)
        return " ".join(s.text.strip() for s in segments).strip()
