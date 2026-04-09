"""Tests for STT engine — wraps pywhispercpp for local speech-to-text."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from bridge.stt import STTEngine


def test_construction():
    with patch("bridge.stt.Model") as MockModel:
        engine = STTEngine(model_name="small")
    assert engine.model_name == "small"


def test_transcribe_with_mock():
    """Feed synthetic audio and verify transcription pipeline."""
    with patch("bridge.stt.Model") as MockModel:
        mock_instance = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "Hallo Welt"
        mock_instance.transcribe.return_value = [mock_segment]
        MockModel.return_value = mock_instance

        engine = STTEngine(model_name="small")
        audio = np.zeros(16000, dtype=np.float32)
        result = engine.transcribe(audio)
    assert result == "Hallo Welt"


def test_transcribe_empty_returns_empty():
    with patch("bridge.stt.Model") as MockModel:
        mock_instance = MagicMock()
        mock_instance.transcribe.return_value = []
        MockModel.return_value = mock_instance

        engine = STTEngine(model_name="small")
        audio = np.zeros(16000, dtype=np.float32)
        result = engine.transcribe(audio)
    assert result == ""


def test_transcribe_multiple_segments_joined():
    with patch("bridge.stt.Model") as MockModel:
        mock_instance = MagicMock()
        seg1, seg2 = MagicMock(), MagicMock()
        seg1.text = " Hallo "
        seg2.text = " Welt "
        mock_instance.transcribe.return_value = [seg1, seg2]
        MockModel.return_value = mock_instance

        engine = STTEngine(model_name="small")
        result = engine.transcribe(np.zeros(16000, dtype=np.float32))
    assert result == "Hallo Welt"
