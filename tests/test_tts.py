"""Tests for TTS engine — wraps Piper as subprocess."""
import asyncio
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock
from bridge.tts import TTSEngine, _run_piper


def test_construction():
    engine = TTSEngine(voice_model=Path("models/de_DE-thorsten-medium.onnx"))
    assert engine.voice_model.name == "de_DE-thorsten-medium.onnx"


def test_construction_with_missing_model_does_not_crash():
    engine = TTSEngine(voice_model=Path("/nonexistent/model.onnx"))
    assert engine.voice_model == Path("/nonexistent/model.onnx")


@pytest.mark.asyncio
async def test_synthesize_returns_wav_bytes():
    """Mock the subprocess to verify the pipeline works without actual Piper."""
    engine = TTSEngine(voice_model=Path("models/test.onnx"))
    fake_wav = b"RIFF" + b"\x00" * 100
    with patch("bridge.tts._run_piper", new_callable=AsyncMock, return_value=fake_wav):
        result = await engine.synthesize("Hallo Welt")
    assert result[:4] == b"RIFF"
    assert len(result) > 10


@pytest.mark.asyncio
async def test_synthesize_empty_text_returns_empty():
    engine = TTSEngine(voice_model=Path("models/test.onnx"))
    with patch("bridge.tts._run_piper", new_callable=AsyncMock, return_value=b""):
        result = await engine.synthesize("")
    assert result == b""
