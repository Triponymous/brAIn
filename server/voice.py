"""Voice endpoints — /api/tts, /api/stt/record, /api/voice-chat.

/api/tts: text -> WAV audio via Piper
/api/stt/record: records from mic for N seconds -> transcribed text
/api/voice-chat: records -> transcribes -> chats -> synthesizes -> returns audio response
"""
from __future__ import annotations
import asyncio
from typing import Any, Callable, Awaitable

import numpy as np
from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel


class TTSRequest(BaseModel):
    text: str


class RecordRequest(BaseModel):
    duration: float = 5.0


def build_voice_router(
    tts_engine: Any,
    stt_engine: Any,
    chat_fn: Callable[[str], Awaitable[dict[str, Any]]],
) -> APIRouter:
    """Build voice API router.

    chat_fn should accept a user message string and return a dict with 'text' key.
    """
    api = APIRouter()

    @api.post("/api/tts")
    async def tts(req: TTSRequest) -> Response:
        wav_bytes = await tts_engine.synthesize(req.text)
        return Response(content=wav_bytes, media_type="audio/wav")

    @api.post("/api/stt/record")
    async def stt_record(req: RecordRequest) -> dict[str, str]:
        """Record from mic and transcribe."""
        try:
            import sounddevice as sd
        except ImportError:
            return {"text": "", "error": "sounddevice not available"}

        sample_rate = 16000
        frames = int(sample_rate * req.duration)
        audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        audio_flat = audio[:, 0]
        text = stt_engine.transcribe(audio_flat, sample_rate=sample_rate)
        return {"text": text}

    @api.post("/api/voice-chat")
    async def voice_chat(req: RecordRequest) -> Response:
        """Record -> STT -> Chat -> TTS -> return audio."""
        try:
            import sounddevice as sd
        except ImportError:
            return Response(content=b"", media_type="audio/wav")

        # Record
        sample_rate = 16000
        frames = int(sample_rate * req.duration)
        audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        audio_flat = audio[:, 0]

        # Transcribe
        user_text = stt_engine.transcribe(audio_flat, sample_rate=sample_rate)
        if not user_text.strip():
            return Response(content=b"", media_type="audio/wav")

        # Chat
        chat_result = await chat_fn(user_text)
        response_text = chat_result.get("text", "")

        # Synthesize response
        wav_bytes = await tts_engine.synthesize(response_text)
        return Response(content=wav_bytes, media_type="audio/wav")

    return api
