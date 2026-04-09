"""Tests for voice endpoints."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from server.voice import build_voice_router


@pytest.mark.asyncio
async def test_tts_endpoint_returns_audio():
    mock_tts = MagicMock()
    mock_tts.synthesize = AsyncMock(return_value=b"RIFF" + b"\x00" * 100)
    mock_stt = MagicMock()
    mock_chat_fn = AsyncMock()

    router = build_voice_router(tts_engine=mock_tts, stt_engine=mock_stt, chat_fn=mock_chat_fn)

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/tts", json={"text": "Hallo"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content[:4] == b"RIFF"


@pytest.mark.asyncio
async def test_tts_empty_text():
    mock_tts = MagicMock()
    mock_tts.synthesize = AsyncMock(return_value=b"")
    router = build_voice_router(tts_engine=mock_tts, stt_engine=MagicMock(), chat_fn=AsyncMock())

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/tts", json={"text": ""})
    assert resp.status_code == 200
