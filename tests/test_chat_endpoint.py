"""Tests for the /api/chat and /api/label endpoints."""
import json
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport

from brain.core import Brain
from bridge.exporter import BrainStateExporter
from bridge.memory_tools import MemoryTools
from bridge.llm_router import HybridLLMRouter
from server.chat import build_chat_router


@pytest.mark.asyncio
async def test_chat_returns_response():
    brain = Brain(num_sensory=8, num_concept=4)
    exporter = BrainStateExporter(brain)
    tools = MemoryTools(brain, exporter)
    router = HybridLLMRouter()

    chat_router = build_chat_router(brain, exporter, tools, router)

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(chat_router)

    with patch.object(router, '_call_ollama', new_callable=AsyncMock, return_value="Alles ruhig."):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/chat", json={"message": "was siehst du?"})

    assert resp.status_code == 200
    body = resp.json()
    assert "text" in body
    assert "backend" in body
    assert body["text"] == "Alles ruhig."


@pytest.mark.asyncio
async def test_chat_with_tool_call():
    """When the LLM returns a tool call, the endpoint executes it."""
    brain = Brain(num_sensory=8, num_concept=4)
    exporter = BrainStateExporter(brain)
    tools = MemoryTools(brain, exporter)
    router = HybridLLMRouter()

    chat_router = build_chat_router(brain, exporter, tools, router)

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(chat_router)

    mock_response = {
        "text": "",
        "tool_calls": [{"name": "current_state", "args": {}}],
        "backend": "cloud",
    }
    with patch.object(router, 'chat', new_callable=AsyncMock, return_value=mock_response):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/chat", json={"message": "/cloud zeig mir alles"})

    assert resp.status_code == 200
    body = resp.json()
    assert "tool_results" in body
    assert len(body["tool_results"]) == 1


@pytest.mark.asyncio
async def test_chat_label_endpoint():
    """POST /api/label sets a concept label."""
    brain = Brain(num_sensory=8, num_concept=4)
    exporter = BrainStateExporter(brain)
    tools = MemoryTools(brain, exporter)
    router = HybridLLMRouter()

    chat_router = build_chat_router(brain, exporter, tools, router)

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(chat_router)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/label", json={"concept_id": 2, "label": "tippen"})

    assert resp.status_code == 200
    assert exporter.get_label(2) == "tippen"
