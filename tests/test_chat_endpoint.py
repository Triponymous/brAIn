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

    with patch.object(router, '_call_ollama', new_callable=AsyncMock, return_value={"text": "Alles ruhig.", "tool_calls": []}):
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
        "tool_calls": [{"name": "current_state", "args": {}, "result": {"tick_count": 0}}],
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
    assert body["tool_results"][0] == {"tool": "current_state", "args": {}, "result": {"tick_count": 0}}


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


@pytest.mark.asyncio
async def test_chat_lets_the_model_ask_its_brain_and_records_it(tmp_path):
    """The endpoint offers the brain tools, tells the model about them, runs
    whatever it calls through the registry, and logs the call as an experience."""
    from fastapi import FastAPI
    from bridge.brain_tools import BrainTools
    from bridge.experience import ExperienceLog
    from capabilities.grants import GrantStore
    from capabilities.registry import ToolRegistry

    brain = Brain(num_sensory=8, num_concept=4)
    brain._experience = ExperienceLog(tmp_path / "experience.db")
    exporter = BrainStateExporter(brain)
    registry = ToolRegistry(brain, exporter, GrantStore(tmp_path / "grants.sqlite"), brain_tools=BrainTools(brain))
    router = HybridLLMRouter()
    app = FastAPI()
    app.include_router(build_chat_router(brain, exporter, registry, router))

    seen = {}

    async def fake_chat(**kw):            # a backend that calls one brain tool, then answers
        seen.update(kw)
        result = await kw["execute"]("brain_state", {})
        return {"text": "Ich bin wach.", "backend": "local",
                "tool_calls": [{"name": "brain_state", "args": {}, "result": result}]}

    with patch.object(router, "chat", new_callable=AsyncMock, side_effect=fake_chat):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            body = (await client.post("/api/chat", json={"message": "wie geht's dir?"})).json()

    assert body["text"] == "Ich bin wach."
    assert body["tool_results"][0]["tool"] == "brain_state" and "felt" in body["tool_results"][0]["result"]
    assert "MEIN GEHIRN BEFRAGEN" in seen["system_prompt"] and "brain_recall" in seen["system_prompt"]
    assert {d["name"] for d in seen["tools"]} >= {"brain_state", "brain_recall", "label_concept"}
    events = [(r["actor"], r["kind"], r["payload"].get("tool")) for r in brain._experience.recent()]
    assert ("llm", "tool_call", "brain_state") in events and ("human", "chat", None) in events
