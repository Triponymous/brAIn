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


@pytest.mark.asyncio
async def test_the_prompt_is_the_brain_protocol_plus_time_and_nothing_read_beside_it(tmp_path, monkeypatch):
    """The system prompt comes from SCP (brain state, personality, conversation), then
    the time. The endpoint reads nothing else for it: an episodes.db in the working
    directory, which may belong to another daemon's checkpoint, stays unread."""
    from fastapi import FastAPI
    from bridge.episode_log import EpisodeLogger

    class _Protocol:
        def __init__(self):
            self.prompts, self.feedback = [], []

        def build_prompt(self, user_message="", history=None):
            self.prompts.append((user_message, history))
            return "SCP-PROMPT"

        def send_feedback(self, kind):
            self.feedback.append(kind)

    monkeypatch.chdir(tmp_path)
    (tmp_path / "checkpoints").mkdir()
    episodes = EpisodeLogger(tmp_path / "checkpoints" / "episodes.db")
    episodes.log(tick=1000, modulators={"DA": 0.1}, active_concepts=[1],
                 sensor_summary={"app": "Safari", "cluster_id": 1, "cluster_label": "flow"}, sleep_mode=False)
    episodes.close()

    brain = Brain(num_sensory=8, num_concept=4)
    brain._scp_client = _Protocol()
    exporter = BrainStateExporter(brain)
    router = HybridLLMRouter()
    app = FastAPI()
    app.include_router(build_chat_router(brain, exporter, MemoryTools(brain, exporter), router))
    seen = {}

    async def fake_chat(**kw):
        seen.update(kw)
        return {"text": "ok", "backend": "local", "tool_calls": []}

    history = [{"role": "user", "content": "vorhin"}]
    with patch.object(router, "chat", new_callable=AsyncMock, side_effect=fake_chat):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/api/chat", json={"message": "hallo", "history": history})

    assert brain._scp_client.prompts == [("hallo", history)]
    assert brain._scp_client.feedback == ["engage"]
    assert seen["system_prompt"].startswith("SCP-PROMPT\n(Zeitpunkt: ")
    assert "Safari" not in seen["system_prompt"] and "flow" not in seen["system_prompt"]
    assert seen["history"] == history
