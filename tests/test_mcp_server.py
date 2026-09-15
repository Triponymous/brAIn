"""server/mcp.py — brAIn's read-only tools over MCP, driven by a real client session in memory."""
import json
from contextlib import asynccontextmanager

import anyio
import httpx
from fastapi import FastAPI
from mcp import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

from brain.core import Brain
from bridge.brain_tools import BrainTools
from bridge.experience import ExperienceLog
from bridge.felt_state import FeltState
from server import mcp as brain_mcp
from server.tools import build_tools_router

FLOW = [0.028, 0.027, 0.058, 0.035, 0.060, 0.069]


def _daemon_app(tmp_path):
    """The daemon's tools endpoint on a small brain, served in-process over ASGI."""
    brain = Brain(num_sensory=8, num_concept=4, num_wm=4)
    brain.felt_state = FeltState()
    brain.felt_state.label("flow", FLOW)
    brain._last_signature = list(FLOW)
    brain._experience = ExperienceLog(tmp_path / "experience.db")
    app = FastAPI()
    app.include_router(build_tools_router(brain, BrainTools(brain, experience=brain._experience)))
    return app, brain


@asynccontextmanager
async def _session(server):
    """A client talking to `server` over in-memory streams, exactly as over stdio."""
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        async with anyio.create_task_group() as tg:
            tg.start_soon(server.run, server_streams[0], server_streams[1],
                          server.create_initialization_options())
            async with ClientSession(client_streams[0], client_streams[1]) as session:
                await session.initialize()
                yield session
            tg.cancel_scope.cancel()


async def test_lists_the_brain_tools_even_without_a_daemon():
    async with _session(brain_mcp.build_server(url="http://127.0.0.1:9")) as session:
        listed = await session.list_tools()
    assert {t.name for t in listed.tools} == BrainTools.NAMES
    recall = next(t for t in listed.tools if t.name == "brain_recall")
    assert "hours" in recall.input_schema["properties"] and recall.description


async def test_a_call_reaches_the_daemon_and_comes_back_as_json(tmp_path):
    app, brain = _daemon_app(tmp_path)
    server = brain_mcp.build_server(url="http://daemon", transport=httpx.ASGITransport(app=app))
    async with _session(server) as session:
        result = await session.call_tool("brain_state", {})
    assert result.is_error is False
    state = json.loads(result.content[0].text)
    assert state["felt"]["label"] == "flow" and "modulators" in state
    row = brain._experience.recent()[0]
    assert row["actor"] == "llm" and row["payload"] == {"tool": "brain_state", "args": {}, "via": "mcp"}


async def test_bad_arguments_are_flagged_as_an_error_for_the_model(tmp_path):
    app, _ = _daemon_app(tmp_path)
    server = brain_mcp.build_server(url="http://daemon", transport=httpx.ASGITransport(app=app))
    async with _session(server) as session:
        result = await session.call_tool("brain_history", {"bogus": 1})
    assert result.is_error is True and "bad arguments" in result.content[0].text


async def test_daemon_down_says_how_to_start_it():
    async with _session(brain_mcp.build_server(url="http://127.0.0.1:9")) as session:
        result = await session.call_tool("brain_state", {})
    assert result.is_error is True
    assert "not running" in result.content[0].text and "8900" in result.content[0].text


async def test_only_read_only_tools_exist_here(tmp_path):
    """Even a client that guesses a name cannot reach a writing tool through this server."""
    app, _ = _daemon_app(tmp_path)
    server = brain_mcp.build_server(url="http://daemon", transport=httpx.ASGITransport(app=app))
    async with _session(server) as session:
        result = await session.call_tool("label_concept", {"concept_id": 1, "label": "x"})
    assert result.is_error is True and "404" in result.content[0].text


def test_console_script_entry_point():
    assert callable(brain_mcp.main)
