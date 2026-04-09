"""Tests for the hybrid LLM router.

In tests we mock both Ollama and Claude to avoid real API calls.
The router logic (which backend to use) is tested via the routing heuristics.
"""
import pytest
from unittest.mock import AsyncMock, patch
from bridge.llm_router import HybridLLMRouter, _should_use_cloud


def test_should_use_cloud_short_message():
    assert _should_use_cloud("hey was siehst du?", {}) is False


def test_should_use_cloud_long_message():
    long = "Kannst du mir erklären warum " + "Concept #12 " * 50 + "immer aktiv ist?"
    assert _should_use_cloud(long, {}) is True


def test_should_use_cloud_reasoning_keyword():
    assert _should_use_cloud("analysiere mein Tagesrhythmus", {}) is True


def test_should_use_cloud_high_ne():
    assert _should_use_cloud("was war das?", {"NE": 0.8}) is True


def test_should_use_cloud_explicit_flag():
    assert _should_use_cloud("/cloud was ist los", {}) is True


def test_router_construction():
    router = HybridLLMRouter()
    assert router.ollama_model == "qwen2.5:7b-instruct"
    assert router.cloud_model == "claude-haiku-4-5-20250404"


@pytest.mark.asyncio
async def test_router_calls_local_for_simple_query():
    router = HybridLLMRouter()
    with patch.object(router, '_call_ollama', new_callable=AsyncMock, return_value="Ich sehe VSCode.") as mock:
        result = await router.chat(
            user_message="was siehst du?",
            system_prompt="Du bist ein Pet.",
            brain_state={"modulators": {"NE": 0.1}},
            tools=[],
        )
    mock.assert_called_once()
    assert result["text"] == "Ich sehe VSCode."
    assert result["backend"] == "local"


@pytest.mark.asyncio
async def test_router_calls_cloud_for_complex_query():
    router = HybridLLMRouter()
    with patch.object(router, '_call_claude', new_callable=AsyncMock, return_value={"text": "Analyse...", "tool_calls": []}) as mock:
        result = await router.chat(
            user_message="analysiere warum Concept #12 und #47 immer zusammen feuern",
            system_prompt="Du bist ein Pet.",
            brain_state={"modulators": {"NE": 0.1}},
            tools=[],
        )
    mock.assert_called_once()
    assert result["backend"] == "cloud"
