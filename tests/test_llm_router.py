"""Tests for the hybrid LLM router.

In tests we mock both Ollama and Claude to avoid real API calls.
The router logic (which backend to use) is tested via the routing heuristics.
"""
import pytest
from unittest.mock import AsyncMock, patch
from bridge.llm_router import HybridLLMRouter, _should_use_cloud


def test_should_use_cloud_short_message():
    assert _should_use_cloud("hey was siehst du?", {}, cloud_enabled=True) is False


def test_should_use_cloud_long_message():
    long = "Kannst du mir erklären warum " + "Concept #12 " * 50 + "immer aktiv ist?"
    assert _should_use_cloud(long, {}, cloud_enabled=True) is True


def test_should_use_cloud_reasoning_keyword():
    assert _should_use_cloud("analysiere mein Tagesrhythmus", {}, cloud_enabled=True) is True


def test_should_use_cloud_high_ne():
    assert _should_use_cloud("was war das?", {"NE": 0.8}, cloud_enabled=True) is True


def test_should_use_cloud_explicit_flag():
    assert _should_use_cloud("/cloud was ist los", {}, cloud_enabled=True) is True


def test_should_use_cloud_disabled():
    # When cloud is disabled, even explicit /cloud flag should return False
    assert _should_use_cloud("/cloud was ist los", {}, cloud_enabled=False) is False


def test_router_construction():
    router = HybridLLMRouter()
    config = router._get_config()
    # Just verify it returns valid config keys, actual model depends on config file
    assert "local_model" in config
    assert "cloud_model" in config
    assert "cloud_enabled" in config


@pytest.mark.asyncio
async def test_router_calls_local_for_simple_query():
    router = HybridLLMRouter()
    with patch.object(router, '_call_ollama', new_callable=AsyncMock, return_value={"text": "Ich sehe VSCode.", "tool_calls": []}) as mock:
        result = await router.chat(
            user_message="was siehst du?",
            system_prompt="Du bist ein Pet.",
            brain_state={"modulators": {"NE": 0.1}},
            tools=[],
        )
    mock.assert_called_once()
    assert result["text"] == "Ich sehe VSCode."
    assert result["backend"].startswith("local")


@pytest.mark.asyncio
async def test_router_calls_cloud_for_complex_query():
    router = HybridLLMRouter()
    # Must enable cloud in config for the router to use it
    with patch.object(router, '_get_config', return_value={
        "local_model": "qwen3:14b",
        "cloud_model": "claude-haiku-4-5-20250404",
        "cloud_enabled": True,
    }):
        with patch.object(router, '_call_claude', new_callable=AsyncMock, return_value={"text": "Analyse...", "tool_calls": []}) as mock:
            result = await router.chat(
                user_message="analysiere warum Concept #12 und #47 immer zusammen feuern",
                system_prompt="Du bist ein Pet.",
                brain_state={"modulators": {"NE": 0.1}},
                tools=[],
            )
        mock.assert_called_once()
        assert result["backend"].startswith("cloud")


@pytest.mark.asyncio
async def test_router_hands_tools_and_execute_to_the_local_backend():
    router = HybridLLMRouter()

    async def execute(name, args):
        return {}

    with patch.object(router, '_call_ollama', new_callable=AsyncMock,
                      return_value={"text": "ok", "tool_calls": [{"name": "brain_state", "args": {}, "result": {}}]}) as mock:
        result = await router.chat(user_message="hi", system_prompt="s", brain_state={},
                                   tools=[{"name": "brain_state"}], execute=execute)
    assert mock.call_args.kwargs["tools"] == [{"name": "brain_state"}]
    assert mock.call_args.kwargs["execute"] is execute
    assert result["tool_calls"][0]["name"] == "brain_state" and result["backend"].startswith("local")
