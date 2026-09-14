"""ollama_chat — the tool loop against a fake Ollama."""
import json

import pytest

from bridge import llm_local
from bridge.llm_local import ollama_chat, to_function_tools

TOOLS = [{"name": "brain_state", "description": "how I am",
          "input_schema": {"type": "object", "properties": {}}}]


class _Resp:
    def __init__(self, message):
        self._message = message

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": self._message}


def _fake_ollama(monkeypatch, responses):
    """httpx.AsyncClient stand-in: post() answers with the given messages in order and records every body."""
    bodies = []

    class Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json):
            bodies.append(json)
            return _Resp(responses[len(bodies) - 1])

    monkeypatch.setattr(llm_local.httpx, "AsyncClient", Client)
    return bodies


def _tool_call(name="brain_state", arguments=None):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": {} if arguments is None else arguments}}]}


async def test_tool_call_round_trip(monkeypatch):
    bodies = _fake_ollama(monkeypatch, [_tool_call(), {"role": "assistant", "content": "Du bist im Flow."}])
    executed = []

    async def execute(name, args):
        executed.append((name, args))
        return {"felt": "flow"}

    out = await ollama_chat("qwen", "sys", "wie geht's?", tools=TOOLS, execute=execute)

    assert out["text"] == "Du bist im Flow."
    assert out["tool_calls"] == [{"name": "brain_state", "args": {}, "result": {"felt": "flow"}}]
    assert executed == [("brain_state", {})]
    assert bodies[0]["tools"] == to_function_tools(TOOLS)
    assert bodies[0]["tools"][0]["function"]["parameters"] == TOOLS[0]["input_schema"]
    # the second request carries the assistant's tool call and the tool's answer
    assert bodies[1]["messages"][-2]["tool_calls"]
    assert bodies[1]["messages"][-1] == {"role": "tool", "tool_name": "brain_state",
                                         "content": json.dumps({"felt": "flow"}, ensure_ascii=False)}


async def test_string_arguments_are_parsed(monkeypatch):
    _fake_ollama(monkeypatch, [_tool_call("brain_recall", '{"hours": 3, "label": "coding"}'),
                               {"role": "assistant", "content": "ok"}])
    seen = {}

    async def execute(name, args):
        seen[name] = args
        return {}

    await ollama_chat("qwen", "sys", "?", tools=TOOLS, execute=execute)
    assert seen == {"brain_recall": {"hours": 3, "label": "coding"}}


async def test_without_tools_a_plain_answer(monkeypatch):
    bodies = _fake_ollama(monkeypatch, [{"role": "assistant", "content": "Alles ruhig."}])
    out = await ollama_chat("qwen", "sys", "hi", history=[{"role": "user", "content": "frueher"},
                                                          {"role": "assistant", "content": "ja"}])
    assert out == {"text": "Alles ruhig.", "tool_calls": []}
    assert "tools" not in bodies[0]
    assert [m["role"] for m in bodies[0]["messages"]] == ["system", "user", "assistant", "user"]


async def test_round_budget_forces_an_answer(monkeypatch):
    bodies = _fake_ollama(monkeypatch, [_tool_call(), _tool_call(), {"role": "assistant", "content": "genug"}])

    async def execute(name, args):
        return {}

    out = await ollama_chat("qwen", "sys", "?", tools=TOOLS, execute=execute, max_rounds=2)
    assert out["text"] == "genug" and len(out["tool_calls"]) == 2
    assert "tools" in bodies[0] and "tools" in bodies[1] and "tools" not in bodies[2]


async def test_tool_call_without_executor_returns_the_text(monkeypatch):
    _fake_ollama(monkeypatch, [_tool_call()])
    out = await ollama_chat("qwen", "sys", "?", tools=TOOLS)
    assert out == {"text": "", "tool_calls": []}
