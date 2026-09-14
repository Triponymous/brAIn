"""claude_chat — the tool loop against a fake Anthropic client."""
from types import SimpleNamespace

from bridge import llm_cloud

TOOLS = [{"name": "brain_state", "description": "how I am",
          "input_schema": {"type": "object", "properties": {}}}]


def _fake_anthropic(monkeypatch, responses):
    calls = []

    class Messages:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return responses[len(calls) - 1]

    class Client:
        def __init__(self, *a, **k):
            self.messages = Messages()

    monkeypatch.setattr(llm_cloud.anthropic, "AsyncAnthropic", Client)
    return calls


def _text(t):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=t)])


def _use(name, input, id="tu_1"):
    return SimpleNamespace(content=[SimpleNamespace(type="tool_use", id=id, name=name, input=input)])


async def test_tool_use_round_trip(monkeypatch):
    calls = _fake_anthropic(monkeypatch, [_use("brain_state", {}), _text("Du bist im Flow.")])

    async def execute(name, args):
        return {"felt": "flow"}

    out = await llm_cloud.claude_chat("claude-x", "sys", "wie geht's?", tools=TOOLS, execute=execute)

    assert out["text"] == "Du bist im Flow."
    assert out["tool_calls"] == [{"name": "brain_state", "args": {}, "result": {"felt": "flow"}}]
    assert calls[0]["tools"] == TOOLS and calls[0]["system"] == "sys"
    msgs = calls[1]["messages"]
    assert msgs[-2]["role"] == "assistant"
    assert msgs[-1]["role"] == "user" and msgs[-1]["content"][0]["tool_use_id"] == "tu_1"
    assert msgs[-1]["content"][0]["type"] == "tool_result"


async def test_plain_answer_without_tools(monkeypatch):
    calls = _fake_anthropic(monkeypatch, [_text("Hallo.")])
    out = await llm_cloud.claude_chat("claude-x", "sys", "hi")
    assert out == {"text": "Hallo.", "tool_calls": []}
    assert "tools" not in calls[0]


async def test_round_budget_stops_the_loop(monkeypatch):
    calls = _fake_anthropic(monkeypatch, [_use("brain_state", {}, "a"), _use("brain_state", {}, "b"),
                                          _use("brain_state", {}, "c")])

    async def execute(name, args):
        return {}

    out = await llm_cloud.claude_chat("claude-x", "sys", "?", tools=TOOLS, execute=execute, max_rounds=2)
    assert len(calls) == 3 and len(out["tool_calls"]) == 2 and out["text"] == ""
