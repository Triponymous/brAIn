"""Ollama HTTP client for local LLM inference, with tool use.

Uses the Ollama HTTP API (http://localhost:11434/api/chat) directly via httpx.
No Ollama SDK needed. Tools go out in OpenAI function format; when the model
calls one, the caller-supplied `execute` runs it and the result goes back as
a tool message, until the model answers in text (or the round budget is
spent, after which it is asked to answer with what it has).
"""
from __future__ import annotations
import json
import random
from typing import Any, Awaitable, Callable

import httpx


OLLAMA_BASE = "http://localhost:11434"

Execute = Callable[[str, dict[str, Any]], Awaitable[Any]]


def to_function_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic-style definitions (name, description, input_schema) in OpenAI/Ollama function format."""
    return [{
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
        },
    } for t in tools]


def _arguments(call: dict[str, Any]) -> dict[str, Any]:
    args = (call.get("function") or {}).get("arguments") or {}
    if isinstance(args, str):  # some Ollama versions send the arguments as a JSON string
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return args if isinstance(args, dict) else {}


async def ollama_chat(
    model: str,
    system_prompt: str,
    user_message: str,
    timeout: float = 300.0,
    history: list[dict] | None = None,
    tools: list[dict[str, Any]] | None = None,
    execute: Execute | None = None,
    max_rounds: int = 4,
) -> dict[str, Any]:
    """Chat with tool use. Returns {"text": ..., "tool_calls": [{name, args, result}, ...]}."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for h in (history or [])[-10:]:
        role, content = h.get("role", "user"), h.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})

    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "options": {
            "num_predict": 200,
            "temperature": 0.85,  # balance variety vs coherence
            "seed": random.randint(1, 999999),  # random seed BUSTS the KV cache
        },
    }
    # Qwen3 has a "thinking" mode that generates thousands of hidden
    # reasoning tokens before responding (30-120s latency). Disable it.
    if "qwen3" in model.lower():
        payload["think"] = False
    function_tools = to_function_tools(tools) if tools else None
    calls: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        async def ask(with_tools: bool) -> dict[str, Any]:
            body = {**payload, "messages": messages}
            if with_tools and function_tools:
                body["tools"] = function_tools
            resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=body)
            resp.raise_for_status()
            return resp.json()["message"]

        for _ in range(max_rounds):
            msg = await ask(with_tools=True)
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls or execute is None:
                return {"text": msg.get("content", ""), "tool_calls": calls}
            messages.append(msg)
            for call in tool_calls:
                name = (call.get("function") or {}).get("name", "")
                args = _arguments(call)
                result = await execute(name, args)
                calls.append({"name": name, "args": args, "result": result})
                messages.append({"role": "tool", "tool_name": name,
                                 "content": json.dumps(result, ensure_ascii=False, default=str)})
        msg = await ask(with_tools=False)  # budget spent: answer with what was gathered
        return {"text": msg.get("content", ""), "tool_calls": calls}
