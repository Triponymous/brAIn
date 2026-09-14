"""Anthropic Claude client for cloud LLM inference with tool use.

Tool definitions are already in the Anthropic shape. When the model calls a
tool, the caller-supplied `execute` runs it and the result goes back as a
tool_result block, until the model answers in text (max_rounds).
"""
from __future__ import annotations
import json
from typing import Any, Awaitable, Callable

import anthropic


Execute = Callable[[str, dict[str, Any]], Awaitable[Any]]


async def claude_chat(
    model: str,
    system_prompt: str,
    user_message: str,
    tools: list[dict[str, Any]] | None = None,
    execute: Execute | None = None,
    max_rounds: int = 4,
) -> dict[str, Any]:
    """Chat with tool use. Returns {"text": ..., "tool_calls": [{name, args, result}, ...]}."""
    client = anthropic.AsyncAnthropic()

    kwargs: dict[str, Any] = {"model": model, "max_tokens": 1024, "system": system_prompt}
    if tools:
        kwargs["tools"] = [
            {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
            for t in tools
        ]
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    calls: list[dict[str, Any]] = []

    for round_ in range(max_rounds + 1):
        response = await client.messages.create(**kwargs, messages=messages)
        text = [b.text for b in response.content if b.type == "text"]
        uses = [b for b in response.content if b.type == "tool_use"]
        if not uses or execute is None or round_ == max_rounds:
            return {"text": "\n".join(text), "tool_calls": calls}
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for b in uses:
            args = dict(b.input or {})
            result = await execute(b.name, args)
            calls.append({"name": b.name, "args": args, "result": result})
            results.append({"type": "tool_result", "tool_use_id": b.id,
                            "content": json.dumps(result, ensure_ascii=False, default=str)})
        messages.append({"role": "user", "content": results})
    return {"text": "", "tool_calls": calls}  # unreachable: the loop returns on its last round
