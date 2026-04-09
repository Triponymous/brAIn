"""Anthropic Claude client for cloud LLM inference with tool-use support."""
from __future__ import annotations
from typing import Any
import anthropic


async def claude_chat(
    model: str,
    system_prompt: str,
    user_message: str,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Send a chat request to Claude and return text + any tool calls."""
    client = anthropic.AsyncAnthropic()

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }
    if tools:
        kwargs["tools"] = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in tools
        ]

    response = await client.messages.create(**kwargs)

    text_parts = []
    tool_calls = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({"name": block.name, "args": block.input})

    return {
        "text": "\n".join(text_parts),
        "tool_calls": tool_calls,
    }
