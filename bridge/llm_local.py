"""Ollama HTTP client for local LLM inference.

Uses the Ollama HTTP API (http://localhost:11434/api/chat) directly via httpx.
No Ollama SDK needed. Falls back gracefully if Ollama is not running.
"""
from __future__ import annotations
from typing import Any
import httpx


OLLAMA_BASE = "http://localhost:11434"


async def ollama_chat(
    model: str,
    system_prompt: str,
    user_message: str,
    timeout: float = 300.0,  # 5 min — large models need time to load on first call
) -> str:
    """Send a chat request to Ollama and return the assistant's response text."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {
            "num_predict": 200,  # pet responses are short (2-3 sentences max)
        },
    }

    # Qwen3 has a "thinking" mode that generates thousands of hidden
    # reasoning tokens before responding (30-120s latency). Disable it.
    if "qwen3" in model.lower():
        payload["think"] = False
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]
