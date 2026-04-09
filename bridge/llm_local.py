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
    # Append /no_think to qwen3 prompts to disable internal chain-of-thought.
    # Qwen3 "thinking" mode generates thousands of hidden tokens before
    # responding, adding 30-120s latency even on M4. Not needed for a pet.
    if "qwen3" in model.lower():
        user_message = user_message + " /no_think"

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
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]
