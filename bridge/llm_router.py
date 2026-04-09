"""Hybrid LLM Router — local Ollama by default, Claude for complex queries.

Routing heuristics:
- Cloud if: message > 200 chars, reasoning keywords, NE > 0.7, /cloud prefix
- Local otherwise (faster, free, private)
"""
from __future__ import annotations
import re
from typing import Any

from bridge.llm_local import ollama_chat
from bridge.llm_cloud import claude_chat


_REASONING_KEYWORDS = re.compile(
    r"plan|recherchier|analysier|warum genau|vergleich|erkl.r mir|zusammenfassung",
    re.IGNORECASE,
)


def _should_use_cloud(message: str, modulators: dict[str, float]) -> bool:
    if message.startswith("/cloud"):
        return True
    if len(message) > 200:
        return True
    if _REASONING_KEYWORDS.search(message):
        return True
    if modulators.get("NE", 0) > 0.7:
        return True
    return False


class HybridLLMRouter:
    def __init__(
        self,
        ollama_model: str = "qwen2.5:7b-instruct",  # TODO: upgrade to 32b when pulled
        cloud_model: str = "claude-haiku-4-5-20250404",
    ) -> None:
        self.ollama_model = ollama_model
        self.cloud_model = cloud_model

    async def chat(
        self,
        user_message: str,
        system_prompt: str,
        brain_state: dict[str, Any],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Route the message to local or cloud LLM."""
        modulators = brain_state.get("modulators", {})
        use_cloud = _should_use_cloud(user_message, modulators)

        if use_cloud:
            result = await self._call_claude(
                system_prompt=system_prompt,
                user_message=user_message.removeprefix("/cloud").strip(),
                tools=tools,
            )
            result["backend"] = "cloud"
            return result
        else:
            text = await self._call_ollama(
                system_prompt=system_prompt,
                user_message=user_message,
            )
            return {"text": text, "tool_calls": [], "backend": "local"}

    async def _call_ollama(self, system_prompt: str, user_message: str) -> str:
        return await ollama_chat(
            model=self.ollama_model,
            system_prompt=system_prompt,
            user_message=user_message,
        )

    async def _call_claude(
        self, system_prompt: str, user_message: str, tools: list[dict]
    ) -> dict[str, Any]:
        return await claude_chat(
            model=self.cloud_model,
            system_prompt=system_prompt,
            user_message=user_message,
            tools=tools or None,
        )
