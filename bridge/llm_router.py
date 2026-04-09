"""Hybrid LLM Router — reads model config from config.json at runtime.

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


def _should_use_cloud(message: str, modulators: dict[str, float], cloud_enabled: bool) -> bool:
    if not cloud_enabled:
        return False
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
    def __init__(self) -> None:
        # No hardcoded defaults — reads from config on every call
        pass

    def _get_config(self) -> dict[str, Any]:
        from server.config import get
        return {
            "local_model": get("llm", "local_model", "qwen2.5:7b-instruct"),
            "cloud_model": get("llm", "cloud_model", "claude-haiku-4-5-20250404"),
            "cloud_enabled": get("llm", "cloud_enabled", False),
        }

    async def chat(
        self,
        user_message: str,
        system_prompt: str,
        brain_state: dict[str, Any],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        cfg = self._get_config()
        modulators = brain_state.get("modulators", {})
        use_cloud = _should_use_cloud(user_message, modulators, cfg["cloud_enabled"])

        if use_cloud:
            result = await self._call_claude(
                model=cfg["cloud_model"],
                system_prompt=system_prompt,
                user_message=user_message.removeprefix("/cloud").strip(),
                tools=tools,
            )
            result["backend"] = f"cloud ({cfg['cloud_model']})"
            return result
        else:
            text = await self._call_ollama(
                model=cfg["local_model"],
                system_prompt=system_prompt,
                user_message=user_message,
            )
            return {"text": text, "tool_calls": [], "backend": f"local ({cfg['local_model']})"}

    async def _call_ollama(self, model: str, system_prompt: str, user_message: str) -> str:
        return await ollama_chat(
            model=model,
            system_prompt=system_prompt,
            user_message=user_message,
        )

    async def _call_claude(
        self, model: str, system_prompt: str, user_message: str, tools: list[dict]
    ) -> dict[str, Any]:
        return await claude_chat(
            model=model,
            system_prompt=system_prompt,
            user_message=user_message,
            tools=tools or None,
        )
