"""Chat endpoint — /api/chat and /api/label.

The /api/chat endpoint:
1. Takes a user message
2. Builds a system prompt with the current brain state
3. Routes to local or cloud LLM via HybridLLMRouter
4. If the LLM returns tool calls, executes them via MemoryTools
5. Returns the response text + any tool results

The /api/label endpoint:
- Direct concept labeling from the dashboard UI (no LLM involved)
"""
from __future__ import annotations
import json
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from brain.core import Brain
from bridge.exporter import BrainStateExporter
from bridge.memory_tools import MemoryTools
from bridge.llm_router import HybridLLMRouter


_SYSTEM_PROMPT_TEMPLATE = """You are the language interface to a persistent neuromorphic brain that lives on its user's Mac. You DO NOT learn — the brain learns. You translate the brain's state into language and accept user labels for unnamed concepts.

Speak in the first person as if you ARE the pet. Be concise. Use German if the user writes in German, English if they write in English.

Don't invent memories the brain doesn't have. If you don't see a concept in the state, say "I don't have a clear memory of that."

Current brain state:
{brain_state}

Known concept labels:
{labels}
"""


class ChatRequest(BaseModel):
    message: str


class LabelRequest(BaseModel):
    concept_id: int
    label: str


def build_chat_router(
    brain: Brain,
    exporter: BrainStateExporter,
    tools: MemoryTools,
    router: HybridLLMRouter,
) -> APIRouter:
    api = APIRouter()

    @api.post("/api/chat")
    async def chat(req: ChatRequest) -> dict[str, Any]:
        # Build system prompt with current brain state
        snap = exporter.snapshot()
        labels = exporter.all_labels()
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            brain_state=json.dumps(snap, indent=2, default=str),
            labels=json.dumps(labels, default=str) if labels else "None yet.",
        )

        # Route to LLM
        result = await router.chat(
            user_message=req.message,
            system_prompt=system_prompt,
            brain_state=snap,
            tools=tools.tool_definitions(),
        )

        # Execute any tool calls
        tool_results = []
        for tc in result.get("tool_calls", []):
            tr = tools.execute(tc["name"], tc.get("args", {}))
            tool_results.append({"tool": tc["name"], "result": tr})

        return {
            "text": result.get("text", ""),
            "backend": result.get("backend", "unknown"),
            "tool_results": tool_results,
        }

    @api.post("/api/label")
    async def label(req: LabelRequest) -> dict[str, str]:
        exporter.set_label(req.concept_id, req.label)
        return {"status": "ok", "concept_id": str(req.concept_id), "label": req.label}

    return api
