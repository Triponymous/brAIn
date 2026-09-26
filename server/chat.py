"""Chat endpoint — /api/chat and /api/label.

The /api/chat endpoint:
1. Takes a user message
2. Builds the system prompt through the brain protocol (SCP): state,
   personality and conversation; past moments are for the brain tools to fetch
3. Routes to local or cloud LLM via HybridLLMRouter
4. If the LLM returns tool calls, executes them via MemoryTools
5. Returns the response text + any tool results

The /api/label endpoint:
- Direct concept labeling (no LLM involved)
"""
from __future__ import annotations
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from brain.core import Brain
from bridge.exporter import BrainStateExporter
from bridge.memory_tools import MemoryTools
from bridge.llm_router import HybridLLMRouter
from bridge.experience import state_of


class ChatRequest(BaseModel):
    message: str
    history: list[dict] | None = None  # conversation history [{role, content}]


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
        log = getattr(brain, "_experience", None)
        if log is not None:  # that a conversation happened, never what was said
            log.record("human", "chat", {"chars": len(req.message)}, state=state_of(brain))
        snap = exporter.snapshot()  # handed to the backend as brain_state

        # SCP v2: build prompt via protocol
        scp_client = getattr(brain, '_scp_client', None)

        if scp_client:
            # Send engagement feedback
            scp_client.send_feedback("engage")

            # Build the system prompt via SCP protocol
            system_prompt = scp_client.build_prompt(
                user_message=req.message,
                history=req.history,
            )
        else:
            # Fallback: minimal prompt without SCP
            system_prompt = "Du bist ein kleines Wesen auf Leons Mac. Antworte kurz und natuerlich."

        # Smart labeling: works with proactive label suggestions
        # The proactive engine suggests labels like "Coding mit Hintergrundmusik"
        # User can confirm ("ja", "passt") or provide alternative ("nenn es X")
        proactive = getattr(brain, '_proactive_engine', None)
        pending = getattr(proactive, '_pending_label_suggestion', None) if proactive else None

        if pending and req.message:
            msg_lower = req.message.lower().strip()
            cluster_id = pending.get("cluster_id")
            suggested = pending.get("suggestion", "")

            # User confirms the suggestion
            confirm_words = ["ja", "passt", "genau", "ok", "jap", "yes", "klar", "mach", "gut", "stimmt", "jo"]
            is_confirm = any(msg_lower.startswith(w) for w in confirm_words) or msg_lower in confirm_words

            # User provides alternative: "nenn es X", "lieber X", "nein, X"
            alt_prefixes = ["nenn", "lieber", "nein,", "nee,", "ne,", "besser"]
            alt_label = None
            for prefix in alt_prefixes:
                if msg_lower.startswith(prefix):
                    alt_label = msg_lower[len(prefix):].strip().strip(",").strip()
                    break

            if is_confirm and suggested:
                brain.concept_tracker.set_label(cluster_id, suggested)
                print(f"[label] cluster {cluster_id} <- '{suggested}' (user confirmed suggestion)")
                proactive._pending_label_suggestion = None
                if scp_client:
                    scp_client.send_feedback("reward")
            elif alt_label and len(alt_label) >= 3:
                final = alt_label[0].upper() + alt_label[1:]
                brain.concept_tracker.set_label(cluster_id, final)
                print(f"[label] cluster {cluster_id} <- '{final}' (user provided alternative)")
                proactive._pending_label_suggestion = None
                if scp_client:
                    scp_client.send_feedback("correct")
            elif not is_confirm:
                # User is talking about something else — clear pending
                proactive._pending_label_suggestion = None

        import datetime
        system_prompt += f"\n(Zeitpunkt: {datetime.datetime.now().strftime('%H:%M:%S')})"

        defs = tools.tool_definitions()
        brain_tool_names = [d["name"] for d in defs if d["name"].startswith("brain_")]
        if brain_tool_names:
            system_prompt += (
                "\n\n=== MEIN GEHIRN BEFRAGEN ===\n"
                "Ich habe Werkzeuge, um mein eigenes Gehirn zu lesen: " + ", ".join(brain_tool_names) + ".\n"
                "Fragt Leon, wie es mir geht, was frueher war, warum ich mich so fuehle, was er sonst um "
                "diese Zeit tut oder wann er zuletzt etwas getan hat, rufe ich ZUERST das passende "
                "Werkzeug auf und antworte dann aus dem Ergebnis. Ich erfinde keine Werte."
            )

        async def execute(name: str, args: dict[str, Any]) -> Any:
            if hasattr(tools, "execute_async"):
                result = await tools.execute_async(name, args)
            else:
                result = tools.execute(name, args)
            if log is not None:  # brain tools take labels and numbers; other tools' args may be content
                log.record("llm", "tool_call",
                           {"tool": name, "args": args} if name.startswith("brain_") else {"tool": name},
                           state=state_of(brain))
            return result

        # Route to LLM (with conversation history for context); the backend runs the tool loop
        try:
            result = await router.chat(
                user_message=req.message,
                system_prompt=system_prompt,
                brain_state=snap,
                tools=defs,
                history=req.history,
                execute=execute,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "text": f"(Mein Gehirn hatte einen Schluckauf: {type(e).__name__})",
                "backend": "error",
                "tool_results": [],
            }

        # What the model asked its brain, and what it was told
        tool_results = [{"tool": c["name"], "args": c.get("args", {}), "result": c.get("result")}
                        for c in result.get("tool_calls", [])]

        return {
            "text": result.get("text", ""),
            "backend": result.get("backend", "unknown"),
            "tool_results": tool_results,
        }

    @api.post("/api/label")
    async def label(req: LabelRequest) -> dict[str, str]:
        # Label a ConceptTracker cluster (stable ID), not a raw neuron
        brain.concept_tracker.set_label(req.concept_id, req.label)
        # Also keep old exporter label for backward compat
        exporter.set_label(req.concept_id, req.label)
        log = getattr(brain, "_experience", None)
        if log is not None:
            log.record("human", "label_concept", {"cluster_id": req.concept_id, "label": req.label},
                       state=state_of(brain))
        return {"status": "ok", "cluster_id": str(req.concept_id), "label": req.label}

    @api.get("/api/concept/{concept_id}")
    async def concept_profile(concept_id: int) -> dict[str, Any]:
        """Get the auto-correlation profile for a concept — what sensors trigger it."""
        profile = exporter.get_concept_profile(concept_id)
        label = exporter.get_label(concept_id)
        return {
            "concept_id": concept_id,
            "label": label,
            **profile,
        }

    return api
