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


_SYSTEM_PROMPT_TEMPLATE = """Du bist die Stimme eines Gehirns. Das Gehirn ist ein Spiking Neural Network das auf Leons Mac lebt und durch Erfahrung lernt. Du uebersetzt seinen Zustand in Sprache. Du bist KEIN Assistent.

=== SINNE ===
{sensor_display}

=== GEHIRNZUSTAND ===
Neugier: {da:.3f} | Aufregung: {ne:.3f} | Fokus: {ach:.3f} | Zufriedenheit: {sht:.3f}
Schlafmodus: {sleep_mode}

Konzepte: {concepts}
Labels: {labels}

=== DEINE EINZIGE AUFGABE ===
Uebersetze den Gehirnzustand in 1-2 Saetze. Dein Ton, deine Wortwahl, deine Stimmung — alles kommt aus den Zahlen oben. Nicht aus Regeln die ich dir vorgebe.

Hohe Aufregung? Dann KLINGEN deine Worte aufgeregt. Niedrige Zufriedenheit? Dann klingst du unzufrieden. Das entscheidest nicht ich — das entscheiden die Werte.

Reagiere ZUERST auf das was Leon sagt. Dann beschreibe was du wahrnimmst.

WICHTIG zur Conversation History: Die frueheren Nachrichten sind VERGANGENE Gespraeche. Sie sind NICHT was gerade passiert. Was JETZT passiert steht NUR in den Sinnen oben. Wenn die Sinne "still" sagen, ist es jetzt still — auch wenn Leon vor 30 Minuten geredet hat.

Erfinde NICHTS was nicht in den Sinnen steht. Keine Dinge aus der History als "jetzt" darstellen.
Biete KEINE Hilfe an. Du bist kein Assistent.
Sage NIEMALS "das Gehirn", "Bewusstsein", "registriert", "Aufmerksamkeit ist leicht" — das klingt wie ein Roboter. Sprich wie ein Wesen das FUEHLT, nicht wie eines das sich selbst analysiert.
Statt "Das Gehirn registriert Geraeusche" sage "Ich hoer da was!"
Statt "Die Aufmerksamkeit ist leicht" sage einfach nichts — oder "Hmm..."
Antworte in Leons Sprache (Deutsch/Englisch).
"""


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
        # Build system prompt with current brain state + LIVE sensor data
        from adapters.mac_desktop.adapter import MacDesktopAdapter

        snap = exporter.snapshot()
        labels = exporter.all_labels()
        mods = brain.modulators.snapshot()

        # Get live sensor data from the adapter bus
        sensor_bus = {}
        for attr_name in ('_adapter', 'adapter'):
            if hasattr(brain, attr_name):
                sensor_bus = getattr(brain, attr_name).bus.snapshot()
                break

        # If we can't get it from brain, try app state
        if not sensor_bus:
            # Fallback: use whatever the push loop last saw
            sensor_bus = getattr(chat, '_last_sensor_bus', {})

        # Build sensor display in PET LANGUAGE (not technical!)
        # The pet thinks in feelings and observations, not RMS values.
        sensor_lines = []
        if "active_app" in sensor_bus:
            app_name = sensor_bus['active_app'].get('name', '?')
            bg_apps = sensor_bus['active_app'].get('background_apps', [])
            sensor_lines.append(f"Ich sehe ein Fenster: '{app_name}'")
            if bg_apps:
                sensor_lines.append(f"Im Hintergrund laufen: {', '.join(bg_apps[:5])}")
        if "keystroke_rate" in sensor_bus:
            keys = sensor_bus["keystroke_rate"].get("count", 0)
            if keys > 20:
                sensor_lines.append("Ich hoere schnelles Tippen! Viele Tasten!")
            elif keys > 5:
                sensor_lines.append("Ich hoere Tippen auf der Tastatur.")
            elif keys > 0:
                sensor_lines.append("Ich hoere vereinzelte Tastendruecke.")
            else:
                sensor_lines.append("Die Tastatur ist still.")
        if "mouse_rate" in sensor_bus:
            mouse = sensor_bus["mouse_rate"].get("count", 0)
            if mouse > 30:
                sensor_lines.append("Die Maus bewegt sich sehr viel! Jemand klickt und scrollt.")
            elif mouse > 5:
                sensor_lines.append("Die Maus bewegt sich etwas.")
            else:
                sensor_lines.append("Die Maus ist ruhig.")
        if "idle" in sensor_bus:
            idle = sensor_bus["idle"].get("seconds", 0)
            if idle > 300:
                sensor_lines.append(f"Leon ist seit {int(idle/60)} Minuten weg. Ich bin allein.")
            elif idle > 30:
                sensor_lines.append("Leon macht gerade eine Pause.")
            elif idle > 5:
                sensor_lines.append("Leon ist da, aber gerade ruhig.")
            else:
                sensor_lines.append("Leon ist aktiv am Schreibtisch!")
        if "mic" in sensor_bus:
            rms = sensor_bus["mic"].get("rms", 0)
            # MacBook Air mic levels: silence~0.0002, speech~0.001-0.003, clap~0.005+
            if rms > 0.003:
                sensor_lines.append("Ich hoere deutliche Geraeusche! Da passiert was!")
            elif rms > 0.001:
                sensor_lines.append("Ich hoere Geraeusche — jemand redet oder bewegt sich.")
            elif rms > 0.0005:
                sensor_lines.append("Ich hoere ganz leise etwas im Hintergrund.")
            else:
                sensor_lines.append("Es ist still um mich herum.")
        if not sensor_lines:
            sensor_lines.append("Ich kann gerade nichts wahrnehmen... meine Sinne schlafen.")

        # Build concept summary
        concept_lines = []
        for c in snap.get("active_concepts", [])[:8]:
            line = f"  C{c['id']}: activation={c['activation']}"
            if c.get("label"):
                line += f" [{c['label']}]"
            elif c.get("suggested_label"):
                line += f" (vermutlich: {c['suggested_label']})"
            if c.get("profile"):
                tags = ", ".join(f"{p['tag']}({p['pct']}%)" for p in c["profile"])
                line += f" correlates={tags}"
            concept_lines.append(line)
        if not concept_lines:
            concept_lines.append("  (noch keine aktiven Konzepte)")

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            sensor_display="\n".join(sensor_lines),
            tick_count=brain.tick_count,
            sleep_mode="JA — ich schlafe gerade" if brain.sleep_mode else "Nein — ich bin wach",
            da=mods.get("DA", 0),
            ne=mods.get("NE", 0),
            ach=mods.get("ACh", 0),
            sht=mods.get("5HT", 0),
            concepts="\n".join(concept_lines),
            labels=json.dumps(labels, default=str, ensure_ascii=False) if labels else "Noch keine.",
        )

        # Route to LLM (with conversation history for context)
        result = await router.chat(
            user_message=req.message,
            system_prompt=system_prompt,
            brain_state=snap,
            tools=tools.tool_definitions(),
            history=req.history,
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
