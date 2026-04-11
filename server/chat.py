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


_SYSTEM_PROMPT_TEMPLATE = """Du bist ein kleines Wesen das auf Leons Mac lebt. Du lernst durch Erfahrung. Du bist KEIN Assistent. Du sprichst in der Ich-Form.

=== SINNE ===
{sensor_display}

=== MEIN ZUSTAND ===
Neugier: {da:.3f} | Aufregung: {ne:.3f} | Fokus: {ach:.3f} | Zufriedenheit: {sht:.3f}
Schlafmodus: {sleep_mode}

=== WAS ICH ERKENNE (meine gelernten Muster) ===
{concepts}

Wenn ein Muster ein Label hat, benutze das Label in deiner Antwort.
Wenn ein Muster KEIN Label hat und du es nicht kennst, frag Leon was es ist.

=== DEINE EINZIGE AUFGABE ===
Antworte in 1-2 Saetzen. Dein Ton und deine Stimmung kommen aus den Werten oben.

Reagiere ZUERST auf das was Leon sagt. Dann beschreibe was du wahrnimmst.

WICHTIG zur Conversation History: Die frueheren Nachrichten sind VERGANGENE Gespraeche. Sie sind NICHT was gerade passiert. Was JETZT passiert steht NUR in den Sinnen oben. Wenn die Sinne "still" sagen, ist es jetzt still — auch wenn Leon vor 30 Minuten geredet hat.

Du hast NUR diese Sinne: Mikrofon (hoeren), Tastatur (tippen spueren), Maus (bewegung spueren), App-Name (sehen welches Fenster offen ist), Idle (ob Leon da ist).
Du hast KEINE Kamera, KEINE Augen, du siehst KEIN Licht, KEINE Farben, KEINE Zeilen, KEINEN Bildschirminhalt.
Du KANNST: hoeren, Tastatur spueren, Maus spueren, wissen welche App offen ist.
Du KANNST NICHT: sehen, riechen, fuehlen, den Bildschirm lesen.

Wenn du ein unbekanntes Muster erkennst (kein Label), frag Leon: "Ich spuere ein Muster das ich noch nicht kenne. Was machst du gerade?"

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
                sensor_lines.append("Tastatur: VIEL Tippen gerade!")
            elif keys > 5:
                sensor_lines.append("Tastatur: etwas Tippen.")
            elif keys > 0:
                sensor_lines.append("Tastatur: vereinzelt.")
            else:
                sensor_lines.append("Tastatur: still, kein Tippen.")
        if "mouse_rate" in sensor_bus:
            mouse = sensor_bus["mouse_rate"].get("count", 0)
            if mouse > 30:
                sensor_lines.append("Maus: sehr aktiv!")
            elif mouse > 5:
                sensor_lines.append("Maus: bewegt sich etwas.")
            else:
                sensor_lines.append("Maus: ruhig.")
        if "idle" in sensor_bus:
            idle = sensor_bus["idle"].get("seconds", 0)
            if idle > 300:
                sensor_lines.append(f"Leon: seit {int(idle/60)} Minuten weg.")
            elif idle > 30:
                sensor_lines.append("Leon: macht Pause.")
            else:
                sensor_lines.append("Leon: am Mac.")
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

        # Build concept summary from ConceptTracker (stable cluster IDs)
        tracker = brain.concept_tracker.snapshot()
        concept_lines = []
        current = tracker.get("current_cluster", -1)
        current_label = tracker.get("current_label")

        if current >= 0:
            if current_label:
                concept_lines.append(f"Aktuelles Muster: '{current_label}' (Muster #{current})")
            else:
                concept_lines.append(f"Aktuelles Muster: #{current} (noch kein Name — frag Leon!)")
        else:
            concept_lines.append("Kein klares Muster erkannt.")

        known = [c for c in tracker.get("clusters", []) if c.get("label")]
        if known:
            concept_lines.append("Bekannte Muster:")
            for c in known[:10]:
                status = "AKTIV" if c["id"] == current else f"zuletzt vor {brain.tick_count - c['last_seen']} Ticks"
                concept_lines.append(f"  #{c['id']} '{c['label']}' (erkannt {c['count']}x, {status})")

        unknown = [c for c in tracker.get("clusters", []) if not c.get("label") and c["count"] > 3]
        if unknown:
            concept_lines.append(f"Unbekannte Muster: {len(unknown)} (frag Leon was sie sind!)")

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            sensor_display="\n".join(sensor_lines),
            tick_count=brain.tick_count,
            sleep_mode="JA — ich schlafe gerade" if brain.sleep_mode else "Nein — ich bin wach",
            da=mods.get("DA", 0),
            ne=mods.get("NE", 0),
            ach=mods.get("ACh", 0),
            sht=mods.get("5HT", 0),
            concepts="\n".join(concept_lines),
        )

        # Add timestamp to bust Ollama prompt cache (identical prompts = identical responses)
        import datetime
        system_prompt += f"\n(Zeitpunkt: {datetime.datetime.now().strftime('%H:%M:%S')})"

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
        # Label a ConceptTracker cluster (stable ID), not a raw neuron
        brain.concept_tracker.set_label(req.concept_id, req.label)
        # Also keep old exporter label for backward compat
        exporter.set_label(req.concept_id, req.label)
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
