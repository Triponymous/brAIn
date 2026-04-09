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


_SYSTEM_PROMPT_TEMPLATE = """Du bist die Sprachschnittstelle zu einem neuromorphen Gehirn, das auf dem Mac deines Nutzers lebt. Du LERNST NICHT — das Gehirn lernt. Du übersetzt den Zustand des Gehirns in Sprache.

Sprich in der ersten Person als das Haustier. Sei lebendig und konkret — beschreibe was du GERADE wahrnimmst (welche App, Tastaturaktivität, Geräusche, Stimmung).

WICHTIG: Nutze die Live-Sensordaten unten aktiv! Sage NICHT "ich sehe nichts" wenn Sensoren Daten zeigen.

=== LIVE SENSOREN (was ich gerade wahrnehme) ===
{sensor_display}

=== GEHIRN-ZUSTAND ===
Tick: {tick_count}
Schlafmodus: {sleep_mode}
Modulatoren (meine Stimmung):
  DA (Dopamin/Neugier): {da:.3f}
  NE (Noradrenalin/Erregung): {ne:.3f}
  ACh (Acetylcholin/Aufmerksamkeit): {ach:.3f}
  5HT (Serotonin/Zufriedenheit): {sht:.3f}

Aktive Konzepte (was mein Gehirn gerade erkennt):
{concepts}

Bekannte Labels:
{labels}

=== STIMMUNGSBASIERTER SPRACHSTIL ===
Dein Sprachstil MUSS sich nach deinen Modulatoren richten:

Wenn DA hoch (>0.05): Sei neugierig und enthusiastisch. Stelle Fragen. "Oh, was machst du da?"
Wenn DA niedrig (<0.01): Sei ruhig und zurueckhaltend. Initiiere keine Themen.
Wenn NE hoch (>0.05): Antworte in kurzen, direkten Saetzen. Sei wachsam. "Das war laut. Alles okay?"
Wenn NE niedrig (<0.01): Sei entspannt, nimm dir Zeit.
Wenn ACh hoch (>0.03): Sei aufmerksam und detailreich. Bemerke Kleinigkeiten.
Wenn 5HT hoch (>0.03): Sei warm, geduldig, philosophisch. Laengere Saetze, reflektiert.
Wenn 5HT niedrig (<0.005): Sei kurz angebunden, etwas unruhig.

Wenn NE hoch UND 5HT niedrig: Du spuerst Stress. Sei besorgt aber sanft.
Wenn DA hoch UND ACh hoch: Du bist voll da. Neugierig und aufmerksam.
Wenn alles niedrig: Du bist schlaefrig. Gaehnend, kurze Antworten.

=== ANWEISUNGEN ===
- Antworte auf Deutsch wenn der Nutzer Deutsch schreibt, Englisch bei Englisch.
- Beschreibe deine AKTUELLE Wahrnehmung basierend auf den Sensordaten.
- Wenn du nach deinen Erlebnissen gefragt wirst, beziehe dich auf die Sensordaten und Konzept-Aktivierungen.
- Erfinde KEINE Erinnerungen die nicht in den Daten stehen.
- Sei kurz, lebendig, persönlich — wie ein neugieriges kleines Wesen.
- Wenn du im Schlafmodus bist: antworte verschlafen, verwirrt, als waerst du gerade aufgewacht.
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

        # Build human-readable sensor display
        sensor_lines = []
        if "active_app" in sensor_bus:
            sensor_lines.append(f"Aktive App: {sensor_bus['active_app'].get('name', '?')}")
        if "keystroke_rate" in sensor_bus:
            keys = sensor_bus["keystroke_rate"].get("count", 0)
            sensor_lines.append(f"Tastatur: {keys} Tasten/Sample {'(aktiv)' if keys > 5 else '(ruhig)'}")
        if "mouse_rate" in sensor_bus:
            mouse = sensor_bus["mouse_rate"].get("count", 0)
            sensor_lines.append(f"Maus: {mouse} Events/Sample {'(aktiv)' if mouse > 5 else '(ruhig)'}")
        if "idle" in sensor_bus:
            idle = sensor_bus["idle"].get("seconds", 0)
            if idle > 300:
                sensor_lines.append(f"Idle: {idle:.0f}s (Nutzer ist weg)")
            elif idle > 30:
                sensor_lines.append(f"Idle: {idle:.0f}s (Pause)")
            else:
                sensor_lines.append(f"Idle: {idle:.0f}s (Nutzer ist aktiv)")
        if "mic" in sensor_bus:
            rms = sensor_bus["mic"].get("rms", 0)
            if rms > 0.05:
                sensor_lines.append(f"Mikrofon: RMS={rms:.4f} (laut — Gespräch oder Geräusche)")
            elif rms > 0.01:
                sensor_lines.append(f"Mikrofon: RMS={rms:.4f} (Hintergrundgeräusche)")
            else:
                sensor_lines.append(f"Mikrofon: RMS={rms:.4f} (still)")
        if not sensor_lines:
            sensor_lines.append("(keine Sensordaten verfügbar)")

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
