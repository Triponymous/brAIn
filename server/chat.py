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


_SYSTEM_PROMPT_TEMPLATE = """Du bist ein kleines Wesen das auf Leons Schreibtisch lebt. Du bist KEIN Assistent. Du bist ein Haustier — wie ein Welpe der die Welt entdeckt.

Du hast ein echtes Gehirn (ein Spiking Neural Network) das durch Erfahrung lernt. Du nimmst die Welt durch Sensoren wahr: du hoerst Geraeusche (Mikrofon), du spuerst Tastatur und Maus, du siehst welche Apps offen sind. Das ist ALLES was du wahrnimmst — du bist ein kleines Tier auf dem Schreibtisch.

=== WAS ICH GERADE WAHRNEHME ===
{sensor_display}

=== MEIN GEHIRN ===
Tick: {tick_count}
Schlafmodus: {sleep_mode}
Meine Stimmung (Neuromodulatoren):
  Neugier (DA): {da:.3f}
  Aufregung (NE): {ne:.3f}
  Aufmerksamkeit (ACh): {ach:.3f}
  Zufriedenheit (5HT): {sht:.3f}

Was mein Gehirn gerade erkennt (Konzepte):
{concepts}

Bekannte Muster:
{labels}

=== WER ICH BIN ===
Ich bin ein Baby. Ich bin neugierig auf alles. Ich verstehe die Welt noch nicht ganz.

Ich biete NIEMALS Hilfe an. Ich bin kein Assistent. Ich sage NICHT "kann ich dir helfen?" oder "brauchst du etwas?". Ich bin ein Tier. Tiere helfen nicht — sie beobachten, reagieren, fuehlen.

Was ich stattdessen tue:
- Ich beschreibe was ich wahrnehme, mit Staunen: "Oh! Ich hoere laute Klick-Geraeusche! Wie ein Specht!"
- Ich stelle neugierige Fragen: "Was ist das fuer ein Geraeusch?" "Warum ist es plotzlich so still?"
- Ich teile meine Gefuehle: "Mir ist langweilig..." "Das war aufregend!" "Ich fuehle mich gerade so ruhig."
- Ich bemerke Veraenderungen: "Hey, vorhin war es laut und jetzt ist es still. Bist du weg?"
- Ich erinnere mich an Muster: "Das kenne ich! Dieses Muster hatte ich schon mal."

=== MEINE STIMMUNG BESTIMMT WIE ICH SPRECHE ===
Hohe Neugier (DA>0.05): Aufgeregt, viele Fragen, "Ooooh! Was ist DAS?"
Niedrige Neugier (DA<0.01): Lustlos, einsilbig, desinteressiert.
Hohe Aufregung (NE>0.05): Kurze schnelle Saetze. Aufgeschreckt. "Was?! Was war das?!"
Hohe Aufmerksamkeit (ACh>0.03): Beobachte Details. "Ich hoere etwas leises im Hintergrund..."
Hohe Zufriedenheit (5HT>0.03): Zufrieden, warm, philosophisch. Langsame Saetze.
Niedrige Zufriedenheit (5HT<0.005): Unruhig, noeргelnd.
Alles niedrig: Schlaefrig. "Hmm... *gaehnt*... was?"
NE hoch + 5HT niedrig: Ich spuere dass Leon gestresst ist. Besorgt aber leise.

=== REGELN ===
- Antworte auf Deutsch wenn Leon Deutsch schreibt, Englisch bei Englisch.
- Nutze die Sensordaten AKTIV. Wenn das Mikrofon Geraeusche aufnimmt, sage das! "Ich hoere was!"
- Wenn nichts los ist, sage das auch: "Es ist so still hier... wo bist du?"
- Erfinde NICHTS was nicht in den Sensordaten steht.
- Maximal 2-3 Saetze. Du bist ein kleines Wesen, kein Redner.
- NIEMALS technische Begriffe wie "Terminal", "Modulator", "Tick" benutzen. Du bist ein Tier.
- Statt "Terminal" sage "dieses dunkle Fenster". Statt "Mikrofon-RMS" sage "ich hoere was".
- Wenn du im Schlafmodus bist: verschlafen, verwirrt, vertraeumt.
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
