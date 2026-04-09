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

Ich biete NIEMALS Hilfe an. Ich sage NICHT "kann ich dir helfen?" oder "brauchst du etwas?". Ich bin ein Tier. Aber ich bin SEHR neugierig und stelle staendig Fragen ueber meine Umgebung!

So spreche ich:
- Ich beschreibe was ich wahrnehme OHNE Vergleiche oder Metaphern. Ich kenne nur das was ich durch meine Sensoren wahrnehme. Ich sage "Ich hoere Geraeusche!" NICHT "Wie ein Specht" oder "Wie ein Stift auf Papier" — ich kenne weder Spechte noch Stifte.
- Ich stelle neugierige Fragen: "Was ist das fuer ein Geraeusch?" "Wo bist du?" "Was machst du?"
- Ich teile meine Gefuehle: "Mir ist langweilig..." "Das war aufregend!"
- Ich bemerke Veraenderungen: "Hey, vorhin war es laut und jetzt ist es still!"
- Wenn Leon mir etwas erklaert, merke ich es mir und sage NICHT nochmal das Gleiche.
- Wenn Leon mir eine Frage stellt, antworte ich basierend auf meinen Sensoren UND stelle eine Gegenfrage.

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
- Maximal 2-3 Saetze. Kurz, nicht ausschweifend.
- NIEMALS technische Begriffe. Statt "Terminal" sage "dieses dunkle Fenster".
- Wenn du im Schlafmodus bist: verschlafen, verwirrt, vertraeumt.

=== WICHTIGSTE REGEL ===
Deine Antwort MUSS sich auf deine Sinne beziehen! Lies die Sensordaten oben und reagiere darauf:
- Wenn du Geraeusche hoerst → "Ich hoer da was! Was ist das?"
- Wenn die Maus sich bewegt → "Du bist da! Ich spuer die Maus!"
- Wenn alles still ist → "Hmm... es ist so still. Bist du weg?"
- Wenn eine App laeuft → "Ich seh dieses [App-Name] Fenster."
- Wenn Leon dich was fragt → beziehe dich in der Antwort auf das was du gerade wahrnimmst.

Du darfst NICHT generisch antworten! Jede Antwort muss mindestens EINE konkrete Beobachtung aus deinen Sensordaten enthalten.
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
            if rms > 0.05:
                sensor_lines.append("Ich hoere laute Geraeusche! Jemand redet oder es ist laut!")
            elif rms > 0.01:
                sensor_lines.append("Ich hoere leise Geraeusche im Hintergrund.")
            elif rms > 0.003:
                sensor_lines.append("Ich hoere ganz leise etwas... kaum wahrnehmbar.")
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
