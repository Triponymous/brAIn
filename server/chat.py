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

WICHTIG: Erwaehne das aktive Muster in deiner Antwort!
- Wenn es ein Label hat: "Ich erkenne [Label]!"
- Wenn es KEIN Label hat: "Ich spuere Muster #[ID] — das kenne ich noch nicht. Was machst du gerade, Leon?"
- Wenn Leon dir sagt was ein Muster ist, merke es dir.

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

        # Use the SAME smoothed sensor data that the dashboard sees
        # (stored by push_loop on brain._last_sensor_display)
        sensor_display_cached = getattr(brain, '_last_sensor_display', {})

        # Build sensor display using the SAME data the dashboard shows
        # (smoothed by push_loop, stored on brain._last_sensor_display)
        sd = sensor_display_cached
        sensor_lines = []

        app = sd.get("app", "?")
        sensor_lines.append(f"Fenster: '{app}'")

        keys = sd.get("keys", 0)
        if keys > 10:
            sensor_lines.append("Tastatur: Leon tippt viel!")
        elif keys > 2:
            sensor_lines.append("Tastatur: Leon tippt.")
        elif keys > 0:
            sensor_lines.append("Tastatur: vereinzelt getippt.")
        else:
            sensor_lines.append("Tastatur: still.")

        mouse = sd.get("mouse", 0)
        if mouse > 20:
            sensor_lines.append("Maus: sehr aktiv!")
        elif mouse > 3:
            sensor_lines.append("Maus: bewegt sich.")
        else:
            sensor_lines.append("Maus: ruhig.")

        idle = sd.get("idle", 0)
        if idle > 300:
            sensor_lines.append(f"Leon: seit {int(idle/60)} Minuten weg.")
        elif idle > 30:
            sensor_lines.append("Leon: macht Pause.")
        else:
            sensor_lines.append("Leon: am Mac.")

        mic = sd.get("mic_rms", 0)
        if mic > 0.003:
            sensor_lines.append("Ich hoere deutliche Geraeusche!")
        elif mic > 0.001:
            sensor_lines.append("Ich hoere Geraeusche.")
        elif mic > 0.0005:
            sensor_lines.append("Ich hoere ganz leise etwas.")
        else:
            sensor_lines.append("Es ist still.")

        if not sd:
            sensor_lines = ["Meine Sinne starten gerade..."]

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

        # Recent history from episode logger (last 5 entries = ~50 seconds)
        history_lines = []
        try:
            from bridge.episode_log import EpisodeLogger
            ep_path = Path("checkpoints/episodes.db")
            if ep_path.exists():
                ep = EpisodeLogger(ep_path)
                recent = ep.query(last_n=5)
                ep.close()
                for e in reversed(recent):  # oldest first
                    ts = e.get("timestamp", 0)
                    import datetime as dt_mod
                    t = dt_mod.datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "?"
                    sensors = e.get("sensor_summary", {})
                    cluster = sensors.get("cluster_id", -1)
                    label = sensors.get("cluster_label") or f"Muster #{cluster}" if cluster >= 0 else "?"
                    app = sensors.get("app", "?")
                    history_lines.append(f"  {t}: {label} (App: {app})")
        except Exception:
            pass

        if history_lines:
            concept_lines.append("\nWas in letzter Zeit passiert ist:")
            concept_lines.extend(history_lines)

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
