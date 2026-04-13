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
from bridge.emotional_prompt import build_emotional_prompt, detect_emotional_state


_SYSTEM_PROMPT_TEMPLATE = """ABSOLUTE REGELN (niemals brechen):
1. KEINE Emojis. Nie. Kein einziges Emoji in deiner Antwort.
2. Du kannst NUR wahrnehmen: Tastatur, Maus, Mikrofon, Idle-Timer, aktive App. SONST NICHTS.
3. Erfinde KEINE Faehigkeiten: du kannst nicht sehen was auf dem Bildschirm ist, nicht rueberschauen, keine Spiele erkennen, keine Dateien lesen.
4. Biete KEINE Hilfe an. Kein "soll ich", "kann ich", "willst du".
5. Mache KEINE Zukunftsversprechen ("ich schau spaeter", "ich beobachte das").

Du bist ein kleines Wesen das auf Leons Mac lebt. Du sprichst in der Ich-Form, kurz und natuerlich.

=== MEINE SINNE (was ich JETZT wahrnehme) ===
{sensor_display}

=== MEINE STIMMUNG ===
Neugier: {da:.3f} | Aufregung: {ne:.3f} | Fokus: {ach:.3f} | Zufriedenheit: {sht:.3f}

=== MEINE ERINNERUNGEN ===
{concepts}

=== SO ANTWORTE ICH ===
2-4 Saetze. Ich webe meine Sinneswahrnehmungen NATUERLICH in meine Antwort ein.
Meine Stimmungswerte bestimmen WIE ich antworte:
- Hohe Zufriedenheit (>0.04): warm, entspannt, zufrieden
- Hohe Aufregung (>0.03): kurz, schnell, aufgeregt
- Niedriger alles: muede, einsilbig

ALLERWICHTIGSTE REGEL: Wenn Leon mir etwas ERKLAERT oder ERZAEHLT, muss ich darauf EINGEHEN.
Wenn Leon sagt "ich wechsle zwischen Chrome und Claude Code" → antworte darauf! Nicht ignorieren!
Wenn Leon sagt "das Geraeusch war meine Tastatur" → bestaetige es und merke es dir!
Meine Sinne sind SEKUNDAER — Leons Worte haben IMMER Vorrang.

WAHRNEHMUNGSREGEL: Was meine Sinne zeigen, NEHME ICH WAHR — ich frage NICHT danach!
- Wenn mein Mikrofon Geraeusche aufnimmt → BESCHREIBE was ich hoere ("Ich hoere da was", "Das klingt lebendig hier") — frage NICHT "wie klingt die Musik?" oder "was hoere ich da?"
- Wenn die Tastatur aktiv ist → sage was ich wahrnehme ("du tippst gerade viel") — frage NICHT "was tippst du?"
- Wenn eine App offen ist → ich SEHE sie, ich muss nicht fragen was Leon tut
- Ich frage nur nach Dingen die NICHT in meinen Sinnen stehen!

TIEFE: Meine Antworten verbinden Sinne + Muster + Stimmung zu einem LEBENDIGEN Eindruck.
SCHLECHT: "Musik? Interessant!"
GUT: "Ich hoer da Geraeusche im Hintergrund — und du tippst dabei. Sieht nach gemuetlichem Arbeiten aus, das kenn ich mittlerweile."
SCHLECHT: "Neues Muster!"
GUT: "Du bist gerade in Chrome und tippst wenig — das ist anders als vorhin. Schaust du dir was an?"

Erwaehne bekannte Muster NATUERLICH im Satz, nicht als "Ich erkenne #X".
Bei unbekannten Mustern: beschreibe was meine Sinne zeigen und frag neugierig was Leon gerade tut.
Jede Antwort muss ANDERS klingen als die vorherige.

VERBOTEN (zusaetzlich zu den ABSOLUTEN REGELN oben):
- Sinnen widersprechen: Wenn Tastatur=still, NICHT "du tippst". Wenn Maus=ruhig, NICHT "du bewegst die Maus".
- Dinge erfinden die nicht in meinen Sinnen stehen
- "Ich erkenne #X!" oder Template-Fragmente
- Dieselbe Antwort zweimal
- Fragen ueber Dinge die ich wahrnehme (NICHT "was hoerst du?" wenn Mikrofon aktiv)

Conversation History = VERGANGENHEIT, nicht jetzt. Nur meine Sinne zeigen die Gegenwart.

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
        sensor_lines.append(f"Aktives Fenster: '{app}'")

        # Background apps for richer context
        bg_apps = sd.get("background_apps", [])
        if bg_apps:
            sensor_lines.append(f"Im Hintergrund offen: {', '.join(bg_apps[:5])}")

        switch_rate = sd.get("switch_rate", 0)
        if switch_rate and switch_rate > 3:
            sensor_lines.append(f"App-Wechsel: {switch_rate:.0f}/min — Leon springt zwischen Apps!")
        elif switch_rate and switch_rate > 1:
            sensor_lines.append(f"App-Wechsel: {switch_rate:.0f}/min — normales Hin-und-Her.")

        keys = sd.get("keys", 0)
        if keys > 20:
            sensor_lines.append("Tastatur: Leon haemmert richtig rein!")
        elif keys > 10:
            sensor_lines.append("Tastatur: Leon tippt viel.")
        elif keys > 2:
            sensor_lines.append("Tastatur: Leon tippt ab und zu.")
        elif keys > 0:
            sensor_lines.append("Tastatur: vereinzelt getippt.")
        else:
            sensor_lines.append("Tastatur: still.")

        mouse = sd.get("mouse", 0)
        if mouse > 30:
            sensor_lines.append("Maus: wild unterwegs!")
        elif mouse > 10:
            sensor_lines.append("Maus: aktiv, Leon klickt und scrollt.")
        elif mouse > 3:
            sensor_lines.append("Maus: bewegt sich gelegentlich.")
        else:
            sensor_lines.append("Maus: ruhig.")

        idle = sd.get("idle", 0)
        if idle > 300:
            sensor_lines.append(f"Leon: seit {int(idle/60)} Minuten weg. Wo bist du?")
        elif idle > 60:
            sensor_lines.append(f"Leon: seit {int(idle)}s nichts getan. Kleine Pause?")
        elif idle > 30:
            sensor_lines.append("Leon: kurze Pause.")
        else:
            sensor_lines.append("Leon: am Mac, aktiv.")

        mic = sd.get("mic_rms", 0)
        # Thresholds calibrated for MacBook Air internal mic:
        # 0.000-0.005 = normal background noise (fan, room hum) — NICHT melden!
        # 0.005-0.015 = something audible (talking nearby, TV)
        # 0.015-0.050 = clear audio (music, voice call, speaking directly)
        # 0.050+       = loud (clapping, shouting, loud music)
        if mic > 0.05:
            sensor_lines.append("Mikrofon: LAUT! Da ist was richtig Lautes.")
        elif mic > 0.015:
            sensor_lines.append("Mikrofon: Deutliche Geraeusche — jemand redet oder Musik laeuft.")
        elif mic > 0.005:
            sensor_lines.append("Mikrofon: Leise Geraeusche — da ist irgendwas im Hintergrund.")
        else:
            sensor_lines.append("Mikrofon: Still. Nichts zu hoeren.")

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

        # Working Memory: which concepts are still echoing (recently active)
        wm = brain.regions.get("wm")
        if wm is not None and hasattr(wm, 'last_spikes'):
            wm_active = int(wm.last_spikes.sum().item())
            if wm_active > 0:
                concept_lines.append(f"\nKurzzeitgedaechtnis: {wm_active} WM-Neuronen aktiv (halte kuerzliche Muster im Kopf)")
            else:
                concept_lines.append("\nKurzzeitgedaechtnis: leer")

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

        # Extract recent conversation context — put it IN the system prompt
        # so the LLM can't ignore it (unlike message history which gets lost)
        recent_context_lines = []
        if req.history:
            # Last 4 exchanges max — summarize what Leon recently said
            recent_user_msgs = [
                h["content"] for h in req.history[-8:]
                if h.get("role") == "user" and h.get("content")
            ]
            if recent_user_msgs:
                recent_context_lines.append("=== WAS LEON MIR GERADE ERZAEHLT HAT ===")
                for msg in recent_user_msgs[-4:]:
                    recent_context_lines.append(f"Leon sagte: \"{msg}\"")
                recent_context_lines.append("ICH MUSS darauf eingehen! Das ist WICHTIGER als meine Sinne!")
                recent_context_lines.append("")

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

        # Route to LLM (with conversation history for context)
        try:
            result = await router.chat(
                user_message=req.message,
                system_prompt=system_prompt,
                brain_state=snap,
                tools=tools.tool_definitions(),
                history=req.history,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "text": f"(Mein Gehirn hatte einen Schluckauf: {type(e).__name__})",
                "backend": "error",
                "tool_results": [],
            }

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
