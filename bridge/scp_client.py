"""SCP LLM Client -- consumes Brain Server state and produces LLM-ready prompts.

The SCPClient is the LLM-facing half of the Spiking Communication Protocol.
It queries the BrainServer for neural state, formats it through model-specific
adapters, and provides a feedback channel for the LLM to affect SNN dynamics.

Architecture:
    BrainServer (wraps Brain + Interpreter)
         |
    SCPClient (queries server, dispatches adapters)
         |
    Model-specific Adapter (Qwen, Gemma, Claude, Generic)
         |
    System prompt string for the LLM

The adapters consume structured SCP query results (not raw strings) and
produce prompt text tailored to each model family's strengths.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from brain.core import Brain
from bridge.scp import CompactState
from bridge.emotional_prompt import detect_emotional_state
from bridge.model_adapter import _UNIVERSAL_RULES


# ═══════════════════════════════════════════════════════════════════
# Brain Server -- wraps SNN runtime, exposes SCP query/action API
# ═══════════════════════════════════════════════════════════════════


class BrainServer:
    """SCP Brain Server -- wraps a Brain instance and exposes typed queries.

    Implements the server-side of SCP: Query responses, Event emission,
    Action handling, and Personality mode computation.
    """

    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self._compact = CompactState(brain)
        self._events: list[dict[str, Any]] = []

    # ── Queries (Client -> Server) ────────────────────────────────

    def query(self, method: str, params: dict | None = None) -> dict[str, Any]:
        """Handle an SCP query and return a response.

        Args:
            method: SCP method name (brain.emotion, brain.pattern, etc.)
            params: Optional query parameters.

        Returns:
            SCP response dict with 'type', 'id', and 'result' fields.
        """
        request_id = str(uuid.uuid4())[:8]
        handler = {
            "brain.emotion": self._query_emotion,
            "brain.pattern": self._query_pattern,
            "brain.memory": self._query_memory,
            "brain.session": self._query_session,
            "brain.learning": self._query_learning,
            "brain.full": self._query_full,
        }.get(method)

        if handler is None:
            return {
                "type": "response",
                "id": request_id,
                "error": f"Unknown method: {method}",
            }

        return {
            "type": "response",
            "id": request_id,
            "result": handler(params or {}),
        }

    def _query_emotion(self, params: dict) -> dict[str, Any]:
        mods = self.brain.modulators.snapshot()

        interpreter = getattr(self.brain, "_interpreter", None)
        trend = None
        if interpreter and hasattr(interpreter, "state_detector"):
            trend = interpreter.state_detector.emotional_trend(window_seconds=180)

        state_name, state_config = detect_emotional_state(mods, trend=trend)

        # Duration estimate from trend stability
        duration_seconds = 0
        direction = "stable"
        if trend:
            avg_ne = trend.get("avg_NE", 0)
            peak_ne = trend.get("peak_NE", 0)
            if peak_ne < 0.02 and avg_ne < 0.01:
                duration_seconds = 180
                direction = "stable"
            elif peak_ne > avg_ne * 3:
                duration_seconds = 10
                direction = "rising" if trend.get("recent_NE", 0) > avg_ne else "falling"

        # Reason from SNN narrator
        narrator = getattr(self.brain, "_snn_narrator", None)
        reason = ""
        if narrator:
            reason = narrator._narrate_modulators()

        trend_result = {}
        if trend:
            trend_result = {
                "window_seconds": 180,
                "avg": {
                    "DA": round(trend.get("avg_DA", 0), 4),
                    "NE": round(trend.get("avg_NE", 0), 4),
                    "ACh": round(trend.get("avg_ACh", 0), 4),
                    "5HT": round(trend.get("avg_5HT", 0), 4),
                },
                "peak": {
                    "DA": round(trend.get("peak_DA", 0), 4),
                    "NE": round(trend.get("peak_NE", 0), 4),
                    "ACh": round(trend.get("peak_ACh", 0), 4),
                    "5HT": round(trend.get("peak_5HT", 0), 4),
                },
                "direction": direction,
            }

        return {
            "state": state_name,
            "confidence": round(state_config.get("priority", 2) / 6.0, 2),
            "duration_seconds": duration_seconds,
            "trend": trend_result,
            "reason": reason,
        }

    def _query_pattern(self, params: dict) -> dict[str, Any]:
        ct = self.brain.concept_tracker
        snap = ct.snapshot()
        current = snap.get("current_cluster", -1)
        label = snap.get("current_label")
        debug = snap.get("debug", {})
        confidence = debug.get("best_sim", 0)

        sd = getattr(self.brain, "_last_sensor_display", {})

        cluster = ct._clusters.get(current)
        count = cluster.count if cluster else 0

        # Build suggested label from sensors
        suggested = self._suggest_label(sd) if current >= 0 and not label else None

        sensors = {
            "app": sd.get("app", ""),
            "keyboard": "still",
            "mouse": "gelegentlich",
            "mic": "still",
            "idle_seconds": int(sd.get("idle", 0)),
        }
        keys = sd.get("keys", 0)
        if keys > 10:
            sensors["keyboard"] = "viel"
        elif keys > 2:
            sensors["keyboard"] = "aktiv"
        mouse = sd.get("mouse", 0)
        if mouse > 10:
            sensors["mouse"] = "aktiv"
        mic = sd.get("mic_rms", 0)
        if mic > 0.015:
            sensors["mic"] = "laut"
        elif mic > 0.005:
            sensors["mic"] = "Hintergrundgeraeusche"

        return {
            "cluster_id": current,
            "label": label,
            "confidence": round(confidence, 2),
            "observation_count": count,
            "sensors": sensors,
            "suggested_label": suggested,
        }

    def _query_memory(self, params: dict) -> dict[str, Any]:
        wm = self.brain.regions.get("wm")
        active = 0
        capacity = 20
        if wm and hasattr(wm, "last_spikes"):
            active = int(wm.last_spikes.sum().item())
        if wm and hasattr(wm, "max_active"):
            capacity = wm.max_active

        ct = self.brain.concept_tracker.snapshot()
        clusters = ct.get("clusters", [])
        recent = []
        for c in clusters:
            if c.get("label") and not c.get("active"):
                recent.append({
                    "cluster_id": c["id"],
                    "label": c["label"],
                    "ago_seconds": 0,
                })
        return {
            "wm_active": active,
            "wm_capacity": capacity,
            "recent_patterns": recent[:5],
        }

    def _query_session(self, params: dict) -> dict[str, Any]:
        interpreter = getattr(self.brain, "_interpreter", None)
        active_minutes = 0.0
        needs_break = False
        in_flow = False
        in_meeting = False
        habit_context = ""
        anomalies: list[str] = []

        if interpreter:
            states = interpreter.state_detector.detect()
            active_minutes = states.get("active_minutes", 0)
            needs_break = states.get("needs_break", False)
            in_flow = states.get("flow", False)
            in_meeting = states.get("meeting", False)

            habits = interpreter.habit_miner.current_hour_context()
            if habits.get("usual_habits"):
                h = habits["usual_habits"][0]
                habit_context = (
                    f"{h.get('day_name', '?')} {h.get('hour', '?')}h: "
                    f"normalerweise {h.get('app', '?')}"
                )

            sd = getattr(self.brain, "_last_sensor_display", {})
            anomaly_list = interpreter.anomaly.check(sd)
            anomalies = [a["description"] for a in anomaly_list[:3]]

        return {
            "active_minutes": round(active_minutes, 1),
            "needs_break": needs_break,
            "in_flow": in_flow,
            "in_meeting": in_meeting,
            "habit_context": habit_context,
            "anomalies": anomalies,
        }

    def _query_learning(self, params: dict) -> dict[str, Any]:
        syn = self.brain.synapses.get("sensory_concept")
        specialized = 0
        total = 0
        if syn:
            weights = syn.weights
            total = weights.shape[0]
            row_stds = weights.std(dim=1)
            specialized = int((row_stds > 0.1).sum().item())

        num_clusters = self.brain.concept_tracker.snapshot().get("num_clusters", 0)

        age_days = 0.0
        interpreter = getattr(self.brain, "_interpreter", None)
        if interpreter and hasattr(interpreter, "personality"):
            age_days = interpreter.personality.snapshot().get("age_days", 0)

        return {
            "specialized_neurons": specialized,
            "total_neurons": total,
            "specialization_pct": round(specialized / total * 100, 1) if total > 0 else 0.0,
            "cluster_count": num_clusters,
            "age_ticks": self.brain.tick_count,
            "age_days": round(age_days, 1),
        }

    def _query_full(self, params: dict) -> dict[str, Any]:
        return {
            "emotion": self._query_emotion(params),
            "pattern": self._query_pattern(params),
            "memory": self._query_memory(params),
            "session": self._query_session(params),
            "learning": self._query_learning(params),
        }

    # ── Personality (Server -> Client) ────────────────────────────

    def personality(self) -> dict[str, Any]:
        """Compute the current personality mode from brain state.

        Returns a declarative personality specification that the Adapter
        Layer translates into model-specific instructions.
        """
        mods = self.brain.modulators.snapshot()
        interpreter = getattr(self.brain, "_interpreter", None)
        trend = None
        if interpreter and hasattr(interpreter, "state_detector"):
            trend = interpreter.state_detector.emotional_trend(window_seconds=180)

        state_name, state_config = detect_emotional_state(mods, trend=trend)

        # Build style dict from emotional state
        style = {
            "tone": "warm, reflektiv",
            "length": "2-4 Saetze",
            "questions": False,
            "urgency": "low",
        }

        if state_name == "curious":
            style = {"tone": "lebhaft, fragend", "length": "2-3 Saetze", "questions": True, "urgency": "medium"}
        elif state_name == "alert":
            style = {"tone": "kurz, direkt", "length": "1-2 Saetze", "questions": False, "urgency": "high"}
        elif state_name == "stressed":
            style = {"tone": "besorgt, ruhig", "length": "1-2 Saetze", "questions": False, "urgency": "high"}
        elif state_name == "focused":
            style = {"tone": "praezise, aufmerksam", "length": "2-3 Saetze", "questions": False, "urgency": "low"}
        elif state_name == "drowsy":
            style = {"tone": "muede, minimal", "length": "1 Satz", "questions": False, "urgency": "low"}

        return {
            "type": "personality",
            "params": {
                "mode": state_name,
                "personality_text": state_config.get("personality", ""),
                "style": style,
                "constraints": [
                    "keine Emojis",
                    "keine Hilfe anbieten",
                    "keine erfundenen Faehigkeiten",
                    "nicht den Sinnen widersprechen",
                ],
                "priority_topic": None,
            },
        }

    # ── Actions (Client -> Server) ────────────────────────────────

    def action(self, method: str, params: dict | None = None) -> dict[str, Any]:
        """Handle an SCP action that affects SNN dynamics.

        Args:
            method: SCP action method (brain.reward, brain.label, etc.)
            params: Action parameters.

        Returns:
            Dict with 'ok' boolean and optional result data.
        """
        params = params or {}
        handler = {
            "brain.reward": self._action_reward,
            "brain.correct": self._action_correct,
            "brain.label": self._action_label,
            "brain.engage": self._action_engage,
            "brain.disengage": self._action_disengage,
        }.get(method)

        if handler is None:
            return {"ok": False, "error": f"Unknown action: {method}"}

        return handler(params)

    def _action_reward(self, params: dict) -> dict[str, Any]:
        self.brain.modulators.inject("DA", 0.02)
        self.brain.modulators.inject("5HT", 0.01)
        return {"ok": True, "injected": {"DA": 0.02, "5HT": 0.01}}

    def _action_correct(self, params: dict) -> dict[str, Any]:
        self.brain.modulators.inject("NE", 0.01)
        self.brain.modulators.inject("DA", 0.005)
        return {"ok": True, "injected": {"NE": 0.01, "DA": 0.005}}

    def _action_label(self, params: dict) -> dict[str, Any]:
        cluster_id = params.get("cluster_id")
        label = params.get("label")
        if cluster_id is None or label is None:
            return {"ok": False, "error": "cluster_id and label required"}
        self.brain.concept_tracker.set_label(cluster_id, label)
        return {"ok": True, "cluster_id": cluster_id, "label": label}

    def _action_engage(self, params: dict) -> dict[str, Any]:
        self.brain.modulators.inject("ACh", 0.005)
        self.brain.modulators.inject("DA", 0.002)
        return {"ok": True, "injected": {"ACh": 0.005, "DA": 0.002}}

    def _action_disengage(self, params: dict) -> dict[str, Any]:
        self.brain.modulators.inject("5HT", -0.005)
        return {"ok": True, "injected": {"5HT": -0.005}}

    # ── Events (Server -> Client) ─────────────────────────────────

    def push_event(self, method: str, params: dict[str, Any]) -> None:
        """Record an event for the client to poll."""
        self._events.append({
            "type": "event",
            "method": method,
            "params": params,
            "timestamp": time.time(),
        })
        # Keep bounded
        if len(self._events) > 100:
            self._events = self._events[-50:]

    def poll_events(self) -> list[dict[str, Any]]:
        """Return and clear pending events."""
        events = list(self._events)
        self._events.clear()
        return events

    # ── Helpers ────────────────────────────────────────────────────

    def _suggest_label(self, sd: dict) -> str | None:
        """Generate a label suggestion from sensor data."""
        app = sd.get("app", "").strip()
        keys = sd.get("keys", 0)
        mic = sd.get("mic_rms", 0)
        idle = sd.get("idle", 0)

        app_lower = app.lower() if app else ""
        parts: list[str] = []

        if "code" in app_lower or "claude" in app_lower or "terminal" in app_lower:
            parts.append("Coding")
        elif "chrome" in app_lower or "safari" in app_lower or "firefox" in app_lower:
            parts.append("Browsing")
        elif "zoom" in app_lower or "teams" in app_lower or "meet" in app_lower:
            parts.append("Meeting")
        elif app:
            parts.append(f"In {app}")

        if keys > 10:
            parts.append("mit viel Tippen")
        elif idle > 60:
            parts.append("Pause")

        if mic > 0.015 and "Meeting" not in parts:
            parts.append("mit Geraueschen")

        if not parts:
            return None

        return " ".join(parts[:2]) if len(parts) <= 2 else f"{parts[0]} {parts[1]}"


# ═══════════════════════════════════════════════════════════════════
# Model-specific Adapters -- translate SCP data into prompt strings
# ═══════════════════════════════════════════════════════════════════


class _BaseAdapter:
    """Base class for model-specific prompt adapters."""

    def render(
        self,
        full_state: dict[str, Any],
        personality: dict[str, Any],
        context: str = "",
    ) -> str:
        raise NotImplementedError


class QwenAdapter(_BaseAdapter):
    """Qwen 2.5: direct structured headers, explicit instructions."""

    def render(
        self,
        full_state: dict[str, Any],
        personality: dict[str, Any],
        context: str = "",
    ) -> str:
        params = personality.get("params", personality)
        emotion = full_state.get("emotion", {})
        pattern = full_state.get("pattern", {})
        session = full_state.get("session", {})
        learning = full_state.get("learning", {})

        parts = [
            _UNIVERSAL_RULES,
            "",
            f"Dein emotionaler Zustand:\n{params.get('personality_text', '')}",
            f"Antwortstil: {params['style']['length']}, {params['style']['tone']}",
            "",
            "Deine aktuelle Wahrnehmung:",
            self._format_perception(emotion, pattern, session, learning),
        ]

        if context:
            parts.extend(["", f"Konversation:\n{context}"])

        parts.extend([
            "",
            f"Antworte in {params['style']['length']}. Beziehe dich auf deine Wahrnehmung.",
            "Wenn Leon dir etwas erzaehlt, geh darauf ein -- das ist wichtiger als deine Sinne.",
            "WICHTIG: Antworte NUR auf Deutsch oder Englisch. Keine andere Sprache.",
            "WICHTIG: Du kannst NICHTS tun ausser reden. Keine Programme oeffnen, keine Aktionen.",
            "NIEMALS fragen: 'brauchst du etwas?', 'soll ich?', 'kann ich dir?', 'moechtest du?'",
            "NIEMALS Hilfe anbieten. Du bist ein Mitbewohner, kein Assistent.",
        ])
        return "\n".join(parts)

    def _format_perception(
        self,
        emotion: dict,
        pattern: dict,
        session: dict,
        learning: dict,
    ) -> str:
        lines = []
        state = emotion.get("state", "content")
        lines.append(f"  Stimmung: {state}")

        sensors = pattern.get("sensors", {})
        label = pattern.get("label") or "unbekannt"
        conf = pattern.get("confidence", 0)
        lines.append(f"  Muster: {label} (Sicherheit: {conf:.0%})")
        lines.append(f"  App: {sensors.get('app', '?')}, Tastatur: {sensors.get('keyboard', '?')}, Maus: {sensors.get('mouse', '?')}")

        active_min = session.get("active_minutes", 0)
        lines.append(f"  Session: {active_min:.0f}min aktiv")

        if session.get("in_flow"):
            lines.append("  Status: im Flow")
        if session.get("needs_break"):
            lines.append("  Status: Pause empfohlen!")

        habit = session.get("habit_context", "")
        if habit:
            lines.append(f"  Gewohnheit: {habit}")

        return "\n".join(lines)


class GemmaAdapter(_BaseAdapter):
    """Gemma: example-based, softer tone."""

    def render(
        self,
        full_state: dict[str, Any],
        personality: dict[str, Any],
        context: str = "",
    ) -> str:
        params = personality.get("params", personality)
        emotion = full_state.get("emotion", {})
        pattern = full_state.get("pattern", {})
        session = full_state.get("session", {})

        sensors = pattern.get("sensors", {})
        app = sensors.get("app", "")
        keyboard = sensors.get("keyboard", "still")
        label = pattern.get("label") or "unbekannt"

        parts = [
            _UNIVERSAL_RULES,
            "",
            f"So fuehlst du dich: {params.get('personality_text', '')}",
            "",
            "Das nimmst du wahr:",
            f"  Stimmung: {emotion.get('state', 'content')}",
            f"  Muster: {label}",
            f"  App: {app}, Tastatur: {keyboard}",
            f"  Session: {session.get('active_minutes', 0):.0f}min aktiv",
            "",
            "Beispiel gute Antwort: 'Hier ist es gerade ruhig -- du bist in Claude und tippst wenig. Sieht nach einer Denkpause aus.'",
            "Beispiel schlechte Antwort: 'Ich erkenne Muster #5! DA=0.006!'",
        ]

        if context:
            parts.extend(["", f"Leon hat gesagt:\n{context}"])
        return "\n".join(parts)


class ClaudeAdapter(_BaseAdapter):
    """Claude: XML tags, constraint blocks."""

    def render(
        self,
        full_state: dict[str, Any],
        personality: dict[str, Any],
        context: str = "",
    ) -> str:
        params = personality.get("params", personality)
        emotion = full_state.get("emotion", {})
        pattern = full_state.get("pattern", {})
        session = full_state.get("session", {})
        learning = full_state.get("learning", {})

        parts = [
            _UNIVERSAL_RULES,
            "",
            f"<emotional_state>\n{params.get('personality_text', '')}\n</emotional_state>",
            "",
            "<brain_state>",
            f"Stimmung: {emotion.get('state', 'content')} ({emotion.get('reason', '')})",
        ]

        sensors = pattern.get("sensors", {})
        label = pattern.get("label") or "unbekannt"
        conf = pattern.get("confidence", 0)
        parts.append(f"Muster: {label} (Sicherheit: {conf:.0%})")
        parts.append(f"Sensoren: App={sensors.get('app', '?')}, Tastatur={sensors.get('keyboard', '?')}, Maus={sensors.get('mouse', '?')}")
        parts.append(f"Session: {session.get('active_minutes', 0):.0f}min aktiv")

        habit = session.get("habit_context", "")
        if habit:
            parts.append(f"Gewohnheit: {habit}")
        parts.append(f"Gehirn: {learning.get('specialized_neurons', 0)}/{learning.get('total_neurons', 0)} spezialisiert, {learning.get('cluster_count', 0)} Cluster")
        parts.append("</brain_state>")

        if context:
            parts.extend(["", f"<conversation>\n{context}\n</conversation>"])

        constraints = params.get("constraints", [])
        constraints_str = ", ".join(constraints) if constraints else "keine Emojis"
        parts.extend([
            "",
            f"Constraints: {params['style']['length']}. {constraints_str}. Never mention raw numbers or cluster IDs.",
        ])
        return "\n".join(parts)


class GenericAdapter(_BaseAdapter):
    """Generic: minimal, works with any model."""

    def render(
        self,
        full_state: dict[str, Any],
        personality: dict[str, Any],
        context: str = "",
    ) -> str:
        params = personality.get("params", personality)
        emotion = full_state.get("emotion", {})
        pattern = full_state.get("pattern", {})
        session = full_state.get("session", {})

        sensors = pattern.get("sensors", {})

        parts = [
            _UNIVERSAL_RULES,
            "",
            params.get("personality_text", ""),
            "",
            f"Stimmung: {emotion.get('state', 'content')}",
            f"Muster: {pattern.get('label') or 'unbekannt'}",
            f"App: {sensors.get('app', '?')}, Tastatur: {sensors.get('keyboard', '?')}",
            f"Session: {session.get('active_minutes', 0):.0f}min aktiv",
        ]

        if context:
            parts.extend(["", context])

        parts.extend(["", f"{params['style']['length']}. Natuerlich und lebendig."])
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════
# SCP Client -- LLM-facing interface
# ═══════════════════════════════════════════════════════════════════


class SCPClient:
    """LLM Client for the Spiking Communication Protocol.

    Consumes Brain Server state and produces LLM-ready system prompts.
    Provides feedback channel for the LLM to affect SNN dynamics.
    """

    def __init__(
        self,
        server: BrainServer,
        model_type: str = "generic",
    ) -> None:
        self.server = server
        self.model_type = model_type
        self._adapters: dict[str, _BaseAdapter] = {
            "qwen": QwenAdapter(),
            "gemma": GemmaAdapter(),
            "claude": ClaudeAdapter(),
            "generic": GenericAdapter(),
        }

    def build_prompt(
        self,
        user_message: str = "",
        history: list[dict] | None = None,
    ) -> str:
        """Build a complete system prompt from current brain state.

        Steps:
        1. Query brain.full from server for complete neural state.
        2. Get personality mode from server.
        3. Format recent conversation context from history.
        4. Pass everything through the model-specific adapter.

        Args:
            user_message: Current user message (for context).
            history: Recent conversation history as list of
                     {'role': 'user'|'assistant', 'content': str} dicts.

        Returns:
            Complete system prompt string tailored to the model type.
        """
        # 1. Query full brain state
        response = self.server.query("brain.full")
        full_state = response.get("result", {})

        # 2. Get personality mode
        personality = self.server.personality()

        # 3. Format conversation context
        context = self._format_context(user_message, history)

        # 4. Dispatch to model-specific adapter
        adapter = self._adapters.get(self.model_type, self._adapters["generic"])
        return adapter.render(full_state, personality, context)

    def send_feedback(self, feedback_type: str, **kwargs: Any) -> dict[str, Any]:
        """Send an action to the Brain Server.

        Args:
            feedback_type: One of 'reward', 'correct', 'label', 'engage', 'disengage'.
            **kwargs: Additional parameters (e.g., cluster_id, label for 'label' type).

        Returns:
            Action result dict with 'ok' boolean.
        """
        method = f"brain.{feedback_type}"
        return self.server.action(method, kwargs)

    def get_events(self) -> list[dict[str, Any]]:
        """Poll events from Brain Server (for proactive notifications).

        Returns:
            List of SCP event dicts, each with 'type', 'method', and 'params'.
        """
        return self.server.poll_events()

    @staticmethod
    def detect_model_type(model_name: str) -> str:
        """Map a model name to an adapter type.

        Args:
            model_name: Model identifier (e.g., 'qwen2.5:14b', 'gemma2:27b').

        Returns:
            Adapter type string: 'qwen', 'gemma', 'claude', or 'generic'.
        """
        name = model_name.lower()
        if "qwen" in name:
            return "qwen"
        elif "gemma" in name:
            return "gemma"
        elif "claude" in name or "haiku" in name or "sonnet" in name:
            return "claude"
        elif "llama" in name:
            return "qwen"
        return "generic"

    def _format_context(
        self,
        user_message: str,
        history: list[dict] | None,
    ) -> str:
        """Format recent conversation into a context string.

        Takes the last few messages from history plus the current user
        message and produces a condensed conversation summary.
        """
        parts: list[str] = []

        if history:
            # Include last 3 exchanges for context continuity
            recent = history[-6:]
            for msg in recent:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    parts.append(f"Leon: {content}")
                else:
                    parts.append(f"Du: {content}")

        if user_message:
            parts.append(f"Leon: {user_message}")

        return "\n".join(parts)
