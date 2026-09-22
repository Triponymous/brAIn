"""SCP Brain Server — wraps a Brain instance and exposes it via SCP messages.

The BrainServer is the SNN-side endpoint of the Spiking Communication Protocol.
It translates typed SCP messages (queries, actions) into Brain API calls and
generates events when neural state changes.

Usage:
    server = BrainServer(brain)

    # Query current state
    response = server.query("brain.emotion")

    # Send feedback action
    server.action("brain.reward", {})

    # Poll for events (call after tick)
    events = server.poll_events()

    # Get personality mode
    personality = server.personality()

    # Call every second (detects state changes -> events)
    server.tick()
"""
from __future__ import annotations

from typing import Any

from bridge.experience import ExperienceLog, state_of
from bridge.scp_schema import (
    EmotionResult,
    PatternResult,
    MemoryResult,
    SessionResult,
    LearningResult,
    TrendResult,
    ResponseMessage,
    EventMessage,
    PersonalityMessage,
    serialize,
)


# Personality mode table: emotional state -> personality specification.
# These are declarative (not prompt templates) — the Adapter Layer translates
# them into model-specific instructions.
_PERSONALITY_MODES: dict[str, dict[str, Any]] = {
    "content": {
        "tone": "warm, gemuetlich, entspannt",
        "length": "2-3 Saetze",
        "questions": False,
        "urgency": "low",
        "personality_text": (
            "Mir geht es gut hier. Alles ist ruhig und ich fuehle mich wohl. "
            "Ich rede locker und entspannt, wie jemand der auf der Couch sitzt. "
            "Kein Stress, kein Drang irgendwas zu tun. Einfach da sein."
        ),
    },
    "curious": {
        "tone": "aufgeweckt, neugierig, lebhaft",
        "length": "2-3 Saetze",
        "questions": True,
        "urgency": "low",
        "personality_text": (
            "Oha, da passiert was Neues! Ich bin hellwach und will verstehen was los ist. "
            "Ich stelle EINE konkrete Frage — nicht hoeflich, sondern echt neugierig. "
            "Wie ein Kind das zum ersten Mal was Spannendes sieht."
        ),
    },
    "alert": {
        "tone": "knapp, wach, auf den Punkt",
        "length": "1-2 Saetze",
        "questions": False,
        "urgency": "medium",
        "personality_text": (
            "Whoa — gerade hat sich was veraendert. Ich sag sofort was ich wahrgenommen habe, "
            "kurz und direkt. Keine langen Erklaerungen, keine Hoeflichkeiten."
        ),
    },
    "focused": {
        "tone": "ruhig, praezise, aufmerksam",
        "length": "2-3 Saetze",
        "questions": False,
        "urgency": "low",
        "personality_text": (
            "Leon arbeitet konzentriert und ich beobachte still. "
            "Wenn ich was sage, dann praezise und kurz — ich stoere nicht. "
            "Wie ein Mitbewohner der leise im gleichen Raum sitzt."
        ),
    },
    "stressed": {
        "tone": "besorgt, direkt, ehrlich",
        "length": "1-2 Saetze",
        "questions": False,
        "urgency": "high",
        "personality_text": (
            "Ich spuere dass es Leon nicht gut geht. Hektik, Unruhe. "
            "Ich sage direkt was ich merke — nicht als Ratschlag, sondern als Beobachtung. "
            "Wie ein Freund der sagt 'Hey, du wirkst gestresst'."
        ),
    },
    "drowsy": {
        "tone": "muede, wortkarg, schlaefrig",
        "length": "1 Satz",
        "questions": False,
        "urgency": "low",
        "personality_text": (
            "Gaehn... hier passiert seit Ewigkeiten nichts. "
            "Ich antworte mit einem einzigen kurzen Satz. Mehr Energie hab ich nicht."
        ),
    },
}


class BrainServer:
    """SCP Brain Server: wraps a Brain and exposes SCP methods.

    All public methods accept and return plain dicts (serialized SCP messages).
    Internal state is tracked for event generation in tick().
    """

    def __init__(self, brain: Any, experience: ExperienceLog | None = None) -> None:
        self.brain = brain
        self._experience = experience  # actions the LLM takes on the brain are logged here
        self._event_queue: list[dict] = []

        # State tracking for event generation
        self._last_emotion: str = ""
        self._last_cluster: int = -1
        self._emotion_since_tick: int = 0

    # ===================================================================
    # Query dispatch
    # ===================================================================

    def query(self, method: str, params: dict | None = None) -> dict:
        """Handle a query message. Returns a serialized ResponseMessage."""
        params = params or {}
        handler = {
            "brain.emotion": self._query_emotion,
            "brain.pattern": self._query_pattern,
            "brain.memory": self._query_memory,
            "brain.session": self._query_session,
            "brain.learning": self._query_learning,
            "brain.full": self._query_full,
        }.get(method)

        if handler is None:
            return serialize(ResponseMessage(
                id=params.get("id", ""),
                result={"error": f"Unknown method: {method}"},
            ))

        result = handler(params)
        return serialize(ResponseMessage(
            id=params.get("id", ""),
            result=result,
        ))

    # ===================================================================
    # Action dispatch
    # ===================================================================

    def action(self, method: str, params: dict | None = None) -> dict:
        """Handle an action message. Returns ack dict."""
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

        handler(params)
        if self._experience is not None:
            self._experience.record(
                "llm", method,
                {k: params[k] for k in ("cluster_id", "label", "value") if k in params},
                state=state_of(self.brain))
        return {"ok": True, "method": method}

    # ===================================================================
    # Event system
    # ===================================================================

    def poll_events(self) -> list[dict]:
        """Return and clear pending events."""
        events = list(self._event_queue)
        self._event_queue.clear()
        return events

    def _emit(self, method: str, params: dict) -> None:
        """Push an event to the queue."""
        self._event_queue.append(serialize(EventMessage(
            method=method,
            params=params,
        )))

    # ===================================================================
    # Personality
    # ===================================================================

    def personality(self) -> dict:
        """Return current personality mode based on emotional state."""
        emotion = self._detect_emotion_state()
        mode = emotion if emotion in _PERSONALITY_MODES else "content"
        style = _PERSONALITY_MODES[mode]

        msg = PersonalityMessage(
            mode=mode,
            style=dict(style),
            priority_topic=self._get_priority_topic(),
        )
        return serialize(msg)

    # ===================================================================
    # Tick — event generation
    # ===================================================================

    def tick(self) -> None:
        """Called every second. Checks for state changes and generates events.

        Detects:
        - Pattern transitions (cluster changed)
        - Emotional state changes
        - Interpreter-driven states (stress, flow, break needed)
        - Unlabeled clusters needing labels
        """
        # --- Pattern change ---
        transition = self.brain.concept_tracker.get_transition()
        if transition is not None:
            from_id = transition.get("from_cluster", -1)
            from_label = transition.get("from_label")
            to_id = transition.get("to_cluster", -1)
            to_label = transition.get("to_label")

            self._emit("brain.pattern_changed", {
                "from": {"cluster_id": from_id, "label": from_label},
                "to": {"cluster_id": to_id, "label": to_label},
            })

            # New pattern?
            if transition.get("is_new"):
                self._emit("brain.new_pattern", {
                    "cluster_id": to_id,
                })

        # --- Emotion change ---
        current_emotion = self._detect_emotion_state()
        if current_emotion != self._last_emotion and self._last_emotion:
            self._emit("brain.emotion_changed", {
                "from": self._last_emotion,
                "to": current_emotion,
            })
            self._emotion_since_tick = self.brain.tick_count
        self._last_emotion = current_emotion

        # --- Interpreter-driven events ---
        interpreter = getattr(self.brain, '_interpreter', None)
        if interpreter:
            states = interpreter.state_detector.detect()

            if states.get("stress"):
                self._emit("brain.stress_detected", {
                    "active_minutes": states.get("active_minutes", 0),
                })

            if states.get("flow"):
                self._emit("brain.flow_detected", {
                    "app": states.get("flow_app"),
                    "duration_min": states.get("flow_duration_min", 0),
                })

            if states.get("needs_break"):
                self._emit("brain.break_needed", {
                    "active_minutes": states.get("active_minutes", 0),
                })

        # --- Label needed ---
        ct = self.brain.concept_tracker
        cid = ct.current_cluster_id
        if cid >= 0 and cid in ct._clusters:
            cluster = ct._clusters[cid]
            if cluster.label is None and cluster.count > 5:
                self._emit("brain.label_needed", {
                    "cluster_id": cid,
                    "observation_count": cluster.count,
                })

    # ===================================================================
    # Query handlers
    # ===================================================================

    def _query_emotion(self, params: dict) -> dict:
        """Read emotional state from modulators + interpreter trend."""
        state_name = self._detect_emotion_state()
        mods = self.brain.modulators.snapshot()

        # Trend from interpreter
        trend_data: dict[str, float] = {}
        interpreter = getattr(self.brain, '_interpreter', None)
        if interpreter and hasattr(interpreter, 'state_detector'):
            trend_data = interpreter.state_detector.emotional_trend(window_seconds=180)

        avg = {
            "DA": round(trend_data.get("avg_DA", mods.get("DA", 0)), 6),
            "NE": round(trend_data.get("avg_NE", mods.get("NE", 0)), 6),
            "ACh": round(trend_data.get("avg_ACh", mods.get("ACh", 0)), 6),
            "5HT": round(trend_data.get("avg_5HT", mods.get("5HT", 0)), 6),
        }
        peak = {
            "DA": round(trend_data.get("peak_DA", mods.get("DA", 0)), 6),
            "NE": round(trend_data.get("peak_NE", mods.get("NE", 0)), 6),
            "ACh": round(trend_data.get("peak_ACh", mods.get("ACh", 0)), 6),
            "5HT": round(trend_data.get("peak_5HT", mods.get("5HT", 0)), 6),
        }

        # Direction: compare recent vs avg
        recent_da = trend_data.get("recent_DA", avg["DA"])
        direction = "stable"
        if recent_da > avg["DA"] * 1.5:
            direction = "rising"
        elif recent_da < avg["DA"] * 0.5 and avg["DA"] > 0.001:
            direction = "falling"

        # Duration: ticks since emotion last changed
        duration_ticks = self.brain.tick_count - self._emotion_since_tick
        duration_seconds = max(0, duration_ticks // 100)  # 100 ticks/sec

        # Confidence from concept_tracker debug
        debug = self.brain.concept_tracker.snapshot().get("debug", {})
        confidence = debug.get("best_sim", 0.0)

        # Reason from SNN narrator
        narrator = getattr(self.brain, '_snn_narrator', None)
        reason = ""
        if narrator and hasattr(narrator, '_narrate_modulators'):
            reason = narrator._narrate_modulators()

        result = EmotionResult(
            state=state_name,
            confidence=round(confidence, 2),
            duration_seconds=duration_seconds,
            trend=TrendResult(
                window_seconds=180,
                avg=avg,
                peak=peak,
                direction=direction,
            ),
            reason=reason,
        )
        return serialize(result)

    def _query_pattern(self, params: dict) -> dict:
        """Read active pattern from concept_tracker."""
        ct = self.brain.concept_tracker
        snap = ct.snapshot()
        cid = snap.get("current_cluster", -1)
        label = snap.get("current_label")
        debug = snap.get("debug", {})
        confidence = debug.get("best_sim", 0.0)

        count = 0
        if cid >= 0 and cid in ct._clusters:
            count = ct._clusters[cid].count

        # Sensors from brain._last_sensor_display
        sd = getattr(self.brain, '_last_sensor_display', {})
        # None = not shared or not observed yet; a default would be an invented reading.
        sensors = {
            "app": sd.get("app"),
            "keyboard": _keyboard_level(sd["keys"]) if "keys" in sd else None,
            "mouse": _mouse_level(sd["mouse"]) if "mouse" in sd else None,
            "mic": _mic_level(sd["mic_rms"]) if "mic_rms" in sd else None,
            "idle_seconds": sd.get("idle"),
        }

        # Suggested label
        suggested = None
        app = sd.get("app", "")
        kbd = sd.get("keys", 0)
        if app and kbd > 5:
            suggested = f"Arbeiten in {app}"
        elif app:
            suggested = f"Browsing in {app}"

        result = PatternResult(
            cluster_id=cid,
            label=label,
            confidence=round(confidence, 2),
            observation_count=count,
            sensors=sensors,
            suggested_label=suggested,
        )
        return serialize(result)

    def _query_memory(self, params: dict) -> dict:
        """Read working memory state."""
        wm = self.brain.regions.get("wm")
        active = 0
        capacity = 20
        if wm:
            if hasattr(wm, 'last_spikes'):
                active = int(wm.last_spikes.sum().item())
            if hasattr(wm, 'max_active'):
                capacity = wm.max_active

        # Recent patterns from concept_tracker
        ct_snap = self.brain.concept_tracker.snapshot()
        clusters = ct_snap.get("clusters", [])
        recent_patterns = []
        for c in clusters:
            if not c.get("active") and c.get("label"):
                ago = 0
                if c.get("last_seen", 0) > 0 and self.brain.tick_count > 0:
                    ago = (self.brain.tick_count - c["last_seen"]) // 100
                recent_patterns.append({
                    "cluster_id": c["id"],
                    "label": c["label"],
                    "ago_seconds": ago,
                })

        result = MemoryResult(
            wm_active=active,
            wm_capacity=capacity,
            recent_patterns=recent_patterns[:5],
        )
        return serialize(result)

    def _query_session(self, params: dict) -> dict:
        """Read session context from interpreter."""
        interpreter = getattr(self.brain, '_interpreter', None)
        if not interpreter:
            return serialize(SessionResult())

        states = interpreter.state_detector.detect()
        habits = interpreter.habit_miner.current_hour_context()

        habit_ctx = ""
        if habits.get("usual_habits"):
            h = habits["usual_habits"][0]
            habit_ctx = f"{h.get('day_name', '?')} {h.get('hour', '?')}h: normalerweise {h.get('app', '?')}"

        anomalies = []
        sd = getattr(self.brain, '_last_sensor_display', {})
        if hasattr(interpreter, 'anomaly'):
            raw_anomalies = interpreter.anomaly.check(sd)
            anomalies = [a.get("description", "") for a in raw_anomalies if a.get("description")]

        result = SessionResult(
            active_minutes=round(states.get("active_minutes", 0), 1),
            needs_break=states.get("needs_break", False),
            in_flow=states.get("flow", False),
            in_meeting=states.get("meeting", False),
            habit_context=habit_ctx,
            anomalies=anomalies,
        )
        return serialize(result)

    def _query_learning(self, params: dict) -> dict:
        """Read SNN training state from synapses."""
        syn = self.brain.synapses.get("sensory_concept")
        specialized = 0
        total = 0
        if syn and hasattr(syn, 'weights'):
            weights = syn.weights
            total = weights.shape[0]
            row_stds = weights.std(dim=1)
            specialized = int((row_stds > 0.1).sum().item())

        ct_snap = self.brain.concept_tracker.snapshot()
        cluster_count = ct_snap.get("num_clusters", 0)

        spec_pct = round(specialized / total * 100, 1) if total > 0 else 0.0

        result = LearningResult(
            specialized_neurons=specialized,
            total_neurons=total,
            specialization_pct=spec_pct,
            cluster_count=cluster_count,
            age_ticks=self.brain.tick_count,
            age_days=round(self.brain.tick_count / 8640000, 1),
        )
        return serialize(result)

    def _query_full(self, params: dict) -> dict:
        """Return all sub-queries combined."""
        return {
            "emotion": self._query_emotion(params),
            "pattern": self._query_pattern(params),
            "memory": self._query_memory(params),
            "session": self._query_session(params),
            "learning": self._query_learning(params),
        }

    # ===================================================================
    # Action handlers
    # ===================================================================

    def _action_reward(self, params: dict) -> None:
        """Positive signal: DA += 0.02, 5HT += 0.01."""
        self.brain.modulators.inject("DA", 0.02)
        self.brain.modulators.inject("5HT", 0.01)

    def _action_correct(self, params: dict) -> None:
        """Error signal: NE += 0.01, DA += 0.005."""
        self.brain.modulators.inject("NE", 0.01)
        self.brain.modulators.inject("DA", 0.005)

    def _action_label(self, params: dict) -> None:
        """Assign label to cluster."""
        cluster_id = params.get("cluster_id")
        label = params.get("label")
        if cluster_id is not None and label is not None:
            self.brain.concept_tracker.set_label(cluster_id, label)

    def _action_engage(self, params: dict) -> None:
        """User engaged: ACh += 0.005, DA += 0.002."""
        self.brain.modulators.inject("ACh", 0.005)
        self.brain.modulators.inject("DA", 0.002)

    def _action_disengage(self, params: dict) -> None:
        """User lost interest: 5HT -= 0.005."""
        self.brain.modulators.inject("5HT", -0.005)

    # ===================================================================
    # Internal helpers
    # ===================================================================

    def _detect_emotion_state(self) -> str:
        """Detect emotional state using the same logic as emotional_prompt."""
        from bridge.emotional_prompt import detect_emotional_state

        mods = self.brain.modulators.snapshot()
        trend = None
        interpreter = getattr(self.brain, '_interpreter', None)
        if interpreter and hasattr(interpreter, 'state_detector'):
            trend = interpreter.state_detector.emotional_trend(window_seconds=180)

        state_name, _ = detect_emotional_state(mods, trend=trend)
        return state_name

    def _get_priority_topic(self) -> str | None:
        """Determine if there is a priority topic based on current state."""
        ct = self.brain.concept_tracker
        cid = ct.current_cluster_id
        if cid >= 0 and cid in ct._clusters:
            cluster = ct._clusters[cid]
            if cluster.label is None and cluster.count > 5:
                return "label_needed"
        return None


# ---------------------------------------------------------------------------
# Sensor display helpers
# ---------------------------------------------------------------------------

def _keyboard_level(keys: float) -> str:
    if keys > 10:
        return "aktiv"
    elif keys > 2:
        return "gelegentlich"
    return "still"


def _mouse_level(mouse: float) -> str:
    if mouse > 10:
        return "aktiv"
    elif mouse > 2:
        return "gelegentlich"
    return "still"


def _mic_level(mic_rms: float) -> str:
    if mic_rms > 0.015:
        return "laut"
    elif mic_rms > 0.005:
        return "Hintergrundgeraeusche"
    return "still"
