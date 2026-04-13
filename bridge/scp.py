"""SNN Communication Protocol — Compact State renderer.

Produces a 6-line structured representation of the entire brain state.
This is the SINGLE interface between SNN and LLM. No prose, no numbers,
just structured facts that any LLM can parse.

Format:
    EMOTION: <state> (<duration>) | Grund: <why>
    PATTERN: <id> <label> (<confidence>, <count>x) | Sensoren: <what>
    CHANGE: <type> <detail> vor <time>
    MEMORY: <active>/<max> WM-Slots | denke an: <recent>
    SESSION: <duration> aktiv | Gewohnheit: <habit_context>
    BRAIN: <specialized>/<total> Neuronen | <clusters> Cluster | Alter: <age>
"""
from __future__ import annotations
from typing import Any

from brain.core import Brain


class CompactState:
    """Renders the brain's complete state in 6 structured lines."""

    def __init__(self, brain: Brain) -> None:
        self.brain = brain

    def render(self) -> str:
        """Produce the compact state string."""
        lines = [
            self._emotion_line(),
            self._pattern_line(),
            self._change_line(),
            self._memory_line(),
            self._session_line(),
            self._brain_line(),
        ]
        return "\n".join(lines)

    def _emotion_line(self) -> str:
        interpreter = getattr(self.brain, '_interpreter', None)
        trend = None
        if interpreter and hasattr(interpreter, 'state_detector'):
            trend = interpreter.state_detector.emotional_trend(window_seconds=180)

        from bridge.emotional_prompt import detect_emotional_state
        mods = self.brain.modulators.snapshot()
        state_name, _ = detect_emotional_state(mods, trend=trend)

        # Duration: how long has the emotional state been stable?
        # Approximate from modulator trend stability
        duration = "kurz"
        if trend:
            avg_ne = trend.get("avg_NE", 0)
            peak_ne = trend.get("peak_NE", 0)
            if peak_ne < 0.02 and avg_ne < 0.01:
                duration = "seit Minuten stabil"
            elif peak_ne > avg_ne * 3:
                duration = "gerade erst gewechselt"

        # Reason from SNN narrator
        narrator = getattr(self.brain, '_snn_narrator', None)
        reason = ""
        if narrator:
            reason = narrator._narrate_modulators()

        return f"EMOTION: {state_name} ({duration}) | Grund: {reason}"

    def _pattern_line(self) -> str:
        ct = self.brain.concept_tracker
        snap = ct.snapshot()
        current = snap.get("current_cluster", -1)
        label = snap.get("current_label")
        debug = snap.get("debug", {})
        confidence = debug.get("best_sim", 0)

        sd = getattr(self.brain, '_last_sensor_display', {})
        sensors = []
        if sd.get("keys", 0) > 10:
            sensors.append("viel Tastatur")
        elif sd.get("keys", 0) > 2:
            sensors.append("Tastatur aktiv")
        if sd.get("mouse", 0) > 10:
            sensors.append("Maus aktiv")
        app = sd.get("app", "")
        if app:
            sensors.append(f"{app} offen")
        if sd.get("mic_rms", 0) > 0.015:
            sensors.append("laut")
        elif sd.get("mic_rms", 0) > 0.005:
            sensors.append("Hintergrundgeraeusche")
        else:
            sensors.append("leise")

        if current < 0:
            return f"PATTERN: kein Muster erkannt | Sensoren: {', '.join(sensors)}"

        cluster = ct._clusters.get(current)
        count = cluster.count if cluster else 0
        label_str = f"'{label}'" if label else "unlabeled"
        conf_pct = f"{confidence:.0%}" if confidence > 0 else "?"

        return f"PATTERN: #{current} {label_str} ({conf_pct}, {count}x) | Sensoren: {', '.join(sensors)}"

    def _change_line(self) -> str:
        transition = self.brain.concept_tracker.get_transition()
        if transition:
            from_label = transition.get("from_label") or f"#{transition.get('from_cluster', '?')}"
            to_label = transition.get("to_label") or f"#{transition.get('to_cluster', '?')}"
            return f"CHANGE: Musterwechsel {from_label} -> {to_label}"

        interpreter = getattr(self.brain, '_interpreter', None)
        if interpreter:
            states = interpreter.state_detector.detect()
            if states.get("stress"):
                return "CHANGE: Stress erkannt -- hektisches Verhalten"
            if states.get("meeting"):
                return f"CHANGE: Meeting seit {states['meeting_duration_min']}min"

        return "CHANGE: keine Aenderung"

    def _memory_line(self) -> str:
        wm = self.brain.regions.get("wm")
        active = 0
        if wm and hasattr(wm, 'last_spikes'):
            active = int(wm.last_spikes.sum().item())

        max_active = 20
        if wm and hasattr(wm, 'max_active'):
            max_active = wm.max_active

        # What's in recent memory?
        recent = ""
        ct = self.brain.concept_tracker.snapshot()
        clusters = ct.get("clusters", [])
        recent_clusters = [c for c in clusters if c.get("label") and not c.get("active")]
        if recent_clusters:
            recent = recent_clusters[0].get("label", "?")

        return f"MEMORY: {active}/{max_active} WM-Slots | denke an: {recent or 'nichts Bestimmtes'}"

    def _session_line(self) -> str:
        interpreter = getattr(self.brain, '_interpreter', None)
        active_min = 0
        habit_ctx = ""

        if interpreter:
            states = interpreter.state_detector.detect()
            active_min = states.get("active_minutes", 0)

            habits = interpreter.habit_miner.current_hour_context()
            if habits.get("usual_habits"):
                h = habits["usual_habits"][0]
                habit_ctx = f"{h.get('day_name', '?')} {h.get('hour', '?')}h normalerweise {h.get('app', '?')}"

        needs_break = ""
        if interpreter:
            states = interpreter.state_detector.detect()
            if states.get("needs_break"):
                needs_break = " | Pause empfohlen!"
            elif states.get("flow"):
                needs_break = f" | im Flow seit {states['flow_duration_min']}min"

        return f"SESSION: {active_min:.0f}min aktiv{needs_break} | Gewohnheit: {habit_ctx or 'noch keine gelernt'}"

    def _brain_line(self) -> str:
        syn = self.brain.synapses.get("sensory_concept")
        specialized = 0
        total = 0
        if syn:
            weights = syn.weights
            total = weights.shape[0]
            row_stds = weights.std(dim=1)
            specialized = int((row_stds > 0.1).sum().item())

        num_clusters = self.brain.concept_tracker.snapshot().get("num_clusters", 0)

        personality = getattr(self.brain, '_interpreter', None)
        age_days = 0
        if personality and hasattr(personality, 'personality'):
            age_days = personality.personality.snapshot().get("age_days", 0)

        return f"BRAIN: {specialized}/{total} Neuronen spezialisiert | {num_clusters} Cluster gelernt | Alter: {age_days:.1f} Tage"
