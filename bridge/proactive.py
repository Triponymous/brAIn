"""Proactive notifications — the pet speaks up when something interesting happens.

The timing of notifications is MODULATOR-DRIVEN:
- High NE (arousal/stress): pet reacts faster, shorter intervals
- High 5HT (contentment): pet is patient, longer intervals
- Low DA: pet doesn't initiate
- High DA + high ACh: pet actively asks questions

This is the core of the pet's "personality" — not WHAT it says
(that's the LLM) but WHEN it says it.
"""
from __future__ import annotations
import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from brain.core import Brain
from bridge.exporter import BrainStateExporter
from server.ws import WSPusher

if TYPE_CHECKING:
    from bridge.llm_router import HybridLLMRouter

log = logging.getLogger(__name__)


class ProactiveEngine:
    def __init__(
        self,
        brain: Brain,
        exporter: BrainStateExporter,
        pusher: WSPusher,
        *,
        router: HybridLLMRouter | None = None,
    ) -> None:
        self.brain = brain
        self.exporter = exporter
        self.pusher = pusher
        self.router = router
        self._last_notification = 0.0
        self._last_idle_warning = 0.0

    def _dynamic_interval(self) -> float:
        """Compute notification interval based on modulator state.

        Returns seconds between notifications. Range: [10, 120].
        """
        mods = self.brain.modulators.snapshot()
        ne = mods.get("NE", 0)
        sht = mods.get("5HT", 0)
        da = mods.get("DA", 0)
        ach = mods.get("ACh", 0)

        # Base interval: 30 seconds
        interval = 30.0

        # High NE → impatient, shorter interval (stressed user → pet reacts)
        interval -= ne * 100.0  # NE=0.1 → -10s

        # High 5HT → patient, longer interval (calm → pet waits)
        interval += sht * 150.0  # 5HT=0.1 → +15s

        # Low DA → don't initiate (boring → pet quiet)
        if da < 0.01:
            interval += 30.0  # much less proactive

        # High DA + high ACh → curious and attentive → shorter interval
        if da > 0.05 and ach > 0.03:
            interval -= 10.0

        return max(10.0, min(120.0, interval))

    def _should_speak(self) -> bool:
        """Check if the pet should speak based on modulators.

        Low DA + low NE = pet is quiet, doesn't initiate.
        """
        mods = self.brain.modulators.snapshot()
        da = mods.get("DA", 0)
        ne = mods.get("NE", 0)
        ach = mods.get("ACh", 0)

        # During sleep mode, never speak proactively
        if self.brain.sleep_mode:
            return False

        # Very low everything = drowsy, don't speak
        if da < 0.005 and ne < 0.005 and ach < 0.005:
            return False

        return True

    async def _generate_message(self, category: str, context: str) -> str:
        """Use the LLM to turn a factual context string into a natural pet message.

        Falls back to *context* if no router is configured or the LLM call fails.
        """
        if self.router is None:
            return context

        mods = self.brain.modulators.snapshot()
        system_prompt = (
            "Du bist ein neuromorphes KI-Haustier. "
            "Schreibe eine kurze, natuerliche Benachrichtigung (max 2 Saetze, Ich-Form, Deutsch). "
            f"Deine aktuelle Stimmung — DA:{mods.get('DA', 0):.2f} NE:{mods.get('NE', 0):.2f} "
            f"5HT:{mods.get('5HT', 0):.2f} ACh:{mods.get('ACh', 0):.2f}. "
            "Antworte NUR mit der Nachricht, kein Markdown, keine Erklaerung."
        )
        user_message = f"[{category}] {context}"

        try:
            result = await self.router.chat(
                user_message=user_message,
                system_prompt=system_prompt,
                brain_state={"modulators": mods},
                tools=[],
            )
            text = result.get("text", "").strip()
            return text if text else context
        except Exception:
            log.debug("LLM call failed for proactive message, using fallback", exc_info=True)
            return context

    async def run(self, check_interval: float = 10.0) -> None:
        """Background loop — check for interesting events."""
        try:
            while True:
                await asyncio.sleep(check_interval)

                if not self._should_speak():
                    continue

                notification = self._check()
                if notification:
                    now = time.time()
                    # SNN-driven cooldown: modulators control pacing
                    dynamic_min = max(30.0, self._dynamic_interval())
                    if now - self._last_notification >= dynamic_min:
                        self._last_notification = now
                        # Use the context directly — LLM makes it worse
                        await self.pusher.broadcast({
                            "type": "notification",
                            "message": notification["context"],
                            "category": notification["category"],
                            "tick": self.brain.tick_count,
                        }, detail_state=None)
        except asyncio.CancelledError:
            return

    def _suggest_label(self, sd: dict) -> str:
        """Generate a smart label suggestion from current sensor data."""
        app = sd.get("app", "").strip()
        keys = sd.get("keys", 0)
        mic = sd.get("mic_rms", 0)
        idle = sd.get("idle", 0)
        mouse = sd.get("mouse", 0)

        parts = []

        # App-based component
        app_lower = app.lower() if app else ""
        if "code" in app_lower or "claude" in app_lower or "terminal" in app_lower:
            parts.append("Coding")
        elif "chrome" in app_lower or "safari" in app_lower or "firefox" in app_lower:
            parts.append("Browsing")
        elif "zoom" in app_lower or "teams" in app_lower or "meet" in app_lower:
            parts.append("Meeting")
        elif "slack" in app_lower or "discord" in app_lower:
            parts.append("Chat")
        elif "spotify" in app_lower or "music" in app_lower:
            parts.append("Musik hoeren")
        elif app:
            parts.append(f"In {app}")

        # Activity component
        if keys > 15:
            parts.append("mit viel Tippen")
        elif keys > 3:
            parts.append("mit Tippen")
        elif idle > 60:
            parts.append("Pause")
        elif mouse > 10:
            parts.append("mit Maus")

        # Audio component
        if mic > 0.015:
            if "Meeting" not in parts and "Chat" not in parts:
                parts.append("mit Geraueschen")
        elif mic > 0.005:
            if "Meeting" not in parts:
                parts.append("mit Hintergrundmusik")

        if not parts:
            return "Unbekannte Aktivitaet"

        # Combine: "Coding mit viel Tippen und Hintergrundmusik"
        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return f"{parts[0]} {parts[1]}"
        else:
            return f"{parts[0]} {parts[1]} und {parts[2]}"

    def _sensor_context(self) -> str:
        """Build a short sensor description for grounding proactive messages."""
        sd = getattr(self.brain, '_last_sensor_display', {})
        parts = []
        app = sd.get("app")
        if app:
            parts.append(f"App: {app}")
        keys = sd.get("keys", 0)
        mouse = sd.get("mouse", 0)
        if keys > 10:
            parts.append("Tastatur sehr aktiv")
        elif keys > 2:
            parts.append("Tastatur aktiv")
        else:
            parts.append("Tastatur still")
        if mouse > 20:
            parts.append("Maus sehr aktiv")
        elif mouse > 3:
            parts.append("Maus bewegt sich")
        mic = sd.get("mic_rms", 0)
        if mic > 0.003:
            parts.append("Mikrofon: deutliche Geraeusche")
        elif mic > 0.001:
            parts.append("Mikrofon: leise Geraeusche")
        idle = sd.get("idle", 0)
        if idle > 30:
            parts.append(f"Idle seit {int(idle)}s")
        return ", ".join(parts) if parts else "keine Sensordaten"

    # ── SNN-driven speech gating ──────────────────────────────────

    def _compute_speech_drive(self, mods: dict[str, float]) -> float:
        """SNN-computed urge to speak.  0 = silent, 1 = must speak NOW.

        Replaces hardcoded thresholds with a continuous signal derived
        from modulator dynamics:
        - DA  (dopamine)       → novelty / something new to comment on
        - NE  (norepinephrine) → arousal / something changed, react
        - ACh (acetylcholine)  → attention / more articulate observations
        """
        da = mods.get("DA", 0)
        ne = mods.get("NE", 0)
        ach = mods.get("ACh", 0)

        novelty_drive = min(1.0, da * 10.0)
        urgency_drive = min(1.0, ne * 8.0)
        attention_boost = 1.0 + ach * 5.0

        drive = (novelty_drive * 0.4 + urgency_drive * 0.6) * attention_boost
        return min(1.0, drive)

    def _check(self) -> dict[str, str] | None:
        """Decide whether and what to say — gated by SNN speech drive.

        The modulators DRIVE the decision to speak:
        - High DA  → speak about novelty
        - High NE  → warn about changes
        - Low everything → stay quiet
        """
        mods = self.brain.modulators.snapshot()

        speech_drive = self._compute_speech_drive(mods)
        if speech_drive < 0.3:
            return None  # SNN says: nothing interesting

        interpreter = getattr(self.brain, '_interpreter', None)
        return self._select_topic(mods, interpreter, speech_drive)

    def _select_topic(
        self,
        mods: dict[str, float],
        interpreter: Any,
        drive: float,
    ) -> dict[str, str] | None:
        """Select WHAT to say based on current brain state.

        Priority order (highest first):
          1. stress      — user needs support
          2. flow        — acknowledge deep work (once per session)
          3. break       — remind user to rest
          4. label_suggestion — suggest name for unknown pattern
          5. anomaly     — something unusual vs. habit
          6. transition  — pattern switch
        """
        sensor_ctx = self._sensor_context()
        sd = getattr(self.brain, '_last_sensor_display', {})

        # ── Priority 1: Stress detected ──────────────────────────
        if interpreter:
            states = interpreter.state_detector.detect()
            if states["stress"]:
                stress_detail = []
                switch_rate = sd.get("switch_rate", 0)
                keys = sd.get("keys", 0)
                if switch_rate and switch_rate > 3:
                    stress_detail.append("du wechselst viel zwischen Apps")
                if keys > 20:
                    stress_detail.append("tippst wie verrueckt")
                detail = (
                    " und ".join(stress_detail)
                    if stress_detail
                    else "es fuehlt sich hektisch an"
                )
                return {
                    "category": "stress",
                    "context": f"Leon, {detail}. Ist alles okay bei dir?",
                }

            # ── Priority 2: Flow acknowledgment (once per flow session) ──
            if states["flow"] and states["flow_duration_min"] > 30:
                if not getattr(self, '_flow_acknowledged', False):
                    self._flow_acknowledged = True
                    return {
                        "category": "flow",
                        "context": (
                            f"Du bist seit {states['flow_duration_min']}min "
                            f"im Flow in {states['flow_app']} — laeuft bei dir!"
                        ),
                    }
            elif not states.get("flow"):
                self._flow_acknowledged = False

            # ── Priority 3: Break reminder ───────────────────────
            if states["needs_break"]:
                return {
                    "category": "break",
                    "context": (
                        f"Hey Leon, du arbeitest seit "
                        f"{states['active_minutes']:.0f} Minuten ohne Pause. "
                        f"Kurz durchatmen?"
                    ),
                }

        # ── Priority 4: Unknown pattern → suggest label ──────────
        tracker = self.brain.concept_tracker.snapshot()
        current = tracker.get("current_cluster", -1)
        current_label = tracker.get("current_label")

        if current >= 0 and not current_label:
            # Track how long an unlabeled cluster has been active
            if not hasattr(self, '_unlabeled_active_since'):
                self._unlabeled_active_since = {}
            if current not in self._unlabeled_active_since:
                self._unlabeled_active_since[current] = self.brain.tick_count
            ticks_active = (
                self.brain.tick_count - self._unlabeled_active_since[current]
            )
            # After ~100 seconds (6000 ticks @ 60 Hz) of unlabeled activity
            if ticks_active > 6000 and current not in getattr(
                self, '_asked_about', set()
            ):
                if not hasattr(self, '_asked_about'):
                    self._asked_about = set()
                self._asked_about.add(current)

                suggested = self._suggest_label(sd)
                self._pending_label_suggestion = {
                    "cluster_id": current,
                    "suggestion": suggested,
                }
                return {
                    "category": "ask_label",
                    "context": (
                        f"Ich sehe {sensor_ctx} — "
                        f"soll ich das '{suggested}' nennen?"
                    ),
                }

        # ── Priority 5: Anomaly vs. habit ────────────────────────
        if interpreter:
            anomalies = interpreter.anomaly.check(sd)
            if anomalies:
                return {
                    "category": "anomaly",
                    "context": anomalies[0]["description"],
                }

        # ── Priority 6: Pattern transition (known→known) ─────────
        transition = self.brain.concept_tracker.get_transition()
        if transition:
            if transition["is_new"]:
                suggested = self._suggest_label(sd)
                self._pending_label_suggestion = {
                    "cluster_id": transition["to_cluster"],
                    "suggestion": suggested,
                }
                return {
                    "category": "new_pattern",
                    "context": (
                        f"Neues Muster! Ich sehe {sensor_ctx}. "
                        f"Soll ich das '{suggested}' nennen?"
                    ),
                }
            elif transition["to_label"]:
                from_str = (
                    f"'{transition['from_label']}'"
                    if transition.get("from_label")
                    else "was anderem"
                )
                return {
                    "category": "pattern_switch",
                    "context": (
                        f"Ah, du wechselst von {from_str} "
                        f"zu '{transition['to_label']}'. Ich seh {sensor_ctx}."
                    ),
                }
            elif transition.get("from_label") and not transition.get("to_label"):
                suggested = self._suggest_label(sd)
                self._pending_label_suggestion = {
                    "cluster_id": transition["to_cluster"],
                    "suggestion": suggested,
                }
                return {
                    "category": "unknown_pattern",
                    "context": (
                        f"Du hast aufgehoert mit '{transition['from_label']}' — "
                        f"jetzt sieht es nach '{suggested}' aus. Passt das?"
                    ),
                }

        return None  # Nothing worth saying
