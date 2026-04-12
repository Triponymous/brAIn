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
                    # Minimum 3 minutes between notifications (not 30s)
                    if now - self._last_notification >= 180:
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

    def _check(self) -> dict[str, str] | None:
        mods = self.brain.modulators.snapshot()
        sensor_ctx = self._sensor_context()

        # ── BrainInterpreter-based triggers (higher-level than raw SNN) ──
        interpreter = getattr(self.brain, '_interpreter', None)
        if interpreter:
            states = interpreter.state_detector.detect()
            if states["flow"] and states["flow_duration_min"] > 60:
                return {
                    "category": "flow",
                    "context": f"Du bist seit {states['flow_duration_min']}min im Flow in {states['flow_app']} — laeuft bei dir!",
                }
            if states["needs_break"]:
                return {
                    "category": "break",
                    "context": f"Hey Leon, du arbeitest seit {states['active_minutes']:.0f} Minuten ohne Pause. Kurz durchatmen?",
                }
            if states["meeting"] and not getattr(self, '_meeting_announced', False):
                self._meeting_announced = True
                return {
                    "category": "meeting",
                    "context": f"Ich seh {sensor_ctx} — bist du in einem Call?",
                }
            elif not states.get("meeting"):
                self._meeting_announced = False

            anomalies = interpreter.anomaly.check(
                getattr(self.brain, '_last_sensor_display', {}))
            if anomalies:
                return {
                    "category": "anomaly",
                    "context": anomalies[0]["description"],
                }

        # 0. ConceptTracker transition — the BEST trigger for proactive messages
        transition = self.brain.concept_tracker.get_transition()
        if transition:
            if transition["is_new"]:
                # New pattern the pet has never seen before!
                return {
                    "category": "new_pattern",
                    "context": f"Hmm, das ist neu — {sensor_ctx}. "
                               f"Das kenn ich noch nicht. Was machst du da, Leon?",
                }
            elif transition["to_label"]:
                # Switched to a known pattern
                from_str = f"'{transition['from_label']}'" if transition['from_label'] else "was anderem"
                return {
                    "category": "pattern_switch",
                    "context": f"Ah, du wechselst von {from_str} zu '{transition['to_label']}'. "
                               f"Ich seh {sensor_ctx}.",
                }
            elif transition["from_label"] and not transition["to_label"]:
                # Left a known pattern for an unknown one
                return {
                    "category": "unknown_pattern",
                    "context": f"Du hast aufgehoert mit '{transition['from_label']}' — "
                               f"jetzt seh ich {sensor_ctx}. Was kommt jetzt, Leon?",
                }

        # 1. Unknown pattern active for a while — ask what Leon is doing
        tracker = self.brain.concept_tracker.snapshot()
        current = tracker.get("current_cluster", -1)
        current_label = tracker.get("current_label")
        if current >= 0 and not current_label:
            # How long has this unlabeled cluster been active?
            if not hasattr(self, '_unlabeled_active_since'):
                self._unlabeled_active_since = {}
            if current not in self._unlabeled_active_since:
                self._unlabeled_active_since[current] = self.brain.tick_count
            ticks_active = self.brain.tick_count - self._unlabeled_active_since[current]
            # Ask after 2 minutes of the SAME unlabeled pattern (12000 ticks)
            if ticks_active > 12000 and current not in getattr(self, '_asked_about', set()):
                if not hasattr(self, '_asked_about'):
                    self._asked_about = set()
                self._asked_about.add(current)
                return {
                    "category": "ask_label",
                    "context": f"Hey Leon — ich seh seit ein paar Minuten {sensor_ctx}. "
                               f"Das ist ein Muster das ich noch nicht kenne. Wie soll ich das nennen?",
                }
        else:
            # Reset tracking when pattern changes
            if hasattr(self, '_unlabeled_active_since'):
                self._unlabeled_active_since = {}

        # 2. High novelty — something changed suddenly
        da = mods.get("DA", 0)
        ne = mods.get("NE", 0)
        sht = mods.get("5HT", 0)
        if da > 0.05 or ne > 0.08:
            return {
                "category": "novelty",
                "context": f"Whoa — gerade hat sich was veraendert! Ich seh {sensor_ctx}. Was ist los?",
            }

        # 3. Stress detection
        if ne > 0.08 and sht < 0.01:
            sd = getattr(self.brain, '_last_sensor_display', {})
            app = sd.get("app", "")
            keys = sd.get("keys", 0)
            switch_rate = sd.get("switch_rate", 0)
            stress_detail = []
            if switch_rate and switch_rate > 3:
                stress_detail.append("du wechselst viel zwischen Apps")
            if keys > 20:
                stress_detail.append("tippst wie verrueckt")
            detail = " und ".join(stress_detail) if stress_detail else "es fuehlt sich hektisch an"
            return {
                "category": "stress",
                "context": f"Leon, {detail}. Ist alles okay bei dir?",
            }

        return None
