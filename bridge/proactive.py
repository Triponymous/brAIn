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

    def _check(self) -> dict[str, str] | None:
        mods = self.brain.modulators.snapshot()

        # 0. ConceptTracker transition — the BEST trigger for proactive messages
        transition = self.brain.concept_tracker.get_transition()
        if transition:
            if transition["is_new"]:
                # New pattern the pet has never seen before!
                return {
                    "category": "new_pattern",
                    "context": f"Neues Muster entdeckt (Muster #{transition['to_cluster']}). "
                               f"Das habe ich noch nie gesehen.",
                }
            elif transition["to_label"]:
                # Switched to a known pattern
                return {
                    "category": "pattern_switch",
                    "context": f"Wechsel erkannt: jetzt '{transition['to_label']}' "
                               f"(vorher: {transition['from_label'] or 'unbekannt'}).",
                }
            elif transition["from_label"] and not transition["to_label"]:
                # Left a known pattern for an unknown one
                return {
                    "category": "unknown_pattern",
                    "context": f"Leon hat aufgehoert mit '{transition['from_label']}'. "
                               f"Jetzt passiert etwas Neues (Muster #{transition['to_cluster']}). "
                               f"Was machst du, Leon?",
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
                    "context": f"Muster #{current} ist seit ein paar Minuten aktiv und hat noch keinen Namen. Was machst du gerade, Leon?",
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
                "context": "Ich spuere dass sich gerade etwas veraendert hat. Was ist passiert, Leon?",
            }

        # 3. Stress detection
        if ne > 0.08 and sht < 0.01:
            return {
                "category": "stress",
                "context": "Du wirkst gerade etwas gestresst. Alles okay?",
            }

        return None
