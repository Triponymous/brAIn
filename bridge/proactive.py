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
        self._seen_concepts: set[int] = set()
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
                    dynamic_min = self._dynamic_interval()
                    if now - self._last_notification >= dynamic_min:
                        self._last_notification = now
                        message = await self._generate_message(
                            notification["category"],
                            notification["context"],
                        )
                        await self.pusher.broadcast({
                            "type": "notification",
                            "message": message,
                            "category": notification["category"],
                            "tick": self.brain.tick_count,
                        }, detail_state=None)
        except asyncio.CancelledError:
            return

    def _check(self) -> dict[str, str] | None:
        mods = self.brain.modulators.snapshot()
        accum = self.brain.concept_spike_accum
        max_val = float(accum.max().item())

        # 1. New concept emerged
        active_ids = (accum > max_val * 0.3).nonzero(as_tuple=True)[0].tolist() if max_val > 0.01 else []
        new_concepts = [cid for cid in active_ids if cid not in self._seen_concepts]
        for cid in active_ids:
            self._seen_concepts.add(cid)

        if new_concepts and len(self._seen_concepts) > 3:
            profile = self.exporter.get_concept_profile(new_concepts[0])
            suggested = profile.get("suggested_label") or f"Concept #{new_concepts[0]}"
            return {
                "category": "new_concept",
                "context": f"Neues Muster entdeckt: {suggested}. Insgesamt {len(self._seen_concepts)} Konzepte bekannt.",
            }

        # 2. High novelty (DA or NE spiked)
        da = mods.get("DA", 0)
        ne = mods.get("NE", 0)
        if da > 0.15 or ne > 0.2:
            return {
                "category": "novelty",
                "context": f"Etwas Unerwartetes passiert. Dopamin={da:.2f}, Noradrenalin={ne:.2f}.",
            }

        # 3. Stress detection (high NE + low 5HT sustained)
        sht = mods.get("5HT", 0)
        if ne > 0.08 and sht < 0.01:
            return {
                "category": "stress",
                "context": f"Anzeichen von Stress erkannt. NE={ne:.2f}, 5HT={sht:.2f}.",
            }

        # 4. Very active labeled concept
        labels = self.exporter.all_labels()
        for cid in active_ids[:5]:
            if cid in labels and float(accum[cid].item()) > max_val * 0.8:
                label = labels[cid]
                return {
                    "category": "activity",
                    "context": f"Konzept '{label}' (#{cid}) ist gerade sehr aktiv.",
                }

        return None
