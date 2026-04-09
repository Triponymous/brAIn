"""Proactive notifications — the pet speaks up when something interesting happens.

Runs as a background coroutine (~every 30s). Checks:
1. Did a new concept emerge (first time active)?
2. Did novelty spike (something unexpected)?
3. Has the user been idle too long?
4. Did a concept get very active that has a label?

Generates a notification message and pushes it via WebSocket.
"""
from __future__ import annotations
import asyncio
import time
from typing import Any

from brain.core import Brain
from bridge.exporter import BrainStateExporter
from server.ws import WSPusher


class ProactiveEngine:
    def __init__(self, brain: Brain, exporter: BrainStateExporter, pusher: WSPusher) -> None:
        self.brain = brain
        self.exporter = exporter
        self.pusher = pusher
        self._seen_concepts: set[int] = set()
        self._last_notification = 0.0
        self._min_interval = 30.0  # don't spam — at most one notification per 30s
        self._last_idle_warning = 0.0

    async def run(self, check_interval: float = 10.0) -> None:
        """Background loop — check for interesting events."""
        try:
            while True:
                await asyncio.sleep(check_interval)
                notification = self._check()
                if notification:
                    now = time.time()
                    if now - self._last_notification >= self._min_interval:
                        self._last_notification = now
                        await self.pusher.broadcast({
                            "type": "notification",
                            "message": notification["message"],
                            "category": notification["category"],
                            "tick": self.brain.tick_count,
                        })
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
                "message": f"Ich habe etwas Neues entdeckt — ein neues Muster ({suggested}). Insgesamt kenne ich jetzt {len(self._seen_concepts)} verschiedene Konzepte.",
            }

        # 2. High novelty (DA or NE spiked)
        da = mods.get("DA", 0)
        ne = mods.get("NE", 0)
        if da > 0.15 or ne > 0.2:
            return {
                "category": "novelty",
                "message": f"Da ist gerade etwas Unerwartetes passiert! Mein Dopamin ist bei {da:.2f}, Noradrenalin bei {ne:.2f}.",
            }

        # 3. Very active labeled concept
        labels = self.exporter.all_labels()
        for cid in active_ids[:5]:
            if cid in labels and float(accum[cid].item()) > max_val * 0.8:
                label = labels[cid]
                return {
                    "category": "activity",
                    "message": f"'{label}' (Concept #{cid}) ist gerade sehr aktiv.",
                }

        return None
