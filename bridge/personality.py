"""PersonalityTracker — models long-term emotional development.

Modulator baselines drift slowly over days/weeks based on the pet's
experiences. A consistently curious user (high DA) raises the DA baseline,
making the pet naturally more curious. This is personality evolution.

Traits are derived from baselines:
- curiosity = DA baseline (normalized 0-1)
- anxiety = NE baseline
- attentiveness = ACh baseline
- patience = 5HT baseline
"""
from __future__ import annotations
from typing import Any


class PersonalityTracker:
    def __init__(
        self,
        drift_rate: float = 0.0001,  # very slow: ~10% shift per 10000 updates
    ) -> None:
        self.drift_rate = drift_rate
        self._baselines = {"DA": 0.0, "NE": 0.0, "ACh": 0.0, "5HT": 0.0}
        self._session_sum = {"DA": 0.0, "NE": 0.0, "ACh": 0.0, "5HT": 0.0}
        self._session_count = 0
        self._age_ticks = 0

    def update(self, modulator_levels: dict[str, float]) -> None:
        """Call periodically (every ~100 ticks) with current modulator snapshot."""
        self._age_ticks += 1
        for name in self._baselines:
            level = modulator_levels.get(name, 0)
            self._session_sum[name] += level
            self._session_count += 1
            # Exponential moving average toward session mean
            self._baselines[name] = (
                self._baselines[name] * (1 - self.drift_rate)
                + level * self.drift_rate
            )

    def snapshot(self) -> dict[str, Any]:
        """Current personality state for LLM prompt."""
        # Normalize baselines to 0-1 trait scale
        # Typical baselines are 0.0-0.1, so scale by 10x and clamp
        def trait(val: float) -> float:
            return min(1.0, max(0.0, val * 10.0))

        return {
            "baselines": dict(self._baselines),
            "traits": {
                "curiosity": round(trait(self._baselines["DA"]), 2),
                "anxiety": round(trait(self._baselines["NE"]), 2),
                "attentiveness": round(trait(self._baselines["ACh"]), 2),
                "patience": round(trait(self._baselines["5HT"]), 2),
            },
            "age_ticks": self._age_ticks,
            "age_days": round(self._age_ticks / 8640000, 1),  # 100Hz * 86400s/day
        }

    def save_state(self) -> dict[str, Any]:
        return {
            "baselines": dict(self._baselines),
            "age_ticks": self._age_ticks,
        }

    def load_state(self, data: dict[str, Any]) -> None:
        self._baselines = data.get("baselines", self._baselines)
        self._age_ticks = data.get("age_ticks", 0)
