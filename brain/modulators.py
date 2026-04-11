"""Neuromodulator system: Dopamine, Noradrenaline, Acetylcholine, Serotonin.

Each modulator has:
- a current level in [-1, 1]
- a baseline (default 0.0)
- a time constant tau for exponential decay toward baseline

Levels are scalars, not per-neuron. Other regions read levels to gate
plasticity (DA modulates STDP), attention (ACh), and global state (NE/5HT).

External code can inject quantities at any time. Decay is applied via
`tick(dt)` from the main brain loop.
"""
from __future__ import annotations
import math


_DEFAULT_TAU = {"DA": 200.0, "NE": 500.0, "ACh": 300.0, "5HT": 1000.0}
_NAMES = ("DA", "NE", "ACh", "5HT")


class Modulators:
    def __init__(
        self,
        baseline: dict[str, float] | None = None,
        tau: dict[str, float] | None = None,
        level_min: float = 0.0,  # modulators can't go negative (no "anti-emotion")
        level_max: float = 1.0,
    ) -> None:
        self.baseline = {n: 0.0 for n in _NAMES}
        if baseline:
            self.baseline.update(baseline)
        self.tau = dict(_DEFAULT_TAU)
        if tau:
            self.tau.update(tau)
        self.level_min = level_min
        self.level_max = level_max
        self._levels: dict[str, float] = {n: self.baseline[n] for n in _NAMES}

    def level(self, name: str) -> float:
        if name not in self._levels:
            raise KeyError(f"Unknown modulator: {name}")
        return self._levels[name]

    def inject(self, name: str, amount: float) -> None:
        if name not in self._levels:
            raise KeyError(f"Unknown modulator: {name}")
        new = self._levels[name] + amount
        self._levels[name] = max(self.level_min, min(self.level_max, new))

    def tick(self, dt: float) -> None:
        """Decay all levels toward their baseline."""
        for name in _NAMES:
            base = self.baseline[name]
            current = self._levels[name]
            decay = math.exp(-dt / self.tau[name])
            self._levels[name] = base + (current - base) * decay

    def snapshot(self) -> dict[str, float]:
        return dict(self._levels)
