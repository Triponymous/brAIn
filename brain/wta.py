"""Winner-Take-All layer with adaptive thresholds (Diehl & Cook 2015).

Like LIFLayer (leak + integrate + threshold) but at most k neurons can
spike per timestep. When more than k cross threshold, only the k with the
highest membrane values spike. Losers above threshold get their membrane
multiplied by an inhibition factor (not full reset) so they retain some
charge for the next round but cannot spike this tick.

This is the simplest practical winner-take-all: deterministic top-k selection
rather than recurrent inhibitory dynamics. Sufficient for sparse concept
formation in Phase 2.
"""
from __future__ import annotations
import math
import torch


class WTALayer:
    """Winner-Take-All with intrinsic plasticity (adaptive thresholds).

    Each neuron's threshold adapts based on its firing rate:
    - Fires too often → threshold goes UP (harder to win next time)
    - Fires too rarely → threshold goes DOWN (easier to win)

    This naturally distributes activity across neurons instead of letting
    the same few winners dominate forever. The target firing rate is
    k/N (the expected rate if all neurons were equally likely to win).
    """
    def __init__(
        self,
        num_neurons: int,
        k: int = 1,
        tau_mem: float = 20.0,
        threshold: float = 1.0,
        inhibit_factor: float = 0.5,
        # Intrinsic plasticity parameters
        ip_rate: float = 0.002,      # threshold increase per spike — slower because lateral inhibition handles discrimination
        ip_tau: float = 10000000.0, # threshold decay tau (Diehl&Cook: 1e7 — quasi-permanent)
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k > num_neurons:
            raise ValueError(f"k={k} cannot exceed num_neurons={num_neurons}")
        self.num_neurons = num_neurons
        self.k = k
        self.tau_mem = tau_mem
        self.threshold = threshold
        self.inhibit_factor = inhibit_factor

        self.membrane = torch.zeros(num_neurons)

        # Intrinsic plasticity: per-neuron adaptive thresholds
        self.ip_rate = ip_rate
        self.ip_tau = ip_tau
        self.thresholds = torch.full((num_neurons,), threshold)
        self._firing_rate = torch.full((num_neurons,), k / num_neurons)
        self._target_rate = k / num_neurons

        # Lateral inhibition weights (learned via anti-Hebbian rule):
        # When two neurons co-fire, their mutual inhibition INCREASES.
        # This pushes them apart → they specialize for different patterns.
        # Shape: [num_neurons, num_neurons], diagonal = 0 (no self-inhibition)
        self.lateral_weights = torch.zeros(num_neurons, num_neurons)
        self._lateral_rate = 0.0005  # slow lateral learning — don't rotate within-pattern

    def step(self, input_current: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        # Lateral inhibition: subtract weighted sum of other neurons' membrane
        # This makes neurons that frequently co-fire compete harder
        lateral_inhib = self.lateral_weights @ self.membrane.clamp(min=0)

        # Standard LIF integrate with lateral inhibition
        leak = -self.membrane / self.tau_mem
        self.membrane = self.membrane + dt * (leak + input_current - lateral_inhib)

        # Find above-threshold neurons (per-neuron adaptive thresholds)
        above = self.membrane >= self.thresholds
        spikes = torch.zeros(self.num_neurons)

        if above.any():
            # Score = membrane value, but only for above-threshold
            scores = torch.where(above, self.membrane, torch.full_like(self.membrane, -float("inf")))
            num_winners = min(self.k, int(above.sum().item()))
            if num_winners > 0:
                _, winner_idx = torch.topk(scores, num_winners)
                spikes[winner_idx] = 1.0

            # Winners: hard reset
            self.membrane = self.membrane * (1.0 - spikes)
            # Losers above threshold (above & not in winners): inhibit
            losers_above = above & (spikes == 0)
            inhibit_mask = losers_above.float() * (self.inhibit_factor - 1.0) + 1.0
            self.membrane = self.membrane * inhibit_mask

        # ── Very gentle adaptive threshold ──
        # Barely raises threshold for frequent winners — just enough to
        # prevent one neuron from claiming ALL patterns.
        # ip_rate=0.002 with tau=10M → very slow adaptation
        decay = math.exp(-dt / self.ip_tau)
        self.thresholds = self.threshold + (self.thresholds - self.threshold) * decay
        self.thresholds = self.thresholds + self.ip_rate * spikes
        alpha = dt / 10000.0
        self._firing_rate = self._firing_rate * (1 - alpha) + spikes * alpha

        # ── Lateral Inhibition Learning (anti-Hebbian) ──
        # If two neurons co-fire (or fire close together), INCREASE mutual inhibition.
        # This pushes co-firing neurons to specialize for different patterns.
        # outer(spikes, spikes) = 1 where both neurons fired this tick.
        if spikes.sum() > 0:
            co_fire = torch.outer(spikes, spikes)
            # Remove self-connections (diagonal)
            co_fire.fill_diagonal_(0)
            # Increase lateral weights where neurons co-fired
            self.lateral_weights = self.lateral_weights + self._lateral_rate * co_fire
            # Slow decay to prevent runaway inhibition
            self.lateral_weights = self.lateral_weights * 0.9999
            # Clamp to [0, 1]
            self.lateral_weights = self.lateral_weights.clamp(0, 1.0)

        return spikes

    def reset(self) -> None:
        self.membrane = torch.zeros(self.num_neurons)
        self.thresholds = torch.full((self.num_neurons,), self.threshold)
        self._firing_rate = torch.full((self.num_neurons,), self._target_rate)
