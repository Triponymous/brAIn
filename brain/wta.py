"""Winner-Take-All layer.

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
import torch


class WTALayer:
    def __init__(
        self,
        num_neurons: int,
        k: int = 1,
        tau_mem: float = 20.0,
        threshold: float = 1.0,
        inhibit_factor: float = 0.5,
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

    def step(self, input_current: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        # Standard LIF integrate
        leak = -self.membrane / self.tau_mem
        self.membrane = self.membrane + dt * (leak + input_current)

        # Find above-threshold neurons
        above = self.membrane >= self.threshold
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

        return spikes

    def reset(self) -> None:
        self.membrane = torch.zeros(self.num_neurons)
