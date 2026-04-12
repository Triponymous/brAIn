"""Working memory layer: LIF with recurrent self-excitation and capacity limit.

Each neuron receives, in addition to external input, a contribution from its
OWN previous spike (`recurrent_gain * last_spike`). With gain > threshold,
this creates self-sustaining persistent activity — the neuron keeps firing.

To prevent saturation (all neurons eventually latching on permanently),
a `max_active` capacity limit enforces that only the K neurons with highest
membrane potential survive each tick. Older/weaker memories get suppressed
when new patterns arrive — natural forgetting via competition.

This models the limited capacity of biological working memory (~7±2 items).
"""
from __future__ import annotations
import torch


class WMLayer:
    def __init__(
        self,
        num_neurons: int,
        recurrent_gain: float = 1.05,
        tau_mem: float = 100.0,
        threshold: float = 1.0,
        max_active: int = 20,  # capacity limit — max simultaneous active WM slots
    ) -> None:
        self.num_neurons = num_neurons
        self.recurrent_gain = recurrent_gain
        self.tau_mem = tau_mem
        self.threshold = threshold
        self.max_active = max_active
        self.membrane = torch.zeros(num_neurons)
        self.last_spikes = torch.zeros(num_neurons)

    def step(self, input_current: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        recurrent_current = self.recurrent_gain * self.last_spikes
        total = input_current + recurrent_current

        leak = -self.membrane / self.tau_mem
        self.membrane = self.membrane + dt * (leak + total)
        spikes = (self.membrane >= self.threshold).float()

        # Capacity limit: if more than max_active neurons want to fire,
        # only keep the strongest (highest membrane). This naturally
        # evicts old/weak WM entries when new ones arrive.
        if spikes.sum() > self.max_active:
            active_indices = spikes.nonzero(as_tuple=True)[0]
            active_membranes = self.membrane[active_indices]
            # Keep only top-K by membrane potential
            _, top_k_local = torch.topk(active_membranes, self.max_active)
            top_k_indices = active_indices[top_k_local]
            mask = torch.zeros_like(spikes)
            mask[top_k_indices] = 1.0
            spikes = spikes * mask

        self.membrane = self.membrane * (1.0 - spikes)
        self.last_spikes = spikes
        return spikes

    def reset(self) -> None:
        self.membrane = torch.zeros(self.num_neurons)
        self.last_spikes = torch.zeros(self.num_neurons)
