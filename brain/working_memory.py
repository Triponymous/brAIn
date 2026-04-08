"""Working memory layer: LIF with recurrent self-excitation.

Each neuron in a WMLayer receives, in addition to external input, a
contribution from its OWN previous spike (`recurrent_gain * last_spike`).
This creates short-term sustained activity after a transient input.

This is the simplest possible WM: per-neuron self-recurrence, no
neuron-to-neuron lateral connections within the layer. Sufficient for
holding context for ~10-30 ticks at typical params.
"""
from __future__ import annotations
import torch


class WMLayer:
    def __init__(
        self,
        num_neurons: int,
        recurrent_gain: float = 0.3,
        tau_mem: float = 20.0,
        threshold: float = 1.0,
    ) -> None:
        self.num_neurons = num_neurons
        self.recurrent_gain = recurrent_gain
        self.tau_mem = tau_mem
        self.threshold = threshold
        self.membrane = torch.zeros(num_neurons)
        self.last_spikes = torch.zeros(num_neurons)

    def step(self, input_current: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        recurrent_current = self.recurrent_gain * self.last_spikes
        total = input_current + recurrent_current

        leak = -self.membrane / self.tau_mem
        self.membrane = self.membrane + dt * (leak + total)
        spikes = (self.membrane >= self.threshold).float()
        self.membrane = self.membrane * (1.0 - spikes)

        self.last_spikes = spikes
        return spikes

    def reset(self) -> None:
        self.membrane = torch.zeros(self.num_neurons)
        self.last_spikes = torch.zeros(self.num_neurons)
