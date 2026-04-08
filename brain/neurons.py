"""Leaky Integrate-and-Fire neuron primitive.

A LIF neuron integrates input over time, leaks toward rest, and emits
a discrete spike when its membrane potential crosses a threshold.
After firing, the membrane resets to zero (hard reset).

Membrane dynamics (Euler discretization):
    dV/dt = -V / tau_mem + I
    V(t+dt) = V(t) + dt * (-V(t)/tau_mem + I(t))
"""
from __future__ import annotations
import torch


class LIFLayer:
    """A vector of independent LIF neurons sharing the same parameters."""

    def __init__(
        self,
        num_neurons: int,
        tau_mem: float = 20.0,
        threshold: float = 1.0,
    ) -> None:
        self.num_neurons = num_neurons
        self.tau_mem = tau_mem
        self.threshold = threshold
        self.membrane = torch.zeros(num_neurons)

    def step(self, input_current: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Advance the layer by one timestep. Returns 0/1 spike vector."""
        leak = -self.membrane / self.tau_mem
        self.membrane = self.membrane + dt * (leak + input_current)
        spikes = (self.membrane >= self.threshold).float()
        # Hard reset on spike
        self.membrane = self.membrane * (1.0 - spikes)
        return spikes

    def reset(self) -> None:
        """Reset all membrane potentials to zero."""
        self.membrane = torch.zeros(self.num_neurons)
