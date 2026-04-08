"""STDP synapse primitive — connects a pre-population to a post-population.

Implements pair-based STDP with exponentially decaying eligibility traces:

    apre  := apre  * exp(-dt/tau_pre)
    apost := apost * exp(-dt/tau_post)

On a pre-synaptic spike:
    apre += a_plus
    weights -= apost            # depress (post happened in the past)

On a post-synaptic spike:
    apost += a_minus
    weights += apre             # potentiate (pre happened in the past)

Weights are clipped to [w_min, w_max].
Forward pass: post_current = weights @ pre_spikes.
"""
from __future__ import annotations
import math
import torch


class STDPSynapse:
    def __init__(
        self,
        num_pre: int,
        num_post: int,
        w_init: float = 0.5,
        a_plus: float = 0.01,
        a_minus: float = 0.01,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
    ) -> None:
        self.num_pre = num_pre
        self.num_post = num_post
        self.weights = torch.full((num_post, num_pre), w_init)
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.w_min = w_min
        self.w_max = w_max
        # Per-neuron traces
        self.apre = torch.zeros(num_pre)
        self.apost = torch.zeros(num_post)

    def forward(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """Compute post-synaptic input current from pre-synaptic spikes."""
        return self.weights @ pre_spikes

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        dt: float = 1.0,
    ) -> None:
        """Apply one STDP update step."""
        # Decay traces (event-driven approximation: decay every step)
        self.apre = self.apre * math.exp(-dt / self.tau_pre)
        self.apost = self.apost * math.exp(-dt / self.tau_post)

        # Pre spikes: increment apre, depress weights via apost (post in the past)
        if pre_spikes.any():
            self.apre = self.apre + self.a_plus * pre_spikes
            # weights[i, j] -= apost[i] for each pre-spike on j
            depression = torch.outer(self.apost, pre_spikes)
            self.weights = self.weights - depression

        # Post spikes: increment apost, potentiate weights via apre (pre in the past)
        if post_spikes.any():
            self.apost = self.apost + self.a_minus * post_spikes
            potentiation = torch.outer(post_spikes, self.apre)
            self.weights = self.weights + potentiation

        # Clip weights
        self.weights = torch.clamp(self.weights, self.w_min, self.w_max)
