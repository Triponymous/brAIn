"""Minimal 2-region brain: Sensory → Feature, fully connected via STDP.

This is a stepping stone toward the full multi-region brain. It exists to
verify that LIFLayer + STDPSynapse can learn meaningful input statistics.
"""
from __future__ import annotations
import torch

from brain.neurons import LIFLayer
from brain.synapses import STDPSynapse


class TwoRegionBrain:
    def __init__(
        self,
        num_sensory: int,
        num_feature: int,
        tau_mem: float = 20.0,
        threshold: float = 1.0,
        a_plus: float = 0.01,
        a_minus: float = 0.012,
        w_init: float = 0.5,
        w_init_jitter: float = 0.0,
    ) -> None:
        self.sensory = LIFLayer(num_sensory, tau_mem=tau_mem, threshold=threshold)
        self.feature = LIFLayer(num_feature, tau_mem=tau_mem, threshold=threshold)
        self.synapse = STDPSynapse(
            num_pre=num_sensory,
            num_post=num_feature,
            w_init=w_init,
            a_plus=a_plus,
            a_minus=a_minus,
        )
        if w_init_jitter > 0.0:
            jitter = (torch.rand_like(self.synapse.weights) - 0.5) * 2 * w_init_jitter
            self.synapse.weights = torch.clamp(
                self.synapse.weights + jitter,
                self.synapse.w_min,
                self.synapse.w_max,
            )

    def tick(self, input_current: torch.Tensor) -> dict[str, torch.Tensor]:
        """Advance the brain by one timestep."""
        sensory_spikes = self.sensory.step(input_current)
        feature_input = self.synapse.forward(sensory_spikes)
        feature_spikes = self.feature.step(feature_input)
        self.synapse.update(sensory_spikes, feature_spikes)
        return {
            "sensory_spikes": sensory_spikes,
            "feature_spikes": feature_spikes,
        }
