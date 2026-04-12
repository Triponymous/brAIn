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
        # Synaptic scaling: target row sum = num_pre * mean(w_init)
        self._synaptic_scaling = False  # DISABLED — BCM metaplasticity handles stability
        self._target_w_sum = torch.tensor(num_pre * w_init, dtype=torch.float32).reshape(1)
        self._scaling_tick_counter = 0

    def forward(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """Compute post-synaptic input current from pre-synaptic spikes."""
        return self.weights @ pre_spikes

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        dt: float = 1.0,
        modulation: float = 1.0,
    ) -> None:
        """Apply one STDP update step.

        Parameters
        ----------
        pre_spikes, post_spikes
            0/1 float spike vectors for pre and post populations.
        dt
            Time step in milliseconds.
        modulation
            Multiplicative gain on the weight delta. 1.0 = unmodulated STDP.
            0.0 freezes learning. >1.0 accelerates. Used by Phase 2 modulators.
        """
        # Decay traces (event-driven approximation: decay every step)
        self.apre = self.apre * math.exp(-dt / self.tau_pre)
        self.apost = self.apost * math.exp(-dt / self.tau_post)

        # Snapshot pre-update traces. Weight changes use ONLY these snapshots,
        # not the post-increment values. This guarantees that simultaneous
        # pre+post spikes on clean traces produce zero weight change (no
        # temporal information yet) and matches the canonical pair-based STDP
        # ordering: read traces -> update weights -> increment traces.
        apre_snapshot = self.apre.clone()
        apost_snapshot = self.apost.clone()

        # ── BCM Metaplasticity: sliding threshold determines LTP vs LTD ──
        # θ_M = running average of post²  (high activity → high threshold → harder to potentiate)
        # If post > θ_M: potentiation is STRONGER (neuron is selective for this input)
        # If post < θ_M: depression is STRONGER (neuron should NOT fire for this input)
        # This creates selectivity: each neuron learns to respond to specific patterns.
        if not hasattr(self, '_bcm_theta'):
            self._bcm_theta = torch.full((self.num_post,), 0.01)  # start low → easy potentiation initially
            self._bcm_tau = 5000.0   # faster adaptation → quicker selectivity

        # Update sliding threshold: θ_M tracks average(post²)
        post_sq = post_spikes * post_spikes  # binary, so same as post_spikes
        alpha_bcm = 1.0 / self._bcm_tau
        self._bcm_theta = self._bcm_theta * (1 - alpha_bcm) + post_sq * alpha_bcm

        # BCM modulation factor per post-neuron: (post - θ_M)
        # Positive = potentiate more, Negative = depress more
        bcm_factor = (post_spikes - self._bcm_theta).unsqueeze(1)  # [num_post, 1]

        # Pre spikes: depress weights, MODULATED by BCM
        if pre_spikes.any():
            depression = torch.outer(apost_snapshot, pre_spikes)
            # BCM: neurons below threshold get MORE depression
            bcm_depression = depression * (1.0 - bcm_factor.clamp(-1, 0))
            self.weights = self.weights - modulation * bcm_depression

        # Post spikes: potentiate weights, MODULATED by BCM
        if post_spikes.any():
            potentiation = torch.outer(post_spikes, apre_snapshot)
            # BCM: neurons above threshold get MORE potentiation
            bcm_potentiation = potentiation * (1.0 + bcm_factor.clamp(0, 2))
            self.weights = self.weights + modulation * bcm_potentiation

        # Now increment traces with the new spikes (these will be read on the
        # NEXT update call).
        if pre_spikes.any():
            self.apre = self.apre + self.a_plus * pre_spikes
        if post_spikes.any():
            self.apost = self.apost + self.a_minus * post_spikes

        # Clip weights — floor at 0.01 (not 0!) so connections never fully die.
        # A fully dead connection (w=0) can NEVER recover via STDP because
        # forward() produces 0 current → post never fires → no potentiation.
        # Floor of 0.01 keeps a tiny signal flowing so learning can revive.
        self.weights = torch.clamp(self.weights, max(self.w_min, 0.01), self.w_max)

        # ── Per-row Weight Normalization ──
        # DISABLED for sensory→concept (BCM + 100:1 ratio handles stability).
        # ENABLED for WM synapses (_synaptic_scaling=True) where weights
        # would otherwise ALL saturate to max over millions of ticks.
        # Normalization preserves RELATIVE differences (selectivity) while
        # keeping total incoming weight per post-neuron stable.
        if self._synaptic_scaling:
            interval = getattr(self, '_scaling_interval', 1000)
            self._scaling_tick_counter += 1
            if self._scaling_tick_counter >= interval:
                self._scaling_tick_counter = 0
                row_sums = self.weights.sum(dim=1, keepdim=True)
                scale = self._target_w_sum / (row_sums + 1e-8)
                self.weights = self.weights * scale
                self.weights = torch.clamp(self.weights, max(self.w_min, 0.01), self.w_max)


class RSTDPSynapse(STDPSynapse):
    """Reward-modulated STDP synapse.

    Identical to STDPSynapse except `update()` takes a `reward` scalar
    instead of `modulation`, and the weight delta is multiplied by `reward`.
    Positive reward -> potentiate causal pairs. Negative reward -> depress them.
    Zero reward -> freeze.

    This is the simplest 3-factor learning rule: pre-trace x post-trace x reward.
    """

    def update(  # type: ignore[override]
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        dt: float = 1.0,
        reward: float = 0.0,
    ) -> None:
        super().update(pre_spikes, post_spikes, dt=dt, modulation=reward)
