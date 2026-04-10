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
        self._synaptic_scaling = True
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

        # Pre spikes: depress weights using OLD apost (post in the past)
        if pre_spikes.any():
            depression = torch.outer(apost_snapshot, pre_spikes)
            self.weights = self.weights - modulation * depression

        # Post spikes: potentiate weights using OLD apre (pre in the past)
        if post_spikes.any():
            potentiation = torch.outer(post_spikes, apre_snapshot)
            self.weights = self.weights + modulation * potentiation

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

        # ── Weight Normalization (Diehl & Cook 2015) ──
        # Normalize per POST-neuron: each post-neuron's total incoming
        # weight (column sum in their notation, row sum in ours since
        # our weight matrix is [post, pre]) stays at target.
        #
        # KEY: This preserves RELATIVE weight differences within a row
        # (some inputs stronger than others = selectivity) while preventing
        # the total from exploding or collapsing.
        # Weight normalization: applied periodically, NOT every tick.
        # Diehl & Cook normalize after each MNIST image (~350ms = ~35 ticks).
        # For our continuous stream, normalize every 500 ticks (~5 seconds).
        # This gives STDP time to create weight differences BEFORE normalization
        # scales the row back to target (preserving relative differences).
        if self._synaptic_scaling and self._scaling_tick_counter % 500 == 0:
            row_sums = self.weights.sum(dim=1, keepdim=True)
            scale = self._target_w_sum / row_sums.clamp(min=1e-8)
            self.weights = (self.weights * scale).clamp(self.w_min, self.w_max)


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
