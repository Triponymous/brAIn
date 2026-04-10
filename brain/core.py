"""Brain core: composes all regions, synapses, and modulators.

Region map:
    sensory   (LIF)        — encodes raw input current to spikes
    feature   (LIF)        — learns low-level features via STDP from sensory
    association (LIF)      — cross-modal binding (currently just from feature)
    concept   (WTA)        — sparse winner-take-all, "the" concept neurons
    wm        (WMLayer)    — recurrent context buffer over concept output
    motor     (LIF)        — output, learned via R-STDP from concept
    meta      (LIF)        — exposes modulator activity (proxy for now)

Synapse map:
    sensory  -> feature      (STDP, modulated by ACh = attention)
    feature  -> association  (STDP, modulated by ACh)
    association -> concept   (STDP, modulated by ACh)
    concept  -> wm           (STDP, slow)
    concept  -> motor        (R-STDP, reward-driven)

The tick() method:
1. Inject reward into Modulators (DA += reward)
2. Decay modulator levels by dt
3. Step sensory with external input
4. Forward sensory→feature, step feature
5. Forward feature→association, step association
6. Forward association→concept, step concept (WTA)
7. Forward concept→wm, step wm (recurrent)
8. Forward concept→motor, step motor
9. Update all STDP synapses (gated by ACh-derived modulation)
10. Update R-STDP synapse (gated by current DA = reward proxy)
11. Increment tick_count

NOTE: WM recurrent_gain=1.1 (>= threshold 1.0) is intentional. This creates
a stable attractor where WM activity persists indefinitely after each spike.
A kill mechanism (modulator-gated recurrence, lateral inhibition) is deferred
to a later phase. For Phase 2, a "loud and persistent" WM is acceptable.
"""
from __future__ import annotations
import torch

from brain.neurons import LIFLayer
from brain.wta import WTALayer
from brain.working_memory import WMLayer
from brain.synapses import STDPSynapse, RSTDPSynapse
from brain.modulators import Modulators
from brain.concept_tracker import ConceptTracker


class Brain:
    def __init__(
        self,
        num_sensory: int = 200,
        num_expansion: int = 500,  # expansion layer for pattern separation
        num_feature: int = 200,
        num_association: int = 500,
        num_concept: int = 200,
        num_wm: int = 100,
        num_motor: int = 50,
        num_meta: int = 10,
        concept_k: int | None = None,
        tau_mem: float = 100.0,
        threshold: float = 1.0,
        a_plus: float = 0.05,     # potentiation rate (5x Diehl&Cook — faster learning for streaming)
        a_minus: float = 0.0005,  # depression rate (100x weaker than potentiation)
        w_init: float = 0.3,
        w_init_std: float = 0.15,  # Gaussian init, not uniform jitter
    ) -> None:
        # Feature & Association are WTA layers too — forces sparse coding at every
        # level. Without this, all-to-all weights cause EVERY feature neuron to fire
        # on every tick (input >> threshold), destroying selectivity.
        # k values: ~10% of layer size = biologically realistic sparsity.
        # concept_k=3: allows multiple neurons to co-fire per tick.
        # Lateral inhibition learning then pushes co-firing neurons apart
        # so they specialize for different patterns. k=1 prevents this
        # because there's no co-firing to learn from.
        if concept_k is None:
            concept_k = 3
        feature_k = max(1, num_feature // 10)   # 20 out of 200
        association_k = max(1, num_association // 10)  # 50 out of 500

        # ── Expansion Layer (Cerebellar Granule Cell inspired) ──
        # Random sparse projection: 200 sensory → 500 expansion neurons.
        # Each expansion neuron connects to ~3% of sensory neurons (6 inputs).
        # Threshold = 2 (need ≥2 active inputs to fire).
        # This DECORRELATES overlapping patterns by projecting them into
        # a higher-dimensional space with sparse, random connectivity.
        # Neuroscience: the cerebellum uses exactly this trick (200 mossy fibers
        # → 100,000 granule cells) for pattern separation.
        # Expansion: 10% connectivity, threshold=1 (fire if ANY connected input is active)
        # This amplifies small differences: if typing has neuron 72 active and zoom doesn't,
        # ~50 expansion neurons connect to 72 and fire for typing but not zoom.
        self._expansion_weights = (torch.rand(num_expansion, num_sensory) < 0.10).float()
        self._expansion_threshold = 1.0
        self.num_expansion = num_expansion

        self.regions: dict[str, object] = {
            "sensory": LIFLayer(num_sensory, tau_mem=tau_mem, threshold=threshold),
            "feature": WTALayer(num_feature, k=feature_k, tau_mem=tau_mem, threshold=threshold * 0.5),
            "association": WTALayer(num_association, k=association_k, tau_mem=tau_mem, threshold=threshold * 0.5),
            "concept": WTALayer(num_concept, k=concept_k, tau_mem=tau_mem, threshold=threshold * 0.5),
            "wm": WMLayer(num_wm, recurrent_gain=0.8, tau_mem=tau_mem, threshold=threshold),
            "motor": LIFLayer(num_motor, tau_mem=tau_mem, threshold=threshold * 0.5),
            "meta": LIFLayer(num_meta, tau_mem=tau_mem, threshold=threshold),
        }

        def _make_stdp(pre: int, post: int) -> STDPSynapse:
            syn = STDPSynapse(
                num_pre=pre, num_post=post,
                w_init=0.0,  # will be overwritten below
                a_plus=a_plus, a_minus=a_minus,
            )
            # Gaussian random init: each neuron starts with different preferences
            # This breaks symmetry so WTA winners depend on input, not noise
            syn.weights = torch.clamp(
                torch.randn(post, pre) * w_init_std + w_init,
                syn.w_min, syn.w_max,
            )
            # Set synaptic scaling target to actual initial mean
            syn._target_w_sum = syn.weights.sum(dim=1, keepdim=True).mean().reshape(1)
            return syn

        def _make_rstdp(pre: int, post: int) -> RSTDPSynapse:
            syn = RSTDPSynapse(
                num_pre=pre, num_post=post,
                w_init=0.0,
                a_plus=a_plus, a_minus=a_minus,
            )
            syn.weights = torch.clamp(
                torch.randn(post, pre) * w_init_std + w_init,
                syn.w_min, syn.w_max,
            )
            syn._target_w_sum = syn.weights.sum(dim=1, keepdim=True).mean().reshape(1)
            return syn

        # ── Diehl & Cook architecture: only ONE learning synapse ──
        # The gold standard for unsupervised STDP (95% MNIST accuracy) uses
        # exactly 2 layers: Input → Excitatory (WTA). Adding intermediate
        # layers (Feature, Association) with STDP dilutes the signal and
        # prevents concept specialization (proven by benchmark failure).
        #
        # Our architecture: Sensory → [Concept via STDP] → WM/Motor
        # Feature/Association still exist for compatibility but their
        # synapses are FIXED (identity-like, no learning).
        # STDP learns on expansion→concept (not sensory→concept)
        # The expansion layer has already separated overlapping patterns
        sensory_concept = _make_stdp(num_expansion, num_concept)

        # Non-learning pass-through synapses for Feature and Association
        # (kept for compatibility with dashboard/visualization code)
        sf_fixed = STDPSynapse(num_sensory, num_feature, w_init=0.3)
        sf_fixed._synaptic_scaling = False
        fa_fixed = STDPSynapse(num_feature, num_association, w_init=0.3)
        fa_fixed._synaptic_scaling = False
        ac_fixed = STDPSynapse(num_association, num_concept, w_init=0.0)
        ac_fixed._synaptic_scaling = False

        self.synapses: dict[str, object] = {
            "sensory_feature": sf_fixed,           # fixed, no learning
            "feature_association": fa_fixed,        # fixed, no learning
            "association_concept": ac_fixed,        # fixed, no learning
            "sensory_concept": sensory_concept,     # THE learning synapse
            "concept_wm": _make_stdp(num_concept, num_wm),
            "concept_motor": _make_rstdp(num_concept, num_motor),
        }

        self.modulators = Modulators()
        self.tick_count = 0
        # Rolling spike counts for visualization (exponential decay)
        self.concept_spike_accum = torch.zeros(num_concept)
        self._spike_decay = 0.95
        # Sleep consolidation state
        self.sleep_mode = False
        self._sleep_noise_scale = 0.3  # amplitude of noise during sleep
        self._sleep_decay_exponent = 0.98
        # Concept tracker: stable cluster IDs from masked sensory fingerprints
        self.concept_tracker = ConceptTracker(expansion_dim=num_sensory)

    def enter_sleep(self) -> None:
        """Enter sleep consolidation mode. Real input is replaced with noise,
        power-law weight decay prunes weak synapses while consolidating strong ones.
        Call exit_sleep() or set sleep_mode=False to resume normal operation."""
        self.sleep_mode = True

    def exit_sleep(self) -> None:
        """Exit sleep mode and resume normal sensory processing."""
        self.sleep_mode = False

    def tick(
        self,
        input_current: torch.Tensor,
        reward: float = 0.0,
        dt: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        # ═══ SLEEP MODE: replace input with noise, apply consolidation ═══
        if self.sleep_mode:
            # Replace real input with low-amplitude Gaussian noise
            # This triggers spontaneous reactivation of learned assemblies
            input_current = torch.randn_like(input_current) * self._sleep_noise_scale

            # Sleep consolidation: DISABLED until STDP weight stability is proven.
            # Both power-law decay and multiplicative decay killed all weights.
            # Sleep mode still replaces input with noise (spontaneous replay)
            # but no weight modification during sleep.

            # Suppress modulators during sleep (calm brain)
            reward = 0.0

        # 1. Reward → DA injection
        if reward != 0.0:
            self.modulators.inject("DA", reward)

        # 2. Modulator decay
        self.modulators.tick(dt)

        # 3. Sensory
        sensory = self.regions["sensory"]
        sensory_spikes = sensory.step(input_current, dt=dt)

        # 4. EXPANSION LAYER (Cerebellar Granule Cell model)
        # Random sparse projection separates overlapping sensory patterns
        # into distinct sparse codes. No learning — fixed random weights.
        #
        # CRITICAL: mask out shared-baseline neurons (time-tonic 148-155,
        # baseline-idle 100, baseline-mic 144) that are identical across
        # all patterns. Only discriminating neurons go through expansion.
        discriminating_spikes = sensory_spikes.clone()
        discriminating_spikes[148:156] = 0  # time-tonic (same for all patterns)
        discriminating_spikes[100] = 0      # idle baseline bin
        discriminating_spikes[144] = 0      # mic RMS baseline bin
        discriminating_spikes[160:164] = 0  # activity level (derived, not unique)

        expansion_input = self._expansion_weights @ discriminating_spikes
        expansion_spikes = (expansion_input >= self._expansion_threshold).float()

        # 5. Feature (fixed pass-through for visualization)
        feature = self.regions["feature"]
        sf = self.synapses["sensory_feature"]
        feature_input = sf.forward(sensory_spikes)
        feature_spikes = feature.step(feature_input, dt=dt)

        # 6. Association (fixed pass-through for visualization)
        association = self.regions["association"]
        fa = self.synapses["feature_association"]
        association_input = fa.forward(feature_spikes)
        association_spikes = association.step(association_input, dt=dt)

        # 7. Concept (WTA) — learns from EXPANSION layer (separated patterns!)
        concept = self.regions["concept"]
        sc = self.synapses["sensory_concept"]
        concept_input = sc.forward(expansion_spikes)
        concept_spikes = concept.step(concept_input, dt=dt)

        # 8. Concept Tracker: cluster MASKED sensory signatures into stable IDs
        # Use discriminating_spikes (shared baseline removed), not expansion
        self.concept_tracker.tick(discriminating_spikes, self.tick_count)

        # Accumulate concept spikes for visualization
        self.concept_spike_accum = self.concept_spike_accum * self._spike_decay + concept_spikes.detach()

        # ═══ NOVELTY & AROUSAL DETECTION ═══
        # Two independent signals:
        # 1. Novelty: how different is current input from prediction?
        # 2. Arousal: how MUCH activity is there, regardless of novelty?
        # Stress = high arousal + high variability (not just novelty)
        # Flow = high arousal + LOW variability (steady focused work)
        if not hasattr(self, '_sensory_avg'):
            self._sensory_avg = torch.zeros_like(input_current)
            self._prev_concept_spikes = torch.zeros(concept.num_neurons)
            self._novelty_smooth = 0.0
            self._arousal_smooth = 0.0
            self._activity_smooth = 0.0

        # Update running average of sensory input (slow exponential)
        self._sensory_avg = self._sensory_avg * 0.995 + input_current * 0.005

        # Prediction error = how different is current input from average
        prediction_error = float((input_current - self._sensory_avg).abs().mean().item())

        # Concept change = how different are current concepts from recent
        concept_change = float((concept_spikes - self._prev_concept_spikes).abs().sum().item())
        self._prev_concept_spikes = concept_spikes.detach().clone()

        # Smooth novelty signal
        novelty = prediction_error * 0.5 + concept_change * 0.1
        self._novelty_smooth = self._novelty_smooth * 0.995 + novelty * 0.005

        # Arousal: total sensory drive magnitude (how much input, period)
        sensory_sum = float(sensory_spikes.sum().item())
        self._activity_smooth = self._activity_smooth * 0.99 + sensory_sum * 0.01

        # Arousal variability: how erratic is the input? (read from encoded features)
        # Neurons 76 (key variability) and 96 (mouse variability) encode this
        key_variability = float(input_current[76].item()) if input_current.shape[0] > 76 else 0
        mouse_variability = float(input_current[96].item()) if input_current.shape[0] > 96 else 0
        input_variability = (key_variability + mouse_variability) / 2.0
        self._arousal_smooth = self._arousal_smooth * 0.99 + input_variability * 0.01

        # ═══ MODULATOR INJECTION ═══
        # Simple, direct, proven approach:
        # Read the ACTUAL sensory spike count and react to CHANGES.
        # No complex smooth/relative thresholds — just absolute levels.
        #
        # Measured values (from user testing):
        #   Silence: sensory = 24-26 spikes/tick
        #   Speaking: sensory = 44-54 spikes/tick
        #   Clapping: sensory = 50-60 spikes/tick (brief)
        #
        # Equilibrium targets at 100Hz tick rate:
        #   Calm: DA~0.02, NE~0, ACh~0.03, 5HT~0.03
        #   Active (speaking): DA~0.05, ACh~0.06
        #   Surprise (clap after silence): NE spikes to ~0.1

        real_activity = sensory_sum - 8  # subtract time tonic neurons
        user_present = real_activity > 5  # more than just app + time

        # Track previous sensory level for change detection
        if not hasattr(self, '_prev_sensory_sum'):
            self._prev_sensory_sum = sensory_sum

        sensory_change = abs(sensory_sum - self._prev_sensory_sum)
        self._prev_sensory_sum = sensory_sum

        # ── 1. User is present and active (speaking, mousing, etc.) ──
        if user_present:
            # ACh: attention scales with activity level
            ach_inject = min(0.0002, real_activity * 0.000005)
            self.modulators.inject("ACh", ach_inject)

            # DA: mild curiosity when active
            self.modulators.inject("DA", 0.00003)

            # 5HT: contentment when activity is steady (low change)
            # Need eq ~0.05 during steady work. 5HT tau=1000.
            # 0.00005/tick × 1000 = 0.05 equilibrium.
            if sensory_change < 5:
                self.modulators.inject("5HT", 0.00005)

        # ── 2. Something CHANGED (sensory spike count jumped) ──
        # Target: NE reaches ~0.15 on loud clap, ~0.05 on speech start.
        # MUST NOT exceed 0.3 sustained — cap injection to prevent saturation.
        if sensory_change > 8:
            scale = min(1.0, sensory_change / 30.0)
            # Only inject if current level is below cap (prevents saturation to 1.0)
            if self.modulators.level("NE") < 0.25:
                self.modulators.inject("NE", 0.002 * scale)
            if self.modulators.level("DA") < 0.25:
                self.modulators.inject("DA", 0.002 * scale)
            self.modulators.inject("ACh", 0.001 * scale)

        if sensory_change > 20:
            scale = min(1.0, sensory_change / 40.0)
            if self.modulators.level("NE") < 0.25:
                self.modulators.inject("NE", 0.005 * scale)
            if self.modulators.level("DA") < 0.25:
                self.modulators.inject("DA", 0.003 * scale)

        # Homeostasis is now handled by weight normalization inside STDP synapse.

        # 7. WM
        wm = self.regions["wm"]
        cw = self.synapses["concept_wm"]
        wm_input = cw.forward(concept_spikes)
        wm_spikes = wm.step(wm_input, dt=dt)

        # 8. Motor
        motor = self.regions["motor"]
        cm = self.synapses["concept_motor"]
        motor_input = cm.forward(concept_spikes)
        motor_spikes = motor.step(motor_input, dt=dt)

        # 9. STDP update — ONLY on sensory→concept (the learning synapse)
        # Feature/Association synapses are fixed (no learning).
        # This is the Diehl & Cook architecture proven to work.
        ach = self.modulators.level("ACh")
        modulation = 1.0 + ach
        sc.update(expansion_spikes, concept_spikes, dt=dt, modulation=modulation)
        cw.update(concept_spikes, wm_spikes, dt=dt, modulation=0.5)

        # 10. R-STDP update for motor (gated by DA = reward proxy)
        da = self.modulators.level("DA")
        cm.update(concept_spikes, motor_spikes, dt=dt, reward=da)

        # 11. Track spike counts + vectors for dashboard (Meso visualization)
        self._last_sensory_spikes = sensory_spikes.sum().item()
        self._last_feature_spikes = feature_spikes.sum().item()
        self._last_association_spikes = association_spikes.sum().item()
        self._last_concept_spikes = concept_spikes.sum().item()
        # Per-region spike vectors (for Meso-level visualization)
        self._last_sensory_spike_vec = sensory_spikes.detach()
        self._last_feature_spike_vec = feature_spikes.detach()
        self._last_association_spike_vec = association_spikes.detach()
        self._last_concept_spike_vec = concept_spikes.detach()
        self._last_wm_spike_vec = wm_spikes.detach()
        self._last_motor_spike_vec = motor_spikes.detach()

        # 12. Tick count
        self.tick_count += 1

        return {
            "sensory": sensory_spikes,
            "feature": feature_spikes,
            "association": association_spikes,
            "concept": concept_spikes,
            "wm": wm_spikes,
            "motor": motor_spikes,
        }
