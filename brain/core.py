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


class Brain:
    def __init__(
        self,
        num_sensory: int = 200,
        num_feature: int = 200,
        num_association: int = 500,
        num_concept: int = 200,
        num_wm: int = 100,
        num_motor: int = 50,
        num_meta: int = 10,
        concept_k: int | None = None,  # defaults to max(1, num_concept // 40)
        tau_mem: float = 20.0,
        threshold: float = 1.0,
        a_plus: float = 0.005,
        a_minus: float = 0.0052,  # slightly asymmetric → mild selectivity, not destructive
        w_init: float = 0.3,
        w_init_std: float = 0.15,  # Gaussian init, not uniform jitter
    ) -> None:
        # Feature & Association are WTA layers too — forces sparse coding at every
        # level. Without this, all-to-all weights cause EVERY feature neuron to fire
        # on every tick (input >> threshold), destroying selectivity.
        # k values: ~10% of layer size = biologically realistic sparsity.
        # Default concept_k: ~2.5% of layer (5 out of 200)
        if concept_k is None:
            concept_k = max(1, num_concept // 40)
        feature_k = max(1, num_feature // 10)   # 20 out of 200
        association_k = max(1, num_association // 10)  # 50 out of 500

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

        self.synapses: dict[str, object] = {
            "sensory_feature": _make_stdp(num_sensory, num_feature),
            "feature_association": _make_stdp(num_feature, num_association),
            "association_concept": _make_stdp(num_association, num_concept),
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
        self._sleep_decay_exponent = 0.98  # power-law: w *= w^0.98

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

            # Power-law weight decay every 100 ticks during sleep:
            # Strong weights stay strong, weak weights get weaker.
            # w *= w^0.02 → for w=0.8: 0.8*0.8^0.02 = 0.796 (barely changes)
            #              → for w=0.1: 0.1*0.1^0.02 = 0.095 (shrinks faster)
            if self.tick_count % 100 == 0:
                for syn in self.synapses.values():
                    w = syn.weights
                    # Power-law: multiply by w^exponent. Avoid log(0).
                    decay = torch.pow(w.clamp(min=1e-6), 1.0 - self._sleep_decay_exponent)
                    syn.weights = (w * decay).clamp(syn.w_min, syn.w_max)

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

        # 4. Feature
        feature = self.regions["feature"]
        sf = self.synapses["sensory_feature"]
        feature_input = sf.forward(sensory_spikes)
        feature_spikes = feature.step(feature_input, dt=dt)

        # 5. Association
        association = self.regions["association"]
        fa = self.synapses["feature_association"]
        association_input = fa.forward(feature_spikes)
        association_spikes = association.step(association_input, dt=dt)

        # 6. Concept (WTA)
        concept = self.regions["concept"]
        ac = self.synapses["association_concept"]
        concept_input = ac.forward(association_spikes)
        concept_spikes = concept.step(concept_input, dt=dt)

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
            if sensory_change < 5:
                self.modulators.inject("5HT", 0.00003)

        # ── 2. Something CHANGED (novelty = sensory spike count jumped) ──
        if sensory_change > 10:
            # Significant change: 10+ more/fewer spikes than last tick
            # e.g. silence→speech, speech→silence, new app
            scale = min(1.0, sensory_change / 30.0)  # normalize to 0-1
            self.modulators.inject("DA", 0.001 * scale)   # curiosity
            self.modulators.inject("NE", 0.0008 * scale)  # alertness
            self.modulators.inject("ACh", 0.0005 * scale)  # attention

        if sensory_change > 20:
            # Large change: loud clap, sudden silence, app switch
            scale = min(1.0, sensory_change / 40.0)
            self.modulators.inject("NE", 0.002 * scale)  # surprise!
            self.modulators.inject("DA", 0.002 * scale)   # what was that?!
            self.modulators.inject("ACh", min(0.001, novelty * 0.02))
            self.modulators.inject("ACh", min(0.03, novelty * 0.5))

        # ═══ SYNAPTIC HOMEOSTASIS — prevents weight drift ═══
        # OLD: every 500 ticks (5s) with 20% scaling → killed all STDP learning
        # NEW: every 30,000 ticks (~5 min) with 2% scaling → gentle guardrails
        #
        # Real brains use homeostatic plasticity on timescales of hours/days.
        # Our version just prevents total weight collapse or explosion.
        if self.tick_count % 30000 == 0 and self.tick_count > 0:
            for syn_name, syn in self.synapses.items():
                w = syn.weights
                row_sums = w.sum(dim=1, keepdim=True)
                row_mean = row_sums / w.shape[1]
                # Only intervene if a row is extremely unbalanced
                # (mean < 0.05 = nearly dead, mean > 0.8 = nearly saturated)
                needs_up = (row_mean < 0.05).float()
                needs_down = (row_mean > 0.8).float()
                scale = torch.ones_like(row_sums)
                scale = scale + needs_up * 0.02    # nudge dead rows up by 2%
                scale = scale - needs_down * 0.02  # nudge saturated rows down by 2%
                syn.weights = (w * scale).clamp(syn.w_min, syn.w_max)

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

        # 9. STDP updates (gated by ACh = attention)
        ach = self.modulators.level("ACh")
        modulation = 1.0 + ach  # ACh boosts learning rate when attention is high
        sf.update(sensory_spikes, feature_spikes, dt=dt, modulation=modulation)
        fa.update(feature_spikes, association_spikes, dt=dt, modulation=modulation)
        ac.update(association_spikes, concept_spikes, dt=dt, modulation=modulation)
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
