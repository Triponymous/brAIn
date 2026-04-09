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
        num_feature: int = 100,
        num_association: int = 250,
        num_concept: int = 50,
        num_wm: int = 50,
        num_motor: int = 50,
        num_meta: int = 10,
        concept_k: int = 3,
        tau_mem: float = 20.0,
        threshold: float = 1.0,
        a_plus: float = 0.005,
        a_minus: float = 0.005,
        w_init: float = 0.4,
        w_init_jitter: float = 0.1,
    ) -> None:
        self.regions: dict[str, object] = {
            "sensory": LIFLayer(num_sensory, tau_mem=tau_mem, threshold=threshold),
            "feature": LIFLayer(num_feature, tau_mem=tau_mem, threshold=threshold * 0.3),
            "association": LIFLayer(num_association, tau_mem=tau_mem, threshold=threshold * 0.3),
            "concept": WTALayer(num_concept, k=concept_k, tau_mem=tau_mem, threshold=threshold * 0.3),
            "wm": WMLayer(num_wm, recurrent_gain=0.8, tau_mem=tau_mem, threshold=threshold),
            "motor": LIFLayer(num_motor, tau_mem=tau_mem, threshold=threshold * 0.5),
            "meta": LIFLayer(num_meta, tau_mem=tau_mem, threshold=threshold),
        }

        def _make_stdp(pre: int, post: int) -> STDPSynapse:
            syn = STDPSynapse(
                num_pre=pre, num_post=post,
                w_init=w_init, a_plus=a_plus, a_minus=a_minus,
            )
            jitter = (torch.rand_like(syn.weights) - 0.5) * 2 * w_init_jitter
            syn.weights = torch.clamp(syn.weights + jitter, syn.w_min, syn.w_max)
            return syn

        def _make_rstdp(pre: int, post: int) -> RSTDPSynapse:
            syn = RSTDPSynapse(
                num_pre=pre, num_post=post,
                w_init=w_init, a_plus=a_plus, a_minus=a_minus,
            )
            jitter = (torch.rand_like(syn.weights) - 0.5) * 2 * w_init_jitter
            syn.weights = torch.clamp(syn.weights + jitter, syn.w_min, syn.w_max)
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
        self._spike_decay = 0.99  # per-tick decay

    def tick(
        self,
        input_current: torch.Tensor,
        reward: float = 0.0,
        dt: float = 1.0,
    ) -> dict[str, torch.Tensor]:
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
        modulation = 1.0 + ach  # ACh boosts learning rate
        sf.update(sensory_spikes, feature_spikes, dt=dt, modulation=modulation)
        fa.update(feature_spikes, association_spikes, dt=dt, modulation=modulation)
        ac.update(association_spikes, concept_spikes, dt=dt, modulation=modulation)
        cw.update(concept_spikes, wm_spikes, dt=dt, modulation=0.5)  # slower

        # 10. R-STDP update for motor (gated by DA = reward proxy)
        da = self.modulators.level("DA")
        cm.update(concept_spikes, motor_spikes, dt=dt, reward=da)

        # 11. Tick count
        self.tick_count += 1

        return {
            "sensory": sensory_spikes,
            "feature": feature_spikes,
            "association": association_spikes,
            "concept": concept_spikes,
            "wm": wm_spikes,
            "motor": motor_spikes,
        }
