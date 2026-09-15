"""Brain core: composes all regions, synapses, and modulators.

Architecture (Diehl & Cook 2015, extended):
    sensory   (LIF, 200)     — encodes raw desktop input to spikes
    expansion (fixed random)  — cerebellar pattern separation (200→500)
    concept   (WTA, 1000)    — sparse winner-take-all concept neurons (STDP)
    wm        (WMLayer, 100) — recurrent short-term memory buffer
                                HOLDS recently active concepts via self-excitation
                                FEEDS BACK to concept layer (temporal context)

Synapse map:
    sensory  -> expansion    (fixed random 10% connectivity, no learning)
    expansion -> concept     (STDP + BCM metaplasticity, three-factor rule)
    concept  -> wm           (STDP, slow learning, modulation=0.5)
    wm       -> concept      (STDP, feedback — recent WM activity biases recognition)

Neuromodulation of SNN dynamics:
    NE  → sensory threshold   (arousal → hypervigilance)
    5HT → expansion threshold (contentment → broader matching)
    ACh → lateral inhibition  (attention → sharper competition)
    DA  → STDP learning rate  (novelty → faster learning)

The tick() method:
1. Inject reward → DA
2. Decay modulator levels
3. Step sensory (NE modulates threshold)
4. Expansion layer (5HT modulates breadth)
5. WM feedback → concept input (temporal context)
6. Step concept WTA (ACh modulates inhibition)
7. ConceptTracker clustering
8. Emergent modulator driver (DA/NE/ACh/5HT from prediction error)
9. Step WM (receives concept spikes, recurrent self-excitation)
10. STDP updates (three-factor: pre × post × DA+ACh)
11. Increment tick_count
"""
from __future__ import annotations
import torch

from brain.neurons import LIFLayer
from brain.wta import WTALayer
from brain.working_memory import WMLayer
from brain.synapses import STDPSynapse
from brain.modulators import Modulators
from brain.concept_tracker import ConceptTracker


class Brain:
    def __init__(
        self,
        num_sensory: int = 200,
        num_expansion: int = 500,
        num_concept: int = 200,
        num_wm: int = 100,
        concept_k: int | None = None,
        tau_mem: float = 100.0,
        threshold: float = 1.0,
        a_plus: float = 0.03,      # potentiation rate (30:1 ratio — balances learning vs saturation)
        a_minus: float = 0.001,    # depression rate (stronger than before to counter potentiation)
        w_init: float = 0.3,
        w_init_std: float = 0.15,
        # Legacy params — accepted but ignored (backward compat with config.json)
        num_feature: int = 200,
        num_association: int = 500,
        num_motor: int = 50,
        num_meta: int = 10,
    ) -> None:
        if concept_k is None:
            concept_k = 3

        # ── Expansion Layer (Cerebellar Granule Cell inspired) ──
        # Random sparse projection: 200 sensory → 500 expansion neurons.
        # 10% connectivity, threshold=1 (fire if ANY connected input active).
        # This DECORRELATES overlapping patterns for STDP to learn from.
        self._expansion_weights = (torch.rand(num_expansion, num_sensory) < 0.10).float()
        self._expansion_threshold = 1.0
        self.num_expansion = num_expansion

        self.regions: dict[str, object] = {
            "sensory": LIFLayer(num_sensory, tau_mem=tau_mem, threshold=threshold),
            "concept": WTALayer(num_concept, k=concept_k, tau_mem=tau_mem, threshold=threshold * 0.5),
            "wm": WMLayer(num_wm, recurrent_gain=1.05, tau_mem=tau_mem, threshold=threshold),
        }

        def _make_stdp(pre: int, post: int) -> STDPSynapse:
            syn = STDPSynapse(
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

        # ── Synapses ──
        # expansion→concept: THE main learning synapse (STDP + BCM, 20:1 ratio)
        # Synaptic scaling enabled: normalizes per-row weight sums every 500 ticks
        # to prevent saturation (59.7% weights at max without scaling).
        sensory_concept = _make_stdp(num_expansion, num_concept)
        sensory_concept._synaptic_scaling = True
        sensory_concept._scaling_interval = 2000  # normalize every 2000 ticks (20s) — less aggressive

        # concept→wm: BALANCED ratio (10:1, not 100:1!) + synaptic scaling.
        # Without this, all weights saturate to max over millions of ticks
        # because WM neurons fire constantly (recurrent) and potentiation
        # always dominates depression. With 10:1 ratio + scaling, only the
        # MOST active concept→WM connections stay strong.
        concept_wm = STDPSynapse(
            num_pre=num_concept, num_post=num_wm,
            w_init=0.0,
            a_plus=a_plus * 0.1,      # 0.005 — slow learning
            a_minus=a_minus * 50.0,   # 0.025 — 5:1 ratio (much more balanced)
        )
        concept_wm.weights = torch.clamp(
            torch.randn(num_wm, num_concept) * w_init_std + w_init,
            concept_wm.w_min, concept_wm.w_max,
        )
        concept_wm._synaptic_scaling = True  # keep total incoming weight stable
        concept_wm._target_w_sum = concept_wm.weights.sum(dim=1, keepdim=True).mean().reshape(1)
        concept_wm._scaling_interval = 100

        # wm→concept: FEEDBACK — WM biases concept recognition with temporal context.
        # Same balanced learning: 5:1 ratio + synaptic scaling.
        wm_concept_feedback = STDPSynapse(
            num_pre=num_wm, num_post=num_concept,
            w_init=0.0,
            a_plus=a_plus * 0.1,      # 0.005
            a_minus=a_minus * 50.0,   # 0.025
        )
        wm_concept_feedback.weights = torch.clamp(
            torch.randn(num_concept, num_wm) * 0.05 + 0.1,
            wm_concept_feedback.w_min, wm_concept_feedback.w_max,
        )
        wm_concept_feedback._synaptic_scaling = True
        wm_concept_feedback._target_w_sum = (
            wm_concept_feedback.weights.sum(dim=1, keepdim=True).mean().reshape(1)
        )
        wm_concept_feedback._scaling_interval = 100

        self.synapses: dict[str, object] = {
            "sensory_concept": sensory_concept,
            "concept_wm": concept_wm,
            "wm_concept": wm_concept_feedback,
        }

        self.modulators = Modulators()
        # ═══ EMERGENT-MODULATOR GAINS — the ONLY tuning surface ═══
        # All DERIVED from measured prediction_error per regime via the steady-state
        # law L* = inject / (1 - exp(-1/tau)) + the classifier bands — NOT by analogy
        # to the old scripted increments. See
        # docs/plans/2026-06-04-phase1-driver-v2-calibrated.md §2.
        self._r_fast = 0.1        # pe_fast EMA rate  (~10-tick memory)
        self._r_slow = 0.02       # pe_slow EMA rate  (~50-tick memory)
        self._k_ne = 0.0035       # surprise        -> NE   (arousal)
        self._k_da = 0.022        # progress*prec.  -> DA   (reward; noisy-TV-safe)
        self._k_ach = 0.03        # progress*prec.  -> ACh  (attention ON NOVELTY)
        self._k_unc = 2000.0      # expected-uncertainty penalty (noise exclusion)
        self._k_5ht = 7.1e-5      # relative calm   -> 5HT  (contentment)
        self._k_calm_s = 200.0    # 5HT surprise sensitivity
        self._k_calm_v = 2000.0   # 5HT variance sensitivity
        self._ne_cap = 0.3        # NE level-cap rail (bounds steady-state level)
        self._da_cap = 0.3        # DA level-cap rail (bounds level + STDP runaway)
        self._ach_clamp = 0.0015  # ACh per-tick rail
        self._sht_setpoint = 0.04 # PHASE-2 SEAM ONLY (interoception). NOT a P1 gate.
        self.tick_count = 0
        self.concept_spike_accum = torch.zeros(num_concept)
        self._spike_decay = 0.95
        self.sleep_mode = False
        self._sleep_noise_scale = 0.3
        self._sleep_decay_exponent = 0.98
        # Track number of WM-active neurons for diagnostics
        self._last_wm_spikes = 0
        # ConceptTracker uses SENSORY spikes (not expansion) because it needs
        # ALL sensors (time, idle, activity) to distinguish real-world situations.
        # Expansion layer masks these for STDP, but tracker needs everything.
        self.concept_tracker = ConceptTracker(input_dim=num_sensory)

    def enter_sleep(self) -> None:
        self.sleep_mode = True

    def exit_sleep(self) -> None:
        self.sleep_mode = False

    def tick(
        self,
        input_current: torch.Tensor,
        reward: float = 0.0,
        dt: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        # ═══ SLEEP MODE ═══
        if self.sleep_mode:
            input_current = torch.randn_like(input_current) * self._sleep_noise_scale
            reward = 0.0

        # 1. Reward → DA
        if reward != 0.0:
            self.modulators.inject("DA", reward)

        # 2. Modulator decay
        self.modulators.tick(dt)

        # ═══ NEUROMODULATION OF SNN DYNAMICS ═══
        da = self.modulators.level("DA")
        ne = self.modulators.level("NE")
        ach = self.modulators.level("ACh")
        sht = self.modulators.level("5HT")

        # 3. Sensory — NE modulates threshold (arousal → hypervigilance)
        sensory = self.regions["sensory"]
        ne_threshold_mod = 1.0 - ne * 2.0
        original_threshold = sensory.threshold
        sensory.threshold = max(0.3, original_threshold * ne_threshold_mod)
        sensory_spikes = sensory.step(input_current, dt=dt)
        sensory.threshold = original_threshold

        # 4. Expansion layer — 5HT modulates pattern breadth
        discriminating_spikes = sensory_spikes.clone()
        n = len(discriminating_spikes)
        if n >= 164:
            discriminating_spikes[148:156] = 0  # time-tonic (masked for expansion)
            discriminating_spikes[100] = 0
            discriminating_spikes[144] = 0
            discriminating_spikes[160:164] = 0

        sht_threshold_mod = 1.0 + sht * 5.0
        expansion_input = self._expansion_weights @ discriminating_spikes
        expansion_spikes = (expansion_input >= self._expansion_threshold * sht_threshold_mod).float()

        # 5. WM feedback → concept (temporal context from recent memory)
        wm = self.regions["wm"]
        wm_feedback_syn = self.synapses["wm_concept"]
        # IMPORTANT: snapshot BEFORE wm.step() overwrites last_spikes
        wm_pre_spikes = wm.last_spikes.clone()
        wm_feedback = wm_feedback_syn.forward(wm_pre_spikes)

        # 6. Concept (WTA) — ACh modulates lateral inhibition
        concept = self.regions["concept"]
        if hasattr(concept, '_lateral_rate'):
            concept._lateral_rate = 0.003 * (1.0 + ach * 10.0)

        sc = self.synapses["sensory_concept"]
        concept_input = sc.forward(expansion_spikes) + wm_feedback * 0.3  # WM bias is subtle
        concept_spikes = concept.step(concept_input, dt=dt)

        # 7. ConceptTracker: clusters BEHAVIORAL patterns.
        # Masking strategy: remove volatile/noisy signals, keep stable behavioral ones.
        #
        # KEEP (stable behavioral signals):
        #   0-39   foreground app (5 neurons, identifies WHAT app)
        #   60-79  keystroke bins + rhythm (typing behavior)
        #   80-99  mouse bins + rhythm (clicking behavior)
        #   101-107 idle bins (away/present)
        #   108-111 pause type (micro/thinking/break/away)
        #   145-147 mic RMS bins (loud/quiet — simple loudness)
        #
        # MASK (volatile/noisy signals):
        #   40-59  background apps (change constantly, not behavioral)
        #   100    idle baseline (always on)
        #   112-143 mic mel-spectrogram (32 dims! music on/off flips 20+ neurons)
        #   144    mic RMS summary (redundant with 145-147)
        #   148-155 time-tonic (same behavior at different times = same cluster)
        #   156-159 app switch rate (transient, not stable)
        #   160-163 activity level (redundant with keystroke/mouse bins)
        tracker_spikes = sensory_spikes.clone()
        if len(tracker_spikes) >= 164:
            tracker_spikes[40:60] = 0     # background apps
            tracker_spikes[100] = 0       # idle baseline
            tracker_spikes[112:145] = 0   # mic mel-spectrogram (too granular!)
            tracker_spikes[148:164] = 0   # time-tonic + switch rate + activity level
        self.concept_tracker.tick(tracker_spikes, self.tick_count)

        # Accumulate concept spikes for visualization
        self.concept_spike_accum = self.concept_spike_accum * self._spike_decay + concept_spikes.detach()

        # ═══ NOVELTY & AROUSAL DETECTION ═══
        if not hasattr(self, '_sensory_avg'):
            self._sensory_avg = torch.zeros_like(input_current)
        if not hasattr(self, '_pe_fast'):
            # Separate guard: a brain loaded from a pre-driver checkpoint has
            # _sensory_avg but not the EMAs — init them WITHOUT clobbering the
            # learned _sensory_avg above.
            self._pe_fast = 0.0   # fast surprise EMA  (current regime)
            self._pe_slow = 0.0   # slow surprise EMA  (lags; exposes error-drop)
            self._pe_var = 0.0    # surprise variance EMA (expected uncertainty)

        self._sensory_avg = self._sensory_avg * 0.995 + input_current * 0.005
        prediction_error = float((input_current - self._sensory_avg).abs().mean().item())
        sensory_sum = float(sensory_spikes.sum().item())   # moved up from old injection block

        # ═══ EMERGENT MODULATOR DRIVER (active inference / free energy) ═══
        # Four modulators EMERGE from prediction error — no if-statement emotions.
        # NE   <- surprise above own baseline            (unexpected uncertainty)
        # DA   <- learning PROGRESS, precision-gated     (reward; noisy-TV-safe)
        # ACh  <- learning PROGRESS, precision-gated     (attention ON NOVELTY)
        # 5HT  <- relative low-surprise (scale-free)     (contentment)
        # Gains DERIVED from measured pe; see
        # docs/plans/2026-06-04-phase1-driver-v2-calibrated.md.
        pe = prediction_error
        self._pe_fast = self._pe_fast * (1 - self._r_fast) + pe * self._r_fast
        self._pe_slow = self._pe_slow * (1 - self._r_slow) + pe * self._r_slow
        self._pe_var = self._pe_var * 0.99 + (pe - self._pe_fast) ** 2 * 0.01

        real_activity = max(0.0, sensory_sum - 8.0)          # ambient floor ~8 spikes

        # precision = inverse expected-uncertainty: the NOISE-EXCLUSION gate shared
        # by the two learning modulators (DA, ACh). High pe_var (pure noise) -> ~0.
        precision = 1.0 / (1.0 + self._pe_var * self._k_unc)

        # NE: surprise relative to this regime's own baseline (self-calibrating),
        # with a level-cap rail so sustained surprise can't pin NE -> 1.
        surprise = max(0.0, pe - self._pe_slow)
        if self.modulators.level("NE") < self._ne_cap:
            self.modulators.inject("NE", min(0.02, surprise * self._k_ne))

        # progress = error dropping = the model is COMPRESSING a new regime.
        # Precision-gated: noisy-TV pe wobble leaks raw progress, but precision ~0
        # there excludes it. Familiar-converged and pure-noise both -> ~0.
        progress = max(0.0, self._pe_slow - self._pe_fast)
        learning = progress * precision

        # DA: learning-progress reward, level-cap rail (also bounds STDP rate).
        if self.modulators.level("DA") < self._da_cap:
            self.modulators.inject("DA", min(0.02, learning * self._k_da))

        # ACh: ATTENTION ON NOVELTY (Yu & Dayan 2005: ACh = expected uncertainty).
        # High while actively learning a structured new regime, falling as it
        # becomes familiar (then 5HT/content takes over). Same scale-free,
        # noise-excluding signal as DA — the only one that separates novel from
        # BOTH familiar and pure noise. Downstream (lateral inhibition, STDP boost)
        # WANTS ACh high during learning. Gate: an absent pet attends to nothing.
        if real_activity > 0.0:
            self.modulators.inject("ACh", min(self._ach_clamp, learning * self._k_ach))

        # 5HT: RELATIVE, scale-free contentment — surprise LOW vs own baseline AND
        # variance low. Steady familiar typing (abs pe high but surprise ~0) -> high
        # 5HT -> content, even though abs pe >> any setpoint (the v1 absolute-gate
        # fix). Baseline 5HT = 0 (blank slate): contentment is continuously earned.
        low_surprise = 1.0 / (1.0 + surprise * self._k_calm_s)
        low_var = 1.0 / (1.0 + self._pe_var * self._k_calm_v)
        calm = low_surprise * low_var
        if real_activity > 0.0:
            self.modulators.inject("5HT", min(0.0002, calm * self._k_5ht))

        # 8. WM — receives concept spikes, recurrent self-excitation holds them
        cw = self.synapses["concept_wm"]
        wm_input = cw.forward(concept_spikes)
        wm_spikes = wm.step(wm_input, dt=dt)

        # 9. STDP — THREE-FACTOR LEARNING RULE (pre × post × neuromodulator)
        da_now = self.modulators.level("DA")
        ach_now = self.modulators.level("ACh")
        learning_modulation = 1.0 + da_now * 5.0 + ach_now * 3.0

        sc.update(expansion_spikes, concept_spikes, dt=dt, modulation=learning_modulation)
        cw.update(concept_spikes, wm_spikes, dt=dt, modulation=0.5)
        # WM→Concept feedback STDP: learns which WM states predict which concepts
        # Uses pre-step snapshot (wm_pre_spikes) for correct STDP timing
        wm_feedback_syn.update(wm_pre_spikes, concept_spikes, dt=dt, modulation=0.3)

        # 10. Track spike counts for dashboard
        self._last_sensory_spikes = sensory_spikes.sum().item()
        self._last_concept_spikes = concept_spikes.sum().item()
        self._last_wm_spikes = wm_spikes.sum().item()
        # Per-region spike vectors (for Meso visualization)
        self._last_sensory_spike_vec = sensory_spikes.detach()
        self._last_concept_spike_vec = concept_spikes.detach()
        self._last_wm_spike_vec = wm_spikes.detach()

        # 11. Tick count
        self.tick_count += 1

        return {
            "sensory": sensory_spikes,
            "expansion": expansion_spikes,
            "concept": concept_spikes,
            "wm": wm_spikes,
        }
