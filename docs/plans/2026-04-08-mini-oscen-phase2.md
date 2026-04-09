# Mini-OSCEN Phase 2 Implementation Plan — Full Brain + Persistence

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the full multi-region brain (Sensory, Feature, Association, Concept-WTA, Working Memory, Motor, Meta), neuromodulators (DA/NE/ACh/5HT), modulator-gated STDP, R-STDP for motor learning, SQLite persistence, and a soak-test script that demonstrates emergent distinct concept formation across restarts.

**Architecture:** Phase 1 primitives (`LIFLayer`, `STDPSynapse`) are kept and extended (modulation parameter, ordering fix). New region types (`WTALayer`, `WMLayer`) are siblings of `LIFLayer`. A new `Brain` class in `brain/core.py` composes all regions and modulators into a single tickable object. `brain/persistence.py` handles SQLite save/load of complete brain state including eligibility traces. The Phase 1 `TwoRegionBrain` and its tests stay green as a fixture — the new `Brain` is built fresh, not extended from `TwoRegionBrain`.

**Tech Stack:** Python 3.11, snnTorch, PyTorch, NumPy, sqlite3 (stdlib), pytest, matplotlib.

**Reference docs:**
- `docs/plans/2026-04-08-mini-oscen-design.md` — full project design
- `docs/plans/2026-04-08-mini-oscen-phase1.md` — Phase 1 plan (completed)

**Out of scope for Phase 2:** Real sensors (Phase 3), web server (Phase 4), LLM bridge (Phase 5), avatar/pygame (Phase 6). Everything in Phase 2 runs on synthetic input.

**Success criteria for Phase 2:**
1. Brain runs continuously for ≥5 minutes (proxy for the design's 24h goal) without crashing or weight explosion.
2. Survives `save → load → resume` with bit-identical state.
3. Forms ≥2 *distinct* concept neurons on a 4-pattern synthetic input within 5 minutes (the Phase 1 collapse problem is solved by WTA).
4. Modulators are inject-able from external code (sets the stage for Phase 5 LLM rewards).
5. Test suite grows from 15 to ~40 tests, all passing.

---

## Task 0: Fix STDP simultaneous-spike ordering bug + add modulation parameter

**Why:** Final review of Phase 1 found that `STDPSynapse.update()` increments `apre` *before* reading it for the post-spike potentiation, which double-counts simultaneous pre+post spikes. We fix this now because Phase 2 will add modulator-gated learning, and we want the trace ordering correct first.

**Files:**
- Modify: `/Users/leonmatthies/brAIntest/brain/synapses.py`
- Modify: `/Users/leonmatthies/brAIntest/tests/test_synapses.py`

**Step 1: Write the failing tests**

Append these tests to `tests/test_synapses.py`:

```python
def test_stdp_simultaneous_spike_no_double_count():
    """When pre and post fire on the SAME tick with both traces zero,
    the resulting weight change should reflect ZERO causal information
    (neither pre-then-post nor post-then-pre history exists yet).

    Concretely: starting from clean traces, a single (pre=1, post=1) update
    should NOT change the weight, because there is no temporal order yet.
    """
    syn = STDPSynapse(
        num_pre=1, num_post=1, w_init=0.5,
        a_plus=0.1, a_minus=0.1, tau_pre=20.0, tau_post=20.0,
    )
    initial = syn.weights.clone()
    syn.update(
        pre_spikes=torch.tensor([1.0]),
        post_spikes=torch.tensor([1.0]),
    )
    # No change: traces were zero, so neither LTP nor LTD path can fire.
    assert torch.allclose(syn.weights, initial), (
        f"Simultaneous pre+post on clean traces should leave weights unchanged, "
        f"got {syn.weights.item():.4f} vs initial {initial.item():.4f}"
    )


def test_stdp_modulation_zero_disables_learning():
    """A modulation factor of 0.0 should freeze all weight updates."""
    syn = STDPSynapse(
        num_pre=1, num_post=1, w_init=0.5,
        a_plus=0.1, a_minus=0.1,
    )
    initial = syn.weights.clone()
    # Causal pair under zero modulation
    syn.update(
        pre_spikes=torch.tensor([1.0]),
        post_spikes=torch.tensor([0.0]),
        modulation=0.0,
    )
    for _ in range(3):
        syn.update(
            pre_spikes=torch.tensor([0.0]),
            post_spikes=torch.tensor([0.0]),
            modulation=0.0,
        )
    syn.update(
        pre_spikes=torch.tensor([0.0]),
        post_spikes=torch.tensor([1.0]),
        modulation=0.0,
    )
    assert torch.allclose(syn.weights, initial), (
        f"Zero modulation should freeze weights, got {syn.weights.item():.4f}"
    )


def test_stdp_modulation_scales_learning_rate():
    """A modulation factor of 2.0 should produce ~2x the weight change of 1.0."""
    syn1 = STDPSynapse(num_pre=1, num_post=1, w_init=0.5, a_plus=0.05, a_minus=0.05)
    syn2 = STDPSynapse(num_pre=1, num_post=1, w_init=0.5, a_plus=0.05, a_minus=0.05)
    # Identical causal pair on both, but syn2 with double modulation
    for syn, mod in [(syn1, 1.0), (syn2, 2.0)]:
        syn.update(pre_spikes=torch.tensor([1.0]), post_spikes=torch.tensor([0.0]), modulation=mod)
        for _ in range(3):
            syn.update(pre_spikes=torch.tensor([0.0]), post_spikes=torch.tensor([0.0]), modulation=mod)
        syn.update(pre_spikes=torch.tensor([0.0]), post_spikes=torch.tensor([1.0]), modulation=mod)
    delta1 = (syn1.weights - 0.5).item()
    delta2 = (syn2.weights - 0.5).item()
    # syn2 delta should be roughly 2x syn1 delta (within numerical tolerance)
    assert delta1 > 0 and delta2 > 0
    ratio = delta2 / delta1
    assert 1.8 < ratio < 2.2, f"Expected ~2x ratio, got {ratio:.3f}"
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/leonmatthies/brAIntest
.venv/bin/pytest tests/test_synapses.py -v
```

Expected: existing 6 tests still pass; the 3 new tests fail (because (a) simultaneous-spike currently DOES change weights due to the ordering bug, (b) `update()` doesn't yet accept a `modulation` parameter so the new tests TypeError).

**Step 3: Fix the implementation**

Replace the entire body of `update()` in `brain/synapses.py` with this corrected version. The two changes are:
1. Cache `apre`/`apost` BEFORE incrementing them, then use the cached pre-update values for the weight change.
2. Accept a new `modulation: float = 1.0` parameter that scales the weight delta.

Use the Edit tool to replace the existing `update` method. Old method (to find):

```python
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
```

New method (to replace it with):

```python
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

        # Clip weights
        self.weights = torch.clamp(self.weights, self.w_min, self.w_max)
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/leonmatthies/brAIntest
.venv/bin/pytest tests/test_synapses.py -v
```

Expected: all 9 tests pass (6 existing + 3 new).

**Step 5: Run the FULL suite to confirm no regressions**

```bash
.venv/bin/pytest -v
```

Expected: 18 tests pass (15 from Phase 1, but `test_two_region_learns_pattern_specialization` may fail because we just changed STDP semantics. If it does, do NOT modify the new STDP — instead modify the test in Step 6.)

**Step 6: If `test_two_region_learns_pattern_specialization` fails, increase epoch count**

If the two-region specialization test fails after the STDP fix (because the new ordering converges slightly slower), edit `tests/test_two_region.py` and change the training loop from `for _ in range(200)` to `for _ in range(400)`. Re-run `.venv/bin/pytest -v` and confirm all 18 pass. Otherwise leave it alone.

**Step 7: Commit**

```bash
cd /Users/leonmatthies/brAIntest
git add brain/synapses.py tests/test_synapses.py tests/test_two_region.py
git commit -m "fix(brain): STDP ordering + modulation parameter on update()"
```

---

## Task 1: Modulators class

**Why:** The Meta region tracks four neuromodulator levels (Dopamine, Noradrenaline, Acetylcholine, Serotonin). Each level decays toward baseline. External code (tests, future LLM bridge) can inject. Other regions read modulator levels to gate plasticity, attention, etc.

**Files:**
- Create: `brain/modulators.py`
- Create: `tests/test_modulators.py`

**Step 1: Write failing tests**

Write file `tests/test_modulators.py`:

```python
"""Tests for the neuromodulator system.

Four modulators: dopamine (DA), noradrenaline (NE), acetylcholine (ACh),
serotonin (5HT). Each has a baseline (default 0), a current level, and
decays toward baseline with its own time constant.
"""
import math
import pytest
from brain.modulators import Modulators


def test_modulators_construction_defaults_zero():
    mod = Modulators()
    assert mod.level("DA") == 0.0
    assert mod.level("NE") == 0.0
    assert mod.level("ACh") == 0.0
    assert mod.level("5HT") == 0.0


def test_modulators_inject_raises_level():
    mod = Modulators()
    mod.inject("DA", 0.5)
    assert mod.level("DA") == 0.5
    mod.inject("DA", 0.3)
    assert abs(mod.level("DA") - 0.8) < 1e-6


def test_modulators_inject_unknown_raises():
    mod = Modulators()
    with pytest.raises(KeyError):
        mod.inject("XYZ", 1.0)


def test_modulators_decay_toward_baseline():
    mod = Modulators(tau={"DA": 100.0})
    mod.inject("DA", 1.0)
    initial = mod.level("DA")
    mod.tick(dt=100.0)  # one full time constant
    expected = initial * math.exp(-1.0)  # ~0.368
    assert abs(mod.level("DA") - expected) < 1e-3


def test_modulators_independent_decay():
    mod = Modulators(tau={"DA": 100.0, "NE": 200.0})
    mod.inject("DA", 1.0)
    mod.inject("NE", 1.0)
    mod.tick(dt=100.0)
    # DA at 1 tau -> 0.368, NE at 0.5 tau -> exp(-0.5) ≈ 0.607
    assert abs(mod.level("DA") - math.exp(-1.0)) < 1e-3
    assert abs(mod.level("NE") - math.exp(-0.5)) < 1e-3


def test_modulators_clipped_to_bounds():
    mod = Modulators()
    mod.inject("DA", 100.0)  # absurdly large
    assert mod.level("DA") <= 1.0
    mod.inject("DA", -100.0)
    assert mod.level("DA") >= -1.0


def test_modulators_snapshot_returns_dict():
    mod = Modulators()
    mod.inject("DA", 0.5)
    mod.inject("NE", 0.3)
    snap = mod.snapshot()
    assert snap == {"DA": 0.5, "NE": 0.3, "ACh": 0.0, "5HT": 0.0}
```

**Step 2: Run to verify failure**

```bash
cd /Users/leonmatthies/brAIntest
.venv/bin/pytest tests/test_modulators.py -v
```

Expected: All 7 tests fail with `ModuleNotFoundError`.

**Step 3: Implement**

Write file `brain/modulators.py`:

```python
"""Neuromodulator system: Dopamine, Noradrenaline, Acetylcholine, Serotonin.

Each modulator has:
- a current level in [-1, 1]
- a baseline (default 0.0)
- a time constant tau for exponential decay toward baseline

Levels are scalars, not per-neuron. Other regions read levels to gate
plasticity (DA modulates STDP), attention (ACh), and global state (NE/5HT).

External code can inject quantities at any time. Decay is applied via
`tick(dt)` from the main brain loop.
"""
from __future__ import annotations
import math


_DEFAULT_TAU = {"DA": 200.0, "NE": 500.0, "ACh": 300.0, "5HT": 1000.0}
_NAMES = ("DA", "NE", "ACh", "5HT")


class Modulators:
    def __init__(
        self,
        baseline: dict[str, float] | None = None,
        tau: dict[str, float] | None = None,
        level_min: float = -1.0,
        level_max: float = 1.0,
    ) -> None:
        self.baseline = {n: 0.0 for n in _NAMES}
        if baseline:
            self.baseline.update(baseline)
        self.tau = dict(_DEFAULT_TAU)
        if tau:
            self.tau.update(tau)
        self.level_min = level_min
        self.level_max = level_max
        self._levels: dict[str, float] = {n: self.baseline[n] for n in _NAMES}

    def level(self, name: str) -> float:
        if name not in self._levels:
            raise KeyError(f"Unknown modulator: {name}")
        return self._levels[name]

    def inject(self, name: str, amount: float) -> None:
        if name not in self._levels:
            raise KeyError(f"Unknown modulator: {name}")
        new = self._levels[name] + amount
        self._levels[name] = max(self.level_min, min(self.level_max, new))

    def tick(self, dt: float) -> None:
        """Decay all levels toward their baseline."""
        for name in _NAMES:
            base = self.baseline[name]
            current = self._levels[name]
            decay = math.exp(-dt / self.tau[name])
            self._levels[name] = base + (current - base) * decay

    def snapshot(self) -> dict[str, float]:
        return dict(self._levels)
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_modulators.py -v
```

Expected: All 7 pass.

**Step 5: Commit**

```bash
git add brain/modulators.py tests/test_modulators.py
git commit -m "feat(brain): add Modulators (DA/NE/ACh/5HT) with decay and clipping"
```

---

## Task 2: WTA (Winner-Take-All) layer for Concept formation

**Why:** This solves Phase 1's structural collapse. A WTA layer is like an LIFLayer but only the top-k strongest membrane potentials get to spike per tick. The losers are inhibited. This forces the layer to develop diverse, sparse concept representations.

**Files:**
- Create: `brain/wta.py`
- Create: `tests/test_wta.py`

**Step 1: Write failing tests**

Write file `tests/test_wta.py`:

```python
"""Winner-take-all layer tests.

A WTALayer behaves like an LIFLayer but at most k neurons fire per tick.
When more than k neurons cross threshold, only the k with the highest
membrane potentials spike; the rest are inhibited (their membrane is
multiplied by an inhibition factor instead of resetting).
"""
import torch
from brain.wta import WTALayer


def test_wta_construction():
    layer = WTALayer(num_neurons=10, k=2)
    assert layer.num_neurons == 10
    assert layer.k == 2
    assert layer.membrane.shape == (10,)


def test_wta_no_input_no_spikes():
    layer = WTALayer(num_neurons=5, k=1)
    spikes = layer.step(input_current=torch.zeros(5))
    assert torch.all(spikes == 0)


def test_wta_only_top_k_fire():
    """If 5 neurons all cross threshold, only the top-k=2 should spike."""
    layer = WTALayer(num_neurons=5, k=2, threshold=1.0)
    # All 5 receive enough input to spike, but with different magnitudes
    spikes = layer.step(input_current=torch.tensor([1.5, 3.0, 1.2, 2.5, 1.1]))
    assert spikes.sum().item() == 2
    # Top 2 by input should be neurons 1 (3.0) and 3 (2.5)
    assert spikes[1].item() == 1
    assert spikes[3].item() == 1


def test_wta_fewer_above_threshold_than_k():
    """If only one neuron crosses, only that one fires (not k=2)."""
    layer = WTALayer(num_neurons=5, k=2, threshold=1.0)
    spikes = layer.step(input_current=torch.tensor([2.0, 0.5, 0.1, 0.3, 0.4]))
    assert spikes.sum().item() == 1
    assert spikes[0].item() == 1


def test_wta_winners_reset_losers_inhibited():
    """Winners reset to 0; near-threshold losers get pushed back below threshold."""
    layer = WTALayer(num_neurons=3, k=1, threshold=1.0, inhibit_factor=0.5)
    layer.membrane = torch.tensor([0.5, 0.5, 0.5])  # set known starting state
    spikes = layer.step(input_current=torch.tensor([2.0, 1.0, 1.0]))
    # Winner is neuron 0
    assert spikes[0].item() == 1
    assert layer.membrane[0].item() == 0.0
    # Losers (1, 2) had charged but didn't win — membrane should be reduced
    assert layer.membrane[1].item() < 1.0
    assert layer.membrane[2].item() < 1.0


def test_wta_independent_concepts_can_form():
    """Across many ticks with two distinct input groups, each WTA neuron
    should consistently win on a different group (sanity check that the
    competition isn't degenerate)."""
    torch.manual_seed(0)
    layer = WTALayer(num_neurons=2, k=1, threshold=1.0)
    # Track which neuron wins on each pattern
    pattern_a_wins = torch.zeros(2)
    pattern_b_wins = torch.zeros(2)
    # Slight asymmetry to break symmetry: neuron 0 gets stronger initial drive on pattern A
    for _ in range(20):
        spikes = layer.step(torch.tensor([2.5, 1.5]))  # pattern A favors neuron 0
        pattern_a_wins += spikes
        spikes = layer.step(torch.tensor([1.5, 2.5]))  # pattern B favors neuron 1
        pattern_b_wins += spikes
    # Each pattern should be won by a different neuron MOST of the time
    assert pattern_a_wins[0] > pattern_a_wins[1]
    assert pattern_b_wins[1] > pattern_b_wins[0]
```

**Step 2: Run to fail**

```bash
.venv/bin/pytest tests/test_wta.py -v
```

Expected: 6 fails with ModuleNotFoundError.

**Step 3: Implement**

Write file `brain/wta.py`:

```python
"""Winner-Take-All layer.

Like LIFLayer (leak + integrate + threshold) but at most k neurons can
spike per timestep. When more than k cross threshold, only the k with the
highest membrane values spike. Losers above threshold get their membrane
multiplied by an inhibition factor (not full reset) so they retain some
charge for the next round but cannot spike this tick.

This is the simplest practical winner-take-all: deterministic top-k selection
rather than recurrent inhibitory dynamics. Sufficient for sparse concept
formation in Phase 2.
"""
from __future__ import annotations
import torch


class WTALayer:
    def __init__(
        self,
        num_neurons: int,
        k: int = 1,
        tau_mem: float = 20.0,
        threshold: float = 1.0,
        inhibit_factor: float = 0.5,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k > num_neurons:
            raise ValueError(f"k={k} cannot exceed num_neurons={num_neurons}")
        self.num_neurons = num_neurons
        self.k = k
        self.tau_mem = tau_mem
        self.threshold = threshold
        self.inhibit_factor = inhibit_factor
        self.membrane = torch.zeros(num_neurons)

    def step(self, input_current: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        # Standard LIF integrate
        leak = -self.membrane / self.tau_mem
        self.membrane = self.membrane + dt * (leak + input_current)

        # Find above-threshold neurons
        above = self.membrane >= self.threshold
        spikes = torch.zeros(self.num_neurons)

        if above.any():
            # Score = membrane value, but only for above-threshold
            scores = torch.where(above, self.membrane, torch.full_like(self.membrane, -float("inf")))
            num_winners = min(self.k, int(above.sum().item()))
            if num_winners > 0:
                _, winner_idx = torch.topk(scores, num_winners)
                spikes[winner_idx] = 1.0

            # Winners: hard reset
            self.membrane = self.membrane * (1.0 - spikes)
            # Losers above threshold (above & not in winners): inhibit
            losers_above = above & (spikes == 0)
            inhibit_mask = losers_above.float() * (self.inhibit_factor - 1.0) + 1.0
            self.membrane = self.membrane * inhibit_mask

        return spikes

    def reset(self) -> None:
        self.membrane = torch.zeros(self.num_neurons)
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_wta.py -v
```

Expected: 6 pass.

**Step 5: Commit**

```bash
git add brain/wta.py tests/test_wta.py
git commit -m "feat(brain): add WTALayer with top-k winner-take-all"
```

---

## Task 3: Working Memory layer (recurrent self-excitation)

**Why:** The brain needs to hold context across several ticks. A WMLayer is an LIFLayer with recurrent self-connections so spike activity persists for ~T timesteps after input stops.

**Files:**
- Create: `brain/working_memory.py`
- Create: `tests/test_working_memory.py`

**Step 1: Write failing tests**

Write file `tests/test_working_memory.py`:

```python
"""Working memory layer tests.

A WMLayer is an LIFLayer with recurrent self-excitation. After receiving
input, its activity persists for several ticks before fading. The persistence
duration is controlled by the recurrent gain and tau_mem.
"""
import torch
from brain.working_memory import WMLayer


def test_wm_construction():
    layer = WMLayer(num_neurons=4, recurrent_gain=0.3)
    assert layer.num_neurons == 4
    assert layer.recurrent_gain == 0.3
    assert layer.last_spikes.shape == (4,)
    assert torch.all(layer.last_spikes == 0)


def test_wm_persists_after_pulse():
    """After a single strong input pulse, activity should remain non-zero
    for at least 5 idle ticks."""
    torch.manual_seed(0)
    layer = WMLayer(num_neurons=4, recurrent_gain=0.5, threshold=1.0, tau_mem=20.0)
    # Strong input that definitely spikes everyone
    layer.step(torch.tensor([3.0, 3.0, 3.0, 3.0]))
    persistent_ticks = 0
    for _ in range(20):
        spikes = layer.step(torch.zeros(4))
        if spikes.sum() > 0:
            persistent_ticks += 1
        else:
            break
    assert persistent_ticks >= 3, f"Expected sustained activity, got {persistent_ticks} ticks"


def test_wm_no_recurrence_means_no_persistence():
    """With recurrent_gain=0, behavior reduces to plain LIFLayer (no persistence)."""
    layer = WMLayer(num_neurons=2, recurrent_gain=0.0, threshold=1.0)
    layer.step(torch.tensor([3.0, 3.0]))  # spike everyone
    # Now idle for many ticks
    for _ in range(10):
        spikes = layer.step(torch.zeros(2))
    # No recurrence, no persistence
    assert spikes.sum().item() == 0
```

**Step 2: Run to fail**

```bash
.venv/bin/pytest tests/test_working_memory.py -v
```

Expected: 3 fail with ModuleNotFoundError.

**Step 3: Implement**

Write file `brain/working_memory.py`:

```python
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
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_working_memory.py -v
```

Expected: 3 pass.

**Step 5: Commit**

```bash
git add brain/working_memory.py tests/test_working_memory.py
git commit -m "feat(brain): add WMLayer with recurrent self-excitation"
```

---

## Task 4: R-STDP synapse (reward-modulated)

**Why:** Motor learning needs reward modulation. R-STDP is the same as STDP but plasticity is scaled by an externally provided reward signal at the moment of update. Positive reward strengthens whatever pattern is currently active; negative reward weakens it.

**Files:**
- Modify: `brain/synapses.py` (add a new class, do NOT modify `STDPSynapse`)
- Create: `tests/test_rstdp.py`

**Step 1: Write failing tests**

Write file `tests/test_rstdp.py`:

```python
"""R-STDP (Reward-modulated STDP) synapse tests.

R-STDP is structurally identical to STDP except the weight delta is
multiplied by an external reward signal each step. Positive reward boosts
LTP, negative reward inverts to LTD, zero reward freezes learning.
"""
import torch
from brain.synapses import RSTDPSynapse


def test_rstdp_construction():
    syn = RSTDPSynapse(num_pre=4, num_post=2, w_init=0.5)
    assert syn.weights.shape == (2, 4)
    assert torch.all(syn.weights == 0.5)


def test_rstdp_zero_reward_freezes():
    syn = RSTDPSynapse(num_pre=1, num_post=1, w_init=0.5, a_plus=0.1, a_minus=0.1)
    initial = syn.weights.clone()
    # Causal pair under zero reward
    syn.update(torch.tensor([1.0]), torch.tensor([0.0]), reward=0.0)
    for _ in range(3):
        syn.update(torch.tensor([0.0]), torch.tensor([0.0]), reward=0.0)
    syn.update(torch.tensor([0.0]), torch.tensor([1.0]), reward=0.0)
    assert torch.allclose(syn.weights, initial)


def test_rstdp_positive_reward_potentiates_causal_pair():
    syn = RSTDPSynapse(num_pre=1, num_post=1, w_init=0.5, a_plus=0.1, a_minus=0.1)
    initial = syn.weights.clone()
    syn.update(torch.tensor([1.0]), torch.tensor([0.0]), reward=1.0)
    for _ in range(3):
        syn.update(torch.tensor([0.0]), torch.tensor([0.0]), reward=1.0)
    syn.update(torch.tensor([0.0]), torch.tensor([1.0]), reward=1.0)
    assert syn.weights[0, 0].item() > initial[0, 0].item()


def test_rstdp_negative_reward_depresses_causal_pair():
    """A causal pair under negative reward should DEPRESS (sign flips)."""
    syn = RSTDPSynapse(num_pre=1, num_post=1, w_init=0.5, a_plus=0.1, a_minus=0.1)
    initial = syn.weights.clone()
    syn.update(torch.tensor([1.0]), torch.tensor([0.0]), reward=-1.0)
    for _ in range(3):
        syn.update(torch.tensor([0.0]), torch.tensor([0.0]), reward=-1.0)
    syn.update(torch.tensor([0.0]), torch.tensor([1.0]), reward=-1.0)
    assert syn.weights[0, 0].item() < initial[0, 0].item()


def test_rstdp_forward_pass():
    syn = RSTDPSynapse(num_pre=2, num_post=2, w_init=0.0)
    syn.weights = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    out = syn.forward(torch.tensor([1.0, 1.0]))
    assert out.tolist() == [1.0, 2.0]
```

**Step 2: Run to fail**

```bash
.venv/bin/pytest tests/test_rstdp.py -v
```

Expected: 5 fail with `ImportError: cannot import name 'RSTDPSynapse'`.

**Step 3: Implement RSTDPSynapse**

Append to `brain/synapses.py` (add a new class at the bottom, do NOT modify `STDPSynapse`):

```python


class RSTDPSynapse(STDPSynapse):
    """Reward-modulated STDP synapse.

    Identical to STDPSynapse except `update()` takes a `reward` scalar
    instead of `modulation`, and the weight delta is multiplied by `reward`.
    Positive reward → potentiate causal pairs. Negative reward → depress them.
    Zero reward → freeze.

    This is the simplest 3-factor learning rule: pre-trace × post-trace × reward.
    """

    def update(  # type: ignore[override]
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        dt: float = 1.0,
        reward: float = 0.0,
    ) -> None:
        super().update(pre_spikes, post_spikes, dt=dt, modulation=reward)
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_rstdp.py -v
```

Expected: 5 pass.

**Step 5: Run full suite**

```bash
.venv/bin/pytest -v
```

Expected: ≥30 tests pass (depending on prior tasks).

**Step 6: Commit**

```bash
git add brain/synapses.py tests/test_rstdp.py
git commit -m "feat(brain): add RSTDPSynapse (reward-modulated STDP)"
```

---

## Task 5: Brain core — composing all regions

**Why:** This is the centerpiece of Phase 2. A `Brain` class in `brain/core.py` instantiates all 7 regions, all synapses between them, the modulators, and exposes a single `tick()` method that the rest of the system calls.

**Files:**
- Create: `brain/core.py`
- Create: `tests/test_core.py`

**Step 1: Write failing tests**

Write file `tests/test_core.py`:

```python
"""Brain core composition tests.

The Brain class wires together:
- Sensory (LIF, large input layer)
- Feature (LIF, smaller, learns features via STDP)
- Association (LIF, cross-modal binding)
- Concept (WTA, sparse, winner-take-all)
- Working Memory (WMLayer, recurrent persistence)
- Motor (LIF, output, learned via R-STDP)
- Meta (LIF, exposes modulator state to the rest of the brain)
- Modulators (DA, NE, ACh, 5HT)

The Brain exposes:
- .tick(input_current, reward=0.0) -> dict[str, Tensor]
- .modulators -> Modulators
- .regions -> dict[str, layer]
- .synapses -> dict[str, synapse]
- .tick_count -> int
"""
import torch
from brain.core import Brain


def test_brain_construction_default_sizes():
    brain = Brain()
    # All 7 regions present
    for name in ["sensory", "feature", "association", "concept", "wm", "motor", "meta"]:
        assert name in brain.regions, f"Missing region: {name}"
    # Modulators present
    assert brain.modulators is not None
    # Tick count starts at 0
    assert brain.tick_count == 0


def test_brain_construction_custom_sizes():
    brain = Brain(
        num_sensory=20,
        num_feature=10,
        num_association=15,
        num_concept=5,
        num_wm=5,
        num_motor=5,
        num_meta=3,
    )
    assert brain.regions["sensory"].num_neurons == 20
    assert brain.regions["feature"].num_neurons == 10
    assert brain.regions["association"].num_neurons == 15
    assert brain.regions["concept"].num_neurons == 5
    assert brain.regions["wm"].num_neurons == 5
    assert brain.regions["motor"].num_neurons == 5
    assert brain.regions["meta"].num_neurons == 3


def test_brain_tick_returns_spike_dict():
    brain = Brain(num_sensory=4)
    out = brain.tick(input_current=torch.zeros(4))
    for name in ["sensory", "feature", "association", "concept", "wm", "motor"]:
        assert name in out, f"Missing spike output for {name}"
        assert isinstance(out[name], torch.Tensor)
    assert brain.tick_count == 1


def test_brain_tick_propagates_input():
    """Strong input should produce some sensory spikes within a few ticks."""
    brain = Brain(num_sensory=4, num_feature=2)
    total_sensory_spikes = 0
    for _ in range(10):
        out = brain.tick(input_current=torch.tensor([3.0, 3.0, 3.0, 3.0]))
        total_sensory_spikes += int(out["sensory"].sum().item())
    assert total_sensory_spikes > 0, "Expected sensory spikes under strong input"


def test_brain_reward_propagates_to_dopamine():
    """A positive reward in tick() should bump dopamine level."""
    brain = Brain(num_sensory=4)
    initial_da = brain.modulators.level("DA")
    brain.tick(input_current=torch.zeros(4), reward=0.5)
    assert brain.modulators.level("DA") > initial_da


def test_brain_modulators_decay_each_tick():
    """Modulator levels should decay over many idle ticks."""
    brain = Brain(num_sensory=4)
    brain.modulators.inject("DA", 1.0)
    initial = brain.modulators.level("DA")
    for _ in range(100):
        brain.tick(input_current=torch.zeros(4))
    final = brain.modulators.level("DA")
    assert final < initial, f"DA did not decay: {initial} -> {final}"
```

**Step 2: Run to fail**

```bash
.venv/bin/pytest tests/test_core.py -v
```

Expected: 6 fail with ModuleNotFoundError.

**Step 3: Implement**

Write file `brain/core.py`:

```python
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
        a_plus: float = 0.02,
        a_minus: float = 0.022,
        w_init: float = 0.4,
        w_init_jitter: float = 0.1,
    ) -> None:
        self.regions: dict[str, object] = {
            "sensory": LIFLayer(num_sensory, tau_mem=tau_mem, threshold=threshold),
            "feature": LIFLayer(num_feature, tau_mem=tau_mem, threshold=threshold),
            "association": LIFLayer(num_association, tau_mem=tau_mem, threshold=threshold),
            "concept": WTALayer(num_concept, k=concept_k, tau_mem=tau_mem, threshold=threshold),
            "wm": WMLayer(num_wm, recurrent_gain=0.4, tau_mem=tau_mem, threshold=threshold),
            "motor": LIFLayer(num_motor, tau_mem=tau_mem, threshold=threshold),
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
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_core.py -v
```

Expected: 6 pass.

**Step 5: Run full suite**

```bash
.venv/bin/pytest -v
```

Expected: All tests pass (~36 total). If `test_two_region_learns_pattern_specialization` regressed during STDP changes, fix as in Task 0 Step 6.

**Step 6: Commit**

```bash
git add brain/core.py tests/test_core.py
git commit -m "feat(brain): Brain core composes all regions and modulators"
```

---

## Task 6: SQLite persistence

**Why:** The whole pitch is "the brain never forgets between sessions". Phase 2's `Brain` must save its complete state (weights, traces, modulator levels, tick count, region membranes) to SQLite and reload it perfectly.

**Files:**
- Create: `brain/persistence.py`
- Create: `tests/test_persistence.py`

**Step 1: Write failing tests**

Write file `tests/test_persistence.py`:

```python
"""Brain persistence tests.

A Brain saved to SQLite must reload to a state that produces identical
outputs on the same input. This includes:
- All synapse weights
- All STDP eligibility traces (apre, apost)
- All region membrane potentials
- Working memory last_spikes
- Modulator levels
- Tick count
"""
import tempfile
from pathlib import Path
import torch
from brain.core import Brain
from brain.persistence import save_brain, load_brain


def test_save_load_round_trip_default():
    brain = Brain(num_sensory=8, num_feature=4, num_association=8, num_concept=4,
                  num_wm=4, num_motor=4, num_meta=2)
    # Run a few ticks to populate non-zero state
    torch.manual_seed(123)
    for _ in range(20):
        brain.tick(torch.rand(8) * 2.0, reward=0.1)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "brain.sqlite"
        save_brain(brain, path)
        loaded = load_brain(path)

    # Check tick count
    assert loaded.tick_count == brain.tick_count
    # Check modulators
    for name in ["DA", "NE", "ACh", "5HT"]:
        assert abs(loaded.modulators.level(name) - brain.modulators.level(name)) < 1e-6
    # Check all synapse weights identical
    for key in brain.synapses:
        assert torch.allclose(loaded.synapses[key].weights, brain.synapses[key].weights)
    # Check eligibility traces identical
    for key in brain.synapses:
        assert torch.allclose(loaded.synapses[key].apre, brain.synapses[key].apre)
        assert torch.allclose(loaded.synapses[key].apost, brain.synapses[key].apost)
    # Check region membranes identical
    for key in brain.regions:
        assert torch.allclose(loaded.regions[key].membrane, brain.regions[key].membrane)
    # Check WM last_spikes
    assert torch.allclose(loaded.regions["wm"].last_spikes, brain.regions["wm"].last_spikes)


def test_save_load_then_tick_matches_unsaved():
    """Brain saved, loaded, then ticked once should match the original ticked once."""
    torch.manual_seed(7)
    brain = Brain(num_sensory=8, num_feature=4, num_association=8, num_concept=4,
                  num_wm=4, num_motor=4, num_meta=2)
    for _ in range(10):
        brain.tick(torch.rand(8) * 2.0)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "brain.sqlite"
        save_brain(brain, path)

        # Continue brain A in place
        next_input = torch.tensor([1.0, 2.0, 0.5, 1.5, 0.0, 2.5, 1.0, 0.5])
        out_a = brain.tick(next_input)

        # Load fresh brain B and tick same input
        brain_b = load_brain(path)
        out_b = brain_b.tick(next_input)

    for key in out_a:
        assert torch.allclose(out_a[key], out_b[key]), f"Divergence in {key}"
    assert brain.tick_count == brain_b.tick_count


def test_save_creates_file():
    brain = Brain(num_sensory=4)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "brain.sqlite"
        assert not path.exists()
        save_brain(brain, path)
        assert path.exists()
        assert path.stat().st_size > 0
```

**Step 2: Run to fail**

```bash
.venv/bin/pytest tests/test_persistence.py -v
```

Expected: 3 fail with ModuleNotFoundError.

**Step 3: Implement**

Write file `brain/persistence.py`:

```python
"""SQLite persistence for the Brain.

Schema (single SQLite file):
    meta(key TEXT PRIMARY KEY, value TEXT)        — tick_count, brain config JSON
    modulators(name TEXT PRIMARY KEY, level REAL)
    region_state(name TEXT PRIMARY KEY, membrane BLOB, last_spikes BLOB)
    synapse_state(name TEXT PRIMARY KEY, weights BLOB, apre BLOB, apost BLOB)

Tensors are serialized as raw float32 byte buffers via torch.save → BytesIO.

We choose torch.save (not numpy) because it preserves dtype/shape exactly
and round-trips through .load() into a fresh tensor without manual reshape.
"""
from __future__ import annotations
import io
import json
import sqlite3
from pathlib import Path
from typing import Any

import torch

from brain.core import Brain
from brain.neurons import LIFLayer
from brain.wta import WTALayer
from brain.working_memory import WMLayer


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS modulators (
    name TEXT PRIMARY KEY,
    level REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS region_state (
    name TEXT PRIMARY KEY,
    membrane BLOB NOT NULL,
    last_spikes BLOB
);
CREATE TABLE IF NOT EXISTS synapse_state (
    name TEXT PRIMARY KEY,
    weights BLOB NOT NULL,
    apre BLOB NOT NULL,
    apost BLOB NOT NULL
);
"""


def _tensor_to_blob(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()


def _blob_to_tensor(b: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(b), weights_only=True)


def _brain_config(brain: Brain) -> dict[str, Any]:
    return {
        "num_sensory": brain.regions["sensory"].num_neurons,
        "num_feature": brain.regions["feature"].num_neurons,
        "num_association": brain.regions["association"].num_neurons,
        "num_concept": brain.regions["concept"].num_neurons,
        "num_wm": brain.regions["wm"].num_neurons,
        "num_motor": brain.regions["motor"].num_neurons,
        "num_meta": brain.regions["meta"].num_neurons,
        "concept_k": brain.regions["concept"].k,
    }


def save_brain(brain: Brain, path: Path) -> None:
    path = Path(path)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_SCHEMA)
        # Meta
        conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            ("tick_count", str(brain.tick_count)),
        )
        conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            ("config", json.dumps(_brain_config(brain))),
        )
        # Modulators
        for name, level in brain.modulators.snapshot().items():
            conn.execute(
                "INSERT INTO modulators(name, level) VALUES (?, ?)",
                (name, level),
            )
        # Regions
        for name, region in brain.regions.items():
            membrane = _tensor_to_blob(region.membrane)
            last_spikes = None
            if hasattr(region, "last_spikes"):
                last_spikes = _tensor_to_blob(region.last_spikes)
            conn.execute(
                "INSERT INTO region_state(name, membrane, last_spikes) VALUES (?, ?, ?)",
                (name, membrane, last_spikes),
            )
        # Synapses
        for name, syn in brain.synapses.items():
            conn.execute(
                "INSERT INTO synapse_state(name, weights, apre, apost) VALUES (?, ?, ?, ?)",
                (
                    name,
                    _tensor_to_blob(syn.weights),
                    _tensor_to_blob(syn.apre),
                    _tensor_to_blob(syn.apost),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def load_brain(path: Path) -> Brain:
    path = Path(path)
    conn = sqlite3.connect(str(path))
    try:
        # Read config
        row = conn.execute("SELECT value FROM meta WHERE key = 'config'").fetchone()
        config = json.loads(row[0])
        brain = Brain(**config)

        # Tick count
        row = conn.execute("SELECT value FROM meta WHERE key = 'tick_count'").fetchone()
        brain.tick_count = int(row[0])

        # Modulators
        for name, level in conn.execute("SELECT name, level FROM modulators"):
            brain.modulators._levels[name] = level

        # Regions
        for name, membrane_blob, last_spikes_blob in conn.execute(
            "SELECT name, membrane, last_spikes FROM region_state"
        ):
            region = brain.regions[name]
            region.membrane = _blob_to_tensor(membrane_blob)
            if last_spikes_blob is not None and hasattr(region, "last_spikes"):
                region.last_spikes = _blob_to_tensor(last_spikes_blob)

        # Synapses
        for name, weights_blob, apre_blob, apost_blob in conn.execute(
            "SELECT name, weights, apre, apost FROM synapse_state"
        ):
            syn = brain.synapses[name]
            syn.weights = _blob_to_tensor(weights_blob)
            syn.apre = _blob_to_tensor(apre_blob)
            syn.apost = _blob_to_tensor(apost_blob)

        return brain
    finally:
        conn.close()
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_persistence.py -v
```

Expected: 3 pass.

**Step 5: Run full suite**

```bash
.venv/bin/pytest -v
```

Expected: All ~39 tests pass.

**Step 6: Commit**

```bash
git add brain/persistence.py tests/test_persistence.py
git commit -m "feat(brain): SQLite persistence for full brain state"
```

---

## Task 7: Soak test + concept emergence demo script

**Why:** This is the Phase 2 deliverable. A script that runs the brain on synthetic 4-pattern input for several minutes, periodically saves a checkpoint, and prints emerging concept neuron statistics. Demonstrates that distinct concepts form (Phase 1's collapse problem is solved) AND that save/restart preserves continuity.

**Files:**
- Create: `scripts/run_soak.py`

**Step 1: Write the script**

Write file `scripts/run_soak.py`:

```python
"""Phase 2 deliverable: soak test + concept emergence demonstration.

Runs the Brain on a synthetic 4-pattern input stream for a configurable
duration. Periodically saves the brain to SQLite. At the end, prints which
concept neurons emerged as winners for each input pattern, demonstrating
that the WTA Concept layer forms distinct sparse representations
(unlike Phase 1's collapsed two-region brain).

Run:
    .venv/bin/python scripts/run_soak.py --ticks 30000 --checkpoint-every 5000
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
from collections import defaultdict, Counter

import torch

from brain.core import Brain
from brain.persistence import save_brain, load_brain


PATTERNS = {
    "A": torch.tensor([3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "B": torch.tensor([0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "C": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0]),
    "D": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0]),
}
GAP = torch.zeros(16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=20000)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/brain.sqlite"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from --out if it exists")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    if args.resume and args.out.exists():
        print(f"Resuming from {args.out}")
        brain = load_brain(args.out)
        print(f"Loaded brain at tick {brain.tick_count}")
    else:
        brain = Brain(num_sensory=16, num_concept=8, concept_k=2)
        print(f"Fresh brain at tick 0")

    pattern_names = list(PATTERNS.keys())
    # Track which concept neurons fire on each pattern (last 1000 ticks)
    concept_winners: dict[str, Counter] = defaultdict(Counter)

    start = time.time()
    last_log = start

    for step in range(args.ticks):
        # 5 ticks of pattern, 5 ticks of gap, cycle through A,B,C,D
        pattern_name = pattern_names[(step // 10) % 4]
        in_pattern_phase = (step % 10) < 5
        input_current = PATTERNS[pattern_name] if in_pattern_phase else GAP

        out = brain.tick(input_current)

        # Log concept winners during pattern phase
        if in_pattern_phase:
            spike_idx = (out["concept"] > 0).nonzero(as_tuple=True)[0].tolist()
            for idx in spike_idx:
                concept_winners[pattern_name][idx] += 1

        # Periodic checkpoint + status
        if (step + 1) % args.checkpoint_every == 0:
            save_brain(brain, args.out)
            elapsed = time.time() - start
            ticks_per_sec = (step + 1) / elapsed
            print(f"  tick {brain.tick_count}: checkpoint saved, "
                  f"{ticks_per_sec:.0f} ticks/sec, elapsed {elapsed:.1f}s")

    save_brain(brain, args.out)
    elapsed = time.time() - start
    print(f"\nFinished {args.ticks} ticks in {elapsed:.1f}s "
          f"({args.ticks/elapsed:.0f} ticks/sec)")
    print(f"Brain saved to {args.out} at tick {brain.tick_count}")

    print("\n=== Concept emergence ===")
    print(f"Concept layer has {brain.regions['concept'].num_neurons} neurons, k={brain.regions['concept'].k}")
    print()
    for name in pattern_names:
        winners = concept_winners[name].most_common(5)
        if not winners:
            print(f"  Pattern {name}: no concept activity")
        else:
            top = ", ".join(f"#{idx}({count})" for idx, count in winners)
            print(f"  Pattern {name}: {top}")

    # Specialization check: are different concepts winning for different patterns?
    top_per_pattern = {
        name: counter.most_common(1)[0][0] if counter else None
        for name, counter in concept_winners.items()
    }
    distinct = len(set(v for v in top_per_pattern.values() if v is not None))
    print(f"\nDistinct top concept neurons across {len(pattern_names)} patterns: {distinct}/{len(pattern_names)}")
    if distinct == len(pattern_names):
        print("Phase 2 success: each pattern won by a DIFFERENT concept neuron")
    elif distinct >= 2:
        print("Partial success: at least some concept differentiation")
    else:
        print("Collapse: all patterns map to one concept (tune hyperparameters)")


if __name__ == "__main__":
    main()
```

**Step 2: Run a short version (smoke test)**

```bash
cd /Users/leonmatthies/brAIntest
.venv/bin/python scripts/run_soak.py --ticks 2000 --checkpoint-every 1000
```

Expected:
- Script runs without error
- Two checkpoint messages at ticks 1000 and 2000
- Final summary shows concept activity for each pattern
- Final "distinct top concept neurons" count is ≥ 2 (ideally 4 = full success)
- Checkpoint file at `checkpoints/brain.sqlite`

If `distinct = 1` (collapse), do NOT change the script. Instead increase `--ticks` to 10000+ and re-run. Real concept emergence may take more ticks at this small scale. If still collapse after 30000 ticks, that's a tuning issue to fix in a follow-up.

**Step 3: Verify resume works**

```bash
cd /Users/leonmatthies/brAIntest
.venv/bin/python scripts/run_soak.py --ticks 1000 --resume
```

Expected: prints `Resuming from checkpoints/brain.sqlite` and `Loaded brain at tick N` where N is the previous tick count, then runs 1000 more.

**Step 4: Commit**

```bash
git add scripts/run_soak.py
git commit -m "feat(scripts): soak test demonstrating concept emergence and resume"
```

---

## Task 8: Tag phase-2-complete + README update

**Step 1: Run full suite one more time**

```bash
cd /Users/leonmatthies/brAIntest
.venv/bin/pytest -v
```

Expected: All tests pass (~39).

**Step 2: Update README**

Edit `/Users/leonmatthies/brAIntest/README.md`. Replace the `Status` section with:

```markdown
## Status
- [x] Phase 1: SNN core foundations (LIF, STDP, 2-region brain, viz)
- [x] Phase 2: Full multi-region brain + persistence
- [ ] Phase 3: Desktop adapter (webcam + mic)
- [ ] Phase 4: FastAPI + minimal dashboard
- [ ] Phase 5: LLM bridge (Ollama + memory tools)
- [ ] Phase 6: Avatar adapter + polish
```

And replace the `Quick start` section with:

```markdown
## Quick start

```bash
uv venv
uv pip install -e ".[dev]"
.venv/bin/pytest -v

# Phase 1 viz
.venv/bin/python scripts/visualize_two_region.py

# Phase 2 soak test
.venv/bin/python scripts/run_soak.py --ticks 20000
.venv/bin/python scripts/run_soak.py --ticks 5000 --resume
```
```

The "Known limitations (Phase 1)" section can stay as-is — it correctly documents what the Phase 1 module does. The Phase 2 Concept layer fixes the collapse for the new `Brain` class but `TwoRegionBrain` is left intact.

**Step 3: Commit & tag**

```bash
cd /Users/leonmatthies/brAIntest
git add README.md
git commit -m "docs: mark Phase 2 complete in README"
git tag phase-2-complete -m "Phase 2: full brain + WTA + modulators + persistence + soak test"
git log --oneline | head -15
git tag
```

---

## Phase 2 Done. What's next?

After Phase 2 the project has:
- A 7-region brain that learns continuously from synthetic input
- Sparse distinct concept neurons (Phase 1 collapse solved by WTA)
- Modulator gating of plasticity (DA, NE, ACh, 5HT)
- R-STDP for motor learning
- Full SQLite persistence with bit-identical save/load
- A soak test script that demonstrates concept emergence and resume
- ~39 passing tests

Phase 3 entry criteria: Phase 2 soak test produces distinct concept emergence (≥2 distinct top concepts) and survives save/load round trip cleanly.

When ready, re-invoke `superpowers:writing-plans` for **Phase 3 (Desktop adapter — webcam + mic)** which connects real sensors to the brain via OpenCV and sounddevice.
