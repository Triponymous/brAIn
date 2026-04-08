# Mini-OSCEN Phase 1 Implementation Plan — SNN Core Foundations

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the foundational LIF neuron model and STDP learning rule, with two regions (Sensory + Feature) wired together, verified by tests showing weights change correctly under controlled spike input.

**Architecture:** Pure Python + snnTorch + PyTorch tensors. No adapters, no server, no UI yet — just the brain math, tested in isolation. End-state: a CLI script that runs the 2-region brain on synthetic input and produces a matplotlib plot showing the weight matrix evolving over time.

**Tech Stack:** Python 3.11, `uv` for deps, `snnTorch`, PyTorch, NumPy, matplotlib, pytest.

**Reference:** See `docs/plans/2026-04-08-mini-oscen-design.md` for the full project context.

**Out of scope for Phase 1:** All other regions (Association, Concept, Motor, Meta, WM), modulators, R-STDP, persistence, adapters, server, UI, real sensors. Those are Phase 2+.

---

## Task 0: Project bootstrap

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.python-version`
- Create: `README.md`
- Create: `brain/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Initialize git repo**

Run:
```bash
cd /Users/leonmatthies/brAIntest
git init
git branch -M main
```

Expected: `Initialized empty Git repository`.

**Step 2: Create `.python-version`**

Write file `/Users/leonmatthies/brAIntest/.python-version` with content:
```
3.11
```

**Step 3: Create `.gitignore`**

Write file `/Users/leonmatthies/brAIntest/.gitignore` with content:
```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.python-version

# uv
.uv-cache/

# Tests
.pytest_cache/
.coverage
htmlcov/

# Editors
.vscode/
.idea/
*.swp
.DS_Store

# Brain artifacts
*.sqlite
*.parquet
checkpoints/
logs/
*.png
!docs/**/*.png
```

**Step 4: Create `pyproject.toml`**

Write file `/Users/leonmatthies/brAIntest/pyproject.toml`:

```toml
[project]
name = "braintest"
version = "0.0.1"
description = "Mini-OSCEN: a persistent neuromorphic brain with LLM bridge"
requires-python = ">=3.11,<3.13"
dependencies = [
    "snntorch>=0.9.1",
    "torch>=2.2",
    "numpy>=1.26",
    "matplotlib>=3.8",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["brain"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v"
```

**Step 5: Create empty `README.md`**

Write file `/Users/leonmatthies/brAIntest/README.md`:
```markdown
# brAIntest — Mini-OSCEN

A persistent neuromorphic brain with LLM bridge.
See `docs/plans/2026-04-08-mini-oscen-design.md` for the design.

## Status
Phase 1 in progress: SNN core foundations.
```

**Step 6: Create empty package files**

Create empty files:
- `/Users/leonmatthies/brAIntest/brain/__init__.py`
- `/Users/leonmatthies/brAIntest/tests/__init__.py`

**Step 7: Install dependencies with uv**

Run:
```bash
cd /Users/leonmatthies/brAIntest
uv venv
uv pip install -e ".[dev]"
```

Expected: venv created at `.venv/`, snnTorch + torch + pytest installed without errors.

**Step 8: Verify imports work**

Run:
```bash
.venv/bin/python -c "import snntorch; import torch; import numpy; print('OK', snntorch.__version__, torch.__version__)"
```

Expected: `OK 0.9.x 2.x.x`.

**Step 9: Commit**

```bash
git add .gitignore .python-version pyproject.toml README.md brain/__init__.py tests/__init__.py docs/
git commit -m "chore: bootstrap project with uv, snntorch, pytest"
```

---

## Task 1: LIF neuron primitive (write the failing test first)

**Files:**
- Create: `tests/test_neurons.py`
- Create: `brain/neurons.py`

**Step 1: Write the failing test**

Write file `/Users/leonmatthies/brAIntest/tests/test_neurons.py`:

```python
"""Tests for the Leaky Integrate-and-Fire (LIF) neuron primitive.

A LIF neuron integrates input current over time, leaks toward rest,
and fires a spike when its membrane potential crosses a threshold.
After firing, the membrane resets.
"""
import torch
from brain.neurons import LIFLayer


def test_lif_layer_construction():
    """A LIF layer of N neurons exposes membrane state of shape (N,)."""
    layer = LIFLayer(num_neurons=10, tau_mem=20.0, threshold=1.0)
    assert layer.num_neurons == 10
    assert layer.membrane.shape == (10,)
    assert torch.all(layer.membrane == 0.0)


def test_lif_no_input_no_spikes():
    """With no input current, a fresh LIF layer produces no spikes."""
    layer = LIFLayer(num_neurons=5, tau_mem=20.0, threshold=1.0)
    spikes = layer.step(input_current=torch.zeros(5), dt=1.0)
    assert spikes.shape == (5,)
    assert torch.all(spikes == 0)
    assert torch.all(layer.membrane == 0.0)


def test_lif_subthreshold_charge_no_spike():
    """Input current below threshold raises membrane but does not spike."""
    layer = LIFLayer(num_neurons=1, tau_mem=20.0, threshold=1.0)
    spikes = layer.step(input_current=torch.tensor([0.5]), dt=1.0)
    assert spikes.item() == 0
    assert layer.membrane.item() > 0.0
    assert layer.membrane.item() < 1.0


def test_lif_suprathreshold_fires_and_resets():
    """Input current above threshold causes a spike and resets membrane."""
    layer = LIFLayer(num_neurons=1, tau_mem=20.0, threshold=1.0)
    spikes = layer.step(input_current=torch.tensor([2.0]), dt=1.0)
    assert spikes.item() == 1
    assert layer.membrane.item() == 0.0


def test_lif_membrane_leaks_toward_zero():
    """Without input, a charged membrane decays toward zero."""
    layer = LIFLayer(num_neurons=1, tau_mem=20.0, threshold=10.0)
    # Charge it manually below threshold
    layer.membrane = torch.tensor([0.5])
    initial = layer.membrane.item()
    for _ in range(10):
        layer.step(input_current=torch.tensor([0.0]), dt=1.0)
    assert layer.membrane.item() < initial
    assert layer.membrane.item() >= 0.0


def test_lif_independent_neurons():
    """Neurons in the same layer evolve independently."""
    layer = LIFLayer(num_neurons=3, tau_mem=20.0, threshold=1.0)
    spikes = layer.step(input_current=torch.tensor([2.0, 0.0, 0.5]), dt=1.0)
    assert spikes[0].item() == 1
    assert spikes[1].item() == 0
    assert spikes[2].item() == 0
```

**Step 2: Run test to verify it fails**

Run:
```bash
.venv/bin/pytest tests/test_neurons.py -v
```

Expected: All tests FAIL with `ModuleNotFoundError: No module named 'brain.neurons'`.

**Step 3: Implement minimal LIF**

Write file `/Users/leonmatthies/brAIntest/brain/neurons.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run:
```bash
.venv/bin/pytest tests/test_neurons.py -v
```

Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add brain/neurons.py tests/test_neurons.py
git commit -m "feat(brain): add LIF neuron layer with leak, threshold, and reset"
```

---

## Task 2: STDP synapse primitive

**Files:**
- Create: `tests/test_synapses.py`
- Create: `brain/synapses.py`

**Step 1: Write the failing test**

Write file `/Users/leonmatthies/brAIntest/tests/test_synapses.py`:

```python
"""Tests for STDP (Spike-Timing-Dependent Plasticity) synapses.

STDP rule:
- If pre-synaptic spike precedes post-synaptic spike (causal): potentiate (LTP).
- If post precedes pre (anti-causal): depress (LTD).
- Magnitude decays exponentially with time difference.

Implementation uses pre/post traces (event-driven, no explicit history):
- Each spike on the pre-side increments apre and weights -= apost
- Each spike on the post-side increments apost and weights += apre
- Both traces decay with their own time constants between events.
"""
import torch
from brain.synapses import STDPSynapse


def test_synapse_construction():
    """An N->M synapse exposes a weight tensor of shape (M, N)."""
    syn = STDPSynapse(num_pre=4, num_post=3, w_init=0.5)
    assert syn.weights.shape == (3, 4)
    assert torch.all(syn.weights == 0.5)


def test_synapse_forward_pass():
    """Forward pass: post-current = weights @ pre-spikes."""
    syn = STDPSynapse(num_pre=2, num_post=2, w_init=0.0)
    syn.weights = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    pre_spikes = torch.tensor([1.0, 1.0])
    current = syn.forward(pre_spikes)
    assert current.shape == (2,)
    assert current[0].item() == 1.0
    assert current[1].item() == 2.0


def test_stdp_potentiation_on_causal_spike_pair():
    """Pre-then-post spike pair should INCREASE the connecting weight."""
    syn = STDPSynapse(
        num_pre=1, num_post=1, w_init=0.5,
        a_plus=0.1, a_minus=0.1, tau_pre=20.0, tau_post=20.0,
    )
    initial = syn.weights.clone()
    # Pre fires at t=0
    syn.update(pre_spikes=torch.tensor([1.0]), post_spikes=torch.tensor([0.0]))
    # Trace decays for a few timesteps (no spikes)
    for _ in range(3):
        syn.update(pre_spikes=torch.tensor([0.0]), post_spikes=torch.tensor([0.0]))
    # Post fires at t=4 (causal: pre before post)
    syn.update(pre_spikes=torch.tensor([0.0]), post_spikes=torch.tensor([1.0]))
    assert syn.weights[0, 0].item() > initial[0, 0].item()


def test_stdp_depression_on_anticausal_spike_pair():
    """Post-then-pre spike pair should DECREASE the connecting weight."""
    syn = STDPSynapse(
        num_pre=1, num_post=1, w_init=0.5,
        a_plus=0.1, a_minus=0.1, tau_pre=20.0, tau_post=20.0,
    )
    initial = syn.weights.clone()
    # Post fires at t=0
    syn.update(pre_spikes=torch.tensor([0.0]), post_spikes=torch.tensor([1.0]))
    for _ in range(3):
        syn.update(pre_spikes=torch.tensor([0.0]), post_spikes=torch.tensor([0.0]))
    # Pre fires at t=4 (anti-causal: post before pre)
    syn.update(pre_spikes=torch.tensor([1.0]), post_spikes=torch.tensor([0.0]))
    assert syn.weights[0, 0].item() < initial[0, 0].item()


def test_stdp_weights_clipped_to_bounds():
    """Weights cannot exceed [w_min, w_max]."""
    syn = STDPSynapse(
        num_pre=1, num_post=1, w_init=0.99,
        a_plus=10.0, a_minus=10.0, w_min=0.0, w_max=1.0,
    )
    # Force a strong potentiation
    syn.update(pre_spikes=torch.tensor([1.0]), post_spikes=torch.tensor([0.0]))
    syn.update(pre_spikes=torch.tensor([0.0]), post_spikes=torch.tensor([1.0]))
    assert syn.weights[0, 0].item() <= 1.0
    assert syn.weights[0, 0].item() >= 0.0


def test_stdp_no_change_when_no_spikes():
    """Weights should not drift on idle steps."""
    syn = STDPSynapse(num_pre=2, num_post=2, w_init=0.5)
    initial = syn.weights.clone()
    for _ in range(50):
        syn.update(pre_spikes=torch.zeros(2), post_spikes=torch.zeros(2))
    assert torch.allclose(syn.weights, initial)
```

**Step 2: Run test to verify it fails**

Run:
```bash
.venv/bin/pytest tests/test_synapses.py -v
```

Expected: All tests FAIL with `ModuleNotFoundError: No module named 'brain.synapses'`.

**Step 3: Implement minimal STDP synapse**

Write file `/Users/leonmatthies/brAIntest/brain/synapses.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run:
```bash
.venv/bin/pytest tests/test_synapses.py -v
```

Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add brain/synapses.py tests/test_synapses.py
git commit -m "feat(brain): add STDP synapse with pre/post traces and weight clipping"
```

---

## Task 3: Two-region wiring (Sensory → Feature)

**Files:**
- Create: `tests/test_two_region.py`
- Create: `brain/two_region.py`

**Step 1: Write the failing test**

Write file `/Users/leonmatthies/brAIntest/tests/test_two_region.py`:

```python
"""Test that a Sensory→Feature 2-region brain learns input correlations.

Setup: 4 sensory neurons, 2 feature neurons, fully connected via STDP.
We drive the sensory neurons with two repeating patterns (A: neurons 0+1,
B: neurons 2+3) and verify that two distinct feature neurons specialize.
"""
import torch
from brain.two_region import TwoRegionBrain


def test_two_region_construction():
    brain = TwoRegionBrain(num_sensory=4, num_feature=2)
    assert brain.sensory.num_neurons == 4
    assert brain.feature.num_neurons == 2
    assert brain.synapse.weights.shape == (2, 4)


def test_two_region_tick_returns_spike_dict():
    brain = TwoRegionBrain(num_sensory=4, num_feature=2)
    out = brain.tick(input_current=torch.zeros(4))
    assert "sensory_spikes" in out
    assert "feature_spikes" in out
    assert out["sensory_spikes"].shape == (4,)
    assert out["feature_spikes"].shape == (2,)


def test_two_region_learns_pattern_specialization():
    """After repeated A/B patterns, the two feature neurons should diverge.

    We feed pattern A (high current to sensory 0+1) and pattern B (high
    current to sensory 2+3) alternately for many epochs and check that the
    weight matrix shows specialization: each feature neuron should respond
    more strongly to one pattern than the other.
    """
    torch.manual_seed(42)
    brain = TwoRegionBrain(
        num_sensory=4,
        num_feature=2,
        a_plus=0.05,
        a_minus=0.05,
        w_init_jitter=0.1,  # break symmetry
    )

    pattern_a = torch.tensor([2.0, 2.0, 0.0, 0.0])
    pattern_b = torch.tensor([0.0, 0.0, 2.0, 2.0])

    # Train for many epochs
    for _ in range(200):
        for _ in range(5):
            brain.tick(pattern_a)
        for _ in range(5):
            brain.tick(torch.zeros(4))  # gap
        for _ in range(5):
            brain.tick(pattern_b)
        for _ in range(5):
            brain.tick(torch.zeros(4))  # gap

    # After training, each feature neuron should prefer one pattern
    w = brain.synapse.weights  # shape (2, 4)
    # Strength of feature 0 for pattern A vs pattern B
    f0_a = w[0, 0:2].sum().item()
    f0_b = w[0, 2:4].sum().item()
    f1_a = w[1, 0:2].sum().item()
    f1_b = w[1, 2:4].sum().item()

    # At least one feature neuron should clearly prefer one pattern
    f0_specializes = abs(f0_a - f0_b) > 0.1
    f1_specializes = abs(f1_a - f1_b) > 0.1
    assert f0_specializes or f1_specializes, (
        f"Neither feature neuron specialized. "
        f"f0: A={f0_a:.3f} B={f0_b:.3f}, f1: A={f1_a:.3f} B={f1_b:.3f}"
    )
```

**Step 2: Run test to verify it fails**

Run:
```bash
.venv/bin/pytest tests/test_two_region.py -v
```

Expected: All tests FAIL with `ModuleNotFoundError: No module named 'brain.two_region'`.

**Step 3: Implement TwoRegionBrain**

Write file `/Users/leonmatthies/brAIntest/brain/two_region.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run:
```bash
.venv/bin/pytest tests/test_two_region.py -v
```

Expected: All 3 tests PASS. The specialization test may be flaky on first try — if it fails, increase epochs from 200 to 400 or adjust `a_plus`/`a_minus` ratio.

**Step 5: Commit**

```bash
git add brain/two_region.py tests/test_two_region.py
git commit -m "feat(brain): two-region brain learns to specialize on input patterns"
```

---

## Task 4: Visualization script — weight matrix evolution

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/visualize_two_region.py`

**Step 1: Create `scripts/__init__.py`**

Empty file.

**Step 2: Write the visualization script**

Write file `/Users/leonmatthies/brAIntest/scripts/visualize_two_region.py`:

```python
"""Visualize the weight matrix evolution of a 2-region brain.

Runs the TwoRegionBrain on alternating A/B patterns and saves three PNGs:
1. weights_before.png — initial weight matrix
2. weights_after.png — weight matrix after training
3. weights_evolution.png — heatmap of |delta| over training epochs
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from brain.two_region import TwoRegionBrain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("logs"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    torch.manual_seed(42)
    brain = TwoRegionBrain(
        num_sensory=4,
        num_feature=2,
        a_plus=0.05,
        a_minus=0.05,
        w_init_jitter=0.1,
    )

    pattern_a = torch.tensor([2.0, 2.0, 0.0, 0.0])
    pattern_b = torch.tensor([0.0, 0.0, 2.0, 2.0])

    weights_before = brain.synapse.weights.clone().numpy()
    history: list[np.ndarray] = []

    for _ in range(args.epochs):
        for _ in range(5):
            brain.tick(pattern_a)
        for _ in range(5):
            brain.tick(torch.zeros(4))
        for _ in range(5):
            brain.tick(pattern_b)
        for _ in range(5):
            brain.tick(torch.zeros(4))
        history.append(brain.synapse.weights.clone().numpy())

    weights_after = brain.synapse.weights.clone().numpy()

    # Plot before
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(weights_before, vmin=0, vmax=1, cmap="viridis")
    ax.set_title("Weights — before training")
    ax.set_xlabel("sensory neuron")
    ax.set_ylabel("feature neuron")
    plt.colorbar(im)
    fig.tight_layout()
    fig.savefig(args.out / "weights_before.png", dpi=120)
    plt.close(fig)

    # Plot after
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(weights_after, vmin=0, vmax=1, cmap="viridis")
    ax.set_title("Weights — after training")
    ax.set_xlabel("sensory neuron")
    ax.set_ylabel("feature neuron")
    plt.colorbar(im)
    fig.tight_layout()
    fig.savefig(args.out / "weights_after.png", dpi=120)
    plt.close(fig)

    # Plot evolution: each feature neuron's weights over time
    history_arr = np.stack(history)  # (epochs, feature, sensory)
    fig, axes = plt.subplots(brain.feature.num_neurons, 1, figsize=(6, 4), sharex=True)
    if brain.feature.num_neurons == 1:
        axes = [axes]
    for f, ax in enumerate(axes):
        for s in range(brain.sensory.num_neurons):
            ax.plot(history_arr[:, f, s], label=f"sensory {s}")
        ax.set_ylabel(f"feature {f} weights")
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("epoch")
    fig.suptitle("Weight evolution during STDP training")
    fig.tight_layout()
    fig.savefig(args.out / "weights_evolution.png", dpi=120)
    plt.close(fig)

    print(f"Wrote 3 plots to {args.out}/")
    print(f"Final weights:\n{weights_after}")


if __name__ == "__main__":
    main()
```

**Step 3: Run the script**

Run:
```bash
.venv/bin/python scripts/visualize_two_region.py
```

Expected:
- `logs/weights_before.png`, `logs/weights_after.png`, `logs/weights_evolution.png` are created.
- Console prints final weight matrix and the path.
- Visually: `weights_after.png` should show that one feature neuron has high weights on the left half (sensory 0+1) and low on the right half, or vice versa — the specialization is visible.

**Step 4: Sanity-check the plots**

Open `logs/weights_after.png`. Verify visually that the matrix is no longer uniform — there should be a clear contrast pattern showing pattern selectivity.

**Step 5: Commit**

```bash
git add scripts/__init__.py scripts/visualize_two_region.py
git commit -m "feat(scripts): visualize STDP weight evolution in two-region brain"
```

---

## Task 5: Run the full test suite and tag Phase 1 complete

**Step 1: Run all tests**

Run:
```bash
.venv/bin/pytest -v
```

Expected: All 15 tests PASS (6 neurons + 6 synapses + 3 two_region).

**Step 2: Tag the milestone**

Run:
```bash
git tag phase-1-complete -m "Phase 1: SNN core (LIF + STDP + 2 regions + viz) complete"
```

**Step 3: Update README**

Modify `/Users/leonmatthies/brAIntest/README.md` (replace the Status line):

```markdown
# brAIntest — Mini-OSCEN

A persistent neuromorphic brain with LLM bridge.
See `docs/plans/2026-04-08-mini-oscen-design.md` for the design.

## Status
- [x] Phase 1: SNN core foundations (LIF, STDP, 2-region brain, viz)
- [ ] Phase 2: Full multi-region brain + persistence
- [ ] Phase 3: Desktop adapter (webcam + mic)
- [ ] Phase 4: FastAPI + minimal dashboard
- [ ] Phase 5: LLM bridge (Ollama + memory tools)
- [ ] Phase 6: Avatar adapter + polish

## Quick start
```bash
uv venv
uv pip install -e ".[dev]"
.venv/bin/pytest -v
.venv/bin/python scripts/visualize_two_region.py
```
```

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: mark Phase 1 complete in README"
```

---

## Phase 1 Done. What's next?

Phase 1 produces:
- A tested LIF neuron primitive
- A tested STDP synapse primitive
- A 2-region brain that learns input statistics
- A visualization showing the learning happens
- A clean repo with tests, ready to grow

Re-invoke `writing-plans` for **Phase 2 (Full multi-region brain + persistence)** when Phase 1 is committed and the visualization confirms STDP is working as expected. Phase 2 adds: Association, Concept (winner-take-all), Working Memory, Motor, Meta regions; modulators (DA/NE/ACh/5HT); the tick loop in `brain/core.py`; and SQLite persistence in `brain/persistence.py`.

The full roadmap (from the design doc):

| Phase | Goal | Entry criteria |
|---|---|---|
| **2** | Full brain + persistence | Phase 1 complete, viz looks right |
| **3** | Desktop adapter (webcam+mic) | Phase 2 brain runs 1h soak, restarts cleanly |
| **4** | FastAPI + minimal dashboard | Phase 3 forms concept neurons consistently on real input |
| **5** | LLM bridge | Phase 4 dashboard renders ≥30 fps with full brain |
| **6** | Avatar + polish | Phase 5 LLM answers ≥3 questions correctly via tools |
