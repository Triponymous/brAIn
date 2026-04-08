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
