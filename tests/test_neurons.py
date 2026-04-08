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
