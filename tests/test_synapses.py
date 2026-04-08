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
