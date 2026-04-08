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
