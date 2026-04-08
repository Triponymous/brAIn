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
