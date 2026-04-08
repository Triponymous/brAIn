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
