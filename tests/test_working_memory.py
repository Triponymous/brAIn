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
    for at least 3 idle ticks. recurrent_gain must be >= threshold so the
    self-feedback alone can re-fire after the membrane reset."""
    torch.manual_seed(0)
    layer = WMLayer(num_neurons=4, recurrent_gain=1.2, threshold=1.0, tau_mem=20.0)
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
