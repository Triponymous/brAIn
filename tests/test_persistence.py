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
    brain = Brain(num_sensory=8, num_concept=4, num_wm=4)
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
    brain = Brain(num_sensory=8, num_concept=4, num_wm=4)
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
