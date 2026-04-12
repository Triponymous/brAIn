"""Brain core composition tests.

The Brain class wires together:
- Sensory (LIF, 200, encodes raw input to spikes)
- Expansion (fixed random, 500, pattern separation)
- Concept (WTA, 1000, sparse winner-take-all with STDP)
- Working Memory (WMLayer, 100, recurrent persistence with feedback)
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
    for name in ["sensory", "concept", "wm"]:
        assert name in brain.regions, f"Missing region: {name}"
    # Removed regions should NOT exist
    for name in ["feature", "association", "motor", "meta"]:
        assert name not in brain.regions, f"Removed region still present: {name}"
    assert brain.modulators is not None
    assert brain.tick_count == 0


def test_brain_construction_custom_sizes():
    brain = Brain(
        num_sensory=20,
        num_concept=5,
        num_wm=5,
    )
    assert brain.regions["sensory"].num_neurons == 20
    assert brain.regions["concept"].num_neurons == 5
    assert brain.regions["wm"].num_neurons == 5


def test_brain_construction_legacy_params():
    """Legacy params from old config.json are accepted but ignored."""
    brain = Brain(
        num_sensory=20,
        num_concept=5,
        num_wm=5,
        num_feature=10,       # legacy — ignored
        num_association=15,   # legacy — ignored
        num_motor=5,          # legacy — ignored
        num_meta=3,           # legacy — ignored
    )
    assert "feature" not in brain.regions
    assert brain.regions["sensory"].num_neurons == 20


def test_brain_tick_returns_spike_dict():
    brain = Brain(num_sensory=4)
    out = brain.tick(input_current=torch.zeros(4))
    for name in ["sensory", "concept", "wm"]:
        assert name in out, f"Missing spike output for {name}"
        assert isinstance(out[name], torch.Tensor)
    assert brain.tick_count == 1


def test_brain_tick_propagates_input():
    """Strong input should produce some sensory spikes within a few ticks."""
    brain = Brain(num_sensory=4)
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


def test_brain_synapses_exist():
    """All expected synapses present, no dead ones."""
    brain = Brain(num_sensory=4)
    expected = {"sensory_concept", "concept_wm", "wm_concept"}
    assert set(brain.synapses.keys()) == expected


def test_wm_feedback_exists():
    """WM → Concept feedback synapse should exist and have correct dimensions."""
    brain = Brain(num_sensory=4, num_concept=10, num_wm=5)
    syn = brain.synapses["wm_concept"]
    assert syn.weights.shape == (10, 5)  # (num_concept, num_wm)
