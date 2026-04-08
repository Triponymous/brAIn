"""Brain core composition tests.

The Brain class wires together:
- Sensory (LIF, large input layer)
- Feature (LIF, smaller, learns features via STDP)
- Association (LIF, cross-modal binding)
- Concept (WTA, sparse, winner-take-all)
- Working Memory (WMLayer, recurrent persistence)
- Motor (LIF, output, learned via R-STDP)
- Meta (LIF, exposes modulator state to the rest of the brain)
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
    # All 7 regions present
    for name in ["sensory", "feature", "association", "concept", "wm", "motor", "meta"]:
        assert name in brain.regions, f"Missing region: {name}"
    # Modulators present
    assert brain.modulators is not None
    # Tick count starts at 0
    assert brain.tick_count == 0


def test_brain_construction_custom_sizes():
    brain = Brain(
        num_sensory=20,
        num_feature=10,
        num_association=15,
        num_concept=5,
        num_wm=5,
        num_motor=5,
        num_meta=3,
    )
    assert brain.regions["sensory"].num_neurons == 20
    assert brain.regions["feature"].num_neurons == 10
    assert brain.regions["association"].num_neurons == 15
    assert brain.regions["concept"].num_neurons == 5
    assert brain.regions["wm"].num_neurons == 5
    assert brain.regions["motor"].num_neurons == 5
    assert brain.regions["meta"].num_neurons == 3


def test_brain_tick_returns_spike_dict():
    brain = Brain(num_sensory=4)
    out = brain.tick(input_current=torch.zeros(4))
    for name in ["sensory", "feature", "association", "concept", "wm", "motor"]:
        assert name in out, f"Missing spike output for {name}"
        assert isinstance(out[name], torch.Tensor)
    assert brain.tick_count == 1


def test_brain_tick_propagates_input():
    """Strong input should produce some sensory spikes within a few ticks."""
    brain = Brain(num_sensory=4, num_feature=2)
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
