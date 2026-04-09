"""Tests for BrainStateExporter — produces LLM-readable JSON snapshots."""
import torch
import pytest
from brain.core import Brain
from bridge.exporter import BrainStateExporter


def test_exporter_construction():
    brain = Brain(num_sensory=8, num_feature=4, num_association=8, num_concept=4,
                  num_wm=4, num_motor=4, num_meta=2)
    exporter = BrainStateExporter(brain)
    assert exporter.brain is brain


def test_snapshot_has_required_keys():
    brain = Brain(num_sensory=8, num_feature=4, num_association=8, num_concept=4,
                  num_wm=4, num_motor=4, num_meta=2)
    torch.manual_seed(0)
    for _ in range(10):
        brain.tick(torch.rand(8) * 3.0)
    exporter = BrainStateExporter(brain)
    snap = exporter.snapshot()
    assert "tick_count" in snap
    assert "modulators" in snap
    assert "active_concepts" in snap
    assert "sensor_summary" in snap
    assert snap["tick_count"] == brain.tick_count


def test_snapshot_active_concepts_is_list():
    brain = Brain(num_sensory=8, num_feature=4, num_association=8, num_concept=4,
                  num_wm=4, num_motor=4, num_meta=2)
    torch.manual_seed(0)
    for _ in range(50):
        brain.tick(torch.rand(8) * 3.0)
    exporter = BrainStateExporter(brain)
    snap = exporter.snapshot()
    assert isinstance(snap["active_concepts"], list)
    for c in snap["active_concepts"]:
        assert "id" in c
        assert "activation" in c


def test_snapshot_modulators_match_brain():
    brain = Brain(num_sensory=8, num_feature=4, num_association=8, num_concept=4,
                  num_wm=4, num_motor=4, num_meta=2)
    brain.modulators.inject("DA", 0.5)
    exporter = BrainStateExporter(brain)
    snap = exporter.snapshot()
    assert abs(snap["modulators"]["DA"] - brain.modulators.level("DA")) < 1e-6


def test_snapshot_with_labels():
    brain = Brain(num_sensory=8, num_feature=4, num_association=8, num_concept=4,
                  num_wm=4, num_motor=4, num_meta=2)
    exporter = BrainStateExporter(brain)
    exporter.set_label(0, "tippen")
    snap = exporter.snapshot()
    assert exporter.get_label(0) == "tippen"


def test_set_get_labels_round_trip():
    brain = Brain(num_sensory=4)
    exporter = BrainStateExporter(brain)
    exporter.set_label(3, "musik")
    exporter.set_label(7, "stille")
    assert exporter.get_label(3) == "musik"
    assert exporter.get_label(7) == "stille"
    assert exporter.get_label(999) is None
    assert exporter.all_labels() == {3: "musik", 7: "stille"}
