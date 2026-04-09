"""Tests for memory tools — functions the LLM can call to query brain state."""
import torch
import pytest
from brain.core import Brain
from bridge.exporter import BrainStateExporter
from bridge.memory_tools import MemoryTools


def test_current_state_returns_snapshot():
    brain = Brain(num_sensory=8, num_concept=4)
    exporter = BrainStateExporter(brain)
    tools = MemoryTools(brain, exporter)
    result = tools.current_state()
    assert "tick_count" in result
    assert "modulators" in result


def test_query_concepts_returns_top_n():
    brain = Brain(num_sensory=8, num_concept=4)
    exporter = BrainStateExporter(brain)
    torch.manual_seed(0)
    for _ in range(20):
        brain.tick(torch.rand(8) * 3.0)
    tools = MemoryTools(brain, exporter)
    result = tools.query_concepts(limit=3)
    assert isinstance(result, list)
    assert len(result) <= 3


def test_label_concept_persists():
    brain = Brain(num_sensory=8, num_concept=4)
    exporter = BrainStateExporter(brain)
    tools = MemoryTools(brain, exporter)
    tools.label_concept(concept_id=2, label="tippen")
    assert exporter.get_label(2) == "tippen"


def test_recall_associations_returns_list():
    brain = Brain(num_sensory=8, num_concept=4)
    exporter = BrainStateExporter(brain)
    tools = MemoryTools(brain, exporter)
    result = tools.recall_associations(concept_id=0)
    assert isinstance(result, list)


def test_tool_definitions_for_llm():
    brain = Brain(num_sensory=8, num_concept=4)
    exporter = BrainStateExporter(brain)
    tools = MemoryTools(brain, exporter)
    defs = tools.tool_definitions()
    assert isinstance(defs, list)
    names = {d["name"] for d in defs}
    assert "current_state" in names
    assert "query_concepts" in names
    assert "label_concept" in names
    assert "recall_associations" in names


def test_execute_dispatches_tool():
    brain = Brain(num_sensory=8, num_concept=4)
    exporter = BrainStateExporter(brain)
    tools = MemoryTools(brain, exporter)
    result = tools.execute("current_state", {})
    assert "tick_count" in result


def test_execute_unknown_tool_returns_error():
    brain = Brain(num_sensory=8, num_concept=4)
    exporter = BrainStateExporter(brain)
    tools = MemoryTools(brain, exporter)
    result = tools.execute("nonexistent", {})
    assert "error" in result
