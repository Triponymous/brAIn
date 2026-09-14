"""Tests for ToolRegistry -- unified dispatch for memory tools + granted capabilities."""
import tempfile
from pathlib import Path
import pytest
from brain.core import Brain
from bridge.exporter import BrainStateExporter
from capabilities.grants import GrantStore
from capabilities.registry import ToolRegistry


def test_registry_lists_memory_tools_always():
    brain = Brain(num_sensory=4, num_concept=4)
    exporter = BrainStateExporter(brain)
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "test.sqlite")
        registry = ToolRegistry(brain, exporter, store)
        defs = registry.tool_definitions()
    names = {d["name"] for d in defs}
    assert "current_state" in names
    assert "query_concepts" in names
    assert "web_search" not in names


def test_registry_includes_granted_tools():
    brain = Brain(num_sensory=4, num_concept=4)
    exporter = BrainStateExporter(brain)
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "test.sqlite")
        store.grant("web_search")
        registry = ToolRegistry(brain, exporter, store)
        defs = registry.tool_definitions()
    names = {d["name"] for d in defs}
    assert "web_search" in names


def test_registry_execute_memory_tool():
    brain = Brain(num_sensory=4, num_concept=4)
    exporter = BrainStateExporter(brain)
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "test.sqlite")
        registry = ToolRegistry(brain, exporter, store)
        result = registry.execute("current_state", {})
    assert "tick_count" in result


@pytest.mark.asyncio
async def test_registry_execute_granted_async_tool():
    brain = Brain(num_sensory=4, num_concept=4)
    exporter = BrainStateExporter(brain)
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "test.sqlite")
        store.grant("shell")
        registry = ToolRegistry(brain, exporter, store)
        result = await registry.execute_async("shell", {"command": "date"})
    assert "stdout" in result


def test_registry_execute_ungranted_tool_errors():
    brain = Brain(num_sensory=4, num_concept=4)
    exporter = BrainStateExporter(brain)
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "test.sqlite")
        registry = ToolRegistry(brain, exporter, store)
        result = registry.execute("web_search", {"query": "test"})
    assert "error" in result


def test_registry_puts_brain_tools_first_and_hides_what_they_supersede():
    from bridge.brain_tools import BrainTools
    brain = Brain(num_sensory=4, num_concept=4)
    exporter = BrainStateExporter(brain)
    with tempfile.TemporaryDirectory() as tmp:
        registry = ToolRegistry(brain, exporter, GrantStore(Path(tmp) / "g.sqlite"), brain_tools=BrainTools(brain))
        defs = registry.tool_definitions()
        names = [d["name"] for d in defs]
        assert names[0] == "brain_state" and set(names) >= BrainTools.NAMES
        assert "current_state" not in names and "query_concepts" not in names   # superseded
        assert "label_concept" in names and "recall_associations" in names       # still useful
        assert "felt" in registry.execute("brain_state", {})
        assert "tick_count" in registry.execute("current_state", {})              # hidden, not gone


@pytest.mark.asyncio
async def test_registry_execute_async_reaches_brain_tools():
    from bridge.brain_tools import BrainTools
    brain = Brain(num_sensory=4, num_concept=4)
    exporter = BrainStateExporter(brain)
    with tempfile.TemporaryDirectory() as tmp:
        registry = ToolRegistry(brain, exporter, GrantStore(Path(tmp) / "g.sqlite"), brain_tools=BrainTools(brain))
        why = await registry.execute_async("brain_why", {"modulator": "DA"})
        assert why["modulator"] == "DA" and "progress" in why["mechanism"]
