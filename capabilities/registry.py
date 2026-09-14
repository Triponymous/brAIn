"""ToolRegistry -- unified tool dispatch for memory tools + granted capabilities."""
from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Any

from brain.core import Brain
from bridge.exporter import BrainStateExporter
from bridge.brain_tools import BrainTools
from bridge.memory_tools import MemoryTools
from capabilities.catalog import get_catalog
from capabilities.grants import GrantStore


_MEMORY_TOOL_NAMES = {"current_state", "query_concepts", "label_concept", "recall_associations"}
# brain_state / brain_history say these better; hidden from the model when brain
# tools are present (fewer, clearer tools help a local model choose), still callable.
_SUPERSEDED_BY_BRAIN_TOOLS = {"current_state", "query_concepts", "episode_search"}


class ToolRegistry:
    """Serves both built-in memory tools and granted capability tools."""

    def __init__(self, brain: Brain, exporter: BrainStateExporter, grant_store: GrantStore,
                 brain_tools: BrainTools | None = None) -> None:
        self.memory_tools = MemoryTools(brain, exporter)
        self.brain_tools = brain_tools
        self.grant_store = grant_store
        self.catalog = get_catalog()
        self._granted_names: set[str] = set(grant_store.list_grants().keys())

    def refresh_grants(self) -> None:
        self._granted_names = set(self.grant_store.list_grants().keys())

    def tool_definitions(self) -> list[dict[str, Any]]:
        defs = list(self.brain_tools.tool_definitions()) if self.brain_tools is not None else []
        for d in self.memory_tools.tool_definitions():
            if self.brain_tools is None or d["name"] not in _SUPERSEDED_BY_BRAIN_TOOLS:
                defs.append(d)
        for name in self._granted_names:
            entry = self.catalog.get(name)
            if entry:
                defs.append(entry.tool_schema)
        return defs

    def execute(self, name: str, args: dict[str, Any]) -> Any:
        """Synchronous dispatch -- for brain tools, memory tools and sync capability tools."""
        if self.brain_tools is not None and name in self.brain_tools.NAMES:
            return self.brain_tools.execute(name, args)
        if name in _MEMORY_TOOL_NAMES:
            return self.memory_tools.execute(name, args)
        if name not in self._granted_names:
            return {"error": f"Tool '{name}' is not granted. Ask the user to approve it first."}
        entry = self.catalog.get(name)
        if entry is None:
            return {"error": f"Unknown tool: {name}"}
        # Handle local_files action dispatch
        if name == "local_files":
            from capabilities.tools.local_files import read_file, write_file, list_files
            action = args.get("action", "list")
            base_dir = args.get("base_dir", str(Path.home() / "Notes"))
            if action == "read":
                return read_file(path=args.get("path", ""), base_dir=base_dir)
            elif action == "write":
                return write_file(path=args.get("path", ""), content=args.get("content", ""), base_dir=base_dir)
            else:
                return list_files(base_dir=base_dir)
        if entry.execute_fn is None:
            return {"error": f"Tool '{name}' has no execute function"}
        if asyncio.iscoroutinefunction(entry.execute_fn):
            return {"error": f"Tool '{name}' is async -- use execute_async()"}
        return entry.execute_fn(**args)

    async def execute_async(self, name: str, args: dict[str, Any]) -> Any:
        """Async dispatch -- for capability tools (web_search, shell)."""
        if self.brain_tools is not None and name in self.brain_tools.NAMES:
            return self.brain_tools.execute(name, args)
        if name in _MEMORY_TOOL_NAMES:
            return self.memory_tools.execute(name, args)
        if name not in self._granted_names:
            return {"error": f"Tool '{name}' is not granted."}
        entry = self.catalog.get(name)
        if entry is None:
            return {"error": f"Unknown tool: {name}"}
        if name == "local_files":
            return self.execute(name, args)
        if entry.execute_fn is None:
            return {"error": f"Tool '{name}' has no execute function"}
        if asyncio.iscoroutinefunction(entry.execute_fn):
            return await entry.execute_fn(**args)
        return entry.execute_fn(**args)
