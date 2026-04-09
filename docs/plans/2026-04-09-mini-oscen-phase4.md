# Mini-OSCEN Phase 4 Implementation Plan — Capability Wishlist + Grant System

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the emergent capability acquisition system: a catalog of 3 tools (web search, shell, local files) all disabled by default, a wish detector that matches stable concept labels to catalog entries, a grant flow where the user approves/denies, and runtime tool registration so granted tools appear in the LLM's toolset and survive restarts.

**Architecture:** The catalog lives in `capabilities/catalog.py` as a registry of tool definitions + implementations. The wish detector is a background coroutine in `capabilities/wish_detector.py` that reads concept labels from the BrainStateExporter and matches against catalog trigger keywords. Grants are stored in SQLite via `capabilities/grants.py`. The existing `bridge/memory_tools.py` is extended into a `ToolRegistry` that serves both memory tools and granted capability tools to the LLM router. A grant-request UI component is added to the dashboard.

**Tech Stack:** Python (httpx for DuckDuckGo, asyncio.subprocess for shell), SQLite, React (grant UI component).

**Reference:** `docs/plans/2026-04-09-mini-oscen-phase4-design.md`

**Out of scope:** Proactive tool use (Phase 5), tool revoke UI, tool parameter config UI, conversation-history-based wishes.

---

## Task 0: Capability package + SQLite schema migration

**Files:**
- Create: `/Users/leonmatthies/brAIntest/capabilities/__init__.py`
- Create: `/Users/leonmatthies/brAIntest/capabilities/grants.py`
- Create: `/Users/leonmatthies/brAIntest/tests/test_grants.py`

**Step 1: Write failing tests**

```python
"""Tests for capability grant persistence."""
import tempfile
from pathlib import Path
import pytest
from capabilities.grants import GrantStore


def test_grant_store_construction():
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "test.sqlite")
        assert store.list_grants() == {}


def test_grant_and_list():
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "test.sqlite")
        store.grant("web_search", concept_label="googlen", reason="User approved")
        grants = store.list_grants()
        assert "web_search" in grants
        assert grants["web_search"]["concept_label"] == "googlen"


def test_grant_persists_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.sqlite"
        store1 = GrantStore(path)
        store1.grant("shell", concept_label="terminal")
        store2 = GrantStore(path)
        assert "shell" in store2.list_grants()


def test_is_granted():
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "test.sqlite")
        assert store.is_granted("web_search") is False
        store.grant("web_search")
        assert store.is_granted("web_search") is True


def test_deny_with_cooldown():
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "test.sqlite")
        store.deny("web_search", cooldown_days=7)
        assert store.is_in_cooldown("web_search") is True


def test_record_wish():
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "test.sqlite")
        wish_id = store.record_wish("web_search", concept_id=12, concept_label="googlen", reason="High activity")
        assert wish_id > 0
        wishes = store.list_wishes()
        assert len(wishes) == 1
        assert wishes[0]["tool_name"] == "web_search"
        assert wishes[0]["status"] == "pending"
```

**Step 2:** Run to fail → implement `capabilities/grants.py`:

```python
"""Grant persistence — SQLite store for capability wishes and grants."""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS capability_wishes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT NOT NULL,
    trigger_concept_id INTEGER,
    trigger_concept_label TEXT,
    reason TEXT,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS capability_grants (
    tool_name TEXT PRIMARY KEY,
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    concept_label TEXT,
    reason TEXT,
    params JSON
);
CREATE TABLE IF NOT EXISTS capability_denials (
    tool_name TEXT PRIMARY KEY,
    denied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cooldown_until TIMESTAMP
);
"""


class GrantStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def grant(self, tool_name: str, concept_label: str = "", reason: str = "", params: dict | None = None) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO capability_grants(tool_name, granted_at, concept_label, reason, params) VALUES (?, ?, ?, ?, ?)",
            (tool_name, datetime.now().isoformat(), concept_label, reason, json.dumps(params or {})),
        )
        # Remove any denial cooldown
        self._conn.execute("DELETE FROM capability_denials WHERE tool_name = ?", (tool_name,))
        self._conn.commit()

    def deny(self, tool_name: str, cooldown_days: int = 7) -> None:
        until = (datetime.now() + timedelta(days=cooldown_days)).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO capability_denials(tool_name, denied_at, cooldown_until) VALUES (?, ?, ?)",
            (tool_name, datetime.now().isoformat(), until),
        )
        self._conn.commit()

    def is_granted(self, tool_name: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM capability_grants WHERE tool_name = ?", (tool_name,)).fetchone()
        return row is not None

    def is_in_cooldown(self, tool_name: str) -> bool:
        row = self._conn.execute("SELECT cooldown_until FROM capability_denials WHERE tool_name = ?", (tool_name,)).fetchone()
        if row is None:
            return False
        return datetime.fromisoformat(row["cooldown_until"]) > datetime.now()

    def list_grants(self) -> dict[str, dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM capability_grants").fetchall()
        return {r["tool_name"]: dict(r) for r in rows}

    def record_wish(self, tool_name: str, concept_id: int = 0, concept_label: str = "", reason: str = "") -> int:
        cur = self._conn.execute(
            "INSERT INTO capability_wishes(tool_name, trigger_concept_id, trigger_concept_label, reason) VALUES (?, ?, ?, ?)",
            (tool_name, concept_id, concept_label, reason),
        )
        self._conn.commit()
        return cur.lastrowid

    def list_wishes(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self._conn.execute("SELECT * FROM capability_wishes WHERE status = ?", (status,)).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM capability_wishes").fetchall()
        return [dict(r) for r in rows]

    def resolve_wish(self, wish_id: int, status: str) -> None:
        self._conn.execute(
            "UPDATE capability_wishes SET status = ?, resolved_at = ? WHERE id = ?",
            (status, datetime.now().isoformat(), wish_id),
        )
        self._conn.commit()
```

**Step 3:** Run tests + full suite, commit.

```bash
.venv/bin/pytest tests/test_grants.py -v
.venv/bin/pytest 2>&1 | tail -3
git add capabilities/__init__.py capabilities/grants.py tests/test_grants.py
git commit -m "feat(capabilities): GrantStore with wish/grant/deny persistence"
```

Expected: 6 new tests, ~140 total.

---

## Task 1: Capability Catalog + Tool Implementations

**Files:**
- Create: `/Users/leonmatthies/brAIntest/capabilities/catalog.py`
- Create: `/Users/leonmatthies/brAIntest/capabilities/tools/`
- Create: `/Users/leonmatthies/brAIntest/capabilities/tools/__init__.py`
- Create: `/Users/leonmatthies/brAIntest/capabilities/tools/web_search.py`
- Create: `/Users/leonmatthies/brAIntest/capabilities/tools/shell.py`
- Create: `/Users/leonmatthies/brAIntest/capabilities/tools/local_files.py`
- Create: `/Users/leonmatthies/brAIntest/tests/test_capability_tools.py`

**Step 1: Write failing tests**

```python
"""Tests for capability tools — web search, shell, local files."""
import asyncio
import tempfile
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, patch
from capabilities.catalog import get_catalog, CatalogEntry
from capabilities.tools.web_search import web_search
from capabilities.tools.shell import safe_shell
from capabilities.tools.local_files import read_file, write_file, list_files


def test_catalog_has_three_entries():
    catalog = get_catalog()
    assert "web_search" in catalog
    assert "shell" in catalog
    assert "local_files" in catalog
    for name, entry in catalog.items():
        assert isinstance(entry, CatalogEntry)
        assert entry.trigger_keywords
        assert entry.tool_schema


@pytest.mark.asyncio
async def test_web_search_mock():
    with patch("capabilities.tools.web_search._fetch_ddg", new_callable=AsyncMock,
               return_value=[{"title": "Test", "url": "https://test.com", "snippet": "A test result."}]):
        result = await web_search(query="test query")
    assert isinstance(result, list)
    assert len(result) >= 1
    assert result[0]["title"] == "Test"


@pytest.mark.asyncio
async def test_shell_allowed_command():
    result = await safe_shell(command="date")
    assert "stdout" in result
    assert result["returncode"] == 0


@pytest.mark.asyncio
async def test_shell_blocked_command():
    result = await safe_shell(command="rm -rf /")
    assert "error" in result
    assert "blocked" in result["error"].lower()


@pytest.mark.asyncio
async def test_shell_blocked_sudo():
    result = await safe_shell(command="sudo ls")
    assert "error" in result


def test_local_files_write_and_read():
    with tempfile.TemporaryDirectory() as tmp:
        result = write_file(path="test.md", content="# Hello", base_dir=tmp)
        assert result["status"] == "ok"
        content = read_file(path="test.md", base_dir=tmp)
        assert content["content"] == "# Hello"


def test_local_files_list():
    with tempfile.TemporaryDirectory() as tmp:
        write_file(path="a.md", content="A", base_dir=tmp)
        write_file(path="b.txt", content="B", base_dir=tmp)
        files = list_files(base_dir=tmp)
        assert len(files["files"]) == 2


def test_local_files_path_traversal_blocked():
    with tempfile.TemporaryDirectory() as tmp:
        result = read_file(path="../../../etc/passwd", base_dir=tmp)
        assert "error" in result
```

**Step 2:** Implement all three tools + the catalog.

`capabilities/catalog.py`:
```python
"""Capability catalog — registry of all available tools."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class CatalogEntry:
    name: str
    description: str
    trigger_keywords: list[str]
    tool_schema: dict[str, Any]
    execute_fn: Callable[..., Any]
    default_enabled: bool = False


def get_catalog() -> dict[str, CatalogEntry]:
    from capabilities.tools.web_search import web_search
    from capabilities.tools.shell import safe_shell
    from capabilities.tools.local_files import read_file, write_file, list_files

    return {
        "web_search": CatalogEntry(
            name="web_search",
            description="Search the web via DuckDuckGo and return top results.",
            trigger_keywords=["google", "suchen", "recherche", "search", "nachschlagen", "web"],
            tool_schema={
                "name": "web_search",
                "description": "Search the web and return top-5 results with title, URL, and snippet.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                    "required": ["query"],
                },
            },
            execute_fn=web_search,
        ),
        "shell": CatalogEntry(
            name="shell",
            description="Run a safe shell command (allowlisted commands only).",
            trigger_keywords=["terminal", "command", "befehl", "shell", "cli", "konsole"],
            tool_schema={
                "name": "shell",
                "description": "Execute a safe shell command. Only allowlisted commands (ls, cat, wc, date, which, brew list, pip list) are permitted.",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string", "description": "Shell command to run"}},
                    "required": ["command"],
                },
            },
            execute_fn=safe_shell,
        ),
        "local_files": CatalogEntry(
            name="local_files",
            description="Read, write, and list text/markdown files in a local directory.",
            trigger_keywords=["notizen", "notes", "datei", "file", "markdown", "schreiben", "lesen"],
            tool_schema={
                "name": "local_files",
                "description": "Read, write, or list text files in the user's notes directory.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["read", "write", "list"]},
                        "path": {"type": "string", "description": "Relative file path (for read/write)"},
                        "content": {"type": "string", "description": "Content to write (for write action)"},
                    },
                    "required": ["action"],
                },
            },
            execute_fn=None,  # dispatched by action in ToolRegistry
        ),
    }
```

`capabilities/tools/web_search.py`:
```python
"""Web search via DuckDuckGo HTML API."""
from __future__ import annotations
from typing import Any
import httpx


async def _fetch_ddg(query: str) -> list[dict[str, str]]:
    """Fetch DuckDuckGo instant answers."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
        )
        data = resp.json()
    results = []
    # Abstract
    if data.get("Abstract"):
        results.append({"title": data.get("Heading", ""), "url": data.get("AbstractURL", ""), "snippet": data["Abstract"]})
    # Related topics
    for topic in data.get("RelatedTopics", [])[:5]:
        if "Text" in topic:
            results.append({"title": topic.get("Text", "")[:80], "url": topic.get("FirstURL", ""), "snippet": topic.get("Text", "")})
    return results[:5]


async def web_search(query: str) -> list[dict[str, str]]:
    return await _fetch_ddg(query)
```

`capabilities/tools/shell.py`:
```python
"""Sandboxed shell execution with allowlist."""
from __future__ import annotations
import asyncio
import shlex
from typing import Any

_ALLOWED_COMMANDS = {"ls", "cat", "wc", "date", "which", "brew", "pip", "echo", "head", "tail", "find", "grep"}
_BLOCKED_PREFIXES = {"rm", "sudo", "mv", "cp", "chmod", "chown", "kill", "pkill", "dd", "mkfs", "fdisk"}


async def safe_shell(command: str) -> dict[str, Any]:
    parts = shlex.split(command)
    if not parts:
        return {"error": "Empty command"}
    base_cmd = parts[0].split("/")[-1]  # handle full paths
    if base_cmd in _BLOCKED_PREFIXES or any(p.startswith("sudo") for p in parts):
        return {"error": f"Blocked: '{base_cmd}' is not allowed for safety reasons."}
    if base_cmd not in _ALLOWED_COMMANDS:
        return {"error": f"Blocked: '{base_cmd}' is not in the allowlist. Allowed: {', '.join(sorted(_ALLOWED_COMMANDS))}"}
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
    return {
        "stdout": stdout.decode("utf-8", errors="replace")[:5000],
        "stderr": stderr.decode("utf-8", errors="replace")[:2000],
        "returncode": proc.returncode,
    }
```

`capabilities/tools/local_files.py`:
```python
"""Local file read/write/list — sandboxed to a base directory."""
from __future__ import annotations
from pathlib import Path
from typing import Any


_DEFAULT_BASE = str(Path.home() / "Notes")


def _safe_path(path: str, base_dir: str) -> Path | None:
    base = Path(base_dir).resolve()
    target = (base / path).resolve()
    if not str(target).startswith(str(base)):
        return None  # path traversal attempt
    return target


def read_file(path: str, base_dir: str = _DEFAULT_BASE) -> dict[str, Any]:
    target = _safe_path(path, base_dir)
    if target is None:
        return {"error": "Path traversal blocked"}
    if not target.exists():
        return {"error": f"File not found: {path}"}
    return {"content": target.read_text(encoding="utf-8"), "path": str(path)}


def write_file(path: str, content: str, base_dir: str = _DEFAULT_BASE) -> dict[str, Any]:
    target = _safe_path(path, base_dir)
    if target is None:
        return {"error": "Path traversal blocked"}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"status": "ok", "path": str(path), "bytes": len(content.encode())}


def list_files(base_dir: str = _DEFAULT_BASE, pattern: str = "*") -> dict[str, Any]:
    base = Path(base_dir)
    if not base.exists():
        return {"files": [], "base_dir": str(base)}
    files = [str(f.relative_to(base)) for f in sorted(base.rglob(pattern)) if f.is_file()]
    return {"files": files[:100], "base_dir": str(base)}
```

**Step 3:** Run tests + commit.

```bash
.venv/bin/pytest tests/test_capability_tools.py -v
.venv/bin/pytest 2>&1 | tail -3
git add capabilities/ tests/test_capability_tools.py
git commit -m "feat(capabilities): catalog + web_search + shell + local_files tools"
```

Expected: 9 new tests, ~149 total.

---

## Task 2: Tool Registry (extends MemoryTools with granted capabilities)

**Files:**
- Create: `/Users/leonmatthies/brAIntest/capabilities/registry.py`
- Create: `/Users/leonmatthies/brAIntest/tests/test_tool_registry.py`

**Step 1: Write failing tests**

```python
"""Tests for ToolRegistry — unified dispatch for memory tools + granted capabilities."""
import tempfile
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, patch
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
    # No capability tools yet (nothing granted)
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
async def test_registry_execute_granted_tool():
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
```

**Step 2:** Implement `capabilities/registry.py`:

```python
"""ToolRegistry — unified tool dispatch for memory tools + granted capabilities."""
from __future__ import annotations
import asyncio
from typing import Any

from brain.core import Brain
from bridge.exporter import BrainStateExporter
from bridge.memory_tools import MemoryTools
from capabilities.catalog import get_catalog
from capabilities.grants import GrantStore


class ToolRegistry:
    """Serves both built-in memory tools and granted capability tools."""

    def __init__(self, brain: Brain, exporter: BrainStateExporter, grant_store: GrantStore) -> None:
        self.memory_tools = MemoryTools(brain, exporter)
        self.grant_store = grant_store
        self.catalog = get_catalog()
        self._granted_names: set[str] = set(grant_store.list_grants().keys())

    def refresh_grants(self) -> None:
        self._granted_names = set(self.grant_store.list_grants().keys())

    def tool_definitions(self) -> list[dict[str, Any]]:
        defs = self.memory_tools.tool_definitions()
        for name in self._granted_names:
            entry = self.catalog.get(name)
            if entry:
                defs.append(entry.tool_schema)
        return defs

    def execute(self, name: str, args: dict[str, Any]) -> Any:
        """Synchronous dispatch — for memory tools."""
        if name in ("current_state", "query_concepts", "label_concept", "recall_associations"):
            return self.memory_tools.execute(name, args)
        if name not in self._granted_names:
            return {"error": f"Tool '{name}' is not granted. Ask the user to approve it first."}
        entry = self.catalog.get(name)
        if entry is None:
            return {"error": f"Unknown tool: {name}"}
        # Sync wrapper for async tools
        if asyncio.iscoroutinefunction(entry.execute_fn):
            return {"error": f"Tool '{name}' is async — use execute_async()"}
        # Handle local_files action dispatch
        if name == "local_files":
            from capabilities.tools.local_files import read_file, write_file, list_files
            action = args.get("action", "list")
            if action == "read":
                return read_file(path=args.get("path", ""), base_dir=args.get("base_dir", str(__import__("pathlib").Path.home() / "Notes")))
            elif action == "write":
                return write_file(path=args.get("path", ""), content=args.get("content", ""), base_dir=args.get("base_dir", str(__import__("pathlib").Path.home() / "Notes")))
            else:
                return list_files(base_dir=args.get("base_dir", str(__import__("pathlib").Path.home() / "Notes")))
        return entry.execute_fn(**args)

    async def execute_async(self, name: str, args: dict[str, Any]) -> Any:
        """Async dispatch — for capability tools (web_search, shell)."""
        if name in ("current_state", "query_concepts", "label_concept", "recall_associations"):
            return self.memory_tools.execute(name, args)
        if name not in self._granted_names:
            return {"error": f"Tool '{name}' is not granted."}
        entry = self.catalog.get(name)
        if entry is None:
            return {"error": f"Unknown tool: {name}"}
        if name == "local_files":
            return self.execute(name, args)
        if asyncio.iscoroutinefunction(entry.execute_fn):
            return await entry.execute_fn(**args)
        return entry.execute_fn(**args)
```

**Step 3:** Run tests + commit.

```bash
.venv/bin/pytest tests/test_tool_registry.py -v
.venv/bin/pytest 2>&1 | tail -3
git add capabilities/registry.py tests/test_tool_registry.py
git commit -m "feat(capabilities): ToolRegistry unifies memory tools + granted capabilities"
```

Expected: 5 new tests, ~154 total.

---

## Task 3: Wish Detector

**Files:**
- Create: `/Users/leonmatthies/brAIntest/capabilities/wish_detector.py`
- Create: `/Users/leonmatthies/brAIntest/tests/test_wish_detector.py`

**Step 1: Write failing tests**

```python
"""Tests for the wish detector — matches concept labels to catalog entries."""
import pytest
from capabilities.wish_detector import match_concepts_to_catalog, WishCandidate
from capabilities.catalog import get_catalog


def test_match_finds_web_search_for_googlen():
    catalog = get_catalog()
    concepts = [{"id": 12, "label": "googlen", "activation": 0.8, "times_active_24h": 15}]
    matches = match_concepts_to_catalog(concepts, catalog, granted=set(), cooldown=set())
    assert len(matches) == 1
    assert matches[0].tool_name == "web_search"
    assert matches[0].concept_label == "googlen"


def test_match_finds_shell_for_terminal():
    catalog = get_catalog()
    concepts = [{"id": 5, "label": "terminal-arbeit", "activation": 0.5, "times_active_24h": 20}]
    matches = match_concepts_to_catalog(concepts, catalog, granted=set(), cooldown=set())
    assert any(m.tool_name == "shell" for m in matches)


def test_match_skips_already_granted():
    catalog = get_catalog()
    concepts = [{"id": 12, "label": "googlen", "activation": 0.8, "times_active_24h": 15}]
    matches = match_concepts_to_catalog(concepts, catalog, granted={"web_search"}, cooldown=set())
    assert len(matches) == 0


def test_match_skips_cooldown():
    catalog = get_catalog()
    concepts = [{"id": 12, "label": "googlen", "activation": 0.8, "times_active_24h": 15}]
    matches = match_concepts_to_catalog(concepts, catalog, granted=set(), cooldown={"web_search"})
    assert len(matches) == 0


def test_match_skips_low_activity():
    catalog = get_catalog()
    concepts = [{"id": 12, "label": "googlen", "activation": 0.8, "times_active_24h": 3}]  # < 10 threshold
    matches = match_concepts_to_catalog(concepts, catalog, granted=set(), cooldown=set())
    assert len(matches) == 0


def test_match_skips_unlabeled():
    catalog = get_catalog()
    concepts = [{"id": 12, "label": None, "activation": 0.8, "times_active_24h": 50}]
    matches = match_concepts_to_catalog(concepts, catalog, granted=set(), cooldown=set())
    assert len(matches) == 0
```

**Step 2:** Implement `capabilities/wish_detector.py`:

```python
"""Wish detector — matches stable concept labels to catalog trigger keywords.

Runs periodically (every 5 min in production). For each labeled concept with
high activity, checks if any catalog entry's trigger keywords match the label.
If so, generates a WishCandidate.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from capabilities.catalog import CatalogEntry


@dataclass
class WishCandidate:
    tool_name: str
    concept_id: int
    concept_label: str
    match_keyword: str
    activity_count: int


def match_concepts_to_catalog(
    concepts: list[dict[str, Any]],
    catalog: dict[str, CatalogEntry],
    granted: set[str],
    cooldown: set[str],
    min_activity: int = 10,
) -> list[WishCandidate]:
    """Match labeled, active concepts against catalog trigger keywords."""
    candidates = []
    for concept in concepts:
        label = concept.get("label")
        if not label:
            continue
        activity = concept.get("times_active_24h", 0)
        if activity < min_activity:
            continue
        label_lower = label.lower()
        for tool_name, entry in catalog.items():
            if tool_name in granted or tool_name in cooldown:
                continue
            for keyword in entry.trigger_keywords:
                if keyword.lower() in label_lower or label_lower in keyword.lower():
                    candidates.append(WishCandidate(
                        tool_name=tool_name,
                        concept_id=concept.get("id", 0),
                        concept_label=label,
                        match_keyword=keyword,
                        activity_count=activity,
                    ))
                    break  # one match per tool per concept
    return candidates
```

**Step 3:** Run tests + commit.

```bash
.venv/bin/pytest tests/test_wish_detector.py -v
.venv/bin/pytest 2>&1 | tail -3
git add capabilities/wish_detector.py tests/test_wish_detector.py
git commit -m "feat(capabilities): wish detector matches concept labels to catalog"
```

Expected: 6 new tests, ~160 total.

---

## Task 4: Wire into daemon + dashboard grant UI

**Files:**
- Modify: `/Users/leonmatthies/brAIntest/server/braind.py` (add grant store + registry + wish loop)
- Modify: `/Users/leonmatthies/brAIntest/server/chat.py` (use ToolRegistry instead of MemoryTools)
- Create: `/Users/leonmatthies/brAIntest/server/grants.py` (grant/deny/wish API endpoints)
- Modify: `/Users/leonmatthies/brAIntest/ui/src/App.tsx` or new component for grant requests
- Create: `/Users/leonmatthies/brAIntest/tests/test_grant_endpoints.py`

**Step 1: Write failing tests for grant endpoints**

```python
"""Tests for grant API endpoints."""
import tempfile
from pathlib import Path
import pytest
from httpx import AsyncClient, ASGITransport
from capabilities.grants import GrantStore
from server.grants import build_grants_router


@pytest.mark.asyncio
async def test_list_wishes_empty():
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "test.sqlite")
        router = build_grants_router(store, refresh_fn=lambda: None)
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/wishes")
        assert resp.status_code == 200
        assert resp.json()["wishes"] == []


@pytest.mark.asyncio
async def test_grant_wish():
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "test.sqlite")
        wish_id = store.record_wish("web_search", concept_id=12, concept_label="googlen")
        refreshed = []
        router = build_grants_router(store, refresh_fn=lambda: refreshed.append(True))
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/wishes/grant", json={"wish_id": wish_id, "tool_name": "web_search"})
        assert resp.status_code == 200
        assert store.is_granted("web_search")
        assert len(refreshed) == 1  # refresh_fn was called
```

**Step 2:** Implement `server/grants.py`:

```python
"""Grant API endpoints — /api/wishes, /api/wishes/grant, /api/wishes/deny."""
from __future__ import annotations
from typing import Any, Callable
from fastapi import APIRouter
from pydantic import BaseModel
from capabilities.grants import GrantStore


class GrantRequest(BaseModel):
    wish_id: int
    tool_name: str


class DenyRequest(BaseModel):
    wish_id: int
    tool_name: str


def build_grants_router(store: GrantStore, refresh_fn: Callable[[], None]) -> APIRouter:
    api = APIRouter()

    @api.get("/api/wishes")
    async def list_wishes() -> dict[str, Any]:
        return {"wishes": store.list_wishes(status="pending")}

    @api.get("/api/grants")
    async def list_grants() -> dict[str, Any]:
        return {"grants": store.list_grants()}

    @api.post("/api/wishes/grant")
    async def grant_wish(req: GrantRequest) -> dict[str, str]:
        store.grant(req.tool_name)
        store.resolve_wish(req.wish_id, "granted")
        refresh_fn()
        return {"status": "granted", "tool_name": req.tool_name}

    @api.post("/api/wishes/deny")
    async def deny_wish(req: DenyRequest) -> dict[str, str]:
        store.deny(req.tool_name, cooldown_days=7)
        store.resolve_wish(req.wish_id, "denied")
        return {"status": "denied", "tool_name": req.tool_name}

    return api
```

**Step 3:** Wire into braind.py — replace MemoryTools with ToolRegistry, add GrantStore, add wish detector loop, add grants router. Wire into chat.py — change tool execution to use `registry.execute_async()` for capability tools.

**Step 4:** Add a `WishBanner` component to the dashboard that polls `/api/wishes` and shows pending wishes with Grant/Deny buttons.

**Step 5:** Run tests + build frontend + commit.

```bash
.venv/bin/pytest tests/test_grant_endpoints.py -v
.venv/bin/pytest 2>&1 | tail -3
cd ui && npm run build && cd ..
git add server/grants.py server/braind.py server/chat.py capabilities/ ui/src/ tests/test_grant_endpoints.py
git commit -m "feat: wire capability system into daemon + dashboard grant UI"
```

Expected: ~2 new tests, ~162 total.

---

## Task 5: Tag phase-4-complete + README

**Step 1:** Run full suite, build frontend.

**Step 2:** Update README status:
```markdown
- [x] Phase 4: Capability wishlist + grant system (web search, shell, local files)
```

**Step 3:** Commit + tag.

```bash
git add README.md
git commit -m "docs: mark Phase 4 complete"
git tag phase-4-complete -m "Phase 4: emergent capability acquisition"
```

---

## Phase 4 Done. What's next?

Phase 4 delivers:
- Capability catalog with 3 tools (web search, sandboxed shell, local files)
- Wish detector matching concept labels to catalog trigger keywords
- Grant/deny flow with SQLite persistence and 7-day cooldown
- ToolRegistry serving memory + granted tools to the LLM
- Dashboard grant-request UI
- ~162 tests, all green

Phase 5: Standing Orders + more tools in catalog.
