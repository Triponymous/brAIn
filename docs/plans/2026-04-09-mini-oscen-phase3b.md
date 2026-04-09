# Mini-OSCEN Phase 3b Implementation Plan — Brain Talks To Me

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the LLM bridge (brain state exporter, memory tools, hybrid Ollama/Claude router, chat endpoint) and the dashboard frontend (React three-column layout with brain canvas visualization, sensor display, and chat panel) so the user can open a browser, see the brain live, and ask "what are you seeing?" and get a real answer derived from brain state.

**Architecture:** The bridge lives in `bridge/` and exposes memory tools as Python functions called by the LLM via tool-use. The hybrid router sends simple queries to local Ollama and complex ones to Claude API. The `/chat` endpoint accepts user messages, augments them with brain state context, calls the router, and streams the response. The dashboard is a Vite+React+TypeScript app in `ui/` that connects to the existing WebSocket for brain state and uses `/chat` REST for conversation. Brain visualization uses plain Canvas 2D.

**Tech Stack:** Python (FastAPI, httpx for Ollama, anthropic SDK for Claude), Vite + React 18 + TypeScript + Tailwind + shadcn/ui, HTML Canvas 2D.

**Reference docs:**
- `docs/plans/2026-04-09-mini-oscen-phase3-design.md` — Phase 3+ design (Sections 3 + 4)
- `docs/plans/2026-04-08-mini-oscen-design.md` — original design

**Out of scope:** Pet face Tauri app (Phase 3c), TTS/STT voice (Phase 3c), agent tools (Phase 4), capability wishlist (Phase 4).

**Dependencies on Phase 3a (already complete):**
- `server/main.py` with `build_app()`, `/ws` endpoint, `push_loop()`
- `brain/core.py` with `Brain` class (regions, synapses, modulators, tick_count)
- `brain/persistence.py` with `save_brain()`/`load_brain()`
- `adapters/mac_desktop/adapter.py` with `MacDesktopAdapter`

**Important carry-over from Phase 3a final review:**
- The broadcast payload at `server/main.py:71-76` needs enrichment (active concepts, sensor summary)
- Use `app.include_router()` to add `/chat` — don't modify `build_app()`
- Time-tonic neurons always fire near threshold (drive≈1.5) → brain may appear always-excited. Consider a tonic-gain parameter or accept as known limitation.

**Success criteria for Phase 3b:**
1. LLM bridge answers ≥3 questions about brain state with correct, tool-derived data.
2. Dashboard shows brain viz (region graph + spike raster + concept cloud), sensor panel, and chat in a three-column layout.
3. Concept labeling works from dashboard (click unnamed concept → type label → saved to DB).
4. Hybrid router uses local Ollama by default and Claude for complex queries.
5. Test count grows from 100 to ~115, all passing.

---

## Task 0: Add Phase 3b dependencies (Ollama client, Anthropic SDK, React toolchain)

**Files:**
- Modify: `/Users/leonmatthies/brAIntest/pyproject.toml`

**Step 1: Add anthropic SDK to Python deps**

Edit `pyproject.toml`. Find the dependencies list and append after the pyobjc entries:

```toml
    # Phase 3b — LLM bridge
    "anthropic>=0.40",
```

**Step 2: Install**

```bash
cd /Users/leonmatthies/brAIntest
uv pip install -e ".[dev]"
```

**Step 3: Verify**

```bash
.venv/bin/python -c "import anthropic; print('anthropic OK', anthropic.__version__)"
```

Note: Ollama uses a plain HTTP API — no SDK needed (we use httpx which is already installed). For local Ollama, the user must have `ollama` installed and a model pulled (`ollama pull qwen2.5:7b-instruct`). Tests will mock the Ollama HTTP calls.

**Step 4: Initialize React project**

```bash
cd /Users/leonmatthies/brAIntest
npm create vite@latest ui -- --template react-ts
cd ui
npm install
npm install -D tailwindcss @tailwindcss/vite
npm install react-markdown
```

Then configure Tailwind. Create/edit `ui/vite.config.ts`:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
})
```

Replace `ui/src/index.css` with:

```css
@import "tailwindcss";
```

**Step 5: Verify React builds**

```bash
cd /Users/leonmatthies/brAIntest/ui
npm run build
```

Expected: build succeeds with no errors.

**Step 6: Run existing Python tests**

```bash
cd /Users/leonmatthies/brAIntest
.venv/bin/pytest 2>&1 | tail -3
```

Expected: 100 tests pass.

**Step 7: Commit**

```bash
cd /Users/leonmatthies/brAIntest
# Add ui/ but exclude node_modules and dist
echo "node_modules/" >> ui/.gitignore
echo "dist/" >> ui/.gitignore
git add pyproject.toml ui/
git commit -m "chore: add Phase 3b deps (anthropic SDK, Vite+React+Tailwind)"
```

---

## Task 1: Brain State Exporter

**Files:**
- Create: `/Users/leonmatthies/brAIntest/bridge/__init__.py`
- Create: `/Users/leonmatthies/brAIntest/bridge/exporter.py`
- Create: `/Users/leonmatthies/brAIntest/tests/test_exporter.py`

**Step 1: Write failing tests**

```python
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
    # Each concept entry has at least id and activation
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
    labeled = [c for c in snap["active_concepts"] if c.get("label") == "tippen"]
    # The label should appear on concept 0 regardless of whether it's active
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
```

**Step 2: Run to fail**

```bash
.venv/bin/pytest tests/test_exporter.py -v
```

**Step 3: Implement**

Create `/Users/leonmatthies/brAIntest/bridge/__init__.py` (empty).

Create `/Users/leonmatthies/brAIntest/bridge/exporter.py`:

```python
"""BrainStateExporter — produces structured LLM-readable snapshots of the brain.

Consumed by the LLM bridge system prompt and by the memory tools. Updated
~1 Hz in production but callable on-demand via snapshot().

Concept labels are stored in-memory and persisted via the label_concept
memory tool (Phase 3b) or directly by the dashboard concept-labeling UI.
"""
from __future__ import annotations
from typing import Any

import torch

from brain.core import Brain


class BrainStateExporter:
    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self._labels: dict[int, str] = {}

    def set_label(self, concept_id: int, label: str) -> None:
        self._labels[concept_id] = label

    def get_label(self, concept_id: int) -> str | None:
        return self._labels.get(concept_id)

    def all_labels(self) -> dict[int, str]:
        return dict(self._labels)

    def snapshot(self, sensor_summary: dict[str, Any] | None = None) -> dict[str, Any]:
        """Produce a structured snapshot of the brain state."""
        brain = self.brain

        # Active concepts: concept layer membrane values as proxy for "activation"
        concept_layer = brain.regions["concept"]
        membrane = concept_layer.membrane
        num_concepts = concept_layer.num_neurons

        active_concepts = []
        for i in range(num_concepts):
            activation = float(membrane[i].item())
            entry: dict[str, Any] = {
                "id": i,
                "activation": round(activation, 4),
            }
            label = self._labels.get(i)
            if label is not None:
                entry["label"] = label
            active_concepts.append(entry)

        # Sort by activation descending, keep top 12
        active_concepts.sort(key=lambda x: x["activation"], reverse=True)
        active_concepts = active_concepts[:12]

        return {
            "tick_count": brain.tick_count,
            "modulators": brain.modulators.snapshot(),
            "active_concepts": active_concepts,
            "sensor_summary": sensor_summary or {},
        }
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_exporter.py -v
```

Expected: 6 pass.

**Step 5: Commit**

```bash
git add bridge/__init__.py bridge/exporter.py tests/test_exporter.py
git commit -m "feat(bridge): BrainStateExporter with concept labels"
```

---

## Task 2: Memory Tools

**Files:**
- Create: `/Users/leonmatthies/brAIntest/bridge/memory_tools.py`
- Create: `/Users/leonmatthies/brAIntest/tests/test_memory_tools.py`

**Step 1: Write failing tests**

```python
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
    # Also appears in query
    state = tools.current_state()
    labeled = [c for c in state["active_concepts"] if c.get("label") == "tippen"]
    # May or may not be in top-12, but the label is stored
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
```

**Step 2: Run to fail**

**Step 3: Implement**

Create `/Users/leonmatthies/brAIntest/bridge/memory_tools.py`:

```python
"""Memory tools — functions the LLM calls via tool-use to query the brain.

Each tool is a method on MemoryTools. tool_definitions() returns the
schema in Anthropic/OpenAI tool-use format so the LLM knows what's available.
execute(name, args) dispatches a tool call by name.
"""
from __future__ import annotations
from typing import Any

import torch

from brain.core import Brain
from bridge.exporter import BrainStateExporter


class MemoryTools:
    def __init__(self, brain: Brain, exporter: BrainStateExporter) -> None:
        self.brain = brain
        self.exporter = exporter

    def current_state(self) -> dict[str, Any]:
        """Full brain state snapshot."""
        return self.exporter.snapshot()

    def query_concepts(self, limit: int = 10, min_activation: float = 0.0) -> list[dict]:
        """Top-N active concept neurons."""
        snap = self.exporter.snapshot()
        concepts = snap["active_concepts"]
        filtered = [c for c in concepts if c["activation"] >= min_activation]
        return filtered[:limit]

    def label_concept(self, concept_id: int, label: str) -> dict[str, str]:
        """Assign a human-readable label to a concept neuron."""
        self.exporter.set_label(concept_id, label)
        return {"status": "ok", "concept_id": concept_id, "label": label}

    def recall_associations(self, concept_id: int) -> list[dict]:
        """Find concepts that are strongly connected to the given concept.

        Reads the association→concept synapse weights to find which other
        concept neurons share strong incoming connections with concept_id.
        """
        syn = self.brain.synapses.get("association_concept")
        if syn is None:
            return []
        weights = syn.weights  # shape (num_concept, num_association)
        target_weights = weights[concept_id]  # shape (num_association,)
        # Cosine similarity between this concept's weight vector and all others
        norms = torch.norm(weights, dim=1)
        target_norm = torch.norm(target_weights)
        if target_norm < 1e-8:
            return []
        similarities = (weights @ target_weights) / (norms * target_norm + 1e-8)
        similarities[concept_id] = -1  # exclude self
        top_k = min(5, len(similarities))
        values, indices = torch.topk(similarities, top_k)
        result = []
        for idx, sim in zip(indices.tolist(), values.tolist()):
            if sim > 0.1:
                entry = {"concept_id": idx, "similarity": round(sim, 4)}
                label = self.exporter.get_label(idx)
                if label:
                    entry["label"] = label
                result.append(entry)
        return result

    def execute(self, name: str, args: dict[str, Any]) -> Any:
        """Dispatch a tool call by name."""
        dispatch = {
            "current_state": lambda: self.current_state(),
            "query_concepts": lambda: self.query_concepts(**args),
            "label_concept": lambda: self.label_concept(**args),
            "recall_associations": lambda: self.recall_associations(**args),
        }
        fn = dispatch.get(name)
        if fn is None:
            return {"error": f"Unknown tool: {name}"}
        return fn()

    def tool_definitions(self) -> list[dict[str, Any]]:
        """Return tool schemas for the LLM system prompt."""
        return [
            {
                "name": "current_state",
                "description": "Get the full current brain state snapshot including modulators, active concepts, and sensor summary.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "query_concepts",
                "description": "Get the top-N most active concept neurons. Optionally filter by minimum activation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 10},
                        "min_activation": {"type": "number", "default": 0.0},
                    },
                },
            },
            {
                "name": "label_concept",
                "description": "Assign a human-readable label to a concept neuron. Use when the user tells you what a concept represents.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "concept_id": {"type": "integer"},
                        "label": {"type": "string"},
                    },
                    "required": ["concept_id", "label"],
                },
            },
            {
                "name": "recall_associations",
                "description": "Find concepts that are strongly connected to a given concept via shared synaptic weights.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "concept_id": {"type": "integer"},
                    },
                    "required": ["concept_id"],
                },
            },
        ]
```

**Step 4: Run tests, commit**

```bash
.venv/bin/pytest tests/test_memory_tools.py -v
git add bridge/memory_tools.py tests/test_memory_tools.py
git commit -m "feat(bridge): memory tools (current_state, query_concepts, label, recall)"
```

---

## Task 3: Hybrid LLM Router (Ollama local + Claude cloud)

**Files:**
- Create: `/Users/leonmatthies/brAIntest/bridge/llm_local.py`
- Create: `/Users/leonmatthies/brAIntest/bridge/llm_cloud.py`
- Create: `/Users/leonmatthies/brAIntest/bridge/llm_router.py`
- Create: `/Users/leonmatthies/brAIntest/tests/test_llm_router.py`

**Step 1: Write failing tests**

```python
"""Tests for the hybrid LLM router.

In tests we mock both Ollama and Claude to avoid real API calls.
The router logic (which backend to use) is tested via the routing heuristics.
"""
import pytest
from unittest.mock import AsyncMock, patch
from bridge.llm_router import HybridLLMRouter, _should_use_cloud


def test_should_use_cloud_short_message():
    assert _should_use_cloud("hey was siehst du?", {}) is False


def test_should_use_cloud_long_message():
    long = "Kannst du mir erklären warum " + "Concept #12 " * 50 + "immer aktiv ist?"
    assert _should_use_cloud(long, {}) is True


def test_should_use_cloud_reasoning_keyword():
    assert _should_use_cloud("analysiere mein Tagesrhythmus", {}) is True


def test_should_use_cloud_high_ne():
    assert _should_use_cloud("was war das?", {"NE": 0.8}) is True


def test_should_use_cloud_explicit_flag():
    assert _should_use_cloud("/cloud was ist los", {}) is True


def test_router_construction():
    router = HybridLLMRouter()
    assert router.ollama_model == "qwen2.5:7b-instruct"
    assert router.cloud_model == "claude-haiku-4-5-20250404"


@pytest.mark.asyncio
async def test_router_calls_local_for_simple_query():
    router = HybridLLMRouter()
    with patch.object(router, '_call_ollama', new_callable=AsyncMock, return_value="Ich sehe VSCode.") as mock:
        result = await router.chat(
            user_message="was siehst du?",
            system_prompt="Du bist ein Pet.",
            brain_state={"modulators": {"NE": 0.1}},
            tools=[],
        )
    mock.assert_called_once()
    assert result["text"] == "Ich sehe VSCode."
    assert result["backend"] == "local"


@pytest.mark.asyncio
async def test_router_calls_cloud_for_complex_query():
    router = HybridLLMRouter()
    with patch.object(router, '_call_claude', new_callable=AsyncMock, return_value={"text": "Analyse...", "tool_calls": []}) as mock:
        result = await router.chat(
            user_message="analysiere warum Concept #12 und #47 immer zusammen feuern",
            system_prompt="Du bist ein Pet.",
            brain_state={"modulators": {"NE": 0.1}},
            tools=[],
        )
    mock.assert_called_once()
    assert result["backend"] == "cloud"
```

**Step 2: Run to fail**

**Step 3: Implement**

Create `/Users/leonmatthies/brAIntest/bridge/llm_local.py`:

```python
"""Ollama HTTP client for local LLM inference.

Uses the Ollama HTTP API (http://localhost:11434/api/chat) directly via httpx.
No Ollama SDK needed. Falls back gracefully if Ollama is not running.
"""
from __future__ import annotations
from typing import Any
import httpx


OLLAMA_BASE = "http://localhost:11434"


async def ollama_chat(
    model: str,
    system_prompt: str,
    user_message: str,
    timeout: float = 30.0,
) -> str:
    """Send a chat request to Ollama and return the assistant's response text."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]
```

Create `/Users/leonmatthies/brAIntest/bridge/llm_cloud.py`:

```python
"""Anthropic Claude client for cloud LLM inference with tool-use support."""
from __future__ import annotations
from typing import Any
import anthropic


async def claude_chat(
    model: str,
    system_prompt: str,
    user_message: str,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Send a chat request to Claude and return text + any tool calls."""
    client = anthropic.AsyncAnthropic()

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }
    if tools:
        # Convert our tool defs to Anthropic format
        kwargs["tools"] = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in tools
        ]

    response = await client.messages.create(**kwargs)

    text_parts = []
    tool_calls = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({"name": block.name, "args": block.input})

    return {
        "text": "\n".join(text_parts),
        "tool_calls": tool_calls,
    }
```

Create `/Users/leonmatthies/brAIntest/bridge/llm_router.py`:

```python
"""Hybrid LLM Router — local Ollama by default, Claude for complex queries.

Routing heuristics:
- Cloud if: message > 200 chars, reasoning keywords, NE > 0.7, /cloud prefix
- Local otherwise (faster, free, private)
"""
from __future__ import annotations
import re
from typing import Any

from bridge.llm_local import ollama_chat
from bridge.llm_cloud import claude_chat


_REASONING_KEYWORDS = re.compile(
    r"plan|recherchier|analysier|warum genau|vergleich|erkl.r mir|zusammenfassung",
    re.IGNORECASE,
)


def _should_use_cloud(message: str, modulators: dict[str, float]) -> bool:
    if message.startswith("/cloud"):
        return True
    if len(message) > 200:
        return True
    if _REASONING_KEYWORDS.search(message):
        return True
    if modulators.get("NE", 0) > 0.7:
        return True
    return False


class HybridLLMRouter:
    def __init__(
        self,
        ollama_model: str = "qwen2.5:7b-instruct",
        cloud_model: str = "claude-haiku-4-5-20250404",
    ) -> None:
        self.ollama_model = ollama_model
        self.cloud_model = cloud_model

    async def chat(
        self,
        user_message: str,
        system_prompt: str,
        brain_state: dict[str, Any],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Route the message to local or cloud LLM."""
        modulators = brain_state.get("modulators", {})
        use_cloud = _should_use_cloud(user_message, modulators)

        if use_cloud:
            result = await self._call_claude(
                system_prompt=system_prompt,
                user_message=user_message.removeprefix("/cloud").strip(),
                tools=tools,
            )
            result["backend"] = "cloud"
            return result
        else:
            text = await self._call_ollama(
                system_prompt=system_prompt,
                user_message=user_message,
            )
            return {"text": text, "tool_calls": [], "backend": "local"}

    async def _call_ollama(self, system_prompt: str, user_message: str) -> str:
        return await ollama_chat(
            model=self.ollama_model,
            system_prompt=system_prompt,
            user_message=user_message,
        )

    async def _call_claude(
        self, system_prompt: str, user_message: str, tools: list[dict]
    ) -> dict[str, Any]:
        return await claude_chat(
            model=self.cloud_model,
            system_prompt=system_prompt,
            user_message=user_message,
            tools=tools or None,
        )
```

**Step 4: Run tests, commit**

```bash
.venv/bin/pytest tests/test_llm_router.py -v
git add bridge/llm_local.py bridge/llm_cloud.py bridge/llm_router.py tests/test_llm_router.py
git commit -m "feat(bridge): hybrid LLM router (Ollama local + Claude cloud)"
```

---

## Task 4: Chat endpoint + enriched WebSocket broadcast

**Files:**
- Create: `/Users/leonmatthies/brAIntest/server/chat.py`
- Modify: `/Users/leonmatthies/brAIntest/server/main.py` (enrich push_loop + mount chat router)
- Modify: `/Users/leonmatthies/brAIntest/server/braind.py` (wire exporter + tools + router)
- Create: `/Users/leonmatthies/brAIntest/tests/test_chat_endpoint.py`

This is the glue task — it connects the bridge to the server. Since it modifies existing files, follow the plan step by step.

**Step 1: Write failing tests**

Create `/Users/leonmatthies/brAIntest/tests/test_chat_endpoint.py`:

```python
"""Tests for the /api/chat endpoint.

We mock the LLM router to avoid real API calls. The test verifies:
- The endpoint accepts a message and returns a response
- The response includes text and backend info
- Tool calls are executed and results included
"""
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport

from brain.core import Brain
from bridge.exporter import BrainStateExporter
from bridge.memory_tools import MemoryTools
from bridge.llm_router import HybridLLMRouter
from server.chat import build_chat_router


@pytest.mark.asyncio
async def test_chat_returns_response():
    brain = Brain(num_sensory=8, num_concept=4)
    exporter = BrainStateExporter(brain)
    tools = MemoryTools(brain, exporter)
    router = HybridLLMRouter()

    chat_router = build_chat_router(brain, exporter, tools, router)

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(chat_router)

    with patch.object(router, '_call_ollama', new_callable=AsyncMock, return_value="Alles ruhig."):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/chat", json={"message": "was siehst du?"})

    assert resp.status_code == 200
    body = resp.json()
    assert "text" in body
    assert "backend" in body
    assert body["text"] == "Alles ruhig."


@pytest.mark.asyncio
async def test_chat_with_tool_call():
    """When the LLM returns a tool call, the endpoint executes it."""
    brain = Brain(num_sensory=8, num_concept=4)
    exporter = BrainStateExporter(brain)
    tools = MemoryTools(brain, exporter)
    router = HybridLLMRouter()

    chat_router = build_chat_router(brain, exporter, tools, router)

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(chat_router)

    # Simulate Claude returning a tool call
    mock_response = {
        "text": "",
        "tool_calls": [{"name": "current_state", "args": {}}],
        "backend": "cloud",
    }
    with patch.object(router, 'chat', new_callable=AsyncMock, return_value=mock_response):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/chat", json={"message": "/cloud zeig mir alles"})

    assert resp.status_code == 200
    body = resp.json()
    assert "tool_results" in body
    assert len(body["tool_results"]) == 1


@pytest.mark.asyncio
async def test_chat_label_endpoint():
    """POST /api/label sets a concept label."""
    brain = Brain(num_sensory=8, num_concept=4)
    exporter = BrainStateExporter(brain)
    tools = MemoryTools(brain, exporter)
    router = HybridLLMRouter()

    chat_router = build_chat_router(brain, exporter, tools, router)

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(chat_router)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/label", json={"concept_id": 2, "label": "tippen"})

    assert resp.status_code == 200
    assert exporter.get_label(2) == "tippen"
```

**Step 2: Run to fail**

**Step 3: Implement chat router**

Create `/Users/leonmatthies/brAIntest/server/chat.py`:

```python
"""Chat endpoint — /api/chat and /api/label.

The /api/chat endpoint:
1. Takes a user message
2. Builds a system prompt with the current brain state
3. Routes to local or cloud LLM via HybridLLMRouter
4. If the LLM returns tool calls, executes them via MemoryTools
5. Returns the response text + any tool results

The /api/label endpoint:
- Direct concept labeling from the dashboard UI (no LLM involved)
"""
from __future__ import annotations
import json
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from brain.core import Brain
from bridge.exporter import BrainStateExporter
from bridge.memory_tools import MemoryTools
from bridge.llm_router import HybridLLMRouter


_SYSTEM_PROMPT_TEMPLATE = """You are the language interface to a persistent neuromorphic brain that lives on its user's Mac. You DO NOT learn — the brain learns. You translate the brain's state into language and accept user labels for unnamed concepts.

Speak in the first person as if you ARE the pet. Be concise. Use German if the user writes in German, English if they write in English.

Don't invent memories the brain doesn't have. If you don't see a concept in the state, say "I don't have a clear memory of that."

Current brain state:
{brain_state}

Known concept labels:
{labels}
"""


class ChatRequest(BaseModel):
    message: str


class LabelRequest(BaseModel):
    concept_id: int
    label: str


def build_chat_router(
    brain: Brain,
    exporter: BrainStateExporter,
    tools: MemoryTools,
    router: HybridLLMRouter,
) -> APIRouter:
    api = APIRouter()

    @api.post("/api/chat")
    async def chat(req: ChatRequest) -> dict[str, Any]:
        # Build system prompt with current brain state
        snap = exporter.snapshot()
        labels = exporter.all_labels()
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            brain_state=json.dumps(snap, indent=2, default=str),
            labels=json.dumps(labels, default=str) if labels else "None yet.",
        )

        # Route to LLM
        result = await router.chat(
            user_message=req.message,
            system_prompt=system_prompt,
            brain_state=snap,
            tools=tools.tool_definitions(),
        )

        # Execute any tool calls
        tool_results = []
        for tc in result.get("tool_calls", []):
            tr = tools.execute(tc["name"], tc.get("args", {}))
            tool_results.append({"tool": tc["name"], "result": tr})

        return {
            "text": result.get("text", ""),
            "backend": result.get("backend", "unknown"),
            "tool_results": tool_results,
        }

    @api.post("/api/label")
    async def label(req: LabelRequest) -> dict[str, str]:
        exporter.set_label(req.concept_id, req.label)
        return {"status": "ok", "concept_id": str(req.concept_id), "label": req.label}

    return api


**Step 4: Wire into braind.py and main.py**

Edit `/Users/leonmatthies/brAIntest/server/braind.py` — in the `_run_daemon` function, after `adapter = MacDesktopAdapter(...)` and before `app = build_app(...)`, add the bridge setup:

```python
    # Bridge setup
    from bridge.exporter import BrainStateExporter
    from bridge.memory_tools import MemoryTools
    from bridge.llm_router import HybridLLMRouter
    from server.chat import build_chat_router

    exporter = BrainStateExporter(brain)
    memory_tools = MemoryTools(brain, exporter)
    llm_router = HybridLLMRouter()
    chat_router = build_chat_router(brain, exporter, memory_tools, llm_router)
```

Then after `app = build_app(...)`, add:

```python
    app.include_router(chat_router)
```

Edit `/Users/leonmatthies/brAIntest/server/main.py` — in the `push_loop` function, enrich the state dict at line 71-75. Replace the state dict with:

```python
            state = {
                "tick": brain.tick_count,
                "modulators": brain.modulators.snapshot(),
                "concept_membrane": brain.regions["concept"].membrane.tolist(),
                "wm_membrane": brain.regions["wm"].membrane.tolist(),
            }
```

**Step 5: Run tests, commit**

```bash
.venv/bin/pytest tests/test_chat_endpoint.py -v
.venv/bin/pytest -v 2>&1 | tail -3
git add server/chat.py server/braind.py server/main.py tests/test_chat_endpoint.py
git commit -m "feat(server): /api/chat + /api/label endpoints with LLM bridge wiring"
```

---

## Task 5: Dashboard frontend — project scaffold + three-column layout

**Files:**
- Create/edit: `ui/src/App.tsx`, `ui/src/components/` (multiple)
- Create: `ui/src/lib/ws.ts`

This task sets up the three-column layout shell and the WebSocket connection. No brain viz yet — that comes in Tasks 6-7.

**Step 1: Create the WebSocket client**

Create `ui/src/lib/ws.ts`:

```typescript
// WebSocket client for brain state stream

export type BrainState = {
  tick: number;
  modulators: Record<string, number>;
  concept_membrane: number[];
  wm_membrane: number[];
};

type Listener = (state: BrainState) => void;

let ws: WebSocket | null = null;
let listeners: Set<Listener> = new Set();
let latestState: BrainState | null = null;

export function connectWS() {
  const url = `ws://${window.location.host}/ws`;
  ws = new WebSocket(url);
  ws.onmessage = (ev) => {
    try {
      const state: BrainState = JSON.parse(ev.data);
      latestState = state;
      listeners.forEach((fn) => fn(state));
    } catch {}
  };
  ws.onclose = () => {
    setTimeout(connectWS, 2000); // reconnect
  };
}

export function subscribe(fn: Listener): () => void {
  listeners.add(fn);
  if (latestState) fn(latestState); // immediate update
  return () => listeners.delete(fn);
}

export function getLatest(): BrainState | null {
  return latestState;
}
```

**Step 2: Create the three-column layout App**

Replace `ui/src/App.tsx` with:

```tsx
import { useEffect, useState } from "react";
import { connectWS, subscribe, BrainState } from "./lib/ws";
import { SensorPanel } from "./components/SensorPanel";
import { BrainPanel } from "./components/BrainPanel";
import { ChatPanel } from "./components/ChatPanel";

export default function App() {
  const [state, setState] = useState<BrainState | null>(null);

  useEffect(() => {
    connectWS();
    return subscribe(setState);
  }, []);

  return (
    <div className="h-screen bg-gray-950 text-gray-100 flex flex-col">
      {/* Header */}
      <header className="h-10 px-4 flex items-center gap-4 bg-gray-900 border-b border-gray-800 text-sm font-mono">
        <span className="text-green-400">●</span>
        <span>braind</span>
        <span className="text-gray-500">
          tick {state?.tick ?? "—"}
        </span>
        <span className="text-gray-500">
          DA {state?.modulators?.DA?.toFixed(2) ?? "—"}
          {" "}NE {state?.modulators?.NE?.toFixed(2) ?? "—"}
          {" "}ACh {state?.modulators?.ACh?.toFixed(2) ?? "—"}
          {" "}5HT {state?.modulators?.["5HT"]?.toFixed(2) ?? "—"}
        </span>
      </header>

      {/* Three columns */}
      <div className="flex-1 flex overflow-hidden">
        <div className="w-64 border-r border-gray-800 overflow-y-auto p-3">
          <SensorPanel state={state} />
        </div>
        <div className="flex-1 overflow-hidden p-3">
          <BrainPanel state={state} />
        </div>
        <div className="w-80 border-l border-gray-800 flex flex-col">
          <ChatPanel />
        </div>
      </div>
    </div>
  );
}
```

**Step 3: Create stub components**

Create `ui/src/components/SensorPanel.tsx`:

```tsx
import type { BrainState } from "../lib/ws";

export function SensorPanel({ state }: { state: BrainState | null }) {
  if (!state) return <div className="text-gray-500">Connecting...</div>;

  return (
    <div className="space-y-4 text-xs font-mono">
      <h3 className="text-gray-400 uppercase tracking-wider text-[10px]">Sensors</h3>
      <div>
        <div className="text-gray-500">Modulators</div>
        {Object.entries(state.modulators).map(([k, v]) => (
          <div key={k} className="flex justify-between">
            <span>{k}</span>
            <span>{typeof v === 'number' ? v.toFixed(3) : '—'}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
```

Create `ui/src/components/BrainPanel.tsx`:

```tsx
import type { BrainState } from "../lib/ws";

export function BrainPanel({ state }: { state: BrainState | null }) {
  if (!state) return <div className="text-gray-500 text-center mt-20">Waiting for brain state...</div>;

  // Concept membrane as simple bars for now — Task 6 replaces with Canvas
  const membrane = state.concept_membrane ?? [];

  return (
    <div className="h-full flex flex-col gap-3">
      <h3 className="text-gray-400 uppercase tracking-wider text-[10px]">Brain</h3>
      <div className="flex-1 bg-gray-900 rounded-lg p-3 overflow-y-auto">
        <div className="text-xs text-gray-500 mb-2">Concept membrane ({membrane.length} neurons)</div>
        <div className="flex gap-0.5 h-20 items-end">
          {membrane.map((v, i) => (
            <div
              key={i}
              className="flex-1 bg-emerald-500 rounded-t"
              style={{ height: `${Math.min(100, Math.max(2, v * 100))}%`, opacity: 0.3 + v * 0.7 }}
              title={`Concept ${i}: ${v.toFixed(3)}`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
```

Create `ui/src/components/ChatPanel.tsx`:

```tsx
import { useState, useRef, useEffect } from "react";

type Message = {
  role: "user" | "assistant";
  text: string;
  backend?: string;
  tool_results?: Array<{ tool: string; result: unknown }>;
};

export function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async () => {
    const msg = input.trim();
    if (!msg || loading) return;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", text: msg }]);
    setLoading(true);

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg }),
      });
      const data = await resp.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.text || "(no response)",
          backend: data.backend,
          tool_results: data.tool_results,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: `Error: ${err}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <h3 className="text-gray-400 uppercase tracking-wider text-[10px] p-3 pb-1">Chat</h3>

      <div className="flex-1 overflow-y-auto p-3 space-y-3 text-sm">
        {messages.map((m, i) => (
          <div key={i} className={m.role === "user" ? "text-blue-300" : "text-gray-200"}>
            <span className="text-gray-500 text-xs">{m.role === "user" ? "You" : "Pet"}</span>
            {m.backend && (
              <span className="text-gray-600 text-[10px] ml-1">[{m.backend}]</span>
            )}
            <div className="mt-0.5">{m.text}</div>
            {m.tool_results && m.tool_results.length > 0 && (
              <details className="mt-1 text-xs text-gray-500">
                <summary className="cursor-pointer">
                  🧠 {m.tool_results.length} tool call(s)
                </summary>
                <pre className="mt-1 text-[10px] overflow-x-auto">
                  {JSON.stringify(m.tool_results, null, 2)}
                </pre>
              </details>
            )}
          </div>
        ))}
        {loading && <div className="text-gray-500 text-xs animate-pulse">Thinking...</div>}
        <div ref={bottomRef} />
      </div>

      <div className="p-3 border-t border-gray-800">
        <div className="flex gap-2">
          <input
            className="flex-1 bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm focus:outline-none focus:border-gray-500"
            placeholder="Ask the pet..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            disabled={loading}
          />
          <button
            className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded text-sm disabled:opacity-50"
            onClick={send}
            disabled={loading || !input.trim()}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
```

**Step 4: Verify the frontend builds**

```bash
cd /Users/leonmatthies/brAIntest/ui
npm run build
```

Expected: builds without TypeScript errors.

**Step 5: Commit**

```bash
cd /Users/leonmatthies/brAIntest
git add ui/src/
git commit -m "feat(ui): dashboard three-column layout with WS client + chat panel"
```

---

## Task 6: Run full suite + tag phase-3b-complete + README

**Step 1: Run Python tests**

```bash
cd /Users/leonmatthies/brAIntest
.venv/bin/pytest -v 2>&1 | tail -5
```

Expected: ~115 tests pass.

**Step 2: Build frontend**

```bash
cd /Users/leonmatthies/brAIntest/ui
npm run build
```

**Step 3: Update README**

Edit README — replace status section:

```markdown
## Status
- [x] Phase 1: SNN core foundations (LIF, STDP, 2-region brain, viz)
- [x] Phase 2: Full multi-region brain + WTA + modulators + R-STDP + SQLite persistence
- [x] Phase 3a: Mac sensor adapter + daemon (FastAPI + WebSocket)
- [x] Phase 3b: LLM bridge + dashboard
- [ ] Phase 3c: Pet face (Tauri) + voice (TTS/STT)
- [ ] Phase 4+: see `docs/plans/2026-04-09-mini-oscen-phase3-design.md`
```

Add to Quick start section:

```markdown
# Dashboard (open in browser alongside running daemon)
cd ui && npm run dev
# Then open http://localhost:5173
```

**Step 4: Commit + tag**

```bash
cd /Users/leonmatthies/brAIntest
git add README.md
git commit -m "docs: mark Phase 3b complete"
git tag phase-3b-complete -m "Phase 3b: LLM bridge + dashboard"
```

---

## Phase 3b Done. What's next?

Phase 3b delivers:
- Brain state exporter with concept labels
- Memory tools (current_state, query_concepts, label_concept, recall_associations)
- Hybrid LLM router (Ollama local / Claude cloud)
- /api/chat + /api/label REST endpoints
- Enriched WebSocket broadcast (concept + WM membrane)
- React dashboard: three columns (sensors / brain bars / chat)
- ~115 tests all green

Phase 3c entry criteria: start the daemon + open the dashboard + verify you can have a real conversation where the pet's answers trace back to brain state. Then: Phase 3c adds the Tauri pet face + TTS/STT voice.
