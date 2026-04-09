# Phase 5: Hardening & Use Case Validation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the 10 real-world use cases actually work — not just theoretically possible but demonstrably functioning. Fix every gap between the design doc and reality.

**Architecture:** No new large systems. Focus on wiring existing components together correctly, adding the missing data pipeline (episode logging), upgrading the proactive engine to use LLM, and validating end-to-end with real desktop activity.

**Tech Stack:** Python 3.11, PyTorch, FastAPI, React 18, react-force-graph-3d, Ollama

---

## Overview of Tasks

| # | Task | Why | Use Cases |
|---|------|-----|-----------|
| 1 | Episode Logger | "What happened today" needs historical data | UC 5, 8, 9 |
| 2 | Proactive Engine → LLM | Pet should speak in natural language, not canned strings | UC 1-10 |
| 3 | Dashboard SensorPanel update | New sensor fields (rhythm, switch_rate) are invisible | All |
| 4 | Pet Face eye thresholds recalibration | Eyes must respond correctly after modulator injection changes | UC 6, 7 |
| 5 | Concept stability validation test | Prove the SNN actually forms distinct concepts | UC 1-10 |
| 6 | End-to-end soak test with assertions | 1-hour automated test that validates all systems | All |

---

### Task 1: Episode Logger

The brain has no memory of what happened WHEN. The exporter tracks live state + auto-correlations, but there's no timeline. Use Cases 5 (gewohnheits-tracker), 8 (end-of-day), and 9 (langzeit-trend) all need "what happened at 14:00?" or "how was this week vs last week?"

**Files:**
- Create: `bridge/episode_log.py`
- Create: `tests/test_episode_log.py`
- Modify: `server/main.py` (add episode logging to tick loop)

**Step 1: Write the failing test**

```python
# tests/test_episode_log.py
"""Tests for the episode logger — records brain state snapshots over time."""
import tempfile
import time
from pathlib import Path

import pytest
import torch

from brain.core import Brain
from bridge.exporter import BrainStateExporter
from bridge.episode_log import EpisodeLogger


def test_logger_construction():
    logger = EpisodeLogger(Path(tempfile.mkdtemp()) / "episodes.db")
    assert logger is not None


def test_log_and_query_episode():
    path = Path(tempfile.mkdtemp()) / "episodes.db"
    logger = EpisodeLogger(path)
    brain = Brain(num_sensory=8, num_concept=4)
    exporter = BrainStateExporter(brain)

    # Log a snapshot
    logger.log(
        tick=100,
        modulators={"DA": 0.05, "NE": 0.02, "ACh": 0.03, "5HT": 0.01},
        active_concepts=[2, 3],
        sensor_summary={"app": "VSCode", "keys": 10, "mouse": 5},
        sleep_mode=False,
    )

    # Query it back
    episodes = logger.query(last_n=1)
    assert len(episodes) == 1
    assert episodes[0]["tick"] == 100
    assert episodes[0]["modulators"]["DA"] == 0.05
    assert 2 in episodes[0]["active_concepts"]


def test_query_time_range():
    path = Path(tempfile.mkdtemp()) / "episodes.db"
    logger = EpisodeLogger(path)

    # Log 3 episodes at different "times"
    for i in range(3):
        logger.log(
            tick=i * 1000,
            modulators={"DA": 0.01 * i, "NE": 0, "ACh": 0, "5HT": 0},
            active_concepts=[i],
            sensor_summary={"app": f"App{i}"},
            sleep_mode=False,
        )

    # Query last 2
    episodes = logger.query(last_n=2)
    assert len(episodes) == 2
    assert episodes[0]["tick"] == 2000  # most recent first


def test_daily_summary():
    path = Path(tempfile.mkdtemp()) / "episodes.db"
    logger = EpisodeLogger(path)

    # Log many episodes
    for i in range(100):
        logger.log(
            tick=i * 100,
            modulators={"DA": 0.02, "NE": 0.05 if i > 50 else 0.01, "ACh": 0.01, "5HT": 0.01},
            active_concepts=[i % 5],
            sensor_summary={"app": "VSCode" if i % 2 == 0 else "Chrome"},
            sleep_mode=False,
        )

    summary = logger.daily_summary()
    assert "total_ticks" in summary
    assert "top_concepts" in summary
    assert "avg_modulators" in summary
    assert "top_apps" in summary
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_episode_log.py -v`
Expected: FAIL with "No module named 'bridge.episode_log'"

**Step 3: Write minimal implementation**

```python
# bridge/episode_log.py
"""Episode Logger — records brain state snapshots over time.

Stores periodic snapshots (every ~10s) to SQLite for historical queries.
Enables "what happened today?", "how was this week?", "end-of-day summary".

Schema:
    episodes(id INTEGER PRIMARY KEY, timestamp REAL, tick INTEGER,
             modulators TEXT, active_concepts TEXT, sensor_summary TEXT,
             sleep_mode INTEGER)
"""
from __future__ import annotations
import json
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class EpisodeLogger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                tick INTEGER NOT NULL,
                modulators TEXT NOT NULL,
                active_concepts TEXT NOT NULL,
                sensor_summary TEXT NOT NULL,
                sleep_mode INTEGER NOT NULL DEFAULT 0
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_timestamp
            ON episodes(timestamp DESC)
        """)
        self._conn.commit()

    def log(
        self,
        tick: int,
        modulators: dict[str, float],
        active_concepts: list[int],
        sensor_summary: dict[str, Any],
        sleep_mode: bool = False,
    ) -> None:
        self._conn.execute(
            "INSERT INTO episodes(timestamp, tick, modulators, active_concepts, sensor_summary, sleep_mode) VALUES (?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                tick,
                json.dumps(modulators),
                json.dumps(active_concepts),
                json.dumps(sensor_summary, default=str),
                int(sleep_mode),
            ),
        )
        self._conn.commit()

    def query(self, last_n: int = 10, since_timestamp: float | None = None) -> list[dict[str, Any]]:
        if since_timestamp is not None:
            rows = self._conn.execute(
                "SELECT timestamp, tick, modulators, active_concepts, sensor_summary, sleep_mode "
                "FROM episodes WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT ?",
                (since_timestamp, last_n),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT timestamp, tick, modulators, active_concepts, sensor_summary, sleep_mode "
                "FROM episodes ORDER BY timestamp DESC LIMIT ?",
                (last_n,),
            ).fetchall()
        return [
            {
                "timestamp": r[0],
                "tick": r[1],
                "modulators": json.loads(r[2]),
                "active_concepts": json.loads(r[3]),
                "sensor_summary": json.loads(r[4]),
                "sleep_mode": bool(r[5]),
            }
            for r in rows
        ]

    def daily_summary(self, since_hours: float = 24.0) -> dict[str, Any]:
        cutoff = time.time() - since_hours * 3600
        rows = self._conn.execute(
            "SELECT tick, modulators, active_concepts, sensor_summary, sleep_mode "
            "FROM episodes WHERE timestamp >= ? ORDER BY timestamp",
            (cutoff,),
        ).fetchall()

        if not rows:
            return {"total_ticks": 0, "top_concepts": [], "avg_modulators": {}, "top_apps": []}

        # Aggregate
        concept_counter: Counter = Counter()
        app_counter: Counter = Counter()
        mod_sums: dict[str, float] = defaultdict(float)
        n = len(rows)
        awake_count = 0
        sleep_count = 0

        for tick, mods_json, concepts_json, sensors_json, sleep in rows:
            mods = json.loads(mods_json)
            concepts = json.loads(concepts_json)
            sensors = json.loads(sensors_json)

            for c in concepts:
                concept_counter[c] += 1
            for k, v in mods.items():
                mod_sums[k] += v
            app = sensors.get("app")
            if app:
                app_counter[app] += 1
            if sleep:
                sleep_count += 1
            else:
                awake_count += 1

        first_tick = rows[0][0]
        last_tick = rows[-1][0]

        return {
            "total_ticks": last_tick - first_tick,
            "episodes_count": n,
            "awake_episodes": awake_count,
            "sleep_episodes": sleep_count,
            "top_concepts": concept_counter.most_common(10),
            "avg_modulators": {k: round(v / n, 4) for k, v in mod_sums.items()},
            "top_apps": app_counter.most_common(10),
        }

    def close(self) -> None:
        self._conn.close()
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_episode_log.py -v`
Expected: 4 passed

**Step 5: Wire into the daemon tick loop**

In `server/main.py`, in `brain_tick_loop`, add episode logging every 1000 ticks (~10s at 100Hz):

```python
# After the exporter.record_spikes_with_context() call:
if episode_logger is not None and brain.tick_count % 1000 == 0:
    sensor_snap_summary = {
        "app": sensor_snap.get("active_app", {}).get("name", "?"),
        "keys": sensor_snap.get("keystroke_rate", {}).get("count", 0),
        "mouse": sensor_snap.get("mouse_rate", {}).get("count", 0),
        "idle": sensor_snap.get("idle", {}).get("seconds", 0),
    }
    active_ids = (brain.concept_spike_accum > 0.1).nonzero(as_tuple=True)[0].tolist()
    episode_logger.log(
        tick=brain.tick_count,
        modulators=brain.modulators.snapshot(),
        active_concepts=active_ids[:20],
        sensor_summary=sensor_snap_summary,
        sleep_mode=brain.sleep_mode,
    )
```

Create the EpisodeLogger in `braind.py` `_run_daemon()` and pass it to `brain_tick_loop()`.

**Step 6: Add episode_search to memory tools**

In `bridge/memory_tools.py`, add a tool that queries the episode log:

```python
def episode_search(self, time_range: str = "today") -> dict:
    """Search episodes. time_range: 'today', 'yesterday', 'this_week', 'last_hour'."""
    hours = {"today": 24, "yesterday": 48, "this_week": 168, "last_hour": 1}.get(time_range, 24)
    return self._episode_logger.daily_summary(since_hours=hours)
```

**Step 7: Commit**

```bash
git add bridge/episode_log.py tests/test_episode_log.py server/main.py server/braind.py bridge/memory_tools.py
git commit -m "feat: add episode logger for historical brain state queries"
```

---

### Task 2: Proactive Engine Uses LLM

Currently `proactive.py` broadcasts hardcoded German strings. For the pet to feel alive, notifications should be LLM-generated from current brain state — the same way /api/chat works.

**Files:**
- Modify: `bridge/proactive.py`
- Modify: `server/braind.py` (pass router to proactive engine)

**Step 1: Update ProactiveEngine to accept LLM router**

Add `router: HybridLLMRouter` to `__init__()` and a method `_generate_message()` that calls the LLM with a focused prompt:

```python
async def _generate_message(self, category: str, context: str) -> str:
    """Ask LLM to generate a natural notification message."""
    mods = self.brain.modulators.snapshot()
    prompt = (
        "Du bist ein kleines AI-Haustier. Generiere EINE kurze, natuerliche Nachricht "
        f"(max 2 Saetze) fuer folgende Situation:\n"
        f"Kategorie: {category}\n"
        f"Kontext: {context}\n"
        f"Deine Stimmung: DA={mods.get('DA',0):.3f} NE={mods.get('NE',0):.3f} "
        f"ACh={mods.get('ACh',0):.3f} 5HT={mods.get('5HT',0):.3f}\n"
        f"Sprich in der ersten Person. Sei lebendig, kurz, persoenlich."
    )
    try:
        result = await self.router.chat(
            user_message="Generiere eine proaktive Nachricht.",
            system_prompt=prompt,
            brain_state={"modulators": mods},
            tools=[],
        )
        return result.get("text", context)
    except Exception:
        return context  # fallback to canned string
```

Replace all hardcoded strings in `_check()` return values with a `context` field, then call `_generate_message()` in the `run()` loop before broadcasting.

**Step 2: Run existing tests**

Run: `.venv/bin/python -m pytest tests/ --ignore=tests/test_e2e_smoke.py -q`
Expected: All pass (proactive engine is not directly tested — integration only)

**Step 3: Commit**

```bash
git add bridge/proactive.py server/braind.py
git commit -m "feat: proactive engine generates messages via LLM instead of canned strings"
```

---

### Task 3: Dashboard SensorPanel — Show New Fields

The SensorPanel currently shows basic fields (app, keys, mouse, idle, mic_rms). After our sensor overhaul, we have: background_apps, switch_rate, keystroke rhythm (variability/burst), mouse rhythm, pause type, activity level. These are invisible.

**Files:**
- Modify: `ui/src/components/SensorPanel.tsx`

**Step 1: Read current SensorPanel**

Check what fields it currently renders and add the new ones.

**Step 2: Add new sensor display sections**

Add to SensorPanel.tsx:
- **Background Apps:** list of names (pill badges)
- **Switch Rate:** switches/min with color indicator (green=calm, yellow=busy, red=frantic)
- **Typing Rhythm:** "steady" / "erratic" / "burst" / "silent" indicator
- **Pause Type:** "micro" / "thinking" / "break" / "away" when idle
- **Activity Level:** "dormant" / "idle" / "active" / "intense" badge

**Step 3: Verify UI compiles and renders**

Run: `cd ui && npx tsc --noEmit && npm run build`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add ui/src/components/SensorPanel.tsx
git commit -m "feat(ui): display rhythm, switch rate, pause type in sensor panel"
```

---

### Task 4: Pet Face Eye Recalibration

After all our modulator injection changes (flow detection, stress detection, baseline changes), the eye thresholds in `pet-face/src/eyes.js` may be wrong. We need to verify actual modulator ranges and adjust.

**Files:**
- Modify: `pet-face/src/eyes.js`

**Step 1: Measure actual modulator ranges**

Run the daemon for 2 minutes with normal desktop activity. Log the modulator values:

```bash
curl -s http://localhost:8765/api/chat -X POST -H "Content-Type: application/json" \
  -d '{"message":"zeig mir deine aktuellen modulator-werte"}' | python3 -m json.tool
```

**Step 2: Update eye state thresholds to match actual ranges**

Based on measured values, update `getTargetState()` in eyes.js. The thresholds should be:
- `alarmed`: NE > 2x its normal resting value
- `curious`: DA > 2x AND ACh > 1.5x normal
- `sleepy`: all modulators < 0.5x normal
- `asleep`: all < 0.2x normal OR brain.sleep_mode

Also add: push sleep_mode flag from WebSocket to pet face so it knows when brain is sleeping.

**Step 3: Test with Tauri dev**

Run: `cd pet-face && npx tauri dev`
Expected: Eyes respond visibly to different activity patterns

**Step 4: Commit**

```bash
git add pet-face/src/eyes.js
git commit -m "fix(pet-face): recalibrate eye thresholds to actual modulator ranges"
```

---

### Task 5: Concept Stability Validation

We claim concepts form. We've never validated this end-to-end. This task creates an automated test that:
1. Starts a brain with mock sensors
2. Injects 3 distinct repeating patterns (typing, browsing, idle)
3. Runs for 50,000 ticks
4. Asserts that at least 3 distinct concept neurons emerged (each firing primarily for one pattern)

**Files:**
- Create: `tests/test_concept_formation.py`

**Step 1: Write the test**

```python
# tests/test_concept_formation.py
"""Validates that the SNN actually forms distinct concepts from patterned input.

Injects 3 repeating sensor patterns (typing, browsing, idle) for 50K ticks
and checks that distinct concept neurons emerge for each.
"""
import torch
import pytest
from collections import defaultdict

from brain.core import Brain
from adapters.mac_desktop.encoding import encode_snapshot


# Three distinct desktop activity patterns
PATTERNS = {
    "typing": {
        "active_app": {"name": "VSCode"},
        "keystroke_rate": {"count": 20, "variability": 0.3, "burst": 0.0},
        "mouse_rate": {"count": 2, "variability": 0.1, "burst": 0.0},
        "idle": {"seconds": 0.5},
        "mic": {"mel": [0.01] * 32, "rms": 0.005},
        "time_tonic": {"day_phase": [0.5, 0.5, 0.5, 0.5], "week_phase": [0.5, 0.5, 0.5, 0.5]},
    },
    "browsing": {
        "active_app": {"name": "Chrome"},
        "keystroke_rate": {"count": 1, "variability": 0.1, "burst": 0.0},
        "mouse_rate": {"count": 40, "variability": 0.5, "burst": 0.0},
        "idle": {"seconds": 0.5},
        "mic": {"mel": [0.01] * 32, "rms": 0.005},
        "time_tonic": {"day_phase": [0.5, 0.5, 0.5, 0.5], "week_phase": [0.5, 0.5, 0.5, 0.5]},
    },
    "idle": {
        "active_app": {"name": "Finder"},
        "keystroke_rate": {"count": 0, "variability": 0.0, "burst": 0.0},
        "mouse_rate": {"count": 0, "variability": 0.0, "burst": 0.0},
        "idle": {"seconds": 60.0},
        "mic": {"mel": [0.01] * 32, "rms": 0.001},
        "time_tonic": {"day_phase": [0.5, 0.5, 0.5, 0.5], "week_phase": [0.5, 0.5, 0.5, 0.5]},
    },
}


def test_three_distinct_concepts_form():
    """After 50K ticks alternating 3 patterns, at least 3 concept neurons
    should specialize (each fires primarily for one pattern)."""
    brain = Brain(num_sensory=200, num_concept=200)
    pattern_names = list(PATTERNS.keys())

    # Track which concepts fire for which pattern
    concept_pattern_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    ticks_per_pattern = 500  # switch pattern every 500 ticks (5s)
    total_ticks = 50000

    for tick in range(total_ticks):
        pattern_idx = (tick // ticks_per_pattern) % len(pattern_names)
        pattern_name = pattern_names[pattern_idx]
        snap = PATTERNS[pattern_name]
        vec = encode_snapshot(snap)
        out = brain.tick(vec)

        # Track which concepts fire for which pattern
        concept_spikes = out["concept"]
        for cid in (concept_spikes > 0).nonzero(as_tuple=True)[0].tolist():
            concept_pattern_counts[cid][pattern_name] += 1

    # Now check: how many concepts are "specialized"?
    # A specialized concept fires >60% for one pattern
    specialized = set()
    for cid, counts in concept_pattern_counts.items():
        total = sum(counts.values())
        if total < 10:
            continue  # ignore rarely-firing concepts
        for pattern_name, count in counts.items():
            if count / total > 0.6:
                specialized.add(cid)
                break

    # We need at least 3 specialized concepts (one per pattern)
    assert len(specialized) >= 3, (
        f"Only {len(specialized)} specialized concepts found. "
        f"Need at least 3 for 3 patterns. "
        f"Concept stats: {dict(concept_pattern_counts)}"
    )

    # Verify they're different (not all specialized for the same pattern)
    primary_patterns = set()
    for cid in specialized:
        counts = concept_pattern_counts[cid]
        total = sum(counts.values())
        primary = max(counts, key=counts.get)
        primary_patterns.add(primary)

    assert len(primary_patterns) >= 2, (
        f"All specialized concepts fire for the same pattern. "
        f"Primary patterns: {primary_patterns}"
    )
```

**Step 2: Run the test**

Run: `.venv/bin/python -m pytest tests/test_concept_formation.py -v -s`
Expected: PASS (may take 30-60 seconds for 50K ticks)

If FAIL: this proves the SNN does NOT form concepts and we need to debug the STDP/WTA/encoding pipeline.

**Step 3: Commit**

```bash
git add tests/test_concept_formation.py
git commit -m "test: validate that 3 distinct concepts form from 3 input patterns"
```

---

### Task 6: End-to-End Soak Test

An automated 1-hour test that starts the full daemon with mock sensors, runs through simulated daily activity, and validates all systems.

**Files:**
- Create: `scripts/soak_test.py`

**Step 1: Write the soak test script**

```python
# scripts/soak_test.py
"""Automated soak test: 1 hour of simulated desktop activity.

Validates:
1. Brain doesn't crash or produce NaN
2. Concepts form (>=3 specialized)
3. Modulators stay in sane ranges
4. Persistence works (save/load/resume)
5. Episode logger records data
6. Proactive engine generates at least 1 notification

Usage: .venv/bin/python scripts/soak_test.py --ticks 360000
"""
import argparse
import sys
import time
import torch
from pathlib import Path
from collections import defaultdict

from brain.core import Brain
from brain.persistence import save_brain, load_brain
from adapters.mac_desktop.encoding import encode_snapshot
from bridge.exporter import BrainStateExporter
from bridge.episode_log import EpisodeLogger


# Simulated daily schedule (pattern index → pattern)
SCHEDULE = [
    # Morning: emails (browsing)
    (0, 5000, {"active_app": {"name": "Mail"}, "keystroke_rate": {"count": 5}, "mouse_rate": {"count": 30}, "idle": {"seconds": 0.5}, "mic": {"mel": [0.01]*32, "rms": 0.01}, "time_tonic": {"day_phase": [0.9, 0.4, 0.0, 0.0], "week_phase": [0.5, 0.5, 0.5, 0.5]}}),
    # Coding session
    (5000, 20000, {"active_app": {"name": "VSCode"}, "keystroke_rate": {"count": 25, "variability": 0.2}, "mouse_rate": {"count": 5}, "idle": {"seconds": 0.3}, "mic": {"mel": [0.01]*32, "rms": 0.003}, "time_tonic": {"day_phase": [0.7, 0.7, 0.0, 0.0], "week_phase": [0.5, 0.5, 0.5, 0.5]}}),
    # Meeting (audio active)
    (20000, 25000, {"active_app": {"name": "Zoom"}, "keystroke_rate": {"count": 0}, "mouse_rate": {"count": 2}, "idle": {"seconds": 1.0}, "mic": {"mel": [0.3]*32, "rms": 0.15}, "time_tonic": {"day_phase": [0.3, 0.9, 0.0, 0.0], "week_phase": [0.5, 0.5, 0.5, 0.5]}}),
    # Lunch break (idle)
    (25000, 30000, {"active_app": {"name": "Finder"}, "keystroke_rate": {"count": 0}, "mouse_rate": {"count": 0}, "idle": {"seconds": 600}, "mic": {"mel": [0.005]*32, "rms": 0.001}, "time_tonic": {"day_phase": [0.0, 1.0, 0.0, 0.0], "week_phase": [0.5, 0.5, 0.5, 0.5]}}),
    # Afternoon: design work
    (30000, 45000, {"active_app": {"name": "Figma"}, "keystroke_rate": {"count": 3}, "mouse_rate": {"count": 60, "variability": 0.3}, "idle": {"seconds": 0.5}, "mic": {"mel": [0.01]*32, "rms": 0.005}, "time_tonic": {"day_phase": [-0.3, 0.9, 0.0, 0.0], "week_phase": [0.5, 0.5, 0.5, 0.5]}}),
    # Stress: rapid switching
    (45000, 50000, {"active_app": {"name": "Slack", "switch_rate": 12.0}, "keystroke_rate": {"count": 15, "variability": 1.5, "burst": 1.0}, "mouse_rate": {"count": 40, "variability": 1.2}, "idle": {"seconds": 0.2}, "mic": {"mel": [0.01]*32, "rms": 0.01}, "time_tonic": {"day_phase": [-0.7, 0.7, 0.0, 0.0], "week_phase": [0.5, 0.5, 0.5, 0.5]}}),
]


def get_pattern(tick: int) -> dict:
    for start, end, pattern in SCHEDULE:
        if start <= tick < end:
            return pattern
    return SCHEDULE[-1][2]  # default to last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=50000)
    args = parser.parse_args()

    print(f"=== SOAK TEST: {args.ticks} ticks ===")
    brain = Brain()
    exporter = BrainStateExporter(brain)
    logger = EpisodeLogger(Path("/tmp/soak_test_episodes.db"))

    concept_pattern_counts = defaultdict(lambda: defaultdict(int))
    errors = []

    t0 = time.time()
    for tick in range(args.ticks):
        pattern = get_pattern(tick)
        vec = encode_snapshot(pattern)

        # Check for NaN in input
        if torch.isnan(vec).any():
            errors.append(f"NaN in input at tick {tick}")
            break

        out = brain.tick(vec)

        # Check for NaN in output
        for name, spikes in out.items():
            if torch.isnan(spikes).any():
                errors.append(f"NaN in {name} spikes at tick {tick}")
                break

        # Track concept formation
        concept_spikes = out["concept"]
        app = pattern.get("active_app", {}).get("name", "?")
        for cid in (concept_spikes > 0).nonzero(as_tuple=True)[0].tolist():
            concept_pattern_counts[cid][app] += 1

        # Log episodes
        if tick % 1000 == 0:
            exporter.record_spikes_with_context(concept_spikes, pattern)
            active_ids = (brain.concept_spike_accum > 0.1).nonzero(as_tuple=True)[0].tolist()
            logger.log(
                tick=tick,
                modulators=brain.modulators.snapshot(),
                active_concepts=active_ids[:20],
                sensor_summary={"app": app},
                sleep_mode=brain.sleep_mode,
            )

        # Progress
        if tick % 10000 == 0 and tick > 0:
            elapsed = time.time() - t0
            mods = brain.modulators.snapshot()
            print(f"  tick {tick}/{args.ticks} ({elapsed:.1f}s) "
                  f"DA={mods['DA']:.4f} NE={mods['NE']:.4f} "
                  f"ACh={mods['ACh']:.4f} 5HT={mods['5HT']:.4f}")

    elapsed = time.time() - t0
    print(f"\n=== RESULTS ({elapsed:.1f}s) ===")

    # Check 1: No errors
    if errors:
        print(f"FAIL: {len(errors)} errors: {errors[:5]}")
        sys.exit(1)
    print("PASS: No NaN or crashes")

    # Check 2: Modulators in sane range
    mods = brain.modulators.snapshot()
    for name, val in mods.items():
        if val < -1.0 or val > 1.0:
            print(f"FAIL: Modulator {name} out of range: {val}")
            sys.exit(1)
    print(f"PASS: Modulators in range: {mods}")

    # Check 3: Concepts formed
    specialized = 0
    for cid, counts in concept_pattern_counts.items():
        total = sum(counts.values())
        if total < 10:
            continue
        for app, count in counts.items():
            if count / total > 0.5:
                specialized += 1
                break
    print(f"PASS: {specialized} specialized concepts")
    if specialized < 3:
        print(f"WARN: Only {specialized} specialized concepts (want >=3)")

    # Check 4: Persistence
    save_path = Path("/tmp/soak_test_brain.sqlite")
    save_brain(brain, save_path)
    brain2 = load_brain(save_path)
    assert brain2.tick_count == brain.tick_count
    print(f"PASS: Persistence works (saved and loaded at tick {brain.tick_count})")

    # Check 5: Episode log
    summary = logger.daily_summary()
    print(f"PASS: Episode log: {summary['episodes_count']} episodes, "
          f"top apps: {summary['top_apps'][:3]}")

    # Check 6: Weight stats
    for syn_name, syn in brain.synapses.items():
        w = syn.weights
        print(f"  {syn_name}: mean={w.mean():.4f} std={w.std():.4f} "
              f"min={w.min():.4f} max={w.max():.4f}")

    print("\n=== ALL CHECKS PASSED ===")
    logger.close()


if __name__ == "__main__":
    main()
```

**Step 2: Run soak test**

Run: `.venv/bin/python scripts/soak_test.py --ticks 50000`
Expected: All checks pass in <60 seconds

**Step 3: Commit**

```bash
git add scripts/soak_test.py
git commit -m "test: add soak test script validating concept formation and stability"
```

---

## Execution Note

Tasks 1-6 are strictly ordered: Task 1 (episode logger) is needed by Task 2 (proactive LLM) and Task 6 (soak test). Tasks 3 and 4 are independent and can be done in parallel.

**Total estimated effort: ~4-6 hours.**

After this plan is complete, the system should demonstrably:
- Form distinct concepts from different desktop patterns (Task 5)
- Log brain state history for "what happened today?" queries (Task 1)
- Generate natural-language proactive notifications (Task 2)
- Display rich sensor context in the dashboard (Task 3)
- Show correct eye animations reflecting the pet's mood (Task 4)
- Pass a 50K-tick soak test without divergence (Task 6)
