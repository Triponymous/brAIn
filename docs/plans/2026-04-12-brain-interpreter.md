# BrainInterpreter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a BrainInterpreter layer that transforms raw SNN output into rich, contextual understanding — enabling all 10 original use cases (habits, personality evolution, flow/stress/meeting detection, anomalies, narratives).

**Architecture:** A `BrainInterpreter` class with 6 composable modules sits between the SNN and LLM. It reads brain state + episode history and produces a structured interpretation dict that gets injected into the system prompt. The proactive engine uses it for intelligent, context-aware notifications.

**Tech Stack:** Python 3.11, torch, sqlite3 (episode log), existing Brain/Modulators/ConceptTracker classes.

**Branch:** `feature/brain-interpreter`

---

## Task 1: StateDetector — Flow, Stress, Meeting, Break Detection

**Files:**
- Create: `bridge/state_detector.py`
- Test: `tests/test_state_detector.py`

**Step 1: Write the failing tests**

```python
# tests/test_state_detector.py
import pytest
from unittest.mock import MagicMock
from bridge.state_detector import StateDetector


def _mock_brain(app="Claude", keys=20, mouse=5, mic=0.001, idle=0,
                switch_rate=0, ne=0.01, sht=0.03):
    brain = MagicMock()
    brain._last_sensor_display = {
        "app": app, "keys": keys, "mouse": mouse,
        "mic_rms": mic, "idle": idle, "switch_rate": switch_rate,
    }
    brain.modulators.snapshot.return_value = {
        "DA": 0.02, "NE": ne, "ACh": 0.03, "5HT": sht,
    }
    brain.tick_count = 100000
    return brain


def test_flow_detection():
    """Same app + steady typing + low switch rate for 30+ min = flow."""
    sd = StateDetector()
    brain = _mock_brain(app="Claude", keys=15, switch_rate=0.2)
    # Simulate 30 minutes of same state (1800 snapshots at 1/sec)
    for _ in range(1800):
        sd.update(brain)
    state = sd.detect()
    assert state["flow"] is True
    assert state["flow_duration_min"] >= 30


def test_no_flow_with_switching():
    """High app switching breaks flow."""
    sd = StateDetector()
    brain = _mock_brain(app="Claude", keys=15, switch_rate=5.0)
    for _ in range(1800):
        sd.update(brain)
    state = sd.detect()
    assert state["flow"] is False


def test_stress_detection():
    """High NE + erratic typing + high switch rate = stress."""
    sd = StateDetector()
    brain = _mock_brain(keys=30, switch_rate=6.0, ne=0.15, sht=0.005)
    for _ in range(120):
        sd.update(brain)
    state = sd.detect()
    assert state["stress"] is True


def test_meeting_detection():
    """Zoom + mic active + no typing = meeting."""
    sd = StateDetector()
    brain = _mock_brain(app="Zoom", keys=0, mic=0.02, idle=0)
    for _ in range(60):
        sd.update(brain)
    state = sd.detect()
    assert state["meeting"] is True


def test_break_needed():
    """Active work for 90+ min without break = needs_break."""
    sd = StateDetector()
    brain = _mock_brain(keys=15, idle=0)
    for _ in range(5400):  # 90 min
        sd.update(brain)
    state = sd.detect()
    assert state["needs_break"] is True


def test_break_reset_after_idle():
    """Idle period > 5 min resets the break timer."""
    sd = StateDetector()
    brain_active = _mock_brain(keys=15, idle=0)
    brain_idle = _mock_brain(keys=0, idle=400)
    for _ in range(5400):
        sd.update(brain_active)
    for _ in range(60):
        sd.update(brain_idle)
    state = sd.detect()
    assert state["needs_break"] is False
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_state_detector.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'bridge.state_detector'"

**Step 3: Write implementation**

```python
# bridge/state_detector.py
"""State detection: Flow, Stress, Meeting, Break-needed.

Reads live sensor data from brain._last_sensor_display and modulator levels.
Maintains rolling history to detect sustained states (not momentary spikes).
"""
from __future__ import annotations
from collections import deque
from typing import Any


_MEETING_APPS = {"zoom", "teams", "meet", "webex", "skype", "facetime", "discord"}


class StateDetector:
    def __init__(
        self,
        flow_min_minutes: float = 30.0,
        flow_max_switch_rate: float = 1.0,
        flow_min_keys: float = 3.0,
        stress_min_seconds: float = 120.0,
        meeting_min_seconds: float = 30.0,
        break_after_minutes: float = 90.0,
        break_reset_idle: float = 300.0,  # 5 min idle resets break timer
    ) -> None:
        self.flow_min_minutes = flow_min_minutes
        self.flow_max_switch_rate = flow_max_switch_rate
        self.flow_min_keys = flow_min_keys
        self.stress_min_seconds = stress_min_seconds
        self.meeting_min_seconds = meeting_min_seconds
        self.break_after_minutes = break_after_minutes
        self.break_reset_idle = break_reset_idle

        # Rolling state
        self._history: deque[dict] = deque(maxlen=7200)  # 2h at 1/sec
        self._last_app: str = ""
        self._same_app_since: int = 0  # snapshot index when app last changed
        self._active_since: int = 0    # snapshot index of last idle reset
        self._snapshot_count: int = 0

    def update(self, brain: Any) -> None:
        """Call once per second (or per snapshot interval)."""
        sd = getattr(brain, '_last_sensor_display', {})
        mods = brain.modulators.snapshot()

        snap = {
            "app": sd.get("app", ""),
            "keys": sd.get("keys", 0),
            "mouse": sd.get("mouse", 0),
            "mic_rms": sd.get("mic_rms", 0),
            "idle": sd.get("idle", 0),
            "switch_rate": sd.get("switch_rate", 0),
            "ne": mods.get("NE", 0),
            "sht": mods.get("5HT", 0),
        }
        self._history.append(snap)
        self._snapshot_count += 1

        # Track app continuity
        if snap["app"] != self._last_app:
            self._last_app = snap["app"]
            self._same_app_since = self._snapshot_count

        # Track active time (reset on long idle)
        if snap["idle"] > self.break_reset_idle:
            self._active_since = self._snapshot_count

    def detect(self) -> dict[str, Any]:
        """Return current detected states."""
        if not self._history:
            return self._empty()

        recent = self._history
        latest = recent[-1]

        # Flow: same app for 30+ min, steady keys, low switching
        same_app_secs = self._snapshot_count - self._same_app_since
        same_app_min = same_app_secs / 60.0
        flow = (
            same_app_min >= self.flow_min_minutes
            and latest["switch_rate"] <= self.flow_max_switch_rate
            and latest["keys"] >= self.flow_min_keys
        )

        # Stress: high NE + low 5HT for sustained period
        stress_window = list(recent)[-int(self.stress_min_seconds):]
        stress = False
        if len(stress_window) >= self.stress_min_seconds:
            avg_ne = sum(s["ne"] for s in stress_window) / len(stress_window)
            avg_sht = sum(s["sht"] for s in stress_window) / len(stress_window)
            avg_switch = sum(s["switch_rate"] for s in stress_window) / len(stress_window)
            stress = avg_ne > 0.05 and avg_sht < 0.02 and avg_switch > 3.0

        # Meeting: meeting app + mic active + no typing
        app_lower = latest["app"].lower() if latest["app"] else ""
        is_meeting_app = any(m in app_lower for m in _MEETING_APPS)
        meeting_secs = 0
        if is_meeting_app:
            for s in reversed(list(recent)):
                s_app = (s["app"] or "").lower()
                if any(m in s_app for m in _MEETING_APPS) and s["mic_rms"] > 0.005:
                    meeting_secs += 1
                else:
                    break
        meeting = is_meeting_app and latest["mic_rms"] > 0.005 and meeting_secs >= self.meeting_min_seconds

        # Break needed: active for 90+ min
        active_secs = self._snapshot_count - self._active_since
        active_min = active_secs / 60.0
        needs_break = active_min >= self.break_after_minutes and latest["idle"] < 30

        return {
            "flow": flow,
            "flow_duration_min": round(same_app_min, 1) if flow else 0,
            "flow_app": latest["app"] if flow else None,
            "stress": stress,
            "meeting": meeting,
            "meeting_duration_min": round(meeting_secs / 60, 1) if meeting else 0,
            "needs_break": needs_break,
            "active_minutes": round(active_min, 1),
        }

    def _empty(self) -> dict[str, Any]:
        return {
            "flow": False, "flow_duration_min": 0, "flow_app": None,
            "stress": False, "meeting": False, "meeting_duration_min": 0,
            "needs_break": False, "active_minutes": 0,
        }
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_state_detector.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add bridge/state_detector.py tests/test_state_detector.py
git commit -m "feat: add StateDetector — flow, stress, meeting, break detection"
```

---

## Task 2: HabitMiner — Tageszeit + Wochen-Rhythmen

**Files:**
- Create: `bridge/habit_miner.py`
- Test: `tests/test_habit_miner.py`

**Step 1: Write the failing tests**

```python
# tests/test_habit_miner.py
import time
import json
import sqlite3
import tempfile
from pathlib import Path
from bridge.episode_log import EpisodeLogger
from bridge.habit_miner import HabitMiner


def _fill_episodes(logger, days=7, entries_per_hour=6):
    """Fill logger with synthetic week of activity."""
    now = time.time()
    for day in range(days):
        for hour in range(9, 18):  # work hours
            for i in range(entries_per_hour):
                ts = now - (days - day) * 86400 + hour * 3600 + i * 600
                # Monday 9am: always Slack
                app = "Slack" if hour == 9 else "VSCode"
                cluster = 0 if hour == 9 else 1
                logger._conn.execute(
                    "INSERT INTO episodes(timestamp, tick, modulators, active_concepts, sensor_summary, sleep_mode) VALUES (?,?,?,?,?,?)",
                    (ts, int((ts - now) * 100), '{"DA":0.02}',
                     json.dumps([cluster]),
                     json.dumps({"app": app, "keys": 15, "cluster_id": cluster}),
                     0),
                )
    logger._conn.commit()


def test_habit_mining_detects_morning_slack():
    """Should detect pattern: 9am → Slack."""
    with tempfile.TemporaryDirectory() as td:
        logger = EpisodeLogger(Path(td) / "test.db")
        _fill_episodes(logger)
        miner = HabitMiner(logger)
        habits = miner.mine_habits()
        # Should find at least one habit involving Slack at 9am
        slack_habits = [h for h in habits if "Slack" in h.get("app", "")]
        assert len(slack_habits) > 0
        assert slack_habits[0]["hour"] == 9
        logger.close()


def test_hourly_profile():
    """Should produce activity profile per hour."""
    with tempfile.TemporaryDirectory() as td:
        logger = EpisodeLogger(Path(td) / "test.db")
        _fill_episodes(logger)
        miner = HabitMiner(logger)
        profile = miner.hourly_profile()
        # Hour 9 should show Slack as top app
        assert profile[9]["top_app"] == "Slack"
        # Hour 10-17 should show VSCode
        assert profile[10]["top_app"] == "VSCode"
        logger.close()
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_habit_miner.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# bridge/habit_miner.py
"""HabitMiner — discovers recurring time-of-day and day-of-week patterns.

Queries the episode log for multi-day history and extracts:
- Hourly app usage patterns ("9am → Slack, 10am → VSCode")
- Weekly rhythms ("Mondays are meeting-heavy")
- Recurring concept clusters at specific times
"""
from __future__ import annotations
import json
import time
import datetime
from collections import Counter, defaultdict
from typing import Any

from bridge.episode_log import EpisodeLogger


class HabitMiner:
    def __init__(self, episode_logger: EpisodeLogger, lookback_days: int = 7) -> None:
        self.logger = episode_logger
        self.lookback_days = lookback_days
        self._cache: dict[str, Any] = {}
        self._cache_ts: float = 0
        self._cache_ttl: float = 300.0  # refresh every 5 min

    def _get_episodes(self) -> list[dict]:
        cutoff = time.time() - self.lookback_days * 86400
        return self.logger.query(last_n=100000, since_timestamp=cutoff)

    def hourly_profile(self) -> dict[int, dict[str, Any]]:
        """Activity profile per hour of day (0-23)."""
        episodes = self._get_episodes()
        hourly: dict[int, list[dict]] = defaultdict(list)

        for ep in episodes:
            ts = ep.get("timestamp", 0)
            hour = datetime.datetime.fromtimestamp(ts).hour
            hourly[hour].append(ep)

        profile = {}
        for hour in range(24):
            eps = hourly.get(hour, [])
            if not eps:
                profile[hour] = {"top_app": None, "avg_activity": 0, "episodes": 0}
                continue

            apps = Counter()
            for ep in eps:
                app = ep.get("sensor_summary", {}).get("app")
                if app:
                    apps[app] += 1

            top_app = apps.most_common(1)[0][0] if apps else None
            avg_keys = sum(ep.get("sensor_summary", {}).get("keys", 0) for ep in eps) / len(eps)

            profile[hour] = {
                "top_app": top_app,
                "avg_activity": round(avg_keys, 1),
                "episodes": len(eps),
                "app_distribution": dict(apps.most_common(5)),
            }
        return profile

    def mine_habits(self) -> list[dict[str, Any]]:
        """Find recurring patterns: same app at same hour across multiple days."""
        episodes = self._get_episodes()
        # Group by (day_of_week, hour) → app
        pattern_counts: dict[tuple[int, int, str], int] = Counter()
        day_counts: dict[tuple[int, int], int] = Counter()

        for ep in episodes:
            ts = ep.get("timestamp", 0)
            dt = datetime.datetime.fromtimestamp(ts)
            dow = dt.weekday()  # 0=Monday
            hour = dt.hour
            app = ep.get("sensor_summary", {}).get("app")
            if app:
                pattern_counts[(dow, hour, app)] += 1
                day_counts[(dow, hour)] += 1

        habits = []
        for (dow, hour, app), count in pattern_counts.most_common(50):
            total = day_counts[(dow, hour)]
            if total < 3:
                continue  # need at least 3 occurrences
            confidence = count / total
            if confidence > 0.5:  # app dominates >50% of that time slot
                habits.append({
                    "day_of_week": dow,
                    "day_name": ["Montag", "Dienstag", "Mittwoch", "Donnerstag",
                                 "Freitag", "Samstag", "Sonntag"][dow],
                    "hour": hour,
                    "app": app,
                    "confidence": round(confidence, 2),
                    "occurrences": count,
                })

        habits.sort(key=lambda h: -h["confidence"])
        return habits[:20]

    def current_habits(self) -> list[dict[str, Any]]:
        """Cached version of mine_habits for real-time use."""
        now = time.time()
        if now - self._cache_ts > self._cache_ttl:
            self._cache["habits"] = self.mine_habits()
            self._cache["hourly"] = self.hourly_profile()
            self._cache_ts = now
        return self._cache.get("habits", [])

    def current_hour_context(self) -> dict[str, Any]:
        """What usually happens right now?"""
        hour = datetime.datetime.now().hour
        dow = datetime.datetime.now().weekday()
        habits = self.current_habits()
        matching = [h for h in habits if h["hour"] == hour]
        today_matching = [h for h in matching if h["day_of_week"] == dow]
        return {
            "hour": hour,
            "day_of_week": dow,
            "usual_habits": today_matching or matching[:3],
        }
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_habit_miner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bridge/habit_miner.py tests/test_habit_miner.py
git commit -m "feat: add HabitMiner — hourly/weekly pattern detection"
```

---

## Task 3: PersonalityTracker — Emotionale Entwicklung

**Files:**
- Create: `bridge/personality.py`
- Test: `tests/test_personality.py`

**Step 1: Write failing tests**

```python
# tests/test_personality.py
from bridge.personality import PersonalityTracker


def test_baseline_drift():
    """Baselines should drift toward session averages over time."""
    pt = PersonalityTracker()
    # Simulate high-DA sessions (curious user)
    for _ in range(1000):
        pt.update({"DA": 0.1, "NE": 0.01, "ACh": 0.03, "5HT": 0.02})
    snap = pt.snapshot()
    assert snap["baselines"]["DA"] > 0.01  # drifted up from 0
    assert snap["traits"]["curiosity"] > 0.5  # high curiosity


def test_personality_persistence():
    """Baselines should be saveable and loadable."""
    pt = PersonalityTracker()
    for _ in range(100):
        pt.update({"DA": 0.08, "NE": 0.02, "ACh": 0.05, "5HT": 0.04})
    data = pt.save_state()
    pt2 = PersonalityTracker()
    pt2.load_state(data)
    assert abs(pt.snapshot()["baselines"]["DA"] - pt2.snapshot()["baselines"]["DA"]) < 0.001


def test_age_tracking():
    """Age should increment with updates."""
    pt = PersonalityTracker()
    assert pt.snapshot()["age_ticks"] == 0
    for _ in range(500):
        pt.update({"DA": 0.05, "NE": 0.01, "ACh": 0.03, "5HT": 0.02})
    assert pt.snapshot()["age_ticks"] == 500
```

**Step 2: Run tests — FAIL**

**Step 3: Write implementation**

```python
# bridge/personality.py
"""PersonalityTracker — models long-term emotional development.

Modulator baselines drift slowly over days/weeks based on the pet's
experiences. A consistently curious user (high DA) raises the DA baseline,
making the pet naturally more curious. This is personality evolution.

Traits are derived from baselines:
- curiosity = DA baseline (normalized 0-1)
- anxiety = NE baseline
- attentiveness = ACh baseline
- patience = 5HT baseline
"""
from __future__ import annotations
from typing import Any


class PersonalityTracker:
    def __init__(
        self,
        drift_rate: float = 0.0001,  # very slow: ~10% shift per 10000 updates
    ) -> None:
        self.drift_rate = drift_rate
        self._baselines = {"DA": 0.0, "NE": 0.0, "ACh": 0.0, "5HT": 0.0}
        self._session_sum = {"DA": 0.0, "NE": 0.0, "ACh": 0.0, "5HT": 0.0}
        self._session_count = 0
        self._age_ticks = 0

    def update(self, modulator_levels: dict[str, float]) -> None:
        """Call periodically (every ~100 ticks) with current modulator snapshot."""
        self._age_ticks += 1
        for name in self._baselines:
            level = modulator_levels.get(name, 0)
            self._session_sum[name] += level
            self._session_count += 1
            # Exponential moving average toward session mean
            self._baselines[name] = (
                self._baselines[name] * (1 - self.drift_rate)
                + level * self.drift_rate
            )

    def snapshot(self) -> dict[str, Any]:
        """Current personality state for LLM prompt."""
        # Normalize baselines to 0-1 trait scale
        # Typical baselines are 0.0-0.1, so scale by 10x and clamp
        def trait(val: float) -> float:
            return min(1.0, max(0.0, val * 10.0))

        return {
            "baselines": dict(self._baselines),
            "traits": {
                "curiosity": round(trait(self._baselines["DA"]), 2),
                "anxiety": round(trait(self._baselines["NE"]), 2),
                "attentiveness": round(trait(self._baselines["ACh"]), 2),
                "patience": round(trait(self._baselines["5HT"]), 2),
            },
            "age_ticks": self._age_ticks,
            "age_days": round(self._age_ticks / 8640000, 1),  # 100Hz * 86400s/day
        }

    def save_state(self) -> dict[str, Any]:
        return {
            "baselines": dict(self._baselines),
            "age_ticks": self._age_ticks,
        }

    def load_state(self, data: dict[str, Any]) -> None:
        self._baselines = data.get("baselines", self._baselines)
        self._age_ticks = data.get("age_ticks", 0)
```

**Step 4: Run tests — PASS**

**Step 5: Commit**

```bash
git add bridge/personality.py tests/test_personality.py
git commit -m "feat: add PersonalityTracker — emotional baseline drift"
```

---

## Task 4: NarrativeBuilder + AnomalyDetector

**Files:**
- Create: `bridge/narrative.py`
- Create: `bridge/anomaly.py`
- Test: `tests/test_narrative.py`

**Step 1-5:** (follow same TDD pattern)

**NarrativeBuilder** reads episode log and produces:
- `today_summary`: "Morgens ruhig angefangen, ab 11h intensiv gecoded, nach dem Meeting um 14h hektisch"
- `yesterday_summary`: Same for yesterday
- `week_trend`: "Diese Woche weniger gestresst als letzte"

**AnomalyDetector** compares current state to habit baseline:
- "Normalerweise tippst du jetzt, aber heute ist alles still"
- "Du warst seltener in Terminal als letzte Woche"
- Reads from HabitMiner.hourly_profile() + current sensor state

```python
# bridge/narrative.py
"""NarrativeBuilder — creates temporal stories from episode data."""
from __future__ import annotations
import datetime
import time
from typing import Any
from bridge.episode_log import EpisodeLogger


class NarrativeBuilder:
    def __init__(self, episode_logger: EpisodeLogger) -> None:
        self.logger = episode_logger

    def today_summary(self) -> str:
        now = time.time()
        midnight = now - (now % 86400)  # approximate today start
        episodes = self.logger.query(last_n=5000, since_timestamp=midnight)
        if not episodes:
            return "Heute noch nicht viel passiert."
        return self._build_narrative(episodes)

    def yesterday_summary(self) -> str:
        now = time.time()
        yesterday_start = now - (now % 86400) - 86400
        yesterday_end = now - (now % 86400)
        episodes = self.logger.query(last_n=5000, since_timestamp=yesterday_start)
        episodes = [e for e in episodes if e["timestamp"] < yesterday_end]
        if not episodes:
            return "Gestern keine Daten."
        return self._build_narrative(episodes)

    def _build_narrative(self, episodes: list[dict]) -> str:
        """Convert episodes into a short German narrative."""
        # Group by hour
        hourly: dict[int, list] = {}
        for ep in episodes:
            hour = datetime.datetime.fromtimestamp(ep["timestamp"]).hour
            hourly.setdefault(hour, []).append(ep)

        parts = []
        for hour in sorted(hourly.keys()):
            eps = hourly[hour]
            apps = [ep.get("sensor_summary", {}).get("app", "?") for ep in eps]
            top_app = max(set(apps), key=apps.count) if apps else "?"
            avg_keys = sum(ep.get("sensor_summary", {}).get("keys", 0) for ep in eps) / max(len(eps), 1)

            if avg_keys > 15:
                activity = "intensiv gearbeitet"
            elif avg_keys > 3:
                activity = "leicht aktiv"
            else:
                activity = "ruhig"
            parts.append(f"{hour}h: {activity} in {top_app}")

        return " | ".join(parts[:8]) if parts else "Keine Aktivitaet."
```

```python
# bridge/anomaly.py
"""AnomalyDetector — detects deviations from learned habits."""
from __future__ import annotations
from typing import Any
from bridge.habit_miner import HabitMiner


class AnomalyDetector:
    def __init__(self, habit_miner: HabitMiner) -> None:
        self.habit_miner = habit_miner

    def check(self, current_sensor: dict[str, Any]) -> list[dict[str, Any]]:
        """Compare current state to what's usual at this time."""
        import datetime
        now = datetime.datetime.now()
        hour = now.hour
        dow = now.weekday()

        habits = self.habit_miner.current_habits()
        anomalies = []

        # Check: expected app at this hour
        expected_at_hour = [h for h in habits if h["hour"] == hour and h["day_of_week"] == dow]
        current_app = current_sensor.get("app", "")

        for habit in expected_at_hour:
            if habit["app"] != current_app and habit["confidence"] > 0.6:
                anomalies.append({
                    "type": "missing_habit",
                    "description": f"Normalerweise bist du um {hour}h in {habit['app']} "
                                   f"({habit['day_name']}), aber heute nicht.",
                    "severity": "medium",
                    "expected": habit["app"],
                    "actual": current_app,
                })

        # Check: unusual inactivity
        hourly = self.habit_miner.hourly_profile()
        usual = hourly.get(hour, {})
        usual_activity = usual.get("avg_activity", 0)
        current_keys = current_sensor.get("keys", 0)

        if usual_activity > 10 and current_keys < 2:
            anomalies.append({
                "type": "unusual_quiet",
                "description": f"Normalerweise tippst du um {hour}h viel "
                               f"(avg {usual_activity:.0f} keys), aber heute ist es still.",
                "severity": "low",
            })

        return anomalies
```

**Commit:**
```bash
git add bridge/narrative.py bridge/anomaly.py tests/test_narrative.py
git commit -m "feat: add NarrativeBuilder + AnomalyDetector"
```

---

## Task 5: SynapseExplainer — LLM versteht das SNN

**Files:**
- Create: `bridge/synapse_explainer.py`
- Test: `tests/test_synapse_explainer.py`

```python
# bridge/synapse_explainer.py
"""SynapseExplainer — translates SNN internals for LLM consumption.

Answers: WHY is NE high? WHAT does concept #5 respond to?
WHICH concepts are associated?
"""
from __future__ import annotations
import torch
from typing import Any
from brain.core import Brain


class SynapseExplainer:
    def __init__(self, brain: Brain) -> None:
        self.brain = brain

    def explain_modulators(self) -> dict[str, str]:
        """Human-readable explanation of current modulator levels."""
        mods = self.brain.modulators.snapshot()
        explanations = {}
        da = mods.get("DA", 0)
        ne = mods.get("NE", 0)
        ach = mods.get("ACh", 0)
        sht = mods.get("5HT", 0)

        if ne > 0.08:
            explanations["NE"] = "Hoch — etwas Ueberraschendes oder Stressiges ist passiert"
        elif ne > 0.03:
            explanations["NE"] = "Leicht erhoeht — aufmerksam"
        else:
            explanations["NE"] = "Niedrig — entspannt"

        if da > 0.05:
            explanations["DA"] = "Hoch — etwas Neues passiert, ich lerne schneller"
        elif da > 0.01:
            explanations["DA"] = "Normal — stetige Wahrnehmung"
        else:
            explanations["DA"] = "Niedrig — nichts Neues, langweilig"

        if sht > 0.04:
            explanations["5HT"] = "Hoch — zufrieden, ruhige Arbeit"
        elif sht > 0.01:
            explanations["5HT"] = "Normal"
        else:
            explanations["5HT"] = "Niedrig — unruhig oder gestresst"

        if ach > 0.05:
            explanations["ACh"] = "Hoch — sehr fokussiert"
        else:
            explanations["ACh"] = "Normal"

        return explanations

    def concept_profile(self, cluster_id: int) -> dict[str, Any]:
        """What sensors typically trigger this concept cluster?"""
        syn = self.brain.synapses.get("sensory_concept")
        if syn is None:
            return {"error": "no synapse"}

        # Get the weight profile for neurons in this cluster
        # (approximation: use the concept tracker centroid)
        ct = self.brain.concept_tracker
        cluster = ct._clusters.get(cluster_id)
        if cluster is None:
            return {"error": f"cluster {cluster_id} not found"}

        # Top active dimensions in the centroid
        centroid = cluster.centroid
        top_dims = torch.topk(centroid, min(10, int(centroid.sum().item()))).indices.tolist()

        # Map dimensions to sensor names
        dim_names = self._dim_to_sensor(top_dims)
        return {
            "cluster_id": cluster_id,
            "label": cluster.label,
            "active_sensors": dim_names,
            "count": cluster.count,
        }

    def _dim_to_sensor(self, dims: list[int]) -> list[str]:
        """Map encoding dimensions to human-readable sensor names."""
        names = []
        for d in dims:
            if 0 <= d < 40:
                names.append(f"App-ID (dim {d})")
            elif 40 <= d < 60:
                names.append(f"Background-App (dim {d})")
            elif 60 <= d < 76:
                names.append(f"Tastatur-Bin (dim {d})")
            elif 76 <= d < 80:
                names.append(f"Tipp-Rhythmus (dim {d})")
            elif 80 <= d < 96:
                names.append(f"Maus-Bin (dim {d})")
            elif 96 <= d < 100:
                names.append(f"Maus-Rhythmus (dim {d})")
            elif 100 <= d < 108:
                names.append(f"Idle-Bin (dim {d})")
            elif 112 <= d < 144:
                names.append(f"Mic-Mel (dim {d})")
            elif 144 <= d < 148:
                names.append(f"Mic-RMS (dim {d})")
            elif 148 <= d < 156:
                names.append(f"Tageszeit (dim {d})")
            elif 156 <= d < 160:
                names.append(f"App-Switch (dim {d})")
            elif 160 <= d < 164:
                names.append(f"Aktivitaets-Level (dim {d})")
            else:
                names.append(f"Reserve (dim {d})")
        return names

    def explain(self) -> dict[str, Any]:
        """Full explanation snapshot for LLM."""
        return {
            "modulator_causes": self.explain_modulators(),
        }
```

**Commit:**
```bash
git add bridge/synapse_explainer.py tests/test_synapse_explainer.py
git commit -m "feat: add SynapseExplainer — LLM reads SNN internals"
```

---

## Task 6: BrainInterpreter — Orchestrator + Integration

**Files:**
- Create: `bridge/interpreter.py`
- Modify: `server/chat.py:236-280` — inject interpreter output into system prompt
- Modify: `bridge/proactive.py:150-262` — use interpreter for smarter triggers
- Modify: `server/braind.py:196-199` — instantiate interpreter, pass to proactive + chat
- Modify: `brain/persistence.py:147-158` — save/load personality state
- Test: `tests/test_interpreter.py`

**Step 1: Create BrainInterpreter**

```python
# bridge/interpreter.py
"""BrainInterpreter — the pet's 'consciousness layer'.

Orchestrates all interpretation modules and produces a unified
understanding of the current situation for the LLM and proactive engine.
"""
from __future__ import annotations
from typing import Any

from brain.core import Brain
from bridge.episode_log import EpisodeLogger
from bridge.state_detector import StateDetector
from bridge.habit_miner import HabitMiner
from bridge.personality import PersonalityTracker
from bridge.narrative import NarrativeBuilder
from bridge.anomaly import AnomalyDetector
from bridge.synapse_explainer import SynapseExplainer


class BrainInterpreter:
    def __init__(self, brain: Brain, episode_logger: EpisodeLogger) -> None:
        self.brain = brain
        self.state_detector = StateDetector()
        self.habit_miner = HabitMiner(episode_logger)
        self.personality = PersonalityTracker()
        self.narrative = NarrativeBuilder(episode_logger)
        self.anomaly = AnomalyDetector(self.habit_miner)
        self.synapse_explainer = SynapseExplainer(brain)

        # Load personality from brain if available
        if hasattr(brain, '_personality_state'):
            self.personality.load_state(brain._personality_state)

    def tick(self) -> None:
        """Call every ~100 ticks (1 second) to update rolling state."""
        self.state_detector.update(self.brain)
        # Update personality every 100 calls (~100 seconds)
        if self.state_detector._snapshot_count % 100 == 0:
            self.personality.update(self.brain.modulators.snapshot())

    def interpret(self) -> dict[str, Any]:
        """Full interpretation for LLM system prompt."""
        sd = getattr(self.brain, '_last_sensor_display', {})
        return {
            "states": self.state_detector.detect(),
            "personality": self.personality.snapshot(),
            "habits": self.habit_miner.current_hour_context(),
            "anomalies": self.anomaly.check(sd),
            "narrative": self.narrative.today_summary(),
            "explanations": self.synapse_explainer.explain(),
        }

    def format_for_prompt(self) -> str:
        """Render interpretation as text for system prompt injection."""
        data = self.interpret()
        lines = ["=== MEIN BEWUSSTSEIN (was ich VERSTEHE) ==="]

        # States
        states = data["states"]
        if states["flow"]:
            lines.append(f"Zustand: Im FLOW seit {states['flow_duration_min']}min in {states['flow_app']}")
        elif states["stress"]:
            lines.append("Zustand: STRESS erkannt — Leon ist hektisch")
        elif states["meeting"]:
            lines.append(f"Zustand: Im MEETING seit {states['meeting_duration_min']}min")
        elif states["needs_break"]:
            lines.append(f"Zustand: Leon arbeitet seit {states['active_minutes']}min ohne Pause")
        else:
            lines.append("Zustand: Normal")

        # Personality
        p = data["personality"]
        traits = p["traits"]
        trait_strs = []
        if traits["curiosity"] > 0.3:
            trait_strs.append("neugierig")
        if traits["patience"] > 0.3:
            trait_strs.append("geduldig")
        if traits["anxiety"] > 0.3:
            trait_strs.append("nervoes")
        if traits["attentiveness"] > 0.3:
            trait_strs.append("aufmerksam")
        if trait_strs:
            lines.append(f"Persoenlichkeit: {', '.join(trait_strs)}")

        # Habits
        habits = data["habits"]
        if habits.get("usual_habits"):
            for h in habits["usual_habits"][:2]:
                lines.append(f"Gewohnheit: Um {h['hour']}h {h['day_name']} → meistens {h['app']}")

        # Anomalies
        for a in data["anomalies"][:2]:
            lines.append(f"Anomalie: {a['description']}")

        # Narrative (short)
        narrative = data["narrative"]
        if narrative and narrative != "Heute noch nicht viel passiert.":
            lines.append(f"Heute bisher: {narrative[:100]}")

        # Modulator explanations
        expl = data["explanations"].get("modulator_causes", {})
        important = {k: v for k, v in expl.items()
                     if "Hoch" in v or "Niedrig" in v}
        if important:
            lines.append("Warum ich mich so fuehle: " +
                         "; ".join(f"{k}={v}" for k, v in important.items()))

        return "\n".join(lines)

    def save_personality(self) -> dict[str, Any]:
        """For persistence — save personality state."""
        return self.personality.save_state()

    def load_personality(self, data: dict[str, Any]) -> None:
        """For persistence — load personality state."""
        self.personality.load_state(data)
```

**Step 2: Integrate into chat.py**

In `server/chat.py`, after the concept_lines block and before `system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(...)`, add:

```python
        # BrainInterpreter: deep understanding for LLM
        interpreter = getattr(brain, '_interpreter', None)
        interpreter_block = ""
        if interpreter:
            interpreter_block = interpreter.format_for_prompt()
```

Then in the `system_prompt` string, inject `interpreter_block` after `{concepts}`:

```python
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            sensor_display="\n".join(sensor_lines),
            ...
            concepts="\n".join(concept_lines) + "\n\n" + interpreter_block,
        )
```

**Step 3: Integrate into proactive.py**

Replace the stress/novelty checks in `_check()` with interpreter-based triggers:

```python
        # Use BrainInterpreter for smarter triggers
        interpreter = getattr(self.brain, '_interpreter', None)
        if interpreter:
            states = interpreter.state_detector.detect()
            if states["flow"] and states["flow_duration_min"] > 60:
                return {
                    "category": "flow",
                    "context": f"Du bist seit {states['flow_duration_min']}min im Flow in {states['flow_app']} — laeuft bei dir!",
                }
            if states["needs_break"]:
                return {
                    "category": "break",
                    "context": f"Hey Leon, du arbeitest seit {states['active_minutes']:.0f} Minuten ohne Pause. Kurz durchatmen?",
                }
            if states["meeting"] and not getattr(self, '_meeting_announced', False):
                self._meeting_announced = True
                return {
                    "category": "meeting",
                    "context": f"Ich seh {sensor_ctx} — bist du in einem Call?",
                }
            elif not states["meeting"]:
                self._meeting_announced = False

            anomalies = interpreter.anomaly.check(
                getattr(self.brain, '_last_sensor_display', {}))
            if anomalies:
                return {
                    "category": "anomaly",
                    "context": anomalies[0]["description"],
                }
```

**Step 4: Integrate into braind.py**

After creating the episode_logger (~line 142), create the interpreter:

```python
    from bridge.interpreter import BrainInterpreter
    interpreter = BrainInterpreter(brain, episode_logger)
    brain._interpreter = interpreter  # accessible from chat endpoint + proactive

    # Start interpreter tick loop
    async def interpreter_tick_loop():
        while True:
            interpreter.tick()
            await asyncio.sleep(1.0)  # 1 Hz update
    interpreter_task = asyncio.create_task(interpreter_tick_loop())
```

**Step 5: Persist personality in save/load**

In `brain/persistence.py` `save_brain()`, after ConceptTracker save:

```python
        # Personality state
        if hasattr(brain, '_interpreter') and brain._interpreter:
            import json as _json
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('personality', ?)",
                (_json.dumps(brain._interpreter.save_personality()),),
            )
```

In `load_brain()`, after ConceptTracker load:

```python
        # Personality state (will be loaded by BrainInterpreter later)
        try:
            row = conn.execute("SELECT value FROM meta WHERE key='personality'").fetchone()
            if row:
                brain._personality_state = json.loads(row[0])
        except (sqlite3.OperationalError, Exception):
            pass
```

**Step 6: Run all tests**

```bash
.venv/bin/python -m pytest tests/test_state_detector.py tests/test_habit_miner.py tests/test_personality.py tests/test_interpreter.py -v
```

**Step 7: Final commit**

```bash
git add bridge/interpreter.py server/chat.py bridge/proactive.py server/braind.py brain/persistence.py tests/test_interpreter.py
git commit -m "feat: BrainInterpreter — orchestrates all interpretation modules

Integrates StateDetector, HabitMiner, PersonalityTracker, NarrativeBuilder,
AnomalyDetector, SynapseExplainer into a unified interpretation layer.

The LLM now sees:
- Flow/stress/meeting/break states
- Personality traits (curiosity, patience, etc.)
- Habit patterns (usual app at this time)
- Anomalies (deviations from habits)
- Temporal narratives (what happened today)
- Modulator explanations (WHY emotions are at current levels)
"
```

---

## Parallelism Map

Tasks 1-5 can be executed by independent parallel agents (no file conflicts):

| Agent | Task | Files |
|-------|------|-------|
| A | StateDetector | bridge/state_detector.py, tests/test_state_detector.py |
| B | HabitMiner | bridge/habit_miner.py, tests/test_habit_miner.py |
| C | PersonalityTracker | bridge/personality.py, tests/test_personality.py |
| D | NarrativeBuilder + Anomaly | bridge/narrative.py, bridge/anomaly.py, tests/test_narrative.py |
| E | SynapseExplainer | bridge/synapse_explainer.py, tests/test_synapse_explainer.py |

Task 6 (BrainInterpreter + Integration) is **sequential** after all 5 complete — it modifies shared files (chat.py, proactive.py, braind.py, persistence.py).
