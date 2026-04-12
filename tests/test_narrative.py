"""Tests for NarrativeBuilder and AnomalyDetector."""
from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bridge.episode_log import EpisodeLogger
from bridge.narrative import NarrativeBuilder, _local_midnight_ts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(
    ts: float,
    app: str = "VSCode",
    keys: int = 20,
    tick: int = 100,
) -> dict:
    """Build an episode dict matching EpisodeLogger.query() output."""
    return {
        "id": 1,
        "timestamp": ts,
        "tick": tick,
        "modulators": {"DA": 0.02, "NE": 0.01, "ACh": 0.03, "5HT": 0.03},
        "active_concepts": [],
        "sensor_summary": {"app": app, "keys": keys, "mouse": 5},
        "sleep_mode": False,
    }


def _episodes_for_hours(
    base_ts: float,
    hour_specs: list[tuple[int, str, int]],
    count_per_hour: int = 6,
) -> list[dict]:
    """Generate episodes for specific hours.

    hour_specs: list of (hour, app, keys) tuples
    base_ts: midnight timestamp to anchor the hours
    """
    episodes = []
    for hour, app, keys in hour_specs:
        for i in range(count_per_hour):
            ts = base_ts + hour * 3600 + i * 10
            episodes.append(_make_episode(ts, app=app, keys=keys))
    return episodes


# ---------------------------------------------------------------------------
# NarrativeBuilder tests
# ---------------------------------------------------------------------------


class TestNarrativeBuilder:
    """Tests for NarrativeBuilder."""

    def test_today_summary_empty(self, tmp_path: Path):
        """No episodes -> default message."""
        logger = EpisodeLogger(tmp_path / "test.db")
        nb = NarrativeBuilder(logger)
        result = nb.today_summary()
        assert result == "Heute noch nicht viel passiert."
        logger.close()

    def test_today_summary_with_episodes(self, tmp_path: Path):
        """Episodes from today produce hour-grouped narrative."""
        logger = EpisodeLogger(tmp_path / "test.db")
        nb = NarrativeBuilder(logger)

        # Insert episodes for the current hour
        now = time.time()
        for i in range(5):
            logger.log(
                tick=100 + i,
                modulators={"DA": 0.02},
                active_concepts=[],
                sensor_summary={"app": "VSCode", "keys": 25, "mouse": 3},
                sleep_mode=False,
            )

        result = nb.today_summary()
        assert "intensiv gearbeitet" in result
        assert "VSCode" in result
        logger.close()

    def test_today_summary_multiple_hours(self, tmp_path: Path):
        """Multiple hours produce pipe-separated blocks."""
        logger = EpisodeLogger(tmp_path / "test.db")
        nb = NarrativeBuilder(logger)

        midnight = _local_midnight_ts(offset_days=0)
        current_hour = datetime.datetime.now().hour

        # Pick two hours that are both today (both <= current hour)
        h1 = max(0, current_hour - 2)
        h2 = max(1, current_hour - 1)

        episodes = _episodes_for_hours(midnight, [
            (h1, "Terminal", 20),
            (h2, "Chrome", 1),
        ])

        # Manually insert with specific timestamps
        for ep in episodes:
            logger._conn.execute(
                "INSERT INTO episodes(timestamp, tick, modulators, active_concepts, "
                "sensor_summary, sleep_mode) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ep["timestamp"],
                    ep["tick"],
                    json.dumps(ep["modulators"]),
                    json.dumps(ep["active_concepts"]),
                    json.dumps(ep["sensor_summary"]),
                    int(ep["sleep_mode"]),
                ),
            )
        logger._conn.commit()

        result = nb.today_summary()
        assert "|" in result
        assert "Terminal" in result
        assert "Chrome" in result
        assert "intensiv gearbeitet" in result
        assert "ruhig" in result
        logger.close()

    def test_yesterday_summary_empty(self, tmp_path: Path):
        """No yesterday episodes -> default message."""
        logger = EpisodeLogger(tmp_path / "test.db")
        nb = NarrativeBuilder(logger)
        result = nb.yesterday_summary()
        assert result == "Gestern keine Daten."
        logger.close()

    def test_yesterday_summary_with_data(self, tmp_path: Path):
        """Episodes from yesterday produce narrative; today's are excluded."""
        logger = EpisodeLogger(tmp_path / "test.db")
        nb = NarrativeBuilder(logger)

        yesterday_midnight = _local_midnight_ts(offset_days=1)
        today_midnight = _local_midnight_ts(offset_days=0)

        # Insert episodes for yesterday at hour 10
        yesterday_episodes = _episodes_for_hours(yesterday_midnight, [
            (10, "Slack", 8),
        ])
        # Insert an episode for today (should be excluded)
        today_episodes = _episodes_for_hours(today_midnight, [
            (9, "VSCode", 25),
        ])

        for ep in yesterday_episodes + today_episodes:
            logger._conn.execute(
                "INSERT INTO episodes(timestamp, tick, modulators, active_concepts, "
                "sensor_summary, sleep_mode) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ep["timestamp"],
                    ep["tick"],
                    json.dumps(ep["modulators"]),
                    json.dumps(ep["active_concepts"]),
                    json.dumps(ep["sensor_summary"]),
                    int(ep["sleep_mode"]),
                ),
            )
        logger._conn.commit()

        result = nb.yesterday_summary()
        assert "Slack" in result
        assert "leicht aktiv" in result
        # Today's data should not appear
        assert "VSCode" not in result
        logger.close()

    def test_build_narrative_activity_levels(self, tmp_path: Path):
        """Test all three activity thresholds."""
        logger = EpisodeLogger(tmp_path / "test.db")
        nb = NarrativeBuilder(logger)

        midnight = _local_midnight_ts(offset_days=0)
        current_hour = datetime.datetime.now().hour
        h = max(0, current_hour - 1)

        # High activity (keys > 15)
        eps_high = _episodes_for_hours(midnight, [(h, "IntelliJ", 30)])
        result = nb._build_narrative(eps_high)
        assert "intensiv gearbeitet" in result

        # Medium activity (3 < keys <= 15)
        eps_med = _episodes_for_hours(midnight, [(h, "Safari", 8)])
        result = nb._build_narrative(eps_med)
        assert "leicht aktiv" in result

        # Low activity (keys <= 3)
        eps_low = _episodes_for_hours(midnight, [(h, "Finder", 1)])
        result = nb._build_narrative(eps_low)
        assert "ruhig" in result

        logger.close()

    def test_build_narrative_caps_at_8_hours(self, tmp_path: Path):
        """Narrative is capped at 8 hourly blocks."""
        logger = EpisodeLogger(tmp_path / "test.db")
        nb = NarrativeBuilder(logger)

        midnight = _local_midnight_ts(offset_days=0)
        specs = [(h, "App", 10) for h in range(12)]
        episodes = _episodes_for_hours(midnight, specs)
        result = nb._build_narrative(episodes)

        blocks = result.split(" | ")
        assert len(blocks) <= 8
        logger.close()

    def test_build_narrative_empty_list(self, tmp_path: Path):
        """Empty episode list produces fallback."""
        logger = EpisodeLogger(tmp_path / "test.db")
        nb = NarrativeBuilder(logger)
        result = nb._build_narrative([])
        assert result == "Keine Aktivitaet."
        logger.close()

    def test_build_narrative_dominant_app(self, tmp_path: Path):
        """When multiple apps share an hour, the most frequent wins."""
        logger = EpisodeLogger(tmp_path / "test.db")
        nb = NarrativeBuilder(logger)

        midnight = _local_midnight_ts(offset_days=0)
        h = max(0, datetime.datetime.now().hour - 1)

        episodes = []
        # 4 episodes in VSCode, 2 in Chrome -> VSCode should win
        for i in range(4):
            episodes.append(_make_episode(midnight + h * 3600 + i * 10, app="VSCode", keys=15))
        for i in range(2):
            episodes.append(_make_episode(midnight + h * 3600 + 40 + i * 10, app="Chrome", keys=10))

        result = nb._build_narrative(episodes)
        assert "VSCode" in result
        logger.close()


class TestLocalMidnightTs:
    """Tests for the _local_midnight_ts helper."""

    def test_today_midnight_is_past(self):
        """Today's midnight must be in the past (or right now)."""
        ts = _local_midnight_ts(offset_days=0)
        assert ts <= time.time()

    def test_yesterday_midnight_before_today(self):
        """Yesterday's midnight must be before today's."""
        yesterday = _local_midnight_ts(offset_days=1)
        today = _local_midnight_ts(offset_days=0)
        assert yesterday < today
        # Difference should be ~86400 seconds
        assert abs((today - yesterday) - 86400) < 7200  # allow DST


# ---------------------------------------------------------------------------
# AnomalyDetector tests
# ---------------------------------------------------------------------------


class TestAnomalyDetector:
    """Tests for AnomalyDetector (bridge/anomaly.py)."""

    def _mock_habit_miner(
        self,
        habits: list[dict] | None = None,
        hourly_profile: dict | None = None,
    ) -> MagicMock:
        """Create a mock HabitMiner with configurable returns."""
        miner = MagicMock()
        miner.current_habits.return_value = habits or []
        miner.hourly_profile.return_value = hourly_profile or {}
        return miner

    def test_no_anomalies_when_matching(self):
        """No anomalies when current state matches habits."""
        now = datetime.datetime.now()

        miner = self._mock_habit_miner(
            habits=[{
                "hour": now.hour,
                "day_of_week": now.weekday(),
                "app": "VSCode",
                "confidence": 0.8,
                "day_name": "Montag",
            }],
            hourly_profile={now.hour: {"avg_activity": 15}},
        )

        # Import here to allow mocking the habit_miner import
        from bridge.anomaly import AnomalyDetector
        detector = AnomalyDetector(miner)
        result = detector.check({"app": "VSCode", "keys": 20})
        assert result == []

    def test_missing_habit_detected(self):
        """Detects when expected app is not in use."""
        now = datetime.datetime.now()

        miner = self._mock_habit_miner(
            habits=[{
                "hour": now.hour,
                "day_of_week": now.weekday(),
                "app": "Slack",
                "confidence": 0.9,
                "day_name": "Dienstag",
            }],
            hourly_profile={},
        )

        from bridge.anomaly import AnomalyDetector
        detector = AnomalyDetector(miner)
        result = detector.check({"app": "Chrome", "keys": 10})

        assert len(result) == 1
        assert result[0]["type"] == "missing_habit"
        assert result[0]["severity"] == "medium"
        assert "Slack" in result[0]["description"]
        assert result[0]["expected"] == "Slack"
        assert result[0]["actual"] == "Chrome"

    def test_low_confidence_habit_ignored(self):
        """Habits with confidence <= 0.6 are not flagged."""
        now = datetime.datetime.now()

        miner = self._mock_habit_miner(
            habits=[{
                "hour": now.hour,
                "day_of_week": now.weekday(),
                "app": "Slack",
                "confidence": 0.4,
                "day_name": "Mittwoch",
            }],
            hourly_profile={},
        )

        from bridge.anomaly import AnomalyDetector
        detector = AnomalyDetector(miner)
        result = detector.check({"app": "Chrome", "keys": 10})
        assert result == []

    def test_unusual_quiet_detected(self):
        """Detects when normally active but currently silent."""
        now = datetime.datetime.now()

        miner = self._mock_habit_miner(
            habits=[],
            hourly_profile={now.hour: {"avg_activity": 20}},
        )

        from bridge.anomaly import AnomalyDetector
        detector = AnomalyDetector(miner)
        result = detector.check({"app": "VSCode", "keys": 0})

        assert len(result) == 1
        assert result[0]["type"] == "unusual_quiet"
        assert result[0]["severity"] == "low"
        assert "tippst" in result[0]["description"]

    def test_no_quiet_anomaly_when_active(self):
        """No unusual_quiet if current keys >= 2."""
        now = datetime.datetime.now()

        miner = self._mock_habit_miner(
            habits=[],
            hourly_profile={now.hour: {"avg_activity": 20}},
        )

        from bridge.anomaly import AnomalyDetector
        detector = AnomalyDetector(miner)
        result = detector.check({"app": "VSCode", "keys": 5})
        assert result == []

    def test_no_quiet_anomaly_when_baseline_low(self):
        """No unusual_quiet if baseline avg_activity <= 10."""
        now = datetime.datetime.now()

        miner = self._mock_habit_miner(
            habits=[],
            hourly_profile={now.hour: {"avg_activity": 8}},
        )

        from bridge.anomaly import AnomalyDetector
        detector = AnomalyDetector(miner)
        result = detector.check({"app": "VSCode", "keys": 0})
        assert result == []

    def test_multiple_anomalies(self):
        """Both missing_habit and unusual_quiet can fire at once."""
        now = datetime.datetime.now()

        miner = self._mock_habit_miner(
            habits=[{
                "hour": now.hour,
                "day_of_week": now.weekday(),
                "app": "Slack",
                "confidence": 0.85,
                "day_name": "Donnerstag",
            }],
            hourly_profile={now.hour: {"avg_activity": 25}},
        )

        from bridge.anomaly import AnomalyDetector
        detector = AnomalyDetector(miner)
        result = detector.check({"app": "Finder", "keys": 0})

        types = {a["type"] for a in result}
        assert "missing_habit" in types
        assert "unusual_quiet" in types

    def test_different_hour_habit_ignored(self):
        """Habits for a different hour than now are not checked."""
        now = datetime.datetime.now()
        other_hour = (now.hour + 5) % 24

        miner = self._mock_habit_miner(
            habits=[{
                "hour": other_hour,
                "day_of_week": now.weekday(),
                "app": "Slack",
                "confidence": 0.9,
                "day_name": "Freitag",
            }],
            hourly_profile={},
        )

        from bridge.anomaly import AnomalyDetector
        detector = AnomalyDetector(miner)
        result = detector.check({"app": "Chrome", "keys": 10})
        assert result == []

    def test_different_day_habit_ignored(self):
        """Habits for a different day of week are not checked."""
        now = datetime.datetime.now()
        other_dow = (now.weekday() + 3) % 7

        miner = self._mock_habit_miner(
            habits=[{
                "hour": now.hour,
                "day_of_week": other_dow,
                "app": "Slack",
                "confidence": 0.9,
                "day_name": "Samstag",
            }],
            hourly_profile={},
        )

        from bridge.anomaly import AnomalyDetector
        detector = AnomalyDetector(miner)
        result = detector.check({"app": "Chrome", "keys": 10})
        assert result == []

    def test_empty_sensor_data(self):
        """Handles missing keys in sensor dict gracefully."""
        miner = self._mock_habit_miner(habits=[], hourly_profile={})

        from bridge.anomaly import AnomalyDetector
        detector = AnomalyDetector(miner)
        result = detector.check({})
        assert result == []
