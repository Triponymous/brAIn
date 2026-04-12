# tests/test_habit_miner.py
import time
import json
import sqlite3
import datetime
import tempfile
from pathlib import Path
from bridge.episode_log import EpisodeLogger
from bridge.habit_miner import HabitMiner


def _fill_episodes(logger, days=7, entries_per_hour=6):
    """Fill logger with synthetic week of activity.

    Anchors timestamps to local midnight so that the intended hour
    maps correctly regardless of when the test runs.
    """
    now = time.time()
    today_midnight = datetime.datetime.now().replace(
        hour=0, minute=0, second=0, microsecond=0
    ).timestamp()
    for day in range(days):
        day_midnight = today_midnight - (days - 1 - day) * 86400
        for hour in range(9, 18):  # work hours
            for i in range(entries_per_hour):
                ts = day_midnight + hour * 3600 + i * 600
                if ts > now:
                    continue  # skip future timestamps
                # Hour 9: always Slack, otherwise VSCode
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
    """Should detect pattern: 9am -> Slack."""
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


def test_current_habits_caching():
    """current_habits() should cache results for 5 minutes."""
    with tempfile.TemporaryDirectory() as td:
        logger = EpisodeLogger(Path(td) / "test.db")
        _fill_episodes(logger)
        miner = HabitMiner(logger)
        h1 = miner.current_habits()
        h2 = miner.current_habits()
        # Second call should return same object (cached)
        assert h1 is h2
        logger.close()


def test_current_hour_context():
    """current_hour_context() should return habits for the current hour."""
    with tempfile.TemporaryDirectory() as td:
        logger = EpisodeLogger(Path(td) / "test.db")
        _fill_episodes(logger)
        miner = HabitMiner(logger)
        ctx = miner.current_hour_context()
        assert "hour" in ctx
        assert "day_of_week" in ctx
        assert "usual_habits" in ctx
        assert isinstance(ctx["usual_habits"], list)
        logger.close()


def test_empty_log_returns_empty():
    """Empty episode log should return no habits and empty profiles."""
    with tempfile.TemporaryDirectory() as td:
        logger = EpisodeLogger(Path(td) / "test.db")
        miner = HabitMiner(logger)
        habits = miner.mine_habits()
        assert habits == []
        profile = miner.hourly_profile()
        assert profile[0]["top_app"] is None
        assert profile[12]["episodes"] == 0
        logger.close()


def test_confidence_threshold():
    """Habits below 50% confidence should be filtered out."""
    with tempfile.TemporaryDirectory() as td:
        logger = EpisodeLogger(Path(td) / "test.db")
        now = time.time()
        today_midnight = datetime.datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()
        # Insert mixed apps at the same hour so no single app dominates
        for day in range(7):
            day_midnight = today_midnight - (6 - day) * 86400
            for i, app in enumerate(["AppA", "AppB", "AppC", "AppD", "AppE", "AppF"]):
                ts = day_midnight + 14 * 3600 + i * 600
                if ts > now:
                    continue
                logger._conn.execute(
                    "INSERT INTO episodes(timestamp, tick, modulators, active_concepts, sensor_summary, sleep_mode) VALUES (?,?,?,?,?,?)",
                    (ts, int((ts - now) * 100), '{"DA":0.02}',
                     json.dumps([0]),
                     json.dumps({"app": app, "keys": 10}),
                     0),
                )
        logger._conn.commit()
        miner = HabitMiner(logger)
        habits = miner.mine_habits()
        # No single app should dominate at hour 14 across the week
        hour14_habits = [h for h in habits if h["hour"] == 14]
        # Each app appears once per day = ~16.7% confidence, all below 50%
        assert len(hour14_habits) == 0
        logger.close()
