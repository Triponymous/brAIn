"""Tests for the episode logger — SQLite-backed brain state history."""
import time
import pytest
from pathlib import Path

from bridge.episode_log import EpisodeLogger


@pytest.fixture
def logger(tmp_path):
    """Create an EpisodeLogger using a temp directory."""
    lg = EpisodeLogger(tmp_path / "test_episodes.db")
    yield lg
    lg.close()


def test_logger_construction(tmp_path):
    """Logger creates without error and DB file exists."""
    lg = EpisodeLogger(tmp_path / "ep.db")
    assert (tmp_path / "ep.db").exists()
    lg.close()


def test_log_and_query_episode(logger):
    """Log one episode, query it back, verify fields."""
    logger.log(
        tick=100,
        modulators={"dopamine": 0.5, "serotonin": 0.3},
        active_concepts=[{"id": 0, "activation": 0.9, "label": "coding"}],
        sensor_summary={"app": "VSCode", "keys": 42},
        sleep_mode=False,
    )
    episodes = logger.query(last_n=10)
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep["tick"] == 100
    assert ep["modulators"]["dopamine"] == 0.5
    assert ep["active_concepts"][0]["label"] == "coding"
    assert ep["sensor_summary"]["app"] == "VSCode"
    assert ep["sleep_mode"] is False


def test_query_time_range(logger):
    """Log 3 episodes, query last 2, verify order (most recent first)."""
    # Log 3 episodes with slightly different timestamps
    for i in range(3):
        logger.log(
            tick=i * 1000,
            modulators={"dopamine": 0.1 * i},
            active_concepts=[{"id": i, "activation": 0.5}],
            sensor_summary={"app": f"App{i}"},
            sleep_mode=False,
        )

    # Query last 2
    episodes = logger.query(last_n=2)
    assert len(episodes) == 2
    # Most recent first
    assert episodes[0]["tick"] == 2000
    assert episodes[1]["tick"] == 1000


def test_daily_summary(logger):
    """Log 100 episodes, verify summary has required keys and sensible values."""
    for i in range(100):
        logger.log(
            tick=i * 100,
            modulators={"dopamine": 0.5, "serotonin": 0.3 + (i % 10) * 0.01},
            active_concepts=[
                {"id": i % 5, "activation": 0.8, "label": f"concept_{i % 5}"},
                {"id": 10, "activation": 0.02},  # below threshold, should be filtered
            ],
            sensor_summary={"app": f"App{i % 3}", "keys": i},
            sleep_mode=i > 90,
        )

    summary = logger.daily_summary(since_hours=24.0)
    assert "total_ticks" in summary
    assert "top_concepts" in summary
    assert "avg_modulators" in summary
    assert "top_apps" in summary
    assert summary["total_ticks"] == 9900  # 99*100 - 0
    assert summary["episode_count"] == 100

    # Check top concepts are present
    concept_names = [c["name"] for c in summary["top_concepts"]]
    assert "concept_0" in concept_names

    # Check avg modulators
    assert "dopamine" in summary["avg_modulators"]
    assert abs(summary["avg_modulators"]["dopamine"] - 0.5) < 0.01

    # Check top apps
    app_names = [a["name"] for a in summary["top_apps"]]
    assert "App0" in app_names
    assert "App1" in app_names
    assert "App2" in app_names
