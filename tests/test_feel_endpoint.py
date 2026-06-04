"""Tests for the /api/feel router (read + teach the felt-state)."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.feel import build_feel_router
from bridge.felt_state import FeltState
from brain.core import Brain


class _StubDetector:
    """Returns a fixed trend = Leon's measured flow signature."""
    def emotional_trend(self, window_seconds: int = 180) -> dict:
        return {"avg_DA": 0.028, "avg_NE": 0.027, "avg_ACh": 0.058,
                "avg_5HT": 0.035, "peak_DA": 0.060, "peak_NE": 0.069}


def _make_app():
    brain = Brain(num_sensory=8, num_concept=4, num_wm=4)
    brain.felt_state = FeltState()
    app = FastAPI()
    app.include_router(build_feel_router(brain, _StubDetector()))
    return TestClient(app)


def test_get_feel_empty():
    c = _make_app()
    r = c.get("/api/feel").json()
    assert r["recognized"] is None
    assert r["known_labels"] == []
    assert set(r["signature"]) == {"avg_DA", "avg_NE", "avg_ACh", "avg_5HT", "peak_DA", "peak_NE"}


def test_post_feel_teaches_then_get_recognizes():
    c = _make_app()
    p = c.post("/api/feel", json={"label": "flow"}).json()
    assert p["labeled"] == "flow"
    assert p["known_labels"] == ["flow"]
    g = c.get("/api/feel").json()
    assert g["recognized"] == "flow"
    assert g["confidence"] > 0.0
