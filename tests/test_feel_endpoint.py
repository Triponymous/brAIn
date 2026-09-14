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


def test_post_labels_frozen_pending_signature_not_live():
    """A pending ask (state-change) freezes the anomaly signature; POSTing a label
    applies to THAT, not the live state Leon is in when he answers."""
    import time as _t
    from fastapi import FastAPI
    brain = Brain(num_sensory=8, num_concept=4, num_wm=4)
    brain.felt_state = FeltState()
    anomaly = [0.001, 0.002, 0.003, 0.05, 0.004, 0.005]   # != the stub's live flow signature
    brain._pending_ask = {"signature": anomaly, "at": _t.time()}
    app = FastAPI()
    app.include_router(build_feel_router(brain, _StubDetector()))
    c = TestClient(app)
    c.post("/api/feel", json={"label": "pause"})
    assert brain.felt_state.recognize(anomaly)[0] == "pause"   # labeled the FROZEN moment
    assert brain._pending_ask is None                          # pending cleared


# ── Experience log ───────────────────────────────────────────────────────────

def _make_app_with_log(tmp_path):
    from bridge.experience import ExperienceLog
    brain = Brain(num_sensory=8, num_concept=4, num_wm=4)
    brain.felt_state = FeltState()
    brain._experience = ExperienceLog(tmp_path / "experience.db")
    app = FastAPI()
    app.include_router(build_feel_router(brain, _StubDetector()))
    return TestClient(app), brain


def test_teaching_records_an_experience(tmp_path):
    c, brain = _make_app_with_log(tmp_path)
    c.post("/api/feel", json={"label": "flow"})
    rows = brain._experience.recent()
    assert len(rows) == 1
    assert rows[0]["actor"] == "human" and rows[0]["kind"] == "teach_felt"
    assert rows[0]["payload"] == {"label": "flow", "answered_ask": False}


def test_answering_an_ask_links_both_events(tmp_path):
    import time as _t
    c, brain = _make_app_with_log(tmp_path)
    ask_id = brain._experience.record("pet", "ask_label", {}, state={"signature": [0.01] * 6})
    brain._pending_ask = {"signature": [0.01] * 6, "cluster": -1, "at": _t.time(), "event_id": ask_id}

    c.post("/api/feel", json={"label": "pause"})

    rows = {r["id"]: r for r in brain._experience.recent()}
    teach = next(r for r in rows.values() if r["kind"] == "teach_felt")
    assert teach["payload"]["answered_ask"] is True
    assert rows[ask_id]["response_kind"] == "answered" and rows[ask_id]["response_id"] == teach["id"]


def test_dismiss_releases_the_frozen_moment_and_records_it(tmp_path):
    import time as _t
    c, brain = _make_app_with_log(tmp_path)
    ask_id = brain._experience.record("pet", "ask_label", {}, state={"signature": [0.01] * 6})
    anomaly = [0.001, 0.002, 0.003, 0.05, 0.004, 0.005]
    brain._pending_ask = {"signature": anomaly, "cluster": -1, "at": _t.time(), "event_id": ask_id}

    assert c.post("/api/feel/dismiss").json() == {"dismissed": True}
    assert brain._pending_ask is None
    rows = {r["id"]: r for r in brain._experience.recent()}
    assert rows[ask_id]["response_kind"] == "dismissed"
    assert any(r["kind"] == "dismiss_ask" for r in rows.values())

    # a label taught now applies to the LIVE state, not the waved-away moment
    c.post("/api/feel", json={"label": "flow"})
    assert brain.felt_state.recognize(anomaly)[0] is None


def test_dismiss_without_a_pending_ask_is_a_noop(tmp_path):
    c, brain = _make_app_with_log(tmp_path)
    assert c.post("/api/feel/dismiss").json() == {"dismissed": False}
    assert brain._experience.recent() == []
