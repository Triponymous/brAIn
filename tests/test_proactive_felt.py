"""ProactiveEngine felt-state active-learning trigger (_check_felt_state)."""
from brain.core import Brain
from bridge.exporter import BrainStateExporter
from bridge.felt_state import FeltState
from bridge.proactive import ProactiveEngine
from server.ws import WSPusher


def _engine():
    brain = Brain(num_sensory=8, num_concept=4, num_wm=4)
    brain.felt_state = FeltState()                              # empty → unknown
    brain._last_signature = [0.02, 0.02, 0.05, 0.03, 0.05, 0.06]
    eng = ProactiveEngine(brain, BrainStateExporter(brain), WSPusher())
    return eng, brain


def test_asks_after_sustained_unknown():
    eng, _ = _engine()
    assert eng._check_felt_state(0) is None          # just became unknown
    assert eng._check_felt_state(30) is not None     # sustained >=25s → ask


def test_silent_once_recognized():
    eng, brain = _engine()
    eng._check_felt_state(0)
    brain.felt_state.label("entspanntes arbeiten", brain._last_signature)
    assert eng._check_felt_state(300) is None        # now recognized → no ask


def test_silent_during_sleep():
    eng, brain = _engine()
    brain.sleep_mode = True
    assert eng._check_felt_state(0) is None
    assert eng._check_felt_state(60) is None


def test_ask_freezes_the_moment():
    eng, brain = _engine()
    eng._check_felt_state(0)
    assert eng._check_felt_state(30) is not None              # fires the ask
    assert getattr(brain, "_pending_ask", None) is not None   # captured the moment
    assert brain._pending_ask["signature"] == brain._last_signature


def test_ask_is_recorded_as_an_experience(tmp_path):
    from bridge.experience import ExperienceLog
    eng, brain = _engine()
    brain._experience = ExperienceLog(tmp_path / "experience.db")
    eng._check_felt_state(0)
    eng._check_felt_state(30)                                 # fires the ask
    rows = brain._experience.recent()
    assert len(rows) == 1 and rows[0]["actor"] == "pet" and rows[0]["kind"] == "ask_label"
    assert rows[0]["ts"] == 30 and rows[0]["signature"] == brain._last_signature
    assert brain._pending_ask["event_id"] == rows[0]["id"]  # the answer can find its ask
