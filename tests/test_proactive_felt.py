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
