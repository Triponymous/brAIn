"""Tests for FeltState — the learned emotional self-model."""
from bridge.felt_state import FeltState, FeltStateWatcher, signature_from_trend, SIGNATURE_KEYS

FLOW = [0.028, 0.027, 0.058, 0.035, 0.060, 0.069]          # measured flow signature
FLOW_NEAR = [0.030, 0.025, 0.055, 0.037, 0.058, 0.071]     # same regime, slight drift
SILENCE = [0.003, 0.005, 0.009, 0.045, 0.005, 0.006]       # clearly different


def test_empty_recognizes_nothing():
    assert FeltState().recognize(FLOW) == (None, 0.0)


def test_label_then_recognize_hits():
    fs = FeltState()
    fs.label("flow", FLOW)
    name, conf = fs.recognize(FLOW_NEAR)
    assert name == "flow" and conf > 0.0


def test_recognize_rejects_far_signature():
    fs = FeltState()
    fs.label("flow", FLOW)
    assert fs.recognize(SILENCE)[0] is None


def test_label_sharpens_prototype():
    fs = FeltState()
    fs.label("flow", FLOW)
    fs.label("flow", FLOW_NEAR)
    assert fs.to_dict()["prototypes"]["flow"]["count"] == 2


def test_known_labels_sorted():
    fs = FeltState()
    fs.label("flow", FLOW)
    fs.label("admin", SILENCE)
    assert fs.known_labels() == ["admin", "flow"]


def test_to_from_dict_roundtrip():
    fs = FeltState()
    fs.label("flow", FLOW)
    fs2 = FeltState.from_dict(fs.to_dict())
    assert fs2.recognize(FLOW_NEAR)[0] == "flow"
    assert fs2.known_labels() == ["flow"]


def test_signature_from_trend_fixed_order():
    trend = {k: i * 0.01 for i, k in enumerate(SIGNATURE_KEYS)}
    assert signature_from_trend(trend) == [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]


# ── FeltStateWatcher: when to ask for a label (active learning) ──

def test_watcher_silent_while_recognized():
    w = FeltStateWatcher(hold_seconds=25, cooldown_seconds=180)
    for t in range(0, 100, 10):
        w.observe("flow", t)
        assert w.should_ask(t) is False


def test_watcher_asks_after_sustained_unknown():
    w = FeltStateWatcher(hold_seconds=25, cooldown_seconds=180)
    w.observe(None, 0)
    assert w.should_ask(10) is False        # only 10s unknown < 25s hold
    w.observe(None, 20)
    assert w.should_ask(30) is True         # 30s unknown ≥ hold, first ask not blocked


def test_watcher_cooldown_blocks_repeat():
    w = FeltStateWatcher(hold_seconds=25, cooldown_seconds=180)
    w.observe(None, 0)
    assert w.should_ask(30) is True
    w.observe(None, 31)
    assert w.should_ask(60) is False        # within cooldown
    w.observe(None, 220)
    assert w.should_ask(230) is True        # cooldown elapsed + sustained


def test_watcher_resets_when_recognized_again():
    w = FeltStateWatcher(hold_seconds=25, cooldown_seconds=180)
    w.observe(None, 0)
    w.observe("entspanntes arbeiten", 10)    # recognized → reset
    assert w.should_ask(40) is False
