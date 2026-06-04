"""Tests for FeltState — the learned emotional self-model."""
from bridge.felt_state import FeltState, signature_from_trend, SIGNATURE_KEYS

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
