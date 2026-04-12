# tests/test_personality.py
from bridge.personality import PersonalityTracker


def test_baseline_drift():
    """Baselines should drift toward session averages over time."""
    pt = PersonalityTracker()
    # Simulate high-DA sessions (curious user)
    # EMA: baseline converges to level*(1-(1-drift)^n)
    # After 5000 updates: DA ~ 0.039, trait ~ 0.39
    for _ in range(5000):
        pt.update({"DA": 0.1, "NE": 0.01, "ACh": 0.03, "5HT": 0.02})
    snap = pt.snapshot()
    assert snap["baselines"]["DA"] > 0.01  # drifted up from 0
    assert snap["traits"]["curiosity"] > 0.3  # meaningful curiosity


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
