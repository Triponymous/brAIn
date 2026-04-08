"""Tests for the neuromodulator system.

Four modulators: dopamine (DA), noradrenaline (NE), acetylcholine (ACh),
serotonin (5HT). Each has a baseline (default 0), a current level, and
decays toward baseline with its own time constant.
"""
import math
import pytest
from brain.modulators import Modulators


def test_modulators_construction_defaults_zero():
    mod = Modulators()
    assert mod.level("DA") == 0.0
    assert mod.level("NE") == 0.0
    assert mod.level("ACh") == 0.0
    assert mod.level("5HT") == 0.0


def test_modulators_inject_raises_level():
    mod = Modulators()
    mod.inject("DA", 0.5)
    assert mod.level("DA") == 0.5
    mod.inject("DA", 0.3)
    assert abs(mod.level("DA") - 0.8) < 1e-6


def test_modulators_inject_unknown_raises():
    mod = Modulators()
    with pytest.raises(KeyError):
        mod.inject("XYZ", 1.0)


def test_modulators_decay_toward_baseline():
    mod = Modulators(tau={"DA": 100.0})
    mod.inject("DA", 1.0)
    initial = mod.level("DA")
    mod.tick(dt=100.0)  # one full time constant
    expected = initial * math.exp(-1.0)  # ~0.368
    assert abs(mod.level("DA") - expected) < 1e-3


def test_modulators_independent_decay():
    mod = Modulators(tau={"DA": 100.0, "NE": 200.0})
    mod.inject("DA", 1.0)
    mod.inject("NE", 1.0)
    mod.tick(dt=100.0)
    # DA at 1 tau -> 0.368, NE at 0.5 tau -> exp(-0.5) ≈ 0.607
    assert abs(mod.level("DA") - math.exp(-1.0)) < 1e-3
    assert abs(mod.level("NE") - math.exp(-0.5)) < 1e-3


def test_modulators_clipped_to_bounds():
    mod = Modulators()
    mod.inject("DA", 100.0)  # absurdly large
    assert mod.level("DA") <= 1.0
    mod.inject("DA", -100.0)
    assert mod.level("DA") >= -1.0


def test_modulators_snapshot_returns_dict():
    mod = Modulators()
    mod.inject("DA", 0.5)
    mod.inject("NE", 0.3)
    snap = mod.snapshot()
    assert snap == {"DA": 0.5, "NE": 0.3, "ACh": 0.0, "5HT": 0.0}
