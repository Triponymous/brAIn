"""Phase 1: prove the neuromodulators EMERGE from prediction error, not if-statements.

These assert the PRINCIPLE (direction of causation), not magnitudes, so they survive
gain re-tuning. The noisy-TV test is the regression guard for the learning-progress
mandate: a constant-high-error stream must NOT reward dopamine.
"""
import pathlib

import torch

from brain.core import Brain


def test_noisy_tv_does_not_reward_dopamine():
    """Irreducible noise = high but FLAT prediction error => no learning progress
    => DA must not climb. The core noisy-TV / Schmidhuber guarantee. Precision
    (high pe_var) is what gates the spurious progress that pe-wobble leaks."""
    brain = Brain(num_sensory=64)
    g = torch.Generator().manual_seed(0)
    for _ in range(400):
        brain.tick(torch.rand(64, generator=g) * 8.0)
    for _ in range(200):
        brain.tick(torch.zeros(64))          # let DA decay to a floor
    base = brain.modulators.level("DA")
    g2 = torch.Generator().manual_seed(1)
    da_peak = base
    for _ in range(400):
        brain.tick(torch.rand(64, generator=g2) * 8.0)
        da_peak = max(da_peak, brain.modulators.level("DA"))
    assert da_peak - base < 0.01, (
        f"noisy-TV rewarded DA: base={base:.4f} peak={da_peak:.4f} "
        f"(DA must track learning progress, not raw surprise)"
    )


def test_surprise_drives_noradrenaline():
    """A sudden unfamiliar input = high prediction error => NE rises. Direction
    only (magnitude is gain-dependent). Proves NE is surprise-driven, not scripted."""
    brain = Brain(num_sensory=64)
    calm = torch.ones(64) * 0.5
    for _ in range(500):
        brain.tick(calm)                     # train _sensory_avg on the calm pattern
    ne_before = brain.modulators.level("NE")
    spike = torch.ones(64) * 6.0             # large deviation from the learned average
    ne_peak = ne_before
    for _ in range(50):
        brain.tick(spike)
        ne_peak = max(ne_peak, brain.modulators.level("NE"))
    assert ne_peak > ne_before, "surprise did not raise NE — driver not emergent"


def test_no_hardcoded_stimulus_thresholds_in_driver():
    """Grep-guard: the driver must not reintroduce magic sensory_change thresholds.
    Emotions emerge from prediction error, not from `if sensory_change > N`. This
    test fails the instant anyone re-grafts a scripted if-statement emotion."""
    src = pathlib.Path(__file__).resolve().parents[1] / "brain" / "core.py"
    text = src.read_text()
    assert "sensory_change" not in text, "scripted sensory_change threshold reintroduced"
    assert "sustained_stress" not in text, "scripted stress heuristic reintroduced"
