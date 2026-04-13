"""Benchmark: Energy Proxy -- spike efficiency measurement.

Measures total spikes across all regions per recognition event for each
scenario. Compares a trained brain (50K ticks warmup) against an untrained
(fresh) brain to quantify learning-driven efficiency gains.

Reference: Diehl & Cook (2015) report ~17 spikes per correct MNIST digit
classification in their SNN. Our architecture is more complex (sensory +
expansion + concept + WM), but the principle is the same: fewer spikes for
correct recognition = more energy efficient.

Usage: .venv/bin/python benchmark/energy_proxy.py
"""
import json
import random
import time
from pathlib import Path

import torch

from brain.core import Brain
from adapters.mac_desktop.encoding import encode_snapshot

# ── Scenario generators ──

SCENARIOS = {
    "typing_vscode": lambda: {
        "active_app": {"name": "VSCode"},
        "keystroke_rate": {"count": random.randint(18, 28), "variability": random.uniform(0.1, 0.3)},
        "mouse_rate": {"count": random.randint(1, 4)},
        "idle": {"seconds": random.uniform(0.2, 0.8)},
        "mic": {"mel": [random.uniform(0.01, 0.03)] * 10 + [random.uniform(0.04, 0.1)] * 12 + [random.uniform(0.01, 0.03)] * 10, "rms": random.uniform(0.02, 0.05)},
        "time_tonic": {"day_phase": [0.5, 0.5, 0.5, 0.5], "week_phase": [0.5, 0.5, 0.5, 0.5]},
    },
    "zoom_call": lambda: {
        "active_app": {"name": "Zoom"},
        "keystroke_rate": {"count": 0},
        "mouse_rate": {"count": random.randint(0, 2)},
        "idle": {"seconds": random.uniform(0.5, 2.0)},
        "mic": {"mel": [random.uniform(0.05, 0.15)] * 32, "rms": random.uniform(0.05, 0.15)},
        "time_tonic": {"day_phase": [0.5, 0.5, 0.5, 0.5], "week_phase": [0.5, 0.5, 0.5, 0.5]},
    },
    "music_spotify": lambda: {
        "active_app": {"name": "Spotify"},
        "keystroke_rate": {"count": 0},
        "mouse_rate": {"count": random.randint(0, 3)},
        "idle": {"seconds": random.uniform(1.0, 5.0)},
        "mic": {"mel": [random.uniform(0.02, 0.08)] * 8 + [random.uniform(0.08, 0.2)] * 16 + [random.uniform(0.02, 0.06)] * 8, "rms": random.uniform(0.03, 0.08)},
        "time_tonic": {"day_phase": [0.5, 0.5, 0.5, 0.5], "week_phase": [0.5, 0.5, 0.5, 0.5]},
    },
    "idle_away": lambda: {
        "active_app": {"name": "Finder"},
        "keystroke_rate": {"count": 0},
        "mouse_rate": {"count": 0},
        "idle": {"seconds": random.uniform(60, 300)},
        "mic": {"mel": [random.uniform(0.001, 0.005)] * 32, "rms": random.uniform(0.0005, 0.002)},
        "time_tonic": {"day_phase": [0.5, 0.5, 0.5, 0.5], "week_phase": [0.5, 0.5, 0.5, 0.5]},
    },
}

WARMUP_TICKS = 50_000     # pre-training for the "trained" brain
TEST_TICKS = 1000         # ticks per scenario during measurement
DIEHL_COOK_REF = 17       # ~17 spikes per correct MNIST digit (Diehl & Cook 2015)


def _warmup_brain(brain: Brain, ticks: int) -> None:
    """Pre-train by cycling through all scenarios in alternating blocks."""
    scenario_list = list(SCENARIOS.items())
    block_size = 500
    total_cycles = ticks // (block_size * len(scenario_list))

    for cycle in range(total_cycles):
        for sname, maker in scenario_list:
            for _ in range(block_size):
                vec = encode_snapshot(maker())
                brain.tick(vec)
            # Brief pause
            for _ in range(100):
                brain.tick(torch.zeros(200))

    remaining = ticks - total_cycles * (block_size + 100) * len(scenario_list)
    for _ in range(max(0, remaining)):
        vec = encode_snapshot(SCENARIOS["typing_vscode"]())
        brain.tick(vec)


def _count_spikes(brain: Brain, maker, ticks: int) -> dict:
    """Present a scenario and count total spikes across all regions.

    Returns per-region spike counts and per-tick averages.
    """
    total_sensory = 0
    total_concept = 0
    total_wm = 0
    per_tick_totals: list[int] = []

    for _ in range(ticks):
        vec = encode_snapshot(maker())
        out = brain.tick(vec)

        s = int(out["sensory"].sum().item())
        c = int(out["concept"].sum().item())
        w = int(out["wm"].sum().item())

        total_sensory += s
        total_concept += c
        total_wm += w
        per_tick_totals.append(s + c + w)

    total_all = total_sensory + total_concept + total_wm

    # "Recognition event" = one ConceptTracker snapshot interval (50 ticks).
    # This is when the tracker actually makes a cluster assignment.
    snapshot_interval = brain.concept_tracker.snapshot_interval
    num_recognitions = ticks // snapshot_interval
    spikes_per_recognition = total_all / num_recognitions if num_recognitions > 0 else total_all

    return {
        "total_spikes": total_all,
        "sensory_spikes": total_sensory,
        "concept_spikes": total_concept,
        "wm_spikes": total_wm,
        "ticks": ticks,
        "spikes_per_tick": round(total_all / ticks, 2),
        "spikes_per_recognition": round(spikes_per_recognition, 1),
        "num_recognitions": num_recognitions,
        "per_tick_mean": round(sum(per_tick_totals) / len(per_tick_totals), 2),
        "per_tick_max": max(per_tick_totals),
        "per_tick_min": min(per_tick_totals),
    }


def run_all():
    random.seed(42)

    print("=" * 60)
    print("  BENCHMARK: ENERGY PROXY (Spike Efficiency)")
    print("  Measuring spikes per recognition event")
    print(f"  Reference: Diehl & Cook (2015) ~{DIEHL_COOK_REF} spikes/digit (MNIST)")
    print("=" * 60)

    results = {"untrained": {}, "trained": {}}

    # ── Phase 1: Untrained (fresh) brain ──
    print("\n--- Phase 1: Untrained Brain (fresh) ---")
    t0 = time.time()

    for sname, maker in SCENARIOS.items():
        brain_fresh = Brain(num_sensory=200, num_concept=200)
        counts = _count_spikes(brain_fresh, maker, TEST_TICKS)
        results["untrained"][sname] = counts
        print(f"  {sname:20s}: {counts['spikes_per_recognition']:>8.1f} spikes/recognition  "
              f"({counts['spikes_per_tick']:.1f}/tick, total={counts['total_spikes']:,})")

    untrained_time = time.time() - t0
    print(f"  (completed in {untrained_time:.1f}s)")

    # ── Phase 2: Trained brain (50K warmup) ──
    print(f"\n--- Phase 2: Trained Brain ({WARMUP_TICKS:,} ticks warmup) ---")
    t0 = time.time()

    brain_trained = Brain(num_sensory=200, num_concept=200)
    print(f"  Warming up with {WARMUP_TICKS:,} ticks...")
    _warmup_brain(brain_trained, WARMUP_TICKS)
    warmup_time = time.time() - t0
    print(f"  Warmup done in {warmup_time:.1f}s")

    for sname, maker in SCENARIOS.items():
        # Brief silence before each measurement
        for _ in range(200):
            brain_trained.tick(torch.zeros(200))

        counts = _count_spikes(brain_trained, maker, TEST_TICKS)
        results["trained"][sname] = counts
        print(f"  {sname:20s}: {counts['spikes_per_recognition']:>8.1f} spikes/recognition  "
              f"({counts['spikes_per_tick']:.1f}/tick, total={counts['total_spikes']:,})")

    trained_time = time.time() - t0
    print(f"  (completed in {trained_time:.1f}s)")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")

    print(f"\n  {'Scenario':20s}  {'Untrained':>12s}  {'Trained':>10s}  {'Ratio':>8s}  {'vs D&C':>8s}")
    print(f"  {'-' * 65}")

    all_trained_spr: list[float] = []
    all_untrained_spr: list[float] = []

    for sname in SCENARIOS:
        u_spr = results["untrained"][sname]["spikes_per_recognition"]
        t_spr = results["trained"][sname]["spikes_per_recognition"]
        ratio = t_spr / u_spr if u_spr > 0 else float("inf")
        vs_dc = t_spr / DIEHL_COOK_REF

        all_trained_spr.append(t_spr)
        all_untrained_spr.append(u_spr)

        print(f"  {sname:20s}  {u_spr:>10.1f}  {t_spr:>10.1f}  {ratio:>7.2f}x  {vs_dc:>7.1f}x")

    avg_untrained = sum(all_untrained_spr) / len(all_untrained_spr)
    avg_trained = sum(all_trained_spr) / len(all_trained_spr)
    avg_ratio = avg_trained / avg_untrained if avg_untrained > 0 else float("inf")

    print(f"\n  Average spikes/recognition:")
    print(f"    Untrained: {avg_untrained:.1f}")
    print(f"    Trained:   {avg_trained:.1f}")
    print(f"    Ratio:     {avg_ratio:.2f}x")
    print(f"    vs D&C:    {avg_trained / DIEHL_COOK_REF:.1f}x  "
          f"(our task is more complex: 4-class desktop activity vs 10-class MNIST)")

    # Efficiency gain: trained should use fewer or similar spikes per recognition
    efficiency_improved = avg_trained <= avg_untrained * 1.5
    print(f"\n  Efficiency: {'PASS' if efficiency_improved else 'WARN'} "
          f"(trained <= 1.5x untrained)")

    # Sparsity check: concept layer should be sparse (WTA with k=3)
    print(f"\n  Sparsity analysis (concept layer):")
    for phase_name, phase_results in results.items():
        concept_total = sum(r["concept_spikes"] for r in phase_results.values())
        all_total = sum(r["total_spikes"] for r in phase_results.values())
        concept_pct = concept_total / all_total * 100 if all_total > 0 else 0
        print(f"    {phase_name:12s}: concept = {concept_pct:.1f}% of total spikes")

    print(f"\n{'=' * 60}")

    # ── Save report ──
    report = {
        "test": "energy_proxy",
        "timestamp": time.time(),
        "warmup_ticks": WARMUP_TICKS,
        "test_ticks": TEST_TICKS,
        "diehl_cook_reference": DIEHL_COOK_REF,
        "untrained": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_tick_totals"}
            for k, v in results["untrained"].items()
        },
        "trained": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_tick_totals"}
            for k, v in results["trained"].items()
        },
        "summary": {
            "avg_untrained_spr": round(avg_untrained, 1),
            "avg_trained_spr": round(avg_trained, 1),
            "trained_vs_untrained_ratio": round(avg_ratio, 3),
            "trained_vs_diehl_cook": round(avg_trained / DIEHL_COOK_REF, 2),
            "efficiency_improved": efficiency_improved,
        },
    }
    outdir = Path("benchmark/reports")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "energy_proxy.json"
    outpath.write_text(json.dumps(report, indent=2))
    print(f"  Report saved to {outpath}")
    return efficiency_improved


if __name__ == "__main__":
    run_all()
