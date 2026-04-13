"""Benchmark: Adaptation Speed -- how fast does the SNN learn new patterns?

Measures ticks-to-stable-cluster for each scenario on both a fresh brain
and a pre-trained brain (100K warmup ticks). Stability is defined as the
ConceptTracker.current_cluster_id being the same for 5 consecutive checks
sampled every 100 ticks.

Target: < 5000 ticks to stable cluster.

Usage: .venv/bin/python benchmark/adaptation_speed.py
"""
import json
import random
import time
from pathlib import Path

import torch

from brain.core import Brain
from adapters.mac_desktop.encoding import encode_snapshot

# ── Scenario generators (same as discrimination_test) ──

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

CHECK_INTERVAL = 100      # check cluster every N ticks
STABILITY_WINDOW = 5      # N consecutive identical checks = stable
MAX_TICKS = 20000         # give up after this many ticks
WARMUP_TICKS = 100_000    # pre-training for the "experienced" brain
TARGET_TICKS = 5000       # pass threshold


def _measure_adaptation(brain: Brain, scenario_name: str, maker) -> dict:
    """Present a scenario and return ticks until cluster_id stabilises."""
    history: list[int] = []
    ticks_to_stable = -1

    for tick in range(1, MAX_TICKS + 1):
        vec = encode_snapshot(maker())
        brain.tick(vec)

        if tick % CHECK_INTERVAL == 0:
            cid = brain.concept_tracker.current_cluster_id
            history.append(cid)

            # Check last STABILITY_WINDOW entries
            if len(history) >= STABILITY_WINDOW:
                window = history[-STABILITY_WINDOW:]
                if len(set(window)) == 1 and window[0] >= 0:
                    ticks_to_stable = tick
                    break

    return {
        "scenario": scenario_name,
        "ticks_to_stable": ticks_to_stable,
        "final_cluster_id": brain.concept_tracker.current_cluster_id,
        "passed": 0 < ticks_to_stable <= TARGET_TICKS,
        "check_history": history[-20:],  # last 20 checks for debugging
    }


def _warmup_brain(brain: Brain, ticks: int) -> None:
    """Pre-train a brain by cycling through all scenarios."""
    scenario_list = list(SCENARIOS.items())
    ticks_per_cycle = ticks // len(scenario_list)
    for sname, maker in scenario_list:
        for _ in range(ticks_per_cycle):
            vec = encode_snapshot(maker())
            brain.tick(vec)
        # Brief pause between scenarios
        for _ in range(200):
            brain.tick(torch.zeros(200))


def run_all():
    random.seed(42)

    print("=" * 60)
    print("  BENCHMARK: ADAPTATION SPEED")
    print("  How fast does the SNN learn new patterns?")
    print(f"  Target: < {TARGET_TICKS} ticks to stable cluster")
    print("=" * 60)

    results = {"fresh_brain": {}, "pretrained_brain": {}}

    # ── Phase 1: Fresh brain ──
    print("\n--- Phase 1: Fresh Brain ---")
    t0 = time.time()

    for sname, maker in SCENARIOS.items():
        brain = Brain(num_sensory=200, num_concept=200)
        res = _measure_adaptation(brain, sname, maker)
        results["fresh_brain"][sname] = res
        status = "PASS" if res["passed"] else "FAIL"
        tts = res["ticks_to_stable"]
        tts_str = str(tts) if tts > 0 else "UNSTABLE"
        print(f"  {sname:20s}: {tts_str:>8s} ticks  [{status}]")

    fresh_time = time.time() - t0
    print(f"  (completed in {fresh_time:.1f}s)")

    # ── Phase 2: Pre-trained brain ──
    print(f"\n--- Phase 2: Pre-trained Brain ({WARMUP_TICKS:,} ticks warmup) ---")
    t0 = time.time()

    pretrained = Brain(num_sensory=200, num_concept=200)
    print(f"  Warming up brain with {WARMUP_TICKS:,} ticks...")
    _warmup_brain(pretrained, WARMUP_TICKS)
    warmup_time = time.time() - t0
    print(f"  Warmup done in {warmup_time:.1f}s")

    # Reset tracker state so we measure fresh adaptation on the pre-trained network
    for sname, maker in SCENARIOS.items():
        # Clone the pre-trained brain's weights by creating a new tracker context
        # We keep the same brain (with learned STDP weights) but inject a new scenario
        # First, present 500 ticks of silence to "clear" working memory
        for _ in range(500):
            pretrained.tick(torch.zeros(200))

        res = _measure_adaptation(pretrained, sname, maker)
        results["pretrained_brain"][sname] = res
        status = "PASS" if res["passed"] else "FAIL"
        tts = res["ticks_to_stable"]
        tts_str = str(tts) if tts > 0 else "UNSTABLE"
        print(f"  {sname:20s}: {tts_str:>8s} ticks  [{status}]")

    pretrained_time = time.time() - t0
    print(f"  (completed in {pretrained_time:.1f}s)")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")

    all_passed = True
    for phase_name, phase_results in results.items():
        phase_pass = all(r["passed"] for r in phase_results.values())
        all_passed = all_passed and phase_pass
        avg_ticks = [r["ticks_to_stable"] for r in phase_results.values() if r["ticks_to_stable"] > 0]
        avg = sum(avg_ticks) / len(avg_ticks) if avg_ticks else -1
        stable_count = sum(1 for r in phase_results.values() if r["ticks_to_stable"] > 0)
        print(f"  {phase_name:20s}: {stable_count}/{len(phase_results)} stabilised, "
              f"avg={avg:.0f} ticks  [{'PASS' if phase_pass else 'FAIL'}]")

    # Compare fresh vs pretrained speed
    print(f"\n  Speed comparison (pretrained vs fresh):")
    for sname in SCENARIOS:
        fresh_t = results["fresh_brain"][sname]["ticks_to_stable"]
        pre_t = results["pretrained_brain"][sname]["ticks_to_stable"]
        if fresh_t > 0 and pre_t > 0:
            speedup = fresh_t / pre_t
            print(f"    {sname:20s}: {fresh_t:>6d} -> {pre_t:>6d}  ({speedup:.1f}x speedup)")
        else:
            print(f"    {sname:20s}: fresh={'UNSTABLE' if fresh_t <= 0 else fresh_t}, "
                  f"pre={'UNSTABLE' if pre_t <= 0 else pre_t}")

    overall = "PASS" if all_passed else "FAIL"
    print(f"\n  OVERALL: {overall}")
    print(f"{'=' * 60}")

    # ── Save report ──
    report = {
        "test": "adaptation_speed",
        "timestamp": time.time(),
        "target_ticks": TARGET_TICKS,
        "check_interval": CHECK_INTERVAL,
        "stability_window": STABILITY_WINDOW,
        "warmup_ticks": WARMUP_TICKS,
        "fresh_brain": {k: {kk: vv for kk, vv in v.items() if kk != "check_history"}
                        for k, v in results["fresh_brain"].items()},
        "pretrained_brain": {k: {kk: vv for kk, vv in v.items() if kk != "check_history"}
                             for k, v in results["pretrained_brain"].items()},
        "overall_pass": all_passed,
    }
    outdir = Path("benchmark/reports")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "adaptation_speed.json"
    outpath.write_text(json.dumps(report, indent=2))
    print(f"  Report saved to {outpath}")
    return all_passed


if __name__ == "__main__":
    run_all()
