"""Benchmark Test 1: Long-term Stability — do weights survive 500K ticks?

Runs 500K ticks (~1.4 hours at 100Hz) with varied activity patterns.
Checks that weights don't collapse to 0 or explode to 1, and that
concept neurons keep firing throughout.

Usage: .venv/bin/python benchmark/stability_test.py
"""
import json
import random
import time
from pathlib import Path

import torch

from brain.core import Brain
from adapters.mac_desktop.encoding import encode_snapshot


PATTERNS = {
    "typing": lambda: {
        "active_app": {"name": "VSCode"},
        "keystroke_rate": {"count": random.randint(15, 30), "variability": random.uniform(0.1, 0.4)},
        "mouse_rate": {"count": random.randint(0, 5)},
        "idle": {"seconds": random.uniform(0.2, 1.0)},
        "mic": {"mel": [random.uniform(0.01, 0.03)] * 10 + [random.uniform(0.04, 0.1)] * 12 + [random.uniform(0.01, 0.03)] * 10, "rms": random.uniform(0.02, 0.05)},
        "time_tonic": {"day_phase": [0.5, 0.5, 0.5, 0.5], "week_phase": [0.5, 0.5, 0.5, 0.5]},
    },
    "browsing": lambda: {
        "active_app": {"name": "Chrome"},
        "keystroke_rate": {"count": random.randint(0, 3)},
        "mouse_rate": {"count": random.randint(25, 60), "variability": random.uniform(0.2, 0.6)},
        "idle": {"seconds": random.uniform(0.2, 1.0)},
        "mic": {"mel": [random.uniform(0.001, 0.005)] * 32, "rms": random.uniform(0.001, 0.005)},
        "time_tonic": {"day_phase": [0.5, 0.5, 0.5, 0.5], "week_phase": [0.5, 0.5, 0.5, 0.5]},
    },
    "idle": lambda: {
        "active_app": {"name": "Finder"},
        "keystroke_rate": {"count": 0},
        "mouse_rate": {"count": 0},
        "idle": {"seconds": random.uniform(30, 180)},
        "mic": {"mel": [random.uniform(0.001, 0.005)] * 32, "rms": random.uniform(0.0005, 0.002)},
        "time_tonic": {"day_phase": [0.5, 0.5, 0.5, 0.5], "week_phase": [0.5, 0.5, 0.5, 0.5]},
    },
}

TOTAL_TICKS = 500000
PHASE_TICKS = 5000  # switch pattern every 5000 ticks


def run_all():
    random.seed(42)
    brain = Brain(num_sensory=200, num_concept=200)
    pattern_names = list(PATTERNS.keys())

    print("=" * 60)
    print("  BENCHMARK TEST 1: LONG-TERM STABILITY (500K ticks)")
    print("=" * 60)

    checkpoints = []
    t0 = time.time()

    for tick in range(TOTAL_TICKS):
        pidx = (tick // PHASE_TICKS) % len(pattern_names)
        vec = encode_snapshot(PATTERNS[pattern_names[pidx]]())
        out = brain.tick(vec)

        if tick % 50000 == 0:
            elapsed = time.time() - t0
            sc = brain.synapses["sensory_concept"]
            w = sc.weights
            concept_firing = int(out["concept"].sum().item())
            mods = brain.modulators.snapshot()
            checkpoint = {
                "tick": tick,
                "elapsed": round(elapsed, 1),
                "w_mean": round(float(w.mean()), 4),
                "w_std": round(float(w.std()), 4),
                "w_min": round(float(w.min()), 4),
                "w_max": round(float(w.max()), 4),
                "dead_rows": int((w.mean(dim=1) < 0.02).sum().item()),
                "concept_firing": concept_firing,
                "DA": round(mods["DA"], 4),
                "NE": round(mods["NE"], 4),
            }
            checkpoints.append(checkpoint)
            print(f"  {tick:7d}/{TOTAL_TICKS} ({elapsed:.0f}s) w_mean={checkpoint['w_mean']} "
                  f"w_std={checkpoint['w_std']} dead={checkpoint['dead_rows']}/200 "
                  f"concept={concept_firing}")

    # Final analysis
    print(f"\n{'=' * 60}")
    print("  ANALYSIS")

    first = checkpoints[0]
    last = checkpoints[-1]

    # Test 1: weights didn't collapse
    w_alive = last["w_mean"] > 0.05
    print(f"  Weight mean: {first['w_mean']} → {last['w_mean']} ({'PASS' if w_alive else 'FAIL — collapsed'})")

    # Test 2: weights didn't explode
    w_sane = last["w_max"] <= 1.0
    print(f"  Weight max: {last['w_max']} ({'PASS' if w_sane else 'FAIL — exploded'})")

    # Test 3: concepts still firing at the end
    c_alive = last["concept_firing"] > 0
    print(f"  Concepts firing at end: {last['concept_firing']} ({'PASS' if c_alive else 'FAIL — dead'})")

    # Test 4: not too many dead rows
    dead_ok = last["dead_rows"] < 100  # less than half
    print(f"  Dead rows: {last['dead_rows']}/200 ({'PASS' if dead_ok else 'FAIL — too many dead'})")

    # Test 5: modulators in range
    mods_ok = all(0 <= brain.modulators.level(k) <= 0.8 for k in ["DA", "NE", "ACh", "5HT"])
    print(f"  Modulators in range: {'PASS' if mods_ok else 'FAIL'}")

    overall = w_alive and w_sane and c_alive and dead_ok and mods_ok
    print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'}")
    print(f"{'=' * 60}")

    report = {"test": "stability", "timestamp": time.time(), "checkpoints": checkpoints, "overall_pass": overall}
    Path("benchmark/reports/stability_test.json").write_text(json.dumps(report, indent=2))
    print(f"  Report saved to benchmark/reports/stability_test.json")
    return overall


if __name__ == "__main__":
    run_all()
