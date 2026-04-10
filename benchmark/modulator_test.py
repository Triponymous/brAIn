"""Benchmark Test 4: Modulator Reactivity — do modulators respond correctly?

Tests:
1. Sudden loud sound → NE spikes within 2 seconds
2. Steady typing → 5HT rises over 30 seconds
3. Silence after activity → all modulators drop
4. Novel pattern after familiar → DA spikes

Usage: .venv/bin/python benchmark/modulator_test.py
"""
import json
import random
import time
from pathlib import Path

import torch

from brain.core import Brain
from adapters.mac_desktop.encoding import encode_snapshot


def make_silence():
    return {
        "active_app": {"name": "Finder"},
        "keystroke_rate": {"count": 0},
        "mouse_rate": {"count": 0},
        "idle": {"seconds": random.uniform(30, 60)},
        "mic": {"mel": [0.001] * 32, "rms": 0.001},
        "time_tonic": {"day_phase": [0.5, 0.5, 0.5, 0.5], "week_phase": [0.5, 0.5, 0.5, 0.5]},
    }


def make_typing():
    return {
        "active_app": {"name": "VSCode"},
        "keystroke_rate": {"count": random.randint(18, 28), "variability": random.uniform(0.1, 0.3)},
        "mouse_rate": {"count": random.randint(1, 4)},
        "idle": {"seconds": random.uniform(0.2, 0.8)},
        "mic": {"mel": [random.uniform(0.01, 0.03)] * 10 + [random.uniform(0.04, 0.1)] * 12 + [random.uniform(0.01, 0.03)] * 10, "rms": random.uniform(0.02, 0.05)},
        "time_tonic": {"day_phase": [0.5, 0.5, 0.5, 0.5], "week_phase": [0.5, 0.5, 0.5, 0.5]},
    }


def make_loud():
    return {
        "active_app": {"name": "Finder"},
        "keystroke_rate": {"count": 0},
        "mouse_rate": {"count": 0},
        "idle": {"seconds": 0.5},
        "mic": {"mel": [random.uniform(0.3, 0.5)] * 32, "rms": random.uniform(0.2, 0.4)},
        "time_tonic": {"day_phase": [0.5, 0.5, 0.5, 0.5], "week_phase": [0.5, 0.5, 0.5, 0.5]},
    }


def run_all():
    random.seed(42)
    brain = Brain(num_sensory=200, num_concept=200)
    results = {}

    print("=" * 60)
    print("  BENCHMARK TEST 4: MODULATOR REACTIVITY")
    print("=" * 60)

    # Warmup: 1000 ticks of silence to establish baseline
    for _ in range(1000):
        brain.tick(encode_snapshot(make_silence()))
    baseline = brain.modulators.snapshot()
    print(f"\nBaseline after 1000 ticks silence:")
    for k, v in baseline.items():
        print(f"  {k}: {v:.4f}")

    # Test 1: Sudden loud sound → NE spike
    print(f"\n--- Test 1: Sudden loud sound → NE should spike ---")
    ne_before = brain.modulators.level("NE")
    ne_max = ne_before
    for tick in range(200):  # 2 seconds
        brain.tick(encode_snapshot(make_loud()))
        ne_now = brain.modulators.level("NE")
        ne_max = max(ne_max, ne_now)
    ne_delta = ne_max - ne_before
    test1_pass = ne_delta > 0.02
    print(f"  NE before: {ne_before:.4f}, NE max: {ne_max:.4f}, delta: {ne_delta:.4f}")
    print(f"  {'PASS' if test1_pass else 'FAIL'} (need delta > 0.02)")
    results["loud_ne_spike"] = {"pass": test1_pass, "ne_before": ne_before, "ne_max": ne_max, "delta": ne_delta}

    # Reset to silence
    for _ in range(2000):
        brain.tick(encode_snapshot(make_silence()))

    # Test 2: Steady typing → 5HT rises
    print(f"\n--- Test 2: 30 seconds steady typing → 5HT should rise ---")
    sht_before = brain.modulators.level("5HT")
    for tick in range(3000):  # 30 seconds
        brain.tick(encode_snapshot(make_typing()))
    sht_after = brain.modulators.level("5HT")
    sht_delta = sht_after - sht_before
    test2_pass = sht_delta > 0.001
    print(f"  5HT before: {sht_before:.4f}, after: {sht_after:.4f}, delta: {sht_delta:.4f}")
    print(f"  {'PASS' if test2_pass else 'FAIL'} (need delta > 0.001)")
    results["typing_sht_rise"] = {"pass": test2_pass, "before": sht_before, "after": sht_after, "delta": sht_delta}

    # Test 3: Silence after activity → all drop
    print(f"\n--- Test 3: Silence after activity → modulators should drop ---")
    mods_before = brain.modulators.snapshot()
    for _ in range(3000):  # 30 seconds silence
        brain.tick(encode_snapshot(make_silence()))
    mods_after = brain.modulators.snapshot()
    all_dropped = all(mods_after[k] <= mods_before[k] + 0.001 for k in ["DA", "NE", "ACh"])
    print(f"  Before: DA={mods_before['DA']:.4f} NE={mods_before['NE']:.4f} ACh={mods_before['ACh']:.4f}")
    print(f"  After:  DA={mods_after['DA']:.4f} NE={mods_after['NE']:.4f} ACh={mods_after['ACh']:.4f}")
    print(f"  {'PASS' if all_dropped else 'FAIL'}")
    results["silence_drop"] = {"pass": all_dropped, "before": mods_before, "after": mods_after}

    # Test 4: Novel pattern after familiar → DA spike
    print(f"\n--- Test 4: Novel pattern after familiar → DA should spike ---")
    # 20 seconds typing (familiar)
    for _ in range(2000):
        brain.tick(encode_snapshot(make_typing()))
    da_before = brain.modulators.level("DA")
    # Sudden switch to loud (novel)
    da_max = da_before
    for tick in range(200):
        brain.tick(encode_snapshot(make_loud()))
        da_max = max(da_max, brain.modulators.level("DA"))
    da_delta = da_max - da_before
    test4_pass = da_delta > 0.01
    print(f"  DA before: {da_before:.4f}, DA max: {da_max:.4f}, delta: {da_delta:.4f}")
    print(f"  {'PASS' if test4_pass else 'FAIL'} (need delta > 0.01)")
    results["novel_da_spike"] = {"pass": test4_pass, "before": da_before, "max": da_max, "delta": da_delta}

    # Summary
    passes = sum(1 for r in results.values() if r["pass"])
    overall = passes >= 3  # 3 of 4 must pass

    print(f"\n{'=' * 60}")
    print(f"  RESULT: {passes}/4 tests passed")
    print(f"  {'PASS' if overall else 'FAIL'}")
    print(f"{'=' * 60}")

    report = {"test": "modulator", "timestamp": time.time(), "results": results, "overall_pass": overall}
    Path("benchmark/reports/modulator_test.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"  Report saved to benchmark/reports/modulator_test.json")
    return overall


if __name__ == "__main__":
    run_all()
