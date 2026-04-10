"""Benchmark Test 3: Discrimination — can the SNN distinguish 4 different activities?

Runs 4 scenarios (typing, zoom, music, idle) for 5000 ticks each,
repeated 3 times. Measures confusion matrix: does each scenario
produce a DIFFERENT dominant concept group?

Usage: .venv/bin/python benchmark/discrimination_test.py
"""
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from brain.core import Brain
from adapters.mac_desktop.encoding import encode_snapshot

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

TICKS_PER_SCENARIO = 5000
REPEATS = 3
PAUSE_TICKS = 2000


def run_all():
    random.seed(42)
    brain = Brain(num_sensory=200, num_concept=200)

    print("=" * 60)
    print("  BENCHMARK TEST 3: DISCRIMINATION")
    print("  Can the SNN distinguish 4 different activities?")
    print("=" * 60)

    # For each scenario x repeat, collect which concepts fired
    scenario_concepts: dict[str, list[set[int]]] = defaultdict(list)

    for repeat in range(REPEATS):
        print(f"\n--- Round {repeat + 1}/{REPEATS} ---")
        for sname, maker in SCENARIOS.items():
            fires = Counter()
            for tick in range(TICKS_PER_SCENARIO):
                vec = encode_snapshot(maker())
                out = brain.tick(vec)
                for cid in (out["concept"] > 0).nonzero(as_tuple=True)[0].tolist():
                    fires[cid] += 1

            # Pause between scenarios
            for tick in range(PAUSE_TICKS):
                brain.tick(torch.zeros(200))

            top3 = set(cid for cid, _ in fires.most_common(3))
            dominant = fires.most_common(1)[0] if fires else (-1, 0)
            print(f"  {sname:20s}: dominant=C{dominant[0]} top3={top3}")
            scenario_concepts[sname].append(top3)

    # Confusion analysis
    print(f"\n{'=' * 60}")
    print("  CONFUSION ANALYSIS")
    print(f"{'=' * 60}")

    # For each pair of scenarios: how much do their top-3 groups overlap?
    scenario_names = list(SCENARIOS.keys())
    confusion = {}
    for i, s1 in enumerate(scenario_names):
        for j, s2 in enumerate(scenario_names):
            if i >= j:
                continue
            # Average overlap across repeats
            overlaps = []
            for r in range(REPEATS):
                overlap = len(scenario_concepts[s1][r] & scenario_concepts[s2][r])
                overlaps.append(overlap / 3.0 * 100)  # as % of 3
            avg_overlap = sum(overlaps) / len(overlaps)
            confusion[(s1, s2)] = avg_overlap
            status = "OK" if avg_overlap <= 33 else "WARN" if avg_overlap <= 67 else "FAIL"
            print(f"  {s1:20s} vs {s2:20s}: {avg_overlap:.0f}% overlap [{status}]")

    # Overall: how many unique concept groups?
    all_groups = []
    for sname in scenario_names:
        merged = set()
        for top3 in scenario_concepts[sname]:
            merged.update(top3)
        all_groups.append((sname, merged))
        print(f"\n  {sname}: merged concepts = {merged}")

    # Count distinct groups (overlap <= 1 between any pair)
    distinct = 0
    for i, (s1, g1) in enumerate(all_groups):
        is_distinct = True
        for j, (s2, g2) in enumerate(all_groups):
            if i != j and len(g1 & g2) > len(g1) * 0.5:
                is_distinct = False
        if is_distinct:
            distinct += 1

    overall = distinct >= 3  # at least 3 of 4 scenarios distinguishable
    avg_confusion = sum(confusion.values()) / max(1, len(confusion))

    print(f"\n{'=' * 60}")
    print(f"  RESULT: {distinct}/4 distinct groups, avg confusion={avg_confusion:.0f}%")
    print(f"  {'PASS' if overall else 'FAIL'} (need >=3 distinct)")
    print(f"{'=' * 60}")

    report = {
        "test": "discrimination",
        "timestamp": time.time(),
        "scenario_concepts": {k: [list(s) for s in v] for k, v in scenario_concepts.items()},
        "confusion": {f"{k[0]}_vs_{k[1]}": v for k, v in confusion.items()},
        "distinct_groups": distinct,
        "overall_pass": overall,
    }
    Path("benchmark/reports/discrimination_test.json").write_text(json.dumps(report, indent=2))
    print(f"  Report saved to benchmark/reports/discrimination_test.json")
    return overall


if __name__ == "__main__":
    run_all()
