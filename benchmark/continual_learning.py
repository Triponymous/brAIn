"""Benchmark: Continual Learning — does the SNN suffer catastrophic forgetting?

Trains 4 sequential activity patterns (typing, zoom, music, idle) and
re-tests earlier patterns after each new one. Measures whether ConceptTracker
still assigns the correct cluster_id to previously learned patterns.

A forgetting curve table is printed at the end. The test PASSES if
pattern A (typing_vscode) retains >70% accuracy after all 4 phases.

Usage: .venv/bin/python benchmark/continual_learning.py
"""
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch

from brain.core import Brain
from adapters.mac_desktop.encoding import encode_snapshot

# ---------------------------------------------------------------------------
# Scenario generators (identical to discrimination_test.py)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAIN_TICKS = 50_000       # ticks per training phase
RETEST_TICKS = 5_000       # ticks when re-testing a previously learned pattern
PAUSE_TICKS = 500          # idle pause between phases
CLUSTER_SAMPLE_WINDOW = 2_000  # last N ticks to sample for dominant cluster


def _idle_pause(brain: Brain, ticks: int) -> None:
    """Feed zeros to let transients settle between phases."""
    zeros = torch.zeros(200)
    for _ in range(ticks):
        brain.tick(zeros)


def _run_pattern(brain: Brain, scenario_name: str, ticks: int) -> int:
    """Run a scenario for N ticks and return the dominant cluster_id
    observed in the final CLUSTER_SAMPLE_WINDOW ticks."""
    maker = SCENARIOS[scenario_name]
    cluster_counts: Counter = Counter()
    for t in range(ticks):
        vec = encode_snapshot(maker())
        brain.tick(vec)
        # Only sample cluster assignment in the tail window
        if t >= ticks - CLUSTER_SAMPLE_WINDOW:
            cid = brain.concept_tracker.current_cluster_id
            if cid >= 0:
                cluster_counts[cid] += 1

    if not cluster_counts:
        return -1
    dominant_id, _ = cluster_counts.most_common(1)[0]
    return dominant_id


def _retest_accuracy(brain: Brain, scenario_name: str, expected_cluster: int, ticks: int) -> float:
    """Run a scenario for N ticks and measure what fraction of cluster
    assignments in the tail window match the expected cluster_id."""
    maker = SCENARIOS[scenario_name]
    match_count = 0
    total_count = 0
    for t in range(ticks):
        vec = encode_snapshot(maker())
        brain.tick(vec)
        if t >= ticks - CLUSTER_SAMPLE_WINDOW:
            cid = brain.concept_tracker.current_cluster_id
            if cid >= 0:
                total_count += 1
                if cid == expected_cluster:
                    match_count += 1

    if total_count == 0:
        return 0.0
    return match_count / total_count


def run_all() -> bool:
    random.seed(42)
    torch.manual_seed(42)
    brain = Brain(num_sensory=200, num_concept=200)

    print("=" * 70)
    print("  BENCHMARK: CONTINUAL LEARNING (Catastrophic Forgetting Test)")
    print("=" * 70)

    pattern_order = ["typing_vscode", "zoom_call", "music_spotify", "idle_away"]
    labels = ["A", "B", "C", "D"]
    cluster_ids: dict[str, int] = {}

    # Forgetting curve: rows = retest moments, cols = patterns
    # Each entry is the accuracy of pattern X after training phase Y
    # Format: forgetting_curve[retest_label][(pattern_label, accuracy)]
    forgetting_curve: list[dict] = []

    t_start = time.time()

    for phase_idx, scenario_name in enumerate(pattern_order):
        phase_label = labels[phase_idx]
        phase_num = phase_idx + 1

        # --- Train new pattern ---
        print(f"\n{'─' * 70}")
        print(f"  PHASE {phase_num}: Train pattern {phase_label} ({scenario_name}) — {TRAIN_TICKS} ticks")
        print(f"{'─' * 70}")

        cluster_id = _run_pattern(brain, scenario_name, TRAIN_TICKS)
        cluster_ids[phase_label] = cluster_id
        print(f"  --> cluster_id_{phase_label} = {cluster_id}")

        # Check distinctness from previous clusters
        for prev_label, prev_cid in cluster_ids.items():
            if prev_label != phase_label and prev_cid == cluster_id:
                print(f"  [WARN] cluster_id_{phase_label} == cluster_id_{prev_label} ({cluster_id})")

        # --- Idle pause ---
        _idle_pause(brain, PAUSE_TICKS)

        # --- Re-test all previously trained patterns ---
        retest_entry = {"after_phase": phase_num, "after_pattern": scenario_name}
        if phase_idx > 0:
            print(f"\n  Re-testing after Phase {phase_num}:")
            for prev_idx in range(phase_idx):
                prev_label = labels[prev_idx]
                prev_scenario = pattern_order[prev_idx]
                expected_cid = cluster_ids[prev_label]

                acc = _retest_accuracy(brain, prev_scenario, expected_cid, RETEST_TICKS)
                key = f"accuracy_{prev_label}"
                retest_entry[key] = round(acc * 100, 1)
                status = "OK" if acc >= 0.70 else "WARN" if acc >= 0.50 else "FAIL"
                print(f"    Pattern {prev_label} ({prev_scenario}): "
                      f"{acc * 100:.1f}% [{status}]  (expected cluster {expected_cid})")

                # Idle pause between retests
                _idle_pause(brain, PAUSE_TICKS)

        forgetting_curve.append(retest_entry)

    elapsed = time.time() - t_start

    # -----------------------------------------------------------------------
    # Forgetting curve table
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  FORGETTING CURVE TABLE")
    print(f"{'=' * 70}")

    # Header
    header = f"  {'After Phase':<20}"
    for lbl in labels:
        header += f"{'Acc ' + lbl:>12}"
    print(header)
    print("  " + "-" * (20 + 12 * len(labels)))

    for entry in forgetting_curve:
        phase_num = entry["after_phase"]
        trained = labels[phase_num - 1]
        row = f"  Phase {phase_num} ({trained} trained)  "
        for lbl in labels:
            key = f"accuracy_{lbl}"
            if key in entry:
                val = entry[key]
                row += f"{val:>10.1f}% "
            elif lbl == trained:
                row += f"{'(trained)':>12}"
            else:
                row += f"{'---':>12}"
        print(row)

    # -----------------------------------------------------------------------
    # Final cluster summary
    # -----------------------------------------------------------------------
    print(f"\n  Cluster assignments: ", end="")
    for lbl in labels:
        print(f"{lbl}={cluster_ids[lbl]}  ", end="")
    print()

    distinct_clusters = len(set(cluster_ids.values()))
    print(f"  Distinct clusters: {distinct_clusters}/{len(labels)}")

    # -----------------------------------------------------------------------
    # PASS / FAIL criterion
    # -----------------------------------------------------------------------
    # Find accuracy_A after the LAST phase (Phase 4)
    last_entry = forgetting_curve[-1]
    accuracy_a_final = last_entry.get("accuracy_A", 0.0)

    # Also gather all final retests
    final_accuracies = {}
    for lbl in labels[:-1]:  # A, B, C (D is the last trained, not retested)
        key = f"accuracy_{lbl}"
        final_accuracies[lbl] = last_entry.get(key, 0.0)

    overall_pass = accuracy_a_final > 70.0

    print(f"\n{'=' * 70}")
    print(f"  RESULT: accuracy_A after all phases = {accuracy_a_final:.1f}%")
    print(f"  {'PASS' if overall_pass else 'FAIL'} (threshold: >70%)")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # -----------------------------------------------------------------------
    # Save JSON report
    # -----------------------------------------------------------------------
    report = {
        "test": "continual_learning",
        "timestamp": time.time(),
        "config": {
            "train_ticks": TRAIN_TICKS,
            "retest_ticks": RETEST_TICKS,
            "pause_ticks": PAUSE_TICKS,
            "cluster_sample_window": CLUSTER_SAMPLE_WINDOW,
        },
        "pattern_order": pattern_order,
        "cluster_ids": cluster_ids,
        "distinct_clusters": distinct_clusters,
        "forgetting_curve": forgetting_curve,
        "final_accuracies": final_accuracies,
        "accuracy_A_final": accuracy_a_final,
        "overall_pass": overall_pass,
        "elapsed_seconds": round(elapsed, 1),
    }

    out_path = Path("benchmark/reports/continual_learning.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"  Report saved to {out_path}")

    return overall_pass


if __name__ == "__main__":
    run_all()
