"""Benchmark: Confusion Matrix -- publication-grade discrimination analysis.

Trains the SNN on 4 scenarios (20K ticks warmup), then tests each scenario
for 5K ticks x 3 repeats. Builds a true confusion matrix mapping actual
scenarios to assigned ConceptTracker clusters. Computes per-class Precision,
Recall, F1, and overall accuracy.

Optionally saves a matplotlib heatmap PNG.

Usage: .venv/bin/python benchmark/confusion_matrix.py
"""
import json
import random
import time
from collections import Counter, defaultdict
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

WARMUP_TICKS = 20_000     # alternating scenarios for training
TEST_TICKS = 5000         # ticks per scenario per test repeat
REPEATS = 3               # repeats per scenario in test phase
PAUSE_TICKS = 500         # silence between scenario blocks
SAMPLE_INTERVAL = 50      # how often to record cluster assignment during test


def _warmup(brain: Brain) -> None:
    """Train by cycling through all scenarios in alternating blocks."""
    scenario_list = list(SCENARIOS.items())
    block_size = 500  # ticks per scenario per cycle
    total_cycles = WARMUP_TICKS // (block_size * len(scenario_list))

    for cycle in range(total_cycles):
        for sname, maker in scenario_list:
            for _ in range(block_size):
                vec = encode_snapshot(maker())
                brain.tick(vec)
            # Brief pause between scenarios within a cycle
            for _ in range(100):
                brain.tick(torch.zeros(200))

    remaining = WARMUP_TICKS - total_cycles * (block_size + 100) * len(scenario_list)
    for _ in range(max(0, remaining)):
        vec = encode_snapshot(SCENARIOS["typing_vscode"]())
        brain.tick(vec)


def _test_scenario(brain: Brain, maker, ticks: int) -> list[int]:
    """Present a scenario for `ticks` and record cluster assignments at intervals."""
    assignments: list[int] = []
    for t in range(1, ticks + 1):
        vec = encode_snapshot(maker())
        brain.tick(vec)
        if t % SAMPLE_INTERVAL == 0:
            assignments.append(brain.concept_tracker.current_cluster_id)
    return assignments


def _majority_cluster(assignments: list[int]) -> int:
    """Return the most frequent cluster ID in the assignment list."""
    if not assignments:
        return -1
    counter = Counter(assignments)
    return counter.most_common(1)[0][0]


def _build_confusion_matrix(test_results: dict[str, list[list[int]]]) -> dict:
    """Build NxN confusion matrix from test results.

    Rows = actual scenarios, Cols = assigned cluster IDs.
    Each cell = number of samples where scenario X was assigned cluster Y.
    """
    scenario_names = list(test_results.keys())

    # Determine the dominant cluster for each scenario (ground-truth mapping).
    # This is needed because cluster IDs are arbitrary -- we need to find
    # which cluster the brain SHOULD assign to each scenario.
    scenario_dominant: dict[str, int] = {}
    all_clusters: set[int] = set()
    for sname, repeats in test_results.items():
        all_assignments = []
        for assignments in repeats:
            all_assignments.extend(assignments)
        dominant = _majority_cluster(all_assignments)
        scenario_dominant[sname] = dominant
        all_clusters.update(all_assignments)

    cluster_ids = sorted(all_clusters - {-1})

    # Build raw count matrix: rows=scenarios, cols=cluster_ids
    # Each entry = how many samples from scenario_i were assigned to cluster_j
    raw_matrix: dict[str, dict[int, int]] = {}
    for sname, repeats in test_results.items():
        counts: Counter = Counter()
        for assignments in repeats:
            counts.update(assignments)
        raw_matrix[sname] = dict(counts)

    # Per-class metrics using the dominant cluster as the "predicted positive"
    per_class: dict[str, dict] = {}
    total_correct = 0
    total_samples = 0

    for sname in scenario_names:
        dominant_cid = scenario_dominant[sname]
        all_assignments = []
        for assignments in test_results[sname]:
            all_assignments.extend(assignments)

        total = len(all_assignments)
        if total == 0:
            per_class[sname] = {"precision": 0, "recall": 0, "f1": 0, "dominant_cluster": dominant_cid, "samples": 0}
            continue

        # True positives: samples from this scenario assigned to its dominant cluster
        tp = sum(1 for a in all_assignments if a == dominant_cid)

        # False positives: samples from OTHER scenarios assigned to this scenario's cluster
        fp = 0
        for other_name, other_repeats in test_results.items():
            if other_name == sname:
                continue
            for assignments in other_repeats:
                fp += sum(1 for a in assignments if a == dominant_cid)

        # False negatives: samples from this scenario NOT assigned to its dominant cluster
        fn = total - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[sname] = {
            "dominant_cluster": dominant_cid,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "samples": total,
        }

        total_correct += tp
        total_samples += total

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    # Format matrix as scenario x cluster_id for readability
    matrix_formatted: dict[str, dict[str, int]] = {}
    for sname in scenario_names:
        row = {}
        for cid in cluster_ids:
            row[f"cluster_{cid}"] = raw_matrix[sname].get(cid, 0)
        matrix_formatted[sname] = row

    return {
        "scenario_names": scenario_names,
        "cluster_ids": cluster_ids,
        "raw_matrix": matrix_formatted,
        "scenario_dominant_cluster": {k: v for k, v in scenario_dominant.items()},
        "per_class": per_class,
        "overall_accuracy": round(overall_accuracy, 4),
    }


def _save_heatmap(confusion: dict, outpath: Path) -> bool:
    """Try to save a matplotlib heatmap. Returns True on success."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available -- skipping heatmap PNG")
        return False

    scenario_names = confusion["scenario_names"]
    cluster_ids = confusion["cluster_ids"]

    # Build numeric matrix
    n_scenarios = len(scenario_names)
    n_clusters = len(cluster_ids)
    matrix = np.zeros((n_scenarios, n_clusters))

    for i, sname in enumerate(scenario_names):
        row = confusion["raw_matrix"][sname]
        row_total = sum(row.values())
        for j, cid in enumerate(cluster_ids):
            count = row.get(f"cluster_{cid}", 0)
            matrix[i, j] = count / row_total if row_total > 0 else 0

    fig, ax = plt.subplots(figsize=(max(8, n_clusters * 1.2), max(5, n_scenarios * 1.0)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels([f"C{cid}" for cid in cluster_ids], rotation=45, ha="right")
    ax.set_yticks(range(n_scenarios))
    ax.set_yticklabels(scenario_names)
    ax.set_xlabel("Assigned Cluster")
    ax.set_ylabel("Actual Scenario")
    ax.set_title("SNN Confusion Matrix (normalized)")

    # Annotate cells
    for i in range(n_scenarios):
        for j in range(n_clusters):
            val = matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    # Mark dominant cluster for each scenario
    for i, sname in enumerate(scenario_names):
        dom_cid = confusion["scenario_dominant_cluster"][sname]
        if dom_cid in cluster_ids:
            j = cluster_ids.index(dom_cid)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False, edgecolor="blue", linewidth=2.5))

    # Per-class F1 annotation on the right
    f1_texts = []
    for sname in scenario_names:
        f1 = confusion["per_class"][sname]["f1"]
        f1_texts.append(f"F1={f1:.2f}")
    ax2 = ax.secondary_yaxis("right")
    ax2.set_yticks(range(n_scenarios))
    ax2.set_yticklabels(f1_texts, fontsize=9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.12)
    cbar.set_label("Proportion")

    # Overall accuracy in title
    acc = confusion["overall_accuracy"]
    ax.set_title(f"SNN Confusion Matrix (normalized)  |  Overall Accuracy: {acc:.1%}")

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def run_all():
    random.seed(42)

    print("=" * 60)
    print("  BENCHMARK: CONFUSION MATRIX")
    print("  Publication-grade discrimination analysis")
    print("=" * 60)

    brain = Brain(num_sensory=200, num_concept=200)

    # ── Phase 1: Training ──
    print(f"\n--- Phase 1: Training ({WARMUP_TICKS:,} ticks) ---")
    t0 = time.time()
    _warmup(brain)
    train_time = time.time() - t0
    print(f"  Training complete in {train_time:.1f}s")
    print(f"  ConceptTracker clusters: {len(brain.concept_tracker._clusters)}")

    # ── Phase 2: Testing ──
    print(f"\n--- Phase 2: Testing ({TEST_TICKS} ticks x {REPEATS} repeats per scenario) ---")
    t0 = time.time()

    test_results: dict[str, list[list[int]]] = defaultdict(list)

    for repeat in range(REPEATS):
        print(f"\n  Round {repeat + 1}/{REPEATS}")
        for sname, maker in SCENARIOS.items():
            # Silence gap before each test block
            for _ in range(PAUSE_TICKS):
                brain.tick(torch.zeros(200))

            assignments = _test_scenario(brain, maker, TEST_TICKS)
            test_results[sname].append(assignments)

            dominant = _majority_cluster(assignments)
            unique = len(set(assignments))
            print(f"    {sname:20s}: dominant=C{dominant}, unique_clusters={unique}, "
                  f"samples={len(assignments)}")

    test_time = time.time() - t0
    print(f"\n  Testing complete in {test_time:.1f}s")

    # ── Phase 3: Confusion matrix analysis ──
    print(f"\n{'=' * 60}")
    print("  CONFUSION MATRIX ANALYSIS")
    print(f"{'=' * 60}")

    confusion = _build_confusion_matrix(dict(test_results))

    # Print per-class metrics
    print(f"\n  {'Scenario':20s}  {'Cluster':>8s}  {'Precision':>10s}  {'Recall':>8s}  {'F1':>6s}")
    print(f"  {'-' * 60}")
    for sname in confusion["scenario_names"]:
        pc = confusion["per_class"][sname]
        print(f"  {sname:20s}  C{pc['dominant_cluster']:>6d}  {pc['precision']:>10.3f}  "
              f"{pc['recall']:>8.3f}  {pc['f1']:>6.3f}")

    print(f"\n  Overall Accuracy: {confusion['overall_accuracy']:.1%}")

    # Check if clusters are distinct (each scenario maps to a different cluster)
    dominant_clusters = list(confusion["scenario_dominant_cluster"].values())
    unique_dominants = len(set(dominant_clusters) - {-1})
    distinct = unique_dominants == len(SCENARIOS)
    print(f"  Distinct clusters: {unique_dominants}/{len(SCENARIOS)} "
          f"[{'PASS' if distinct else 'WARN: some scenarios share a cluster'}]")

    # Macro-average F1
    f1_values = [confusion["per_class"][s]["f1"] for s in confusion["scenario_names"]]
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
    print(f"  Macro-average F1: {macro_f1:.3f}")

    overall_pass = confusion["overall_accuracy"] >= 0.5 and macro_f1 >= 0.4
    print(f"\n  OVERALL: {'PASS' if overall_pass else 'FAIL'} "
          f"(accuracy >= 50% and macro-F1 >= 0.4)")
    print(f"{'=' * 60}")

    # ── Save outputs ──
    outdir = Path("benchmark/reports")
    outdir.mkdir(parents=True, exist_ok=True)

    # JSON report
    report = {
        "test": "confusion_matrix",
        "timestamp": time.time(),
        "warmup_ticks": WARMUP_TICKS,
        "test_ticks_per_scenario": TEST_TICKS,
        "repeats": REPEATS,
        "sample_interval": SAMPLE_INTERVAL,
        "confusion_matrix": confusion["raw_matrix"],
        "scenario_dominant_cluster": confusion["scenario_dominant_cluster"],
        "cluster_ids": confusion["cluster_ids"],
        "per_class": confusion["per_class"],
        "overall_accuracy": confusion["overall_accuracy"],
        "macro_f1": round(macro_f1, 4),
        "distinct_clusters": unique_dominants,
        "overall_pass": overall_pass,
    }
    json_path = outdir / "confusion_matrix.json"
    json_path.write_text(json.dumps(report, indent=2))
    print(f"  Report saved to {json_path}")

    # Heatmap PNG
    png_path = outdir / "confusion_matrix.png"
    if _save_heatmap(confusion, png_path):
        print(f"  Heatmap saved to {png_path}")

    return overall_pass


if __name__ == "__main__":
    run_all()
