"""University-grade SNN Benchmark: 5 Real-World Scenarios.

Tests the 5 core promises of the brAIntest system:
1. Stress Detection (sustained hektisches Tippen → NE rises)
2. Flow Protection (deep focus → low NE, high ACh, high 5HT)
3. Rhythm Learning (different clusters for different activities)
4. Memory Persistence (labels survive restart)
5. Uniqueness (different seeds → different brains)

Each test has PASS/FAIL criteria based on measurable metrics.
Results are logged to benchmark/reports/ for longitudinal tracking.

Usage: .venv/bin/python benchmark/scenario_benchmark.py
"""
import json
import random
import time
from pathlib import Path
from collections import Counter, defaultdict

import torch

from brain.core import Brain
from adapters.mac_desktop.encoding import encode_snapshot
from benchmark.scenario_simulator import (
    calm_coding, stressed_coding, focus_flow, zoom_call,
    browsing, idle_away, music_listening, get_workday_pattern,
)


def test_stress_detection() -> dict:
    """Szenario 1: Sustained stress → NE rises, 5HT drops."""
    print("\n" + "=" * 60)
    print("  SZENARIO 1: STRESS-ERKENNUNG")
    print("  5 Minuten ruhig coden, dann 5 Minuten Stress")
    print("=" * 60)

    random.seed(42)
    brain = Brain(num_sensory=200, num_concept=200)

    # Phase 1: 5 min calm coding (30000 ticks)
    for tick in range(30000):
        brain.tick(encode_snapshot(calm_coding()))

    mods_calm = brain.modulators.snapshot()
    print(f"  Nach 5min ruhig: NE={mods_calm['NE']:.4f} 5HT={mods_calm['5HT']:.4f} ACh={mods_calm['ACh']:.4f}")

    # Phase 2: 5 min stressed coding (30000 ticks)
    for tick in range(30000):
        brain.tick(encode_snapshot(stressed_coding()))

    mods_stress = brain.modulators.snapshot()
    print(f"  Nach 5min Stress: NE={mods_stress['NE']:.4f} 5HT={mods_stress['5HT']:.4f} ACh={mods_stress['ACh']:.4f}")

    ne_rose = mods_stress["NE"] > mods_calm["NE"] + 0.01
    sht_dropped = mods_stress["5HT"] < mods_calm["5HT"]

    passed = ne_rose and sht_dropped
    print(f"  NE stieg: {ne_rose} ({mods_calm['NE']:.4f} → {mods_stress['NE']:.4f})")
    print(f"  5HT sank: {sht_dropped} ({mods_calm['5HT']:.4f} → {mods_stress['5HT']:.4f})")
    print(f"  → {'PASS' if passed else 'FAIL'}")

    return {"name": "stress_detection", "pass": passed,
            "ne_calm": mods_calm["NE"], "ne_stress": mods_stress["NE"],
            "sht_calm": mods_calm["5HT"], "sht_stress": mods_stress["5HT"]}


def test_flow_protection() -> dict:
    """Szenario 2: Deep focus → ACh high, 5HT high, NE low."""
    print("\n" + "=" * 60)
    print("  SZENARIO 2: FLOW-ERKENNUNG")
    print("  10 Minuten ununterbrochenes fokussiertes Coden")
    print("=" * 60)

    random.seed(42)
    brain = Brain(num_sensory=200, num_concept=200)

    # 10 min deep focus (60000 ticks)
    for tick in range(60000):
        brain.tick(encode_snapshot(focus_flow()))

    mods = brain.modulators.snapshot()
    print(f"  Nach 10min Flow: DA={mods['DA']:.4f} NE={mods['NE']:.4f} ACh={mods['ACh']:.4f} 5HT={mods['5HT']:.4f}")

    ach_high = mods["ACh"] > 0.02
    sht_high = mods["5HT"] > 0.03
    ne_low = mods["NE"] < 0.02

    passed = ach_high and sht_high and ne_low
    print(f"  ACh hoch (>{0.02}): {ach_high} ({mods['ACh']:.4f})")
    print(f"  5HT hoch (>{0.03}): {sht_high} ({mods['5HT']:.4f})")
    print(f"  NE niedrig (<{0.02}): {ne_low} ({mods['NE']:.4f})")
    print(f"  → {'PASS' if passed else 'FAIL'}")

    return {"name": "flow_protection", "pass": passed, "modulators": mods}


def test_rhythm_learning() -> dict:
    """Szenario 3: Different activities → different clusters."""
    print("\n" + "=" * 60)
    print("  SZENARIO 3: RHYTHMUS-LERNEN")
    print("  Simulierter Arbeitstag: 8 verschiedene Aktivitäten")
    print("=" * 60)

    random.seed(42)
    brain = Brain(num_sensory=200, num_concept=200)

    # Run full workday (70000 ticks ≈ 11.7 min simulated as full day)
    phase_clusters = defaultdict(list)

    for tick in range(70000):
        phase_name, snap = get_workday_pattern(tick)
        brain.tick(encode_snapshot(snap))

        # Record cluster at phase boundaries
        if tick % 5000 == 4999:
            cluster = brain.concept_tracker.current_cluster_id
            phase_clusters[phase_name].append(cluster)

    # How many unique cluster groups?
    unique_phases = set()
    for phase, clusters in phase_clusters.items():
        if clusters:
            dominant = max(set(clusters), key=clusters.count)
            unique_phases.add(dominant)
            print(f"  {phase:20s}: cluster #{dominant} (from {clusters})")

    num_distinct = len(unique_phases)
    total_phases = len(phase_clusters)
    passed = num_distinct >= 3  # at least 3 distinct activity types recognized

    print(f"\n  Erkannte Phasen: {num_distinct} von {total_phases}")
    print(f"  ConceptTracker total: {brain.concept_tracker.snapshot()['num_clusters']} clusters")
    print(f"  → {'PASS' if passed else 'FAIL'} (braucht ≥3 verschiedene)")

    return {"name": "rhythm_learning", "pass": passed,
            "distinct_phases": num_distinct, "total_phases": total_phases}


def test_memory_persistence() -> dict:
    """Szenario 4: Labels + weights survive save/load."""
    print("\n" + "=" * 60)
    print("  SZENARIO 4: GEDÄCHTNIS-PERSISTENZ")
    print("  Labels und Cluster überleben Restart")
    print("=" * 60)

    from brain.persistence import save_brain, load_brain

    random.seed(42)
    brain = Brain(num_sensory=200, num_concept=200)

    # Train different patterns
    for phase, fn in [("typing", calm_coding), ("call", zoom_call), ("idle", idle_away)]:
        for tick in range(5000):
            brain.tick(encode_snapshot(fn()))

    # Label current clusters
    tracker = brain.concept_tracker.snapshot()
    clusters_before = tracker["num_clusters"]
    brain.concept_tracker.set_label(0, "Coding")
    brain.concept_tracker.set_label(1, "Call")

    tick_before = brain.tick_count

    # Save
    path = Path("/tmp/benchmark_persistence_test.sqlite")
    save_brain(brain, path)

    # Load
    brain2 = load_brain(path)
    tracker2 = brain2.concept_tracker.snapshot()
    clusters_after = tracker2["num_clusters"]
    tick_after = brain2.tick_count

    # Check labels survived
    labels_survived = False
    for cl in tracker2["clusters"]:
        if cl.get("label") == "Coding":
            labels_survived = True
            break

    tick_match = tick_before == tick_after
    cluster_match = clusters_before == clusters_after

    passed = labels_survived and tick_match and cluster_match
    print(f"  Ticks: {tick_before} → {tick_after} ({'match' if tick_match else 'MISMATCH'})")
    print(f"  Clusters: {clusters_before} → {clusters_after} ({'match' if cluster_match else 'MISMATCH'})")
    print(f"  Labels überlebt: {labels_survived}")
    print(f"  → {'PASS' if passed else 'FAIL'}")

    path.unlink(missing_ok=True)
    return {"name": "memory_persistence", "pass": passed}


def test_uniqueness() -> dict:
    """Szenario 5: Different random seeds → different brains."""
    print("\n" + "=" * 60)
    print("  SZENARIO 5: EINZIGARTIGKEIT")
    print("  3 Brains mit verschiedenen Seeds → verschiedene Cluster")
    print("=" * 60)

    results = []
    for seed in [1, 2, 3]:
        random.seed(seed)
        torch.manual_seed(seed)
        brain = Brain(num_sensory=200, num_concept=200)

        # Train each on same pattern
        for tick in range(10000):
            brain.tick(encode_snapshot(calm_coding()))

        cluster = brain.concept_tracker.current_cluster_id
        # Get expansion weight fingerprint (first 10 values)
        exp_fp = brain._expansion_weights[0, :10].tolist()
        results.append({"seed": seed, "cluster": cluster,
                        "exp_fingerprint": [round(v, 2) for v in exp_fp]})
        print(f"  Seed {seed}: cluster #{cluster}, expansion[0,:10]={results[-1]['exp_fingerprint']}")

    # Check that expansion weights differ
    fps = [tuple(r["exp_fingerprint"]) for r in results]
    all_different = len(set(fps)) == len(fps)

    passed = all_different
    print(f"\n  Alle Fingerprints verschieden: {all_different}")
    print(f"  → {'PASS' if passed else 'FAIL'}")

    return {"name": "uniqueness", "pass": passed}


def run_all():
    print("=" * 60)
    print("  brAIntest SZENARIO-BENCHMARK (Forschungsniveau)")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = [
        test_stress_detection(),
        test_flow_protection(),
        test_rhythm_learning(),
        test_memory_persistence(),
        test_uniqueness(),
    ]

    print("\n" + "=" * 60)
    print("  ERGEBNISSE")
    print("=" * 60)

    for r in results:
        icon = "✅" if r["pass"] else "❌"
        print(f"  {icon} {r['name']}")

    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    print(f"\n  {passed}/{total} BESTANDEN")

    if passed == total:
        print("  🎉 ALLE SZENARIEN BESTANDEN!")
    else:
        print(f"  ⚠️ {total - passed} FEHLGESCHLAGEN")

    report = {
        "timestamp": time.time(),
        "results": results,
        "passed": passed,
        "total": total,
    }
    report_path = Path("benchmark/reports/scenario_benchmark.json")
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Report: {report_path}")

    return passed == total


if __name__ == "__main__":
    run_all()
