"""Benchmark Test 2: Replay Test — does the SNN recognize repeated patterns?

The most convincing proof that the SNN learns: play the same sensor pattern
10 times with pauses between. Measure:
1. Does a SPECIFIC concept neuron fire consistently after 3-5 replays?
2. Does the response get faster? (concept fires earlier in the replay)
3. Does novelty (DA) decrease with repetition? (it's no longer new)

This test runs entirely in-process with a fresh Brain — no daemon needed.

Usage: .venv/bin/python benchmark/replay_test.py
"""
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from brain.core import Brain
from adapters.mac_desktop.encoding import encode_snapshot


# Three distinct patterns to replay
PATTERNS = {
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
    "idle_away": lambda: {
        "active_app": {"name": "Finder"},
        "keystroke_rate": {"count": 0},
        "mouse_rate": {"count": 0},
        "idle": {"seconds": random.uniform(60, 180)},
        "mic": {"mel": [random.uniform(0.001, 0.005)] * 32, "rms": random.uniform(0.0005, 0.002)},
        "time_tonic": {"day_phase": [0.5, 0.5, 0.5, 0.5], "week_phase": [0.5, 0.5, 0.5, 0.5]},
    },
}

REPLAY_COUNT = 20           # play each pattern 20 times
TICKS_PER_REPLAY = 500      # each replay = 500 ticks (5 seconds — realistic exposure)
PAUSE_TICKS = 500           # 500 ticks pause (5 seconds)
WARMUP_TICKS = 100000       # 100K ticks warmup (lots of alternating training)


def run_replay_test(pattern_name: str, brain: Brain) -> dict:
    """Run a single replay test for one pattern."""
    make_pattern = PATTERNS[pattern_name]

    # Track which concepts fire during each replay
    replay_results = []

    for replay_idx in range(REPLAY_COUNT):
        concept_fires = defaultdict(int)
        first_concept_tick = None
        da_values = []

        # Play the pattern
        for tick in range(TICKS_PER_REPLAY):
            snap = make_pattern()
            vec = encode_snapshot(snap)
            out = brain.tick(vec)

            concept_spikes = out["concept"]
            fired = (concept_spikes > 0).nonzero(as_tuple=True)[0].tolist()
            for cid in fired:
                concept_fires[cid] += 1
                if first_concept_tick is None:
                    first_concept_tick = tick

            da_values.append(brain.modulators.level("DA"))

        # Pause (silence between replays)
        for tick in range(PAUSE_TICKS):
            silence = encode_snapshot(PATTERNS["idle_away"]())
            brain.tick(silence)

        # Find dominant concept for this replay
        if concept_fires:
            dominant = max(concept_fires, key=concept_fires.get)
            dominant_count = concept_fires[dominant]
            total_fires = sum(concept_fires.values())
        else:
            dominant = -1
            dominant_count = 0
            total_fires = 0

        replay_results.append({
            "replay": replay_idx + 1,
            "dominant_concept": dominant,
            "dominant_pct": round(dominant_count / max(1, total_fires) * 100, 1),
            "total_fires": total_fires,
            "first_concept_tick": first_concept_tick,
            "avg_da": round(sum(da_values) / len(da_values), 6),
            "unique_concepts": len(concept_fires),
            "_all_fires": dict(concept_fires),  # for group analysis
        })

    # Analysis — use TOP-3 GROUP consistency (not single-neuron)
    # With k=3, a "concept" is a GROUP of 3 neurons, not one neuron.
    # Measure: do the top-3 neurons across all replays form a stable set?
    all_top3_sets = []
    for r in replay_results:
        # Get top-3 most frequent concepts for this replay
        top3 = set(sorted(r.get("_all_fires", {}).keys(),
                          key=lambda c: r.get("_all_fires", {}).get(c, 0), reverse=True)[:3])
        all_top3_sets.append(top3)

    # Consistency: average pairwise Jaccard similarity between replay top-3 sets
    if len(all_top3_sets) >= 2:
        jaccard_sum = 0
        count = 0
        for i in range(len(all_top3_sets)):
            for j in range(i + 1, len(all_top3_sets)):
                inter = len(all_top3_sets[i] & all_top3_sets[j])
                union = len(all_top3_sets[i] | all_top3_sets[j])
                jaccard_sum += inter / max(1, union)
                count += 1
        consistency = jaccard_sum / count * 100
    else:
        consistency = 0

    # Most common dominant for backward compat
    dominants = [r["dominant_concept"] for r in replay_results if r["dominant_concept"] >= 0]
    most_common = max(set(dominants), key=dominants.count) if dominants else -1

    # Response time trend: does first_concept_tick decrease?
    response_times = [r["first_concept_tick"] for r in replay_results if r["first_concept_tick"] is not None]
    if len(response_times) >= 2:
        first_half = sum(response_times[:len(response_times)//2]) / max(1, len(response_times)//2)
        second_half = sum(response_times[len(response_times)//2:]) / max(1, len(response_times) - len(response_times)//2)
        speedup = round((first_half - second_half) / max(1, first_half) * 100, 1)
    else:
        speedup = 0

    # Novelty habituation: does DA decrease with repetition?
    da_values_all = [r["avg_da"] for r in replay_results]
    if len(da_values_all) >= 2:
        da_first = sum(da_values_all[:3]) / min(3, len(da_values_all))
        da_last = sum(da_values_all[-3:]) / min(3, len(da_values_all))
        da_decrease = round((da_first - da_last) / max(0.0001, da_first) * 100, 1)
    else:
        da_decrease = 0

    return {
        "pattern": pattern_name,
        "replays": replay_results,
        "analysis": {
            "dominant_concept": most_common,
            "consistency_pct": round(consistency, 1),
            "response_speedup_pct": speedup,
            "da_habituation_pct": da_decrease,
        },
        "pass": consistency >= 60,  # 60% = same concept fires for most replays
    }


def run_all():
    random.seed(42)
    brain = Brain(num_sensory=200, num_concept=200)

    print("=" * 60)
    print("  BENCHMARK TEST 2: REPLAY TEST")
    print("  Does the SNN recognize repeated patterns?")

    # Warmup: train on alternating patterns so STDP has time to differentiate
    print(f"\n  Warmup: {WARMUP_TICKS} ticks on alternating patterns...")
    pattern_names_all = list(PATTERNS.keys())
    for tick in range(WARMUP_TICKS):
        pidx = (tick // 5000) % len(pattern_names_all)
        pname = pattern_names_all[pidx]
        if pname == "idle_away":
            pidx = 0
            pname = pattern_names_all[0]
        vec = encode_snapshot(PATTERNS[pname]())
        brain.tick(vec)
    print(f"  Warmup complete. Tick: {brain.tick_count}")
    print("=" * 60)

    results = {}

    for pattern_name in PATTERNS:
        if pattern_name == "idle_away":
            continue  # skip idle — used as pause

        print(f"\n--- Pattern: {pattern_name} ---")
        print(f"  {REPLAY_COUNT} replays x {TICKS_PER_REPLAY} ticks, {PAUSE_TICKS} tick pauses")

        result = run_replay_test(pattern_name, brain)
        results[pattern_name] = result

        print(f"\n  Per-replay results:")
        for r in result["replays"]:
            print(f"    Replay {r['replay']:2d}: dominant=C{r['dominant_concept']} "
                  f"({r['dominant_pct']}%), first_tick={r['first_concept_tick']}, "
                  f"DA={r['avg_da']:.4f}")

        a = result["analysis"]
        print(f"\n  Analysis:")
        print(f"    Dominant concept: C{a['dominant_concept']}")
        print(f"    Consistency: {a['consistency_pct']}% (target: >60%)")
        print(f"    Response speedup: {a['response_speedup_pct']}% (target: >0%)")
        print(f"    DA habituation: {a['da_habituation_pct']}% (target: >0%)")
        print(f"    Result: {'PASS' if result['pass'] else 'FAIL'}")

    # Discrimination: do different patterns produce different concepts?
    dominant_concepts = [r["analysis"]["dominant_concept"] for r in results.values()]
    all_different = len(set(dominant_concepts)) == len(dominant_concepts)

    print(f"\n{'=' * 60}")
    print(f"  DISCRIMINATION: Different patterns → different concepts?")
    print(f"  Dominant concepts: {dominant_concepts}")
    print(f"  All different: {all_different}")
    print(f"  Result: {'PASS' if all_different else 'FAIL'}")

    # Overall
    passes = sum(1 for r in results.values() if r["pass"])
    total = len(results)
    overall = passes == total and all_different

    print(f"\n{'=' * 60}")
    print(f"  OVERALL: {passes}/{total} patterns consistent + {'discrimination OK' if all_different else 'discrimination FAIL'}")
    print(f"  {'PASS' if overall else 'FAIL'}")
    print(f"{'=' * 60}")

    # Save report
    report = {
        "test": "replay",
        "timestamp": time.time(),
        "results": results,
        "discrimination": all_different,
        "overall_pass": overall,
    }
    report_path = Path("benchmark/reports/replay_test.json")
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Report saved to {report_path}")

    return overall


if __name__ == "__main__":
    run_all()
