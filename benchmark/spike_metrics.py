"""Benchmark: Spike Metrics — per-region firing statistics.

Measures spike counts, firing rates, sparsity, per-neuron distributions,
and dead neuron detection across 4 scenarios x 10000 ticks each.

Usage: .venv/bin/python benchmark/spike_metrics.py
"""
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from brain.core import Brain
from adapters.mac_desktop.encoding import encode_snapshot

# ── Scenarios (inlined from discrimination_test.py) ──────────────────────────

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

TICKS_PER_SCENARIO = 10_000

# Region name -> (size, key in tick output)
REGIONS = {
    "sensory": (200, "sensory"),
    "concept": (200, "concept"),
    "wm":      (100, "wm"),
}


def make_histogram(counts: list[int], num_bins: int = 10) -> dict:
    """Build a histogram of per-neuron total fire counts."""
    if not counts:
        return {"bins": [], "edges": []}
    lo = min(counts)
    hi = max(counts)
    if hi == lo:
        return {"bins": [len(counts)], "edges": [lo, lo + 1]}
    step = max(1, (hi - lo) // num_bins)
    edges = list(range(lo, hi + step, step))
    if edges[-1] < hi:
        edges.append(hi + 1)
    bins = [0] * (len(edges) - 1)
    for c in counts:
        idx = min((c - lo) // step, len(bins) - 1)
        bins[idx] += 1
    return {"bins": bins, "edges": edges}


def run_scenario(brain: Brain, scenario_name: str, maker) -> dict:
    """Run one scenario for TICKS_PER_SCENARIO ticks, collecting spike stats."""
    num_neurons = {name: size for name, (size, _) in REGIONS.items()}

    # Per-tick spike counts (for mean firing rate)
    tick_spike_counts: dict[str, list[int]] = {r: [] for r in REGIONS}
    # Per-neuron total fire counts (for distribution + dead detection)
    neuron_fire_totals: dict[str, torch.Tensor] = {
        r: torch.zeros(num_neurons[r]) for r in REGIONS
    }

    for t in range(TICKS_PER_SCENARIO):
        vec = encode_snapshot(maker())
        out = brain.tick(vec)

        for region_name, (size, key) in REGIONS.items():
            spikes = out[key]
            spike_count = int(spikes.sum().item())
            tick_spike_counts[region_name].append(spike_count)
            neuron_fire_totals[region_name] += spikes.detach()

    # ── Compute metrics per region ──
    results = {}
    for region_name, (size, _) in REGIONS.items():
        counts = tick_spike_counts[region_name]
        totals = neuron_fire_totals[region_name].tolist()
        totals_int = [int(x) for x in totals]

        total_spikes = sum(counts)
        mean_rate = total_spikes / TICKS_PER_SCENARIO
        # Sparsity: average fraction of neurons active per tick
        sparsity_per_tick = [c / size * 100.0 for c in counts]
        mean_sparsity = sum(sparsity_per_tick) / len(sparsity_per_tick)

        dead_neurons = [i for i, t in enumerate(totals_int) if t == 0]
        dead_count = len(dead_neurons)
        dead_pct = dead_count / size * 100.0

        histogram = make_histogram(totals_int)

        results[region_name] = {
            "num_neurons": size,
            "total_spikes": total_spikes,
            "mean_rate": round(mean_rate, 3),
            "mean_sparsity_pct": round(mean_sparsity, 3),
            "min_tick_spikes": min(counts),
            "max_tick_spikes": max(counts),
            "dead_neurons": dead_count,
            "dead_pct": round(dead_pct, 2),
            "dead_neuron_ids": dead_neurons[:20],  # first 20 for brevity
            "histogram": histogram,
        }

    return results


def print_table(all_results: dict):
    """Print a formatted summary table."""
    header = f"{'Scenario':<20} {'Region':<10} {'Neurons':>7} {'Total':>9} {'Rate':>7} {'Sparse%':>8} {'Min':>5} {'Max':>5} {'Dead':>5} {'Dead%':>6}"
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for scenario_name, regions in all_results.items():
        first = True
        for region_name, m in regions.items():
            label = scenario_name if first else ""
            first = False
            print(
                f"{label:<20} {region_name:<10} {m['num_neurons']:>7d} "
                f"{m['total_spikes']:>9d} {m['mean_rate']:>7.2f} "
                f"{m['mean_sparsity_pct']:>7.2f}% {m['min_tick_spikes']:>5d} "
                f"{m['max_tick_spikes']:>5d} {m['dead_neurons']:>5d} "
                f"{m['dead_pct']:>5.1f}%"
            )
        print(sep)


def print_dead_summary(all_results: dict):
    """Print dead neuron summary across all scenarios."""
    print("\n  DEAD NEURON SUMMARY (never fired in any scenario)")

    # Collect per-region: neurons dead in ALL scenarios
    region_names = list(REGIONS.keys())
    for region_name in region_names:
        dead_sets = []
        for scenario_name, regions in all_results.items():
            dead_sets.append(set(regions[region_name]["dead_neuron_ids"]))
        # Neurons dead across ALL scenarios
        always_dead = dead_sets[0]
        for ds in dead_sets[1:]:
            always_dead = always_dead & ds
        total_neurons = REGIONS[region_name][0]
        print(
            f"    {region_name:<10}: {len(always_dead)}/{total_neurons} "
            f"always-dead ({len(always_dead)/total_neurons*100:.1f}%)"
        )


def run_all():
    random.seed(42)
    torch.manual_seed(42)
    brain = Brain(num_sensory=200, num_concept=200)

    print("=" * 72)
    print("  SPIKE METRICS BENCHMARK")
    print(f"  {len(SCENARIOS)} scenarios x {TICKS_PER_SCENARIO} ticks each")
    print("=" * 72)

    all_results = {}
    t0 = time.time()

    for i, (sname, maker) in enumerate(SCENARIOS.items()):
        print(f"\n  [{i+1}/{len(SCENARIOS)}] Running {sname} ({TICKS_PER_SCENARIO} ticks)...", end="", flush=True)
        st = time.time()
        all_results[sname] = run_scenario(brain, sname, maker)
        elapsed = time.time() - st
        print(f" done ({elapsed:.1f}s)")

    total_time = time.time() - t0

    # ── Print results ──
    print_table(all_results)
    print_dead_summary(all_results)

    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Ticks/sec: {TICKS_PER_SCENARIO * len(SCENARIOS) / total_time:.0f}")

    # ── Save JSON report ──
    report = {
        "test": "spike_metrics",
        "timestamp": time.time(),
        "config": {
            "num_sensory": 200,
            "num_concept": 200,
            "ticks_per_scenario": TICKS_PER_SCENARIO,
            "scenarios": list(SCENARIOS.keys()),
        },
        "results": all_results,
        "total_time_s": round(total_time, 2),
    }

    out_path = Path("benchmark/reports/spike_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved to {out_path}")


if __name__ == "__main__":
    run_all()
