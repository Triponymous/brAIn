"""Benchmark: STDP Weight Analysis — tracks learning quality over 100K ticks.

Creates a Brain, runs 4 scenarios alternating for 100K ticks, and snapshots
the sensory_concept (expansion->concept) weight matrix every 10K ticks.

Measures:
    - Weight distribution stats (mean, std, min, max, percentiles)
    - Sparsity (% weights near floor 0.01)
    - Saturation (% weights near max 1.0)
    - Selectivity Index per concept neuron (Diehl & Cook style)

Usage: .venv/bin/python benchmark/weight_analysis.py
"""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from brain.core import Brain
from adapters.mac_desktop.encoding import encode_snapshot

# ── Scenario definitions (same patterns as discrimination_test) ──

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

TOTAL_TICKS = 100_000
SNAPSHOT_INTERVAL = 10_000
SCENARIO_NAMES = list(SCENARIOS.keys())

# Thresholds for sparsity/saturation detection
FLOOR_THRESHOLD = 0.02     # weights <= this are "near floor" (actual floor is 0.01)
CEILING_THRESHOLD = 0.98   # weights >= this are "near max" (max is 1.0)

# Diehl & Cook selectivity threshold
SELECTIVITY_THRESHOLD = 0.5


def snapshot_weights(weights: torch.Tensor) -> dict:
    """Compute distribution stats for a weight matrix."""
    w = weights.detach().float()
    total = w.numel()
    return {
        "mean": float(w.mean()),
        "std": float(w.std()),
        "min": float(w.min()),
        "max": float(w.max()),
        "p10": float(w.quantile(0.1)),
        "p50": float(w.quantile(0.5)),
        "p90": float(w.quantile(0.9)),
        "sparsity_pct": float((w <= FLOOR_THRESHOLD).sum() / total * 100),
        "saturation_pct": float((w >= CEILING_THRESHOLD).sum() / total * 100),
    }


def compute_selectivity(
    response_matrix: dict[str, torch.Tensor],
    num_concept: int,
) -> dict:
    """Compute Diehl & Cook selectivity index per concept neuron.

    For each concept neuron j:
        SI_j = max_s(R_j,s) / sum_s(R_j,s)

    where R_j,s is the total spike count of neuron j during scenario s.
    A neuron with SI > 0.5 is considered selective (responds mainly to one scenario).

    Returns dict with per-neuron SI, fraction selective, mean SI, and per-scenario
    assignment counts.
    """
    # Stack responses: shape [num_scenarios, num_concept]
    scenario_names = sorted(response_matrix.keys())
    responses = torch.stack([response_matrix[s] for s in scenario_names], dim=0)  # [S, C]

    total_response = responses.sum(dim=0)  # [C]
    max_response = responses.max(dim=0)    # values [C], indices [C]

    # Avoid division by zero for silent neurons
    active_mask = total_response > 0
    si = torch.zeros(num_concept)
    si[active_mask] = max_response.values[active_mask] / total_response[active_mask]

    selective_mask = si > SELECTIVITY_THRESHOLD
    preferred_scenario = max_response.indices  # which scenario each neuron prefers

    # Count how many selective neurons are assigned to each scenario
    assignment_counts = {}
    for idx, sname in enumerate(scenario_names):
        count = int(((preferred_scenario == idx) & selective_mask).sum())
        assignment_counts[sname] = count

    return {
        "selectivity_index": si.tolist(),
        "mean_si": float(si[active_mask].mean()) if active_mask.any() else 0.0,
        "fraction_selective": float(selective_mask.sum() / num_concept),
        "num_selective": int(selective_mask.sum()),
        "num_active": int(active_mask.sum()),
        "num_silent": int((~active_mask).sum()),
        "assignment_counts": assignment_counts,
    }


def try_plot(snapshots: list[dict], report_dir: Path) -> bool:
    """Attempt to save weight evolution plot. Returns True on success."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    ticks = [s["tick"] for s in snapshots]
    means = [s["stats"]["mean"] for s in snapshots]
    stds = [s["stats"]["std"] for s in snapshots]
    p10 = [s["stats"]["p10"] for s in snapshots]
    p50 = [s["stats"]["p50"] for s in snapshots]
    p90 = [s["stats"]["p90"] for s in snapshots]
    sparsity = [s["stats"]["sparsity_pct"] for s in snapshots]
    saturation = [s["stats"]["saturation_pct"] for s in snapshots]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("STDP Weight Evolution (expansion -> concept)", fontsize=14, fontweight="bold")

    # Top left: weight distribution percentiles
    ax = axes[0, 0]
    ax.fill_between(ticks, p10, p90, alpha=0.2, color="steelblue", label="10th-90th pctl")
    ax.plot(ticks, p50, "o-", color="steelblue", linewidth=2, label="Median")
    ax.plot(ticks, means, "s--", color="darkorange", linewidth=1.5, label="Mean")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Weight")
    ax.set_title("Weight Distribution Over Time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Top right: mean +/- std
    ax = axes[0, 1]
    ax.errorbar(ticks, means, yerr=stds, fmt="o-", color="steelblue", capsize=4, linewidth=1.5)
    ax.axhline(y=0.01, color="red", linestyle=":", alpha=0.5, label="Floor (0.01)")
    ax.axhline(y=1.0, color="red", linestyle=":", alpha=0.5, label="Ceiling (1.0)")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Weight")
    ax.set_title("Mean +/- Std")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom left: sparsity and saturation
    ax = axes[1, 0]
    ax.plot(ticks, sparsity, "o-", color="mediumpurple", linewidth=2, label=f"Sparse (<={FLOOR_THRESHOLD})")
    ax.plot(ticks, saturation, "s-", color="crimson", linewidth=2, label=f"Saturated (>={CEILING_THRESHOLD})")
    ax.set_xlabel("Tick")
    ax.set_ylabel("% of weights")
    ax.set_title("Sparsity & Saturation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Bottom right: selectivity evolution (if available)
    si_means = [s.get("selectivity_mean_si") for s in snapshots]
    si_fracs = [s.get("selectivity_fraction") for s in snapshots]
    if si_means[0] is not None:
        ax = axes[1, 1]
        ax2 = ax.twinx()
        l1, = ax.plot(ticks, si_means, "o-", color="teal", linewidth=2, label="Mean SI")
        l2, = ax2.plot(ticks, [f * 100 for f in si_fracs], "s-", color="coral", linewidth=2, label="% Selective")
        ax.set_xlabel("Tick")
        ax.set_ylabel("Mean Selectivity Index", color="teal")
        ax2.set_ylabel("% Selective Neurons", color="coral")
        ax.set_title("Selectivity Evolution")
        ax.legend(handles=[l1, l2], fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "No selectivity data", ha="center", va="center",
                        transform=axes[1, 1].transAxes, fontsize=12, color="gray")
        axes[1, 1].set_title("Selectivity Evolution")

    plt.tight_layout()
    out_path = report_dir / "weight_evolution.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return True


def run():
    random.seed(42)
    torch.manual_seed(42)

    num_sensory = 200
    num_concept = 200

    brain = Brain(num_sensory=num_sensory, num_concept=num_concept)
    report_dir = Path("benchmark/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  STDP WEIGHT ANALYSIS")
    print(f"  Brain: sensory={num_sensory}, concept={num_concept}")
    print(f"  Total ticks: {TOTAL_TICKS:,}, snapshots every {SNAPSHOT_INTERVAL:,}")
    print(f"  Scenarios: {', '.join(SCENARIO_NAMES)}")
    print("=" * 70)

    # The main STDP synapse is expansion->concept, stored as "sensory_concept"
    sc_synapse = brain.synapses["sensory_concept"]

    # Take initial snapshot
    snapshots = []
    initial = snapshot_weights(sc_synapse.weights)
    snapshots.append({"tick": 0, "stats": initial})
    print(f"\n  [tick      0] mean={initial['mean']:.4f}  std={initial['std']:.4f}  "
          f"sparse={initial['sparsity_pct']:.1f}%  sat={initial['saturation_pct']:.1f}%")

    # Accumulate per-scenario spike counts for selectivity
    # Reset every snapshot interval, but also keep a global accumulator
    scenario_spikes_global: dict[str, torch.Tensor] = {
        s: torch.zeros(num_concept) for s in SCENARIO_NAMES
    }
    scenario_spikes_interval: dict[str, torch.Tensor] = {
        s: torch.zeros(num_concept) for s in SCENARIO_NAMES
    }

    t_start = time.time()
    scenario_cycle = list(SCENARIOS.items())  # deterministic order

    for tick in range(1, TOTAL_TICKS + 1):
        # Round-robin scenario selection (alternating every tick)
        sname, maker = scenario_cycle[tick % len(scenario_cycle)]

        vec = encode_snapshot(maker())
        out = brain.tick(vec)
        concept_spikes = out["concept"].detach()

        # Accumulate spikes
        scenario_spikes_global[sname] += concept_spikes
        scenario_spikes_interval[sname] += concept_spikes

        # Snapshot at interval boundaries
        if tick % SNAPSHOT_INTERVAL == 0:
            stats = snapshot_weights(sc_synapse.weights)

            # Compute interval selectivity
            sel = compute_selectivity(scenario_spikes_interval, num_concept)
            stats_entry = {
                "tick": tick,
                "stats": stats,
                "selectivity_mean_si": sel["mean_si"],
                "selectivity_fraction": sel["fraction_selective"],
                "selectivity_num_selective": sel["num_selective"],
                "selectivity_num_active": sel["num_active"],
                "selectivity_assignment": sel["assignment_counts"],
            }
            snapshots.append(stats_entry)

            elapsed = time.time() - t_start
            tps = tick / elapsed
            print(f"  [tick {tick:>6,}] mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
                  f"sparse={stats['sparsity_pct']:.1f}%  sat={stats['saturation_pct']:.1f}%  "
                  f"SI={sel['mean_si']:.3f}  sel={sel['fraction_selective']*100:.1f}%  "
                  f"({tps:.0f} ticks/s)")

            # Reset interval accumulator
            scenario_spikes_interval = {s: torch.zeros(num_concept) for s in SCENARIO_NAMES}

    elapsed_total = time.time() - t_start

    # ── Final selectivity (global) ──
    global_selectivity = compute_selectivity(scenario_spikes_global, num_concept)

    # ── Print formatted results ──
    print(f"\n{'=' * 70}")
    print("  WEIGHT DISTRIBUTION EVOLUTION")
    print(f"{'=' * 70}")
    print(f"  {'Tick':>8s}  {'Mean':>7s}  {'Std':>7s}  {'Min':>7s}  {'Max':>7s}  "
          f"{'P10':>7s}  {'P50':>7s}  {'P90':>7s}  {'Sparse%':>8s}  {'Sat%':>6s}")
    print(f"  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  "
          f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*6}")
    for s in snapshots:
        st = s["stats"]
        print(f"  {s['tick']:>8,}  {st['mean']:>7.4f}  {st['std']:>7.4f}  {st['min']:>7.4f}  {st['max']:>7.4f}  "
              f"{st['p10']:>7.4f}  {st['p50']:>7.4f}  {st['p90']:>7.4f}  {st['sparsity_pct']:>7.1f}%  {st['saturation_pct']:>5.1f}%")

    print(f"\n{'=' * 70}")
    print("  SELECTIVITY ANALYSIS (Diehl & Cook)")
    print(f"{'=' * 70}")
    print(f"  Selectivity threshold: SI > {SELECTIVITY_THRESHOLD}")
    print(f"  Active neurons:   {global_selectivity['num_active']}/{num_concept}")
    print(f"  Silent neurons:   {global_selectivity['num_silent']}/{num_concept}")
    print(f"  Selective neurons: {global_selectivity['num_selective']}/{num_concept} "
          f"({global_selectivity['fraction_selective']*100:.1f}%)")
    print(f"  Mean SI (active):  {global_selectivity['mean_si']:.4f}")

    print(f"\n  Per-scenario selective neuron assignments:")
    for sname, count in global_selectivity["assignment_counts"].items():
        print(f"    {sname:20s}: {count} neurons")

    # Selectivity evolution
    print(f"\n  {'Tick':>8s}  {'Mean SI':>8s}  {'% Selective':>12s}  {'Num Select':>11s}  {'Num Active':>11s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*11}  {'-'*11}")
    for s in snapshots:
        if "selectivity_mean_si" in s and s["selectivity_mean_si"] is not None:
            print(f"  {s['tick']:>8,}  {s['selectivity_mean_si']:>8.4f}  "
                  f"{s['selectivity_fraction']*100:>11.1f}%  "
                  f"{s['selectivity_num_selective']:>11d}  "
                  f"{s['selectivity_num_active']:>11d}")

    # ── Healthy learning diagnostics ──
    final_stats = snapshots[-1]["stats"]
    initial_stats = snapshots[0]["stats"]

    checks = []
    # Weight drift: std growth shows learning happened (mean may stay stable with synaptic scaling)
    std_growth = abs(final_stats["std"] - initial_stats["std"])
    mean_drift = abs(final_stats["mean"] - initial_stats["mean"])
    drift_metric = max(std_growth, mean_drift)  # either std or mean change counts
    checks.append(("Weight learning (std or mean changed)", drift_metric > 0.001, f"{drift_metric:.4f}"))

    # Std should be positive (differentiation happened)
    checks.append(("Weight differentiation (std > 0.05)", final_stats["std"] > 0.05, f"{final_stats['std']:.4f}"))

    # Not fully sparse (network collapsed)
    checks.append(("Not collapsed (sparse < 90%)", final_stats["sparsity_pct"] < 90, f"{final_stats['sparsity_pct']:.1f}%"))

    # Not fully saturated
    checks.append(("Not saturated (sat < 50%)", final_stats["saturation_pct"] < 50, f"{final_stats['saturation_pct']:.1f}%"))

    # Some selectivity emerged
    checks.append(("Selectivity emerged (>10% selective)", global_selectivity["fraction_selective"] > 0.1,
                    f"{global_selectivity['fraction_selective']*100:.1f}%"))

    # Multiple scenarios have assigned neurons
    scenarios_with_neurons = sum(1 for c in global_selectivity["assignment_counts"].values() if c > 0)
    checks.append(("Multi-scenario coverage (>=2 scenarios)", scenarios_with_neurons >= 2,
                    f"{scenarios_with_neurons}/4"))

    print(f"\n{'=' * 70}")
    print("  HEALTH CHECKS")
    print(f"{'=' * 70}")
    all_pass = True
    for name, passed, value in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}: {value}")

    overall = "PASS" if all_pass else "FAIL"
    print(f"\n  Overall: {overall}")
    print(f"  Runtime: {elapsed_total:.1f}s ({TOTAL_TICKS/elapsed_total:.0f} ticks/s)")

    # ── Plot ──
    plot_saved = try_plot(snapshots, report_dir)
    if plot_saved:
        print(f"\n  Plot saved to benchmark/reports/weight_evolution.png")
    else:
        print(f"\n  matplotlib not available, skipping plot")

    # ── JSON report ──
    report = {
        "test": "weight_analysis",
        "timestamp": time.time(),
        "config": {
            "num_sensory": num_sensory,
            "num_concept": num_concept,
            "total_ticks": TOTAL_TICKS,
            "snapshot_interval": SNAPSHOT_INTERVAL,
            "scenarios": SCENARIO_NAMES,
            "floor_threshold": FLOOR_THRESHOLD,
            "ceiling_threshold": CEILING_THRESHOLD,
            "selectivity_threshold": SELECTIVITY_THRESHOLD,
        },
        "snapshots": snapshots,
        "global_selectivity": {
            "mean_si": global_selectivity["mean_si"],
            "fraction_selective": global_selectivity["fraction_selective"],
            "num_selective": global_selectivity["num_selective"],
            "num_active": global_selectivity["num_active"],
            "num_silent": global_selectivity["num_silent"],
            "assignment_counts": global_selectivity["assignment_counts"],
        },
        "health_checks": {name: {"passed": passed, "value": value} for name, passed, value in checks},
        "overall_pass": all_pass,
        "runtime_seconds": elapsed_total,
    }

    json_path = report_dir / "weight_analysis.json"
    json_path.write_text(json.dumps(report, indent=2))
    print(f"  Report saved to {json_path}")

    print(f"{'=' * 70}")
    return all_pass


if __name__ == "__main__":
    run()
