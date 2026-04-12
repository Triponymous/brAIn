"""Ablation study: quantify the contribution of each Brain subsystem.

Disables one mechanism at a time and measures discrimination quality
(distinct concept groups, pairwise overlap) compared to the full baseline.

7 conditions x 3 seeds x 4 scenarios x 5000 ticks each.

Usage: .venv/bin/python benchmark/ablation_study.py
"""
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from brain.core import Brain
from adapters.mac_desktop.encoding import encode_snapshot

# ── Scenario pattern generators (copied from discrimination_test.py) ────────

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
PAUSE_TICKS = 2000

# ── Discrimination measurement ──────────────────────────────────────────────


def measure_discrimination(brain: Brain) -> dict:
    """Run all 4 scenarios and return discrimination metrics.

    Returns dict with:
        distinct_groups: int (0-4) -- how many scenarios are uniquely represented
        avg_overlap: float -- mean pairwise top-3 overlap as percentage
        scenario_top3: {name: set of top-3 concept ids}
        pairwise: {pair_key: overlap %}
    """
    scenario_names = list(SCENARIOS.keys())
    scenario_top3: dict[str, set[int]] = {}

    for sname, maker in SCENARIOS.items():
        fires = Counter()
        for _ in range(TICKS_PER_SCENARIO):
            vec = encode_snapshot(maker())
            out = brain.tick(vec)
            for cid in (out["concept"] > 0).nonzero(as_tuple=True)[0].tolist():
                fires[cid] += 1

        # Pause between scenarios to let membrane potentials settle
        for _ in range(PAUSE_TICKS):
            brain.tick(torch.zeros(200))

        top3 = set(cid for cid, _ in fires.most_common(3)) if fires else set()
        scenario_top3[sname] = top3

    # Pairwise overlap
    pairwise: dict[str, float] = {}
    for i, s1 in enumerate(scenario_names):
        for j, s2 in enumerate(scenario_names):
            if i >= j:
                continue
            overlap = len(scenario_top3[s1] & scenario_top3[s2])
            pct = overlap / 3.0 * 100 if scenario_top3[s1] and scenario_top3[s2] else 0.0
            pairwise[f"{s1}_vs_{s2}"] = pct

    avg_overlap = sum(pairwise.values()) / max(1, len(pairwise))

    # Count distinct groups: a scenario is "distinct" if no other scenario
    # shares more than 50% of its top-3 concepts
    all_groups = [(sname, scenario_top3[sname]) for sname in scenario_names]
    distinct = 0
    for i, (s1, g1) in enumerate(all_groups):
        if not g1:
            continue
        is_distinct = True
        for j, (s2, g2) in enumerate(all_groups):
            if i != j and g2 and len(g1 & g2) > len(g1) * 0.5:
                is_distinct = False
                break
        if is_distinct:
            distinct += 1

    return {
        "distinct_groups": distinct,
        "avg_overlap": avg_overlap,
        "scenario_top3": {k: list(v) for k, v in scenario_top3.items()},
        "pairwise": pairwise,
    }


# ── Ablation condition builders ─────────────────────────────────────────────


def make_baseline() -> Brain:
    """Normal Brain -- full architecture."""
    return Brain(num_sensory=200, num_concept=200)


def make_no_ip() -> Brain:
    """Disable intrinsic plasticity (adaptive thresholds) in concept WTA."""
    brain = Brain(num_sensory=200, num_concept=200)
    brain.regions["concept"].ip_rate = 0
    return brain


def make_no_bcm() -> Brain:
    """Freeze BCM metaplasticity theta to a constant (no sliding threshold)."""
    brain = Brain(num_sensory=200, num_concept=200)
    syn = brain.synapses["sensory_concept"]
    # Force BCM theta initialization so the attribute exists
    if not hasattr(syn, '_bcm_theta'):
        syn._bcm_theta = torch.full((syn.num_post,), 0.01)
        syn._bcm_tau = 5000.0
    # Freeze: replace _bcm_theta with a constant and monkey-patch update
    syn._bcm_theta = torch.full((syn.num_post,), 0.01)
    _original_update = syn.update

    def frozen_bcm_update(pre_spikes, post_spikes, dt=1.0, modulation=1.0):
        saved_theta = syn._bcm_theta.clone()
        _original_update(pre_spikes, post_spikes, dt=dt, modulation=modulation)
        syn._bcm_theta = saved_theta  # restore -- theta never moves

    syn.update = frozen_bcm_update
    return brain


def make_no_lateral() -> Brain:
    """Disable lateral inhibition learning in concept WTA."""
    brain = Brain(num_sensory=200, num_concept=200)
    brain.regions["concept"]._lateral_rate = 0
    return brain


def make_no_modulators() -> Brain:
    """Disable neuromodulatory system (DA/NE/ACh/5HT all frozen at baseline)."""
    brain = Brain(num_sensory=200, num_concept=200)
    brain.modulators.inject = lambda name, amount: None  # no-op
    return brain


def make_no_wm() -> Brain:
    """Zero out working memory connections (concept<->wm weights)."""
    brain = Brain(num_sensory=200, num_concept=200)
    brain.synapses["concept_wm"].weights.zero_()
    brain.synapses["wm_concept"].weights.zero_()
    return brain


def make_no_expansion() -> Brain:
    """Replace random expansion projection with identity-like pass-through."""
    brain = Brain(num_sensory=200, num_concept=200)
    num_exp = brain.num_expansion  # 500
    num_sen = 200
    # Build identity-like matrix: each expansion neuron maps to one sensory neuron
    # (round-robin assignment so every sensory neuron is represented)
    identity_like = torch.zeros(num_exp, num_sen)
    for i in range(num_exp):
        identity_like[i, i % num_sen] = 1.0
    brain._expansion_weights = identity_like
    return brain


CONDITIONS = {
    "Baseline":       make_baseline,
    "No IP":          make_no_ip,
    "No BCM":         make_no_bcm,
    "No Lateral":     make_no_lateral,
    "No Modulators":  make_no_modulators,
    "No WM":          make_no_wm,
    "No Expansion":   make_no_expansion,
}

SEEDS = [42, 123, 456]


# ── Main ────────────────────────────────────────────────────────────────────


def run_ablation():
    print("=" * 70)
    print("  ABLATION STUDY")
    print("  7 conditions x 3 seeds x 4 scenarios x 5000 ticks")
    print("=" * 70)

    results: dict[str, dict] = {}
    t_start = time.time()

    for cond_name, builder in CONDITIONS.items():
        print(f"\n{'─' * 70}")
        print(f"  Condition: {cond_name}")
        print(f"{'─' * 70}")

        seed_results = []
        for seed in SEEDS:
            random.seed(seed)
            torch.manual_seed(seed)
            brain = builder()
            t0 = time.time()
            metrics = measure_discrimination(brain)
            elapsed = time.time() - t0
            seed_results.append(metrics)
            print(
                f"    seed={seed:3d}  distinct={metrics['distinct_groups']}/4  "
                f"overlap={metrics['avg_overlap']:5.1f}%  ({elapsed:.1f}s)"
            )

        # Average across seeds
        avg_distinct = sum(r["distinct_groups"] for r in seed_results) / len(seed_results)
        avg_overlap = sum(r["avg_overlap"] for r in seed_results) / len(seed_results)

        # Per-pair average overlap across seeds
        pair_keys = list(seed_results[0]["pairwise"].keys())
        avg_pairwise = {}
        for pk in pair_keys:
            avg_pairwise[pk] = sum(r["pairwise"][pk] for r in seed_results) / len(seed_results)

        results[cond_name] = {
            "avg_distinct_groups": round(avg_distinct, 2),
            "avg_overlap": round(avg_overlap, 2),
            "avg_pairwise": {k: round(v, 2) for k, v in avg_pairwise.items()},
            "per_seed": [
                {
                    "seed": seed,
                    "distinct_groups": r["distinct_groups"],
                    "avg_overlap": round(r["avg_overlap"], 2),
                    "scenario_top3": r["scenario_top3"],
                }
                for seed, r in zip(SEEDS, seed_results)
            ],
        }

        print(f"    >>> avg distinct={avg_distinct:.2f}/4  avg overlap={avg_overlap:.1f}%")

    total_time = time.time() - t_start

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  ABLATION RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Condition':<18s} {'Distinct (avg)':>14s} {'Overlap (avg)':>14s} {'Delta vs Base':>14s}")
    print(f"  {'─' * 18} {'─' * 14} {'─' * 14} {'─' * 14}")

    baseline_distinct = results["Baseline"]["avg_distinct_groups"]
    baseline_overlap = results["Baseline"]["avg_overlap"]

    for cond_name, data in results.items():
        d = data["avg_distinct_groups"]
        o = data["avg_overlap"]
        delta_d = d - baseline_distinct
        delta_o = o - baseline_overlap
        sign_d = "+" if delta_d >= 0 else ""
        sign_o = "+" if delta_o >= 0 else ""
        print(
            f"  {cond_name:<18s} {d:>10.2f} /4  {o:>10.1f} %  "
            f"{sign_d}{delta_d:>6.2f}d / {sign_o}{delta_o:>5.1f}%"
        )

    print(f"\n  Total time: {total_time:.0f}s")
    print(f"{'=' * 70}")

    # ── Save JSON report ─────────────────────────────────────────────────
    report = {
        "test": "ablation_study",
        "timestamp": time.time(),
        "seeds": SEEDS,
        "ticks_per_scenario": TICKS_PER_SCENARIO,
        "pause_ticks": PAUSE_TICKS,
        "scenarios": list(SCENARIOS.keys()),
        "conditions": results,
        "baseline_distinct": baseline_distinct,
        "baseline_overlap": baseline_overlap,
        "total_time_seconds": round(total_time, 1),
    }

    report_dir = Path("benchmark/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "ablation_study.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  Report saved to {report_path}")

    return results


if __name__ == "__main__":
    run_ablation()
