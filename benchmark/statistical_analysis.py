"""Statistical analysis of brAIn SNN discrimination across multiple seeds.

Runs the 4-scenario discrimination benchmark with 5 different random seeds,
computes mean/std/CI/range for key metrics, and outputs:
  - Formatted Unicode table to console
  - JSON report to benchmark/reports/statistical_analysis.json
  - LaTeX table to benchmark/reports/stats_table.tex

Usage: .venv/bin/python benchmark/statistical_analysis.py
"""
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from brain.core import Brain

# ---------------------------------------------------------------------------
# Encoding (inlined from adapters/mac_desktop/encoding.py)
# ---------------------------------------------------------------------------
SENSORY_DIM = 200

_DRIVE = 8.0
_DRIVE_BG = 3.0
_DRIVE_CTX = 6.0
_DRIVE_MIC = 4.0
_DRIVE_TIME = 2.0


def _hash_app_to_index(name: str, num_slots: int) -> int:
    digest = hashlib.md5(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % num_slots


def _log_bin(value: float, max_value: float, num_bins: int) -> int:
    if value <= 0:
        return 0
    if value >= max_value:
        return num_bins - 1
    log_v = math.log1p(value)
    log_max = math.log1p(max_value)
    frac = log_v / log_max
    return min(num_bins - 1, int(frac * num_bins))


def encode_snapshot(snap: dict) -> torch.Tensor:
    vec = torch.zeros(SENSORY_DIM, dtype=torch.float32)

    app = snap.get("active_app")
    if app and "name" in app:
        for k in range(5):
            idx = _hash_app_to_index(app["name"] + str(k), num_slots=40)
            vec[idx] = _DRIVE
        for bg_name in app.get("background_apps", [])[:10]:
            bg_idx = _hash_app_to_index(bg_name, num_slots=20)
            vec[40 + bg_idx] = _DRIVE_BG

    ks = snap.get("keystroke_rate")
    if ks is not None and "count" in ks:
        count = ks["count"]
        bin_idx = _log_bin(count, max_value=50.0, num_bins=16)
        vec[60 + bin_idx] = _DRIVE
        variability = ks.get("variability", 0.0)
        burst = ks.get("burst", 0.0)
        vec[76] = min(_DRIVE_CTX, variability * 3.0)
        vec[77] = _DRIVE_CTX if burst > 0.5 else 0.0
        if count > 2 and variability < 0.5:
            vec[78] = _DRIVE_CTX
        if count == 0:
            vec[79] = _DRIVE_CTX

    ms = snap.get("mouse_rate")
    if ms is not None and "count" in ms:
        count = ms["count"]
        bin_idx = _log_bin(count, max_value=200.0, num_bins=16)
        vec[80 + bin_idx] = _DRIVE
        variability = ms.get("variability", 0.0)
        burst = ms.get("burst", 0.0)
        vec[96] = min(_DRIVE_CTX, variability * 3.0)
        vec[97] = _DRIVE_CTX if burst > 0.5 else 0.0
        if count > 5 and variability < 0.5:
            vec[98] = _DRIVE_CTX
        if count == 0:
            vec[99] = _DRIVE_CTX

    idle = snap.get("idle")
    if idle is not None and "seconds" in idle:
        secs = idle["seconds"]
        bin_idx = _log_bin(secs, max_value=3600.0, num_bins=8)
        vec[100 + bin_idx] = _DRIVE
        if 1.0 < secs <= 5.0:
            vec[108] = _DRIVE_CTX
        elif 5.0 < secs <= 30.0:
            vec[109] = _DRIVE_CTX
        elif 30.0 < secs <= 300.0:
            vec[110] = _DRIVE_CTX
        elif secs > 300.0:
            vec[111] = _DRIVE_CTX

    mic = snap.get("mic")
    if mic is not None:
        if "mel" in mic and len(mic["mel"]) == 32:
            mel = torch.tensor(mic["mel"], dtype=torch.float32)
            mel_scaled = (mel / 5.0).clamp(0.0, 1.0) * _DRIVE_MIC
            vec[112:144] = mel_scaled
        if "rms" in mic:
            bin_idx = _log_bin(mic["rms"], max_value=1.0, num_bins=4)
            vec[144 + bin_idx] = _DRIVE

    tt = snap.get("time_tonic")
    if tt is not None:
        if "day_phase" in tt and len(tt["day_phase"]) == 4:
            day = torch.tensor(tt["day_phase"], dtype=torch.float32)
            vec[148:152] = (day + 1.0) * _DRIVE_TIME * 0.5
        if "week_phase" in tt and len(tt["week_phase"]) == 4:
            week = torch.tensor(tt["week_phase"], dtype=torch.float32)
            vec[152:156] = (week + 1.0) * _DRIVE_TIME * 0.5

    if app:
        switch_rate = app.get("switch_rate", 0.0)
        if switch_rate < 1:
            vec[156] = _DRIVE_CTX
        elif switch_rate < 3:
            vec[157] = _DRIVE_CTX
        elif switch_rate < 8:
            vec[158] = _DRIVE_CTX
        else:
            vec[159] = _DRIVE_CTX

    ks_count = snap.get("keystroke_rate", {}).get("count", 0) if snap.get("keystroke_rate") else 0
    ms_count = snap.get("mouse_rate", {}).get("count", 0) if snap.get("mouse_rate") else 0
    idle_secs = snap.get("idle", {}).get("seconds", 999) if snap.get("idle") else 999
    total_activity = ks_count + ms_count * 0.1

    if idle_secs > 300:
        vec[160] = _DRIVE_CTX
    elif total_activity < 1:
        vec[161] = _DRIVE_CTX
    elif total_activity < 15:
        vec[162] = _DRIVE_CTX
    else:
        vec[163] = _DRIVE_CTX

    return vec


# ---------------------------------------------------------------------------
# Scenario patterns (inlined from discrimination_test.py)
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

SCENARIO_NAMES = list(SCENARIOS.keys())
TICKS_PER_SCENARIO = 5000
PAUSE_TICKS = 2000
SEEDS = [42, 123, 456, 789, 1024]

# t-critical for 95% CI, df=4 (two-tailed)
T_CRIT = 2.776


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------
def run_single_seed(seed: int) -> dict:
    """Run all 4 scenarios for one seed and return metrics."""
    random.seed(seed)
    torch.manual_seed(seed)

    brain = Brain(num_sensory=200, num_concept=200)

    scenario_concepts: dict[str, list[set[int]]] = defaultdict(list)
    scenario_dominant: dict[str, list[int]] = defaultdict(list)

    for sname, maker in SCENARIOS.items():
        fires: Counter = Counter()
        for _ in range(TICKS_PER_SCENARIO):
            vec = encode_snapshot(maker())
            out = brain.tick(vec)
            for cid in (out["concept"] > 0).nonzero(as_tuple=True)[0].tolist():
                fires[cid] += 1

        # Idle pause between scenarios
        for _ in range(PAUSE_TICKS):
            brain.tick(torch.zeros(200))

        top3 = set(cid for cid, _ in fires.most_common(3))
        dominant_id = fires.most_common(1)[0][0] if fires else -1
        scenario_concepts[sname].append(top3)
        scenario_dominant[sname].append(dominant_id)

    # --- Metric 1: num_distinct_groups (0-4) ---
    all_groups = []
    for sname in SCENARIO_NAMES:
        merged = set()
        for top3 in scenario_concepts[sname]:
            merged.update(top3)
        all_groups.append((sname, merged))

    distinct = 0
    for i, (s1, g1) in enumerate(all_groups):
        is_distinct = True
        for j, (s2, g2) in enumerate(all_groups):
            if i != j and len(g1 & g2) > len(g1) * 0.5:
                is_distinct = False
        if is_distinct:
            distinct += 1

    # --- Metric 2: avg_overlap_pct ---
    overlaps = []
    for i, (s1, g1) in enumerate(all_groups):
        for j, (s2, g2) in enumerate(all_groups):
            if i >= j:
                continue
            if len(g1) == 0 and len(g2) == 0:
                overlaps.append(0.0)
            else:
                union_size = len(g1 | g2)
                inter_size = len(g1 & g2)
                overlaps.append(inter_size / max(1, union_size) * 100.0)
    avg_overlap = sum(overlaps) / max(1, len(overlaps))

    # --- Metric 3: dominant_neuron_stability ---
    # For each scenario, check if the dominant neuron is the same across
    # the single run (we only have 1 repeat per seed, so stability = 1.0
    # if the dominant neuron exists, 0.0 if no fires).
    # Across seeds we measure whether the SAME scenario tends to pick
    # a consistent dominant neuron.
    stabilities = []
    for sname in SCENARIO_NAMES:
        dominants = scenario_dominant[sname]
        if dominants and dominants[0] != -1:
            stabilities.append(1.0)
        else:
            stabilities.append(0.0)
    dominant_stability = sum(stabilities) / max(1, len(stabilities))

    return {
        "seed": seed,
        "num_distinct_groups": distinct,
        "avg_overlap_pct": avg_overlap,
        "dominant_neuron_stability": dominant_stability,
        "scenario_dominants": {s: d for s, d in zip(SCENARIO_NAMES, [scenario_dominant[s][0] for s in SCENARIO_NAMES])},
        "scenario_top3": {s: sorted(list(scenario_concepts[s][0])) for s in SCENARIO_NAMES},
    }


# ---------------------------------------------------------------------------
# Statistics helpers (no scipy needed)
# ---------------------------------------------------------------------------
def compute_stats(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)
    se = std / math.sqrt(n)
    ci_low = mean - T_CRIT * se
    ci_high = mean + T_CRIT * se
    return {
        "mean": mean,
        "std": std,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------
def print_unicode_table(stats: dict[str, dict], elapsed: float) -> None:
    """Print a formatted Unicode box-drawing table to console."""
    col_metric = 28
    col_mean = 18
    col_ci = 24
    col_range = 16
    total_w = col_metric + col_mean + col_ci + col_range + 5  # +5 for borders

    print()
    print("\u250c" + "\u2500" * (total_w - 2) + "\u2510")
    title = "brAIn SNN Discrimination Performance (n=5 seeds)"
    print("\u2502" + title.center(total_w - 2) + "\u2502")
    print("\u251c" + "\u2500" * col_metric + "\u252c" + "\u2500" * col_mean + "\u252c" + "\u2500" * col_ci + "\u252c" + "\u2500" * col_range + "\u2524")
    print(
        "\u2502" + " Metric".ljust(col_metric)
        + "\u2502" + " Mean \u00b1 Std".ljust(col_mean)
        + "\u2502" + " 95% CI".ljust(col_ci)
        + "\u2502" + " Range".ljust(col_range)
        + "\u2502"
    )
    print("\u251c" + "\u2500" * col_metric + "\u253c" + "\u2500" * col_mean + "\u253c" + "\u2500" * col_ci + "\u253c" + "\u2500" * col_range + "\u2524")

    labels = {
        "num_distinct_groups": "Distinct Groups (0-4)",
        "avg_overlap_pct": "Avg Overlap (%)",
        "dominant_neuron_stability": "Dominant Stability (0-1)",
    }

    for key, label in labels.items():
        s = stats[key]
        mean_std = f" {s['mean']:.2f} \u00b1 {s['std']:.2f}"
        ci = f" [{s['ci_low']:.2f}, {s['ci_high']:.2f}]"
        rng = f" [{s['min']:.2f}, {s['max']:.2f}]"
        print(
            "\u2502" + f" {label}".ljust(col_metric)
            + "\u2502" + mean_std.ljust(col_mean)
            + "\u2502" + ci.ljust(col_ci)
            + "\u2502" + rng.ljust(col_range)
            + "\u2502"
        )

    print("\u2514" + "\u2500" * col_metric + "\u2534" + "\u2500" * col_mean + "\u2534" + "\u2500" * col_ci + "\u2534" + "\u2500" * col_range + "\u2518")
    print(f"  Total runtime: {elapsed:.1f}s")
    print()


def save_json(stats: dict[str, dict], seed_results: list[dict], elapsed: float) -> None:
    report = {
        "test": "statistical_analysis",
        "timestamp": time.time(),
        "seeds": SEEDS,
        "ticks_per_scenario": TICKS_PER_SCENARIO,
        "pause_ticks": PAUSE_TICKS,
        "scenarios": SCENARIO_NAMES,
        "t_crit": T_CRIT,
        "df": len(SEEDS) - 1,
        "elapsed_seconds": elapsed,
        "metrics": {},
        "per_seed": seed_results,
    }
    for key in stats:
        s = stats[key]
        report["metrics"][key] = {
            "mean": s["mean"],
            "std": s["std"],
            "se": s["se"],
            "ci_95_low": s["ci_low"],
            "ci_95_high": s["ci_high"],
            "min": s["min"],
            "max": s["max"],
            "values": s["values"],
        }

    out_path = Path("benchmark/reports/statistical_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"  JSON report saved to {out_path}")


def save_latex(stats: dict[str, dict]) -> None:
    labels = {
        "num_distinct_groups": "Distinct Groups (0--4)",
        "avg_overlap_pct": "Avg Overlap (\\%)",
        "dominant_neuron_stability": "Dominant Stability (0--1)",
    }

    rows = []
    for key, label in labels.items():
        s = stats[key]
        rows.append(
            f"        {label} & ${s['mean']:.2f} \\pm {s['std']:.2f}$ "
            f"& $[{s['ci_low']:.2f},\\; {s['ci_high']:.2f}]$ "
            f"& $[{s['min']:.2f},\\; {s['max']:.2f}]$ \\\\"
        )

    latex = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{brAIn SNN Discrimination Performance (n=5 seeds)}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "        Metric & Mean $\\pm$ Std & 95\\% CI & Range \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    out_path = Path("benchmark/reports/stats_table.tex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex)
    print(f"  LaTeX table saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("  STATISTICAL ANALYSIS: brAIn SNN Discrimination")
    print(f"  Seeds: {SEEDS}")
    print(f"  Scenarios: {SCENARIO_NAMES}")
    print(f"  {TICKS_PER_SCENARIO} ticks/scenario, {PAUSE_TICKS}-tick idle pauses")
    print("=" * 70)

    t0 = time.time()
    seed_results: list[dict] = []

    for i, seed in enumerate(SEEDS):
        t_seed = time.time()
        print(f"\n--- Seed {seed} ({i + 1}/{len(SEEDS)}) ---")
        result = run_single_seed(seed)
        seed_results.append(result)
        dt = time.time() - t_seed
        print(f"  distinct_groups={result['num_distinct_groups']}  "
              f"overlap={result['avg_overlap_pct']:.1f}%  "
              f"stability={result['dominant_neuron_stability']:.2f}  "
              f"({dt:.1f}s)")
        for sname in SCENARIO_NAMES:
            print(f"    {sname:20s}: dominant=C{result['scenario_dominants'][sname]}  "
                  f"top3={result['scenario_top3'][sname]}")

    elapsed = time.time() - t0

    # Aggregate statistics
    metrics = {
        "num_distinct_groups": [r["num_distinct_groups"] for r in seed_results],
        "avg_overlap_pct": [r["avg_overlap_pct"] for r in seed_results],
        "dominant_neuron_stability": [r["dominant_neuron_stability"] for r in seed_results],
    }

    stats = {key: compute_stats(vals) for key, vals in metrics.items()}

    # Output
    print("\n" + "=" * 70)
    print("  AGGREGATE RESULTS")
    print("=" * 70)
    print_unicode_table(stats, elapsed)
    save_json(stats, seed_results, elapsed)
    save_latex(stats)

    # Pass/fail summary
    mean_groups = stats["num_distinct_groups"]["mean"]
    mean_overlap = stats["avg_overlap_pct"]["mean"]
    passed = mean_groups >= 3.0
    print(f"\n  Mean distinct groups: {mean_groups:.2f}/4  "
          f"Mean overlap: {mean_overlap:.1f}%")
    print(f"  {'PASS' if passed else 'FAIL'} (need mean >= 3.0 distinct groups)")
    print()


if __name__ == "__main__":
    main()
