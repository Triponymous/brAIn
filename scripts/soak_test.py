#!/usr/bin/env python3
"""Soak test: run the Brain for 50K ticks with simulated daily activity.

Validates stability, correctness of modulators, spike propagation,
persistence round-trip, and episode logging over a long run.

Usage:
    python scripts/soak_test.py [--ticks N]
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import tempfile
import time
from pathlib import Path

import torch

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.core import Brain
from brain.persistence import save_brain, load_brain
from bridge.exporter import BrainStateExporter
from bridge.episode_log import EpisodeLogger
from adapters.mac_desktop.encoding import encode_snapshot


# ---------------------------------------------------------------------------
# Schedule phases — each returns a noisy sensor-bus snapshot dict
# ---------------------------------------------------------------------------

def _morning_emails(tick: int) -> dict:
    """Low keys, medium mouse, Chrome."""
    return {
        "active_app": {"name": "Google Chrome", "background_apps": ["Mail"], "switch_rate": random.uniform(0.5, 2.0)},
        "keystroke_rate": {"count": random.randint(1, 6), "variability": random.uniform(0.1, 0.4), "burst": random.uniform(0.0, 0.3)},
        "mouse_rate": {"count": random.randint(5, 20), "variability": random.uniform(0.1, 0.5), "burst": random.uniform(0.0, 0.2)},
        "idle": {"seconds": random.uniform(0, 3)},
        "mic": {"rms": random.uniform(0.001, 0.02), "mel": [random.uniform(0, 0.5) for _ in range(32)]},
        "time_tonic": {
            "day_phase": [math.sin(0.3), math.cos(0.3), math.sin(0.6), math.cos(0.6)],
            "week_phase": [math.sin(0.1), math.cos(0.1), math.sin(0.2), math.cos(0.2)],
        },
    }


def _coding(tick: int) -> dict:
    """High keys, low mouse, VSCode."""
    return {
        "active_app": {"name": "Code", "background_apps": ["Terminal", "Google Chrome"], "switch_rate": random.uniform(0.2, 1.0)},
        "keystroke_rate": {"count": random.randint(15, 45), "variability": random.uniform(0.05, 0.3), "burst": random.uniform(0.0, 0.2)},
        "mouse_rate": {"count": random.randint(1, 8), "variability": random.uniform(0.1, 0.3), "burst": 0.0},
        "idle": {"seconds": random.uniform(0, 2)},
        "mic": {"rms": random.uniform(0.0, 0.008), "mel": [random.uniform(0, 0.2) for _ in range(32)]},
        "time_tonic": {
            "day_phase": [math.sin(0.5), math.cos(0.5), math.sin(1.0), math.cos(1.0)],
            "week_phase": [math.sin(0.1), math.cos(0.1), math.sin(0.2), math.cos(0.2)],
        },
    }


def _meeting(tick: int) -> dict:
    """No keys, no mouse, Zoom, high mic."""
    return {
        "active_app": {"name": "zoom.us", "background_apps": ["Slack"], "switch_rate": random.uniform(0.0, 0.5)},
        "keystroke_rate": {"count": 0, "variability": 0.0, "burst": 0.0},
        "mouse_rate": {"count": random.randint(0, 3), "variability": random.uniform(0.0, 0.2), "burst": 0.0},
        "idle": {"seconds": random.uniform(0, 10)},
        "mic": {"rms": random.uniform(0.05, 0.3), "mel": [random.uniform(0.5, 3.0) for _ in range(32)]},
        "time_tonic": {
            "day_phase": [math.sin(0.8), math.cos(0.8), math.sin(1.6), math.cos(1.6)],
            "week_phase": [math.sin(0.1), math.cos(0.1), math.sin(0.2), math.cos(0.2)],
        },
    }


def _lunch_break(tick: int) -> dict:
    """No activity, high idle."""
    return {
        "active_app": {"name": "Finder", "background_apps": [], "switch_rate": 0.0},
        "keystroke_rate": {"count": 0, "variability": 0.0, "burst": 0.0},
        "mouse_rate": {"count": 0, "variability": 0.0, "burst": 0.0},
        "idle": {"seconds": random.uniform(300, 1800)},
        "mic": {"rms": random.uniform(0.0, 0.005), "mel": [random.uniform(0, 0.1) for _ in range(32)]},
        "time_tonic": {
            "day_phase": [math.sin(1.0), math.cos(1.0), math.sin(2.0), math.cos(2.0)],
            "week_phase": [math.sin(0.1), math.cos(0.1), math.sin(0.2), math.cos(0.2)],
        },
    }


def _design_work(tick: int) -> dict:
    """Low keys, high mouse, Figma."""
    return {
        "active_app": {"name": "Figma", "background_apps": ["Google Chrome", "Slack"], "switch_rate": random.uniform(0.3, 1.5)},
        "keystroke_rate": {"count": random.randint(0, 5), "variability": random.uniform(0.1, 0.5), "burst": random.uniform(0.0, 0.2)},
        "mouse_rate": {"count": random.randint(30, 120), "variability": random.uniform(0.1, 0.4), "burst": random.uniform(0.0, 0.3)},
        "idle": {"seconds": random.uniform(0, 4)},
        "mic": {"rms": random.uniform(0.0, 0.015), "mel": [random.uniform(0, 0.3) for _ in range(32)]},
        "time_tonic": {
            "day_phase": [math.sin(1.3), math.cos(1.3), math.sin(2.6), math.cos(2.6)],
            "week_phase": [math.sin(0.1), math.cos(0.1), math.sin(0.2), math.cos(0.2)],
        },
    }


def _stress_period(tick: int) -> dict:
    """High keys, fast switching, high variability."""
    apps = ["Code", "Google Chrome", "Terminal", "Slack", "zoom.us", "Figma"]
    return {
        "active_app": {"name": random.choice(apps), "background_apps": random.sample(apps, 3), "switch_rate": random.uniform(8, 20)},
        "keystroke_rate": {"count": random.randint(10, 50), "variability": random.uniform(2.0, 6.0), "burst": random.uniform(0.5, 1.0)},
        "mouse_rate": {"count": random.randint(10, 80), "variability": random.uniform(2.0, 5.0), "burst": random.uniform(0.5, 1.0)},
        "idle": {"seconds": random.uniform(0, 1)},
        "mic": {"rms": random.uniform(0.01, 0.1), "mel": [random.uniform(0.2, 2.0) for _ in range(32)]},
        "time_tonic": {
            "day_phase": [math.sin(1.6), math.cos(1.6), math.sin(3.2), math.cos(3.2)],
            "week_phase": [math.sin(0.1), math.cos(0.1), math.sin(0.2), math.cos(0.2)],
        },
    }


# Ordered schedule: each phase gets ~1/6 of total ticks
PHASES = [
    ("Morning emails", _morning_emails),
    ("Coding", _coding),
    ("Meeting", _meeting),
    ("Lunch break", _lunch_break),
    ("Design work", _design_work),
    ("Stress period", _stress_period),
]


def get_phase(tick: int, total_ticks: int):
    """Return (phase_name, snapshot_fn) for the current tick."""
    phase_len = total_ticks // len(PHASES)
    idx = min(tick // phase_len, len(PHASES) - 1)
    return PHASES[idx]


# ---------------------------------------------------------------------------
# Main soak test
# ---------------------------------------------------------------------------

def run_soak_test(total_ticks: int) -> bool:
    print(f"=== Soak Test: {total_ticks} ticks, {len(PHASES)} phases ===\n")
    random.seed(42)
    torch.manual_seed(42)

    # 1. Create Brain with default params
    brain = Brain(num_sensory=200, num_concept=200)

    # 2. Create exporter and episode logger
    exporter = BrainStateExporter(brain)
    tmp_dir = Path(tempfile.mkdtemp(prefix="soak_"))
    episode_db = tmp_dir / "episodes.db"
    logger = EpisodeLogger(episode_db)

    t0 = time.time()
    total_concept_spikes = 0
    phase_ticks: dict[str, int] = {}

    # 3. Run ticks
    for t in range(total_ticks):
        phase_name, snap_fn = get_phase(t, total_ticks)
        phase_ticks[phase_name] = phase_ticks.get(phase_name, 0) + 1

        snapshot = snap_fn(t)
        input_current = encode_snapshot(snapshot)
        result = brain.tick(input_current)

        concept_spikes = result["concept"]
        total_concept_spikes += int(concept_spikes.sum().item())

        # Record with context for auto-correlation
        exporter.record_spikes_with_context(concept_spikes, snapshot)

        # Log episode every 1000 ticks
        if t % 1000 == 0:
            mods = brain.modulators.snapshot()
            active = exporter.snapshot().get("active_concepts", [])
            logger.log(
                tick=brain.tick_count,
                modulators=mods,
                active_concepts=active,
                sensor_summary={"app": snapshot.get("active_app", {}).get("name", "")},
                sleep_mode=False,
            )

        # Progress every 10K ticks
        if (t + 1) % 10000 == 0:
            elapsed = time.time() - t0
            mods = brain.modulators.snapshot()
            print(f"  [{t + 1:>6d}/{total_ticks}] {elapsed:5.1f}s | "
                  f"phase={phase_name:<16s} | "
                  f"DA={mods.get('DA', 0):+.4f} NE={mods.get('NE', 0):+.4f} "
                  f"ACh={mods.get('ACh', 0):+.4f} 5HT={mods.get('5HT', 0):+.4f} | "
                  f"concept_spikes_total={total_concept_spikes}")

    elapsed = time.time() - t0
    print(f"\n  Completed {total_ticks} ticks in {elapsed:.1f}s "
          f"({total_ticks / elapsed:.0f} ticks/s)\n")

    # --------------- VALIDATION ---------------
    failures = []

    # Check 1: No NaN in weights or spikes
    print("CHECK 1: No NaN in weights or spike accumulators...")
    for name, syn in brain.synapses.items():
        if torch.isnan(syn.weights).any():
            failures.append(f"  NaN found in synapse '{name}' weights")
    if torch.isnan(brain.concept_spike_accum).any():
        failures.append("  NaN found in concept_spike_accum")
    for name, region in brain.regions.items():
        if torch.isnan(region.membrane).any():
            failures.append(f"  NaN found in region '{name}' membrane")
    if not failures:
        print("  PASSED\n")
    else:
        for f in failures:
            print(f)
        print()

    # Check 2: Modulators in [-1, 1]
    print("CHECK 2: Modulators in [-1, 1] range...")
    mods = brain.modulators.snapshot()
    for name, val in mods.items():
        if val < -1.0 or val > 1.0:
            failures.append(f"  Modulator '{name}' out of range: {val}")
    if not any("Modulator" in f for f in failures):
        print(f"  PASSED (DA={mods.get('DA', 0):.4f}, NE={mods.get('NE', 0):.4f}, "
              f"ACh={mods.get('ACh', 0):.4f}, 5HT={mods.get('5HT', 0):.4f})\n")
    else:
        for f in failures:
            if "Modulator" in f:
                print(f)
        print()

    # Check 3: At least some concepts fired
    print("CHECK 3: Concept spikes accumulated...")
    accum_sum = float(brain.concept_spike_accum.sum().item())
    # The rolling accum decays (0.95^N), so for long runs it approaches zero.
    # Use total_concept_spikes (non-decayed counter) as the primary check.
    if total_concept_spikes <= 0:
        failures.append(f"  total concept spikes is {total_concept_spikes} (expected > 0)")
        print(f"  FAILED: total_concept_spikes={total_concept_spikes}\n")
    else:
        print(f"  PASSED (total_spikes={total_concept_spikes}, rolling_accum={accum_sum:.4f})\n")

    # Check 4: Persistence round-trip
    print("CHECK 4: Persistence (save -> load -> verify)...")
    save_path = tmp_dir / "brain_soak.db"
    save_brain(brain, save_path)
    brain2 = load_brain(save_path)
    if brain2.tick_count != brain.tick_count:
        failures.append(f"  Tick count mismatch: saved={brain.tick_count}, loaded={brain2.tick_count}")
        print(f"  FAILED: tick mismatch\n")
    else:
        # Also verify a synapse survived
        w_orig = brain.synapses["sensory_concept"].weights
        w_loaded = brain2.synapses["sensory_concept"].weights
        diff = float((w_orig - w_loaded).abs().max().item())
        if diff > 1e-5:
            failures.append(f"  Weight drift after load: max_diff={diff}")
            print(f"  FAILED: weight drift {diff}\n")
        else:
            print(f"  PASSED (tick_count={brain2.tick_count}, weight_diff={diff:.2e})\n")

    # Check 5: Episode logger has entries
    print("CHECK 5: Episode logger has entries...")
    episodes = logger.query(last_n=1000)
    logger.close()
    if len(episodes) == 0:
        failures.append("  Episode logger has 0 entries")
        print("  FAILED: no episodes\n")
    else:
        print(f"  PASSED ({len(episodes)} episodes logged)\n")

    # Check 6: Weight stats are sane
    print("CHECK 6: Weight stats sanity (no all-zero rows, no all-max rows)...")
    for syn_name, syn in brain.synapses.items():
        w = syn.weights
        row_sums = w.sum(dim=1)
        # Check for all-zero rows
        zero_rows = int((row_sums == 0).sum().item())
        if zero_rows > 0:
            failures.append(f"  Synapse '{syn_name}' has {zero_rows} all-zero rows")
        # Check for all-max rows (every weight at w_max)
        w_max = syn.w_max
        max_possible_sum = w.shape[1] * w_max
        saturated_rows = int((row_sums >= max_possible_sum * 0.99).sum().item())
        if saturated_rows > 0:
            failures.append(f"  Synapse '{syn_name}' has {saturated_rows} nearly-saturated rows")
        # Print stats
        mean_w = float(w.mean().item())
        std_w = float(w.std().item())
        min_w = float(w.min().item())
        max_w = float(w.max().item())
        print(f"  {syn_name:>25s}: mean={mean_w:.4f} std={std_w:.4f} "
              f"min={min_w:.4f} max={max_w:.4f} "
              f"zero_rows={zero_rows} saturated_rows={saturated_rows}")

    if not any("Synapse" in f for f in failures):
        print("  PASSED\n")
    else:
        for f in failures:
            if "Synapse" in f:
                print(f)
        print()

    # --------------- SUMMARY ---------------
    print("=" * 60)
    print(f"Phase distribution: {phase_ticks}")
    print(f"Total concept spikes: {total_concept_spikes}")
    print(f"Wall time: {elapsed:.1f}s")
    print()

    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
        print("\nSOAK TEST FAILED")
        return False
    else:
        print("ALL CHECKS PASSED")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain soak test")
    parser.add_argument("--ticks", type=int, default=50000, help="Number of ticks to run")
    args = parser.parse_args()

    success = run_soak_test(args.ticks)
    sys.exit(0 if success else 1)
