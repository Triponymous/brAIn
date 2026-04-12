#!/usr/bin/env python3
"""Diagnostic script: verify the SNN brain is actually learning.

Usage:
    python scripts/check_learning.py
    python scripts/check_learning.py /path/to/other/checkpoint.sqlite
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `brain.*` imports work.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from brain.persistence import load_brain

# ── Defaults ──
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "braind.sqlite"

# Initial weight parameters (from Brain.__init__ / _make_stdp)
MAIN_W_INIT = 0.3
MAIN_W_INIT_STD = 0.15
FEEDBACK_W_INIT = 0.1
FEEDBACK_W_INIT_STD = 0.05
EXPANSION_SPARSITY = 0.10

# Thresholds for pass/fail
SPECIALIZATION_VAR_THRESHOLD = 0.01  # per-neuron weight variance to count as "specialized"
WEIGHT_DRIFT_THRESHOLD = 0.02        # mean must differ from init by at least this
EXPANSION_SPARSITY_TOL = 0.03        # expansion sparsity must be within this of 10%

results: list[tuple[str, bool, str]] = []  # (name, passed, detail)


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def weight_stats(w: torch.Tensor, label: str) -> dict:
    """Print and return weight statistics for a synapse matrix."""
    total = w.numel()
    near_floor = (w <= 0.015).sum().item()
    near_ceil = (w >= 0.99).sum().item()
    sparsity = near_floor / total if total > 0 else 0.0
    stats = {
        "mean": w.mean().item(),
        "std": w.std().item(),
        "min": w.min().item(),
        "max": w.max().item(),
        "shape": tuple(w.shape),
        "sparsity": sparsity,
        "near_floor": near_floor,
        "near_ceil": near_ceil,
    }
    print(f"  {label} weights {stats['shape']}:")
    print(f"    mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
          f"min={stats['min']:.4f}  max={stats['max']:.4f}")
    print(f"    sparsity (<=0.015): {sparsity:.1%} ({near_floor}/{total})")
    print(f"    saturated (>=0.99): {near_ceil}/{total}")
    return stats


def check_sensory_concept(brain) -> None:
    section("1. SENSORY_CONCEPT synapse (expansion -> concept, STDP+BCM)")
    syn = brain.synapses["sensory_concept"]
    w = syn.weights  # [num_concept, num_expansion]
    stats = weight_stats(w, "sensory_concept")

    # Check drift from initial values
    mean_drift = abs(stats["mean"] - MAIN_W_INIT)
    std_drift = abs(stats["std"] - MAIN_W_INIT_STD)
    print(f"\n  Drift from init (w_init={MAIN_W_INIT}, w_init_std={MAIN_W_INIT_STD}):")
    print(f"    mean drift: {mean_drift:.4f}  (threshold: {WEIGHT_DRIFT_THRESHOLD})")
    print(f"    std drift:  {std_drift:.4f}")
    has_drifted = mean_drift > WEIGHT_DRIFT_THRESHOLD or std_drift > WEIGHT_DRIFT_THRESHOLD
    drift_status = "DRIFTED (learning occurred)" if has_drifted else "NO DRIFT (no learning detected)"
    print(f"    --> {drift_status}")
    results.append(("sensory_concept weight drift", has_drifted,
                     f"mean_drift={mean_drift:.4f}, std_drift={std_drift:.4f}"))

    # Per-neuron specialization: variance of weights across inputs
    per_neuron_var = w.var(dim=1)  # variance across pre-synaptic weights for each post neuron
    specialized_mask = per_neuron_var > SPECIALIZATION_VAR_THRESHOLD
    num_specialized = specialized_mask.sum().item()
    num_concept = w.shape[0]
    print(f"\n  Neuron specialization (weight variance > {SPECIALIZATION_VAR_THRESHOLD}):")
    print(f"    {num_specialized}/{num_concept} neurons specialized "
          f"({num_specialized / num_concept:.0%})")
    has_specialization = num_specialized > 0
    results.append(("sensory_concept specialization", has_specialization,
                     f"{num_specialized}/{num_concept} neurons"))

    # Top 5 most specialized neurons
    top_k = min(5, num_concept)
    top_var, top_idx = per_neuron_var.topk(top_k)
    print(f"\n  Top {top_k} most specialized neurons:")
    for rank, (idx, var_val) in enumerate(zip(top_idx.tolist(), top_var.tolist())):
        row = w[idx]
        row_min = row.min().item()
        row_max = row.max().item()
        row_mean = row.mean().item()
        strong_inputs = (row > 0.5).sum().item()
        weak_inputs = (row < 0.1).sum().item()
        print(f"    #{rank + 1}  neuron {idx}: var={var_val:.4f}  "
              f"mean={row_mean:.4f}  range=[{row_min:.3f}, {row_max:.3f}]  "
              f"strong(>0.5)={strong_inputs}  weak(<0.1)={weak_inputs}")

    # BCM theta check
    if hasattr(syn, '_bcm_theta'):
        theta = syn._bcm_theta
        print(f"\n  BCM metaplasticity theta:")
        print(f"    mean={theta.mean().item():.6f}  std={theta.std().item():.6f}  "
              f"min={theta.min().item():.6f}  max={theta.max().item():.6f}")


def check_concept_tracker(brain) -> None:
    section("2. CONCEPT TRACKER (cluster stability)")
    ct = brain.concept_tracker
    num_clusters = len(ct.centroids)
    print(f"  Total clusters: {num_clusters}/{ct.max_clusters}")
    print(f"  Similarity threshold: {ct.similarity_threshold}")
    print(f"  Snapshot interval: {ct.snapshot_interval}")

    has_clusters = num_clusters > 0
    results.append(("concept_tracker has clusters", has_clusters,
                     f"{num_clusters} clusters"))

    if num_clusters == 0:
        print("  (no clusters yet)")
        return

    # Print all clusters
    print(f"\n  {'ID':>3}  {'Label':<25} {'Count':>6}  {'Last Seen':>10}")
    print(f"  {'-' * 3}  {'-' * 25} {'-' * 6}  {'-' * 10}")
    total_count = 0
    labeled_count = 0
    for i in range(num_clusters):
        label = ct.cluster_labels[i] or "(unlabeled)"
        count = ct.cluster_counts[i]
        last_seen = ct.cluster_last_seen[i]
        active = " <-- ACTIVE" if i == ct._current_cluster else ""
        print(f"  {i:>3}  {label:<25} {count:>6}  {last_seen:>10}{active}")
        total_count += count
        if ct.cluster_labels[i]:
            labeled_count += 1

    print(f"\n  Total observations: {total_count}")
    print(f"  Labeled clusters: {labeled_count}/{num_clusters}")

    # Stability check: ratio of total observations to cluster count
    # A stable system has high obs/cluster ratio (clusters aren't churning)
    if num_clusters > 0:
        obs_per_cluster = total_count / num_clusters
        print(f"  Avg observations per cluster: {obs_per_cluster:.1f}")
        is_stable = obs_per_cluster >= 3.0
        stability_msg = ("STABLE (clusters are reused)" if is_stable
                         else "UNSTABLE (many new clusters with few observations)")
        print(f"  --> {stability_msg}")
        results.append(("concept_tracker stability", is_stable,
                         f"avg obs/cluster={obs_per_cluster:.1f}"))

    # Debug info
    debug = getattr(ct, '_last_debug', {})
    if debug:
        print(f"\n  Last assignment debug:")
        for k, v in debug.items():
            print(f"    {k}: {v}")


def check_wm_concept(brain) -> None:
    section("3. WM_CONCEPT feedback synapse (wm -> concept)")
    syn = brain.synapses["wm_concept"]
    w = syn.weights  # [num_concept, num_wm]
    stats = weight_stats(w, "wm_concept")

    # Check drift from initial values (low init: mean=0.1, std=0.05)
    mean_drift = abs(stats["mean"] - FEEDBACK_W_INIT)
    std_drift = abs(stats["std"] - FEEDBACK_W_INIT_STD)
    print(f"\n  Drift from init (w_init={FEEDBACK_W_INIT}, w_init_std={FEEDBACK_W_INIT_STD}):")
    print(f"    mean drift: {mean_drift:.4f}")
    print(f"    std drift:  {std_drift:.4f}")
    has_drifted = mean_drift > WEIGHT_DRIFT_THRESHOLD or std_drift > WEIGHT_DRIFT_THRESHOLD
    drift_status = "DRIFTED (learning occurred)" if has_drifted else "NO DRIFT (no learning detected)"
    print(f"    --> {drift_status}")
    results.append(("wm_concept weight drift", has_drifted,
                     f"mean_drift={mean_drift:.4f}, std_drift={std_drift:.4f}"))


def check_concept_wm(brain) -> None:
    section("4. CONCEPT_WM synapse (concept -> wm)")
    syn = brain.synapses["concept_wm"]
    w = syn.weights  # [num_wm, num_concept]
    stats = weight_stats(w, "concept_wm")

    mean_drift = abs(stats["mean"] - MAIN_W_INIT)
    std_drift = abs(stats["std"] - MAIN_W_INIT_STD)
    print(f"\n  Drift from init (w_init={MAIN_W_INIT}, w_init_std={MAIN_W_INIT_STD}):")
    print(f"    mean drift: {mean_drift:.4f}")
    print(f"    std drift:  {std_drift:.4f}")
    has_drifted = mean_drift > WEIGHT_DRIFT_THRESHOLD or std_drift > WEIGHT_DRIFT_THRESHOLD
    drift_status = "DRIFTED (learning occurred)" if has_drifted else "NO DRIFT (no learning detected)"
    print(f"    --> {drift_status}")
    results.append(("concept_wm weight drift", has_drifted,
                     f"mean_drift={mean_drift:.4f}, std_drift={std_drift:.4f}"))


def check_expansion(brain) -> None:
    section("5. EXPANSION LAYER (fixed random, should NOT change)")
    w = brain._expansion_weights  # [num_expansion, num_sensory], binary 0/1
    total = w.numel()
    nonzero = (w > 0).sum().item()
    actual_sparsity = nonzero / total
    print(f"  Expansion weights shape: {tuple(w.shape)}")
    print(f"  Non-zero connections: {nonzero}/{total} ({actual_sparsity:.1%})")
    print(f"  Expected sparsity: ~{EXPANSION_SPARSITY:.0%}")

    # Check all values are 0 or 1 (binary)
    is_binary = torch.all((w == 0) | (w == 1)).item()
    print(f"  Binary (all 0 or 1): {is_binary}")

    sparsity_ok = abs(actual_sparsity - EXPANSION_SPARSITY) < EXPANSION_SPARSITY_TOL
    print(f"  Sparsity within tolerance ({EXPANSION_SPARSITY_TOL:.0%}): {sparsity_ok}")
    print(f"    --> {'CORRECT' if sparsity_ok and is_binary else 'UNEXPECTED'} "
          f"(fixed random projection intact)")
    results.append(("expansion layer integrity", sparsity_ok and is_binary,
                     f"sparsity={actual_sparsity:.3f}, binary={is_binary}"))


def check_meta(brain) -> None:
    section("0. BRAIN METADATA")
    print(f"  Tick count: {brain.tick_count}")
    print(f"  Regions: {list(brain.regions.keys())}")
    print(f"  Synapses: {list(brain.synapses.keys())}")
    print(f"  Modulators: {brain.modulators.snapshot()}")
    concept = brain.regions["concept"]
    print(f"  Concept layer: {concept.num_neurons} neurons, k={concept.k}")
    sensory = brain.regions["sensory"]
    print(f"  Sensory layer: {sensory.num_neurons} neurons")
    wm = brain.regions["wm"]
    print(f"  WM layer: {wm.num_neurons} neurons")

    has_ticks = brain.tick_count > 0
    results.append(("brain has run", has_ticks, f"tick_count={brain.tick_count}"))


def print_summary() -> None:
    section("SUMMARY")
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total = len(results)

    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {detail}")

    print(f"\n  {passed}/{total} checks passed, {failed}/{total} failed")
    if failed == 0:
        print("\n  ** ALL CHECKS PASSED -- brain is learning **")
    else:
        print(f"\n  ** {failed} CHECK(S) FAILED -- investigate above **")


def main() -> None:
    checkpoint = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CHECKPOINT
    print(f"Loading brain from: {checkpoint}")

    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found at {checkpoint}")
        sys.exit(1)

    brain = load_brain(checkpoint)
    print(f"Brain loaded successfully.")

    check_meta(brain)
    check_sensory_concept(brain)
    check_concept_tracker(brain)
    check_wm_concept(brain)
    check_concept_wm(brain)
    check_expansion(brain)
    print_summary()

    # Exit code: 0 if all pass, 1 if any fail
    any_failed = any(not ok for _, ok, _ in results)
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
