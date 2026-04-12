"""Benchmark: Baseline Comparison — SNN vs classical methods on activity discrimination.

Compares 4 approaches on the same 4-scenario data:
  1. Online k-Means (MiniBatchKMeans, unsupervised)
  2. k-NN (supervised, train on first half, test on second half)
  3. Threshold Classifier (hand-coded rules)
  4. SNN (Brain with ConceptTracker)

Metrics: accuracy/discrimination (can it distinguish 4 patterns?),
         adaptation speed (how many samples to converge?).

Usage: .venv/bin/python benchmark/baseline_comparison.py
"""
from __future__ import annotations

import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import numpy as np

from brain.core import Brain
from adapters.mac_desktop.encoding import encode_snapshot

# ---------------------------------------------------------------------------
# Scenario definitions (identical to discrimination_test.py)
# ---------------------------------------------------------------------------

SCENARIOS = {
    "typing_vscode": lambda: {
        "active_app": {"name": "VSCode"},
        "keystroke_rate": {"count": random.randint(18, 28), "variability": random.uniform(0.1, 0.3)},
        "mouse_rate": {"count": random.randint(1, 4)},
        "idle": {"seconds": random.uniform(0.2, 0.8)},
        "mic": {
            "mel": [random.uniform(0.01, 0.03)] * 10
                 + [random.uniform(0.04, 0.1)] * 12
                 + [random.uniform(0.01, 0.03)] * 10,
            "rms": random.uniform(0.02, 0.05),
        },
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
        "mic": {
            "mel": [random.uniform(0.02, 0.08)] * 8
                 + [random.uniform(0.08, 0.2)] * 16
                 + [random.uniform(0.02, 0.06)] * 8,
            "rms": random.uniform(0.03, 0.08),
        },
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
LABEL_MAP = {name: idx for idx, name in enumerate(SCENARIO_NAMES)}
NUM_SAMPLES = 5000
NUM_SCENARIOS = len(SCENARIOS)

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate encoded vectors for each scenario.

    Returns:
        vectors: (NUM_SCENARIOS * NUM_SAMPLES, 200) float32 array
        labels:  (NUM_SCENARIOS * NUM_SAMPLES,)     int array (0..3)
        order:   list of scenario names matching label indices
    """
    print("Generating encoded data ...")
    all_vectors = []
    all_labels = []
    for sname, maker in SCENARIOS.items():
        label = LABEL_MAP[sname]
        for _ in range(NUM_SAMPLES):
            vec = encode_snapshot(maker())
            all_vectors.append(vec.numpy())
            all_labels.append(label)
    vectors = np.array(all_vectors, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    print(f"  Generated {len(vectors)} vectors ({NUM_SCENARIOS} scenarios x {NUM_SAMPLES} each)")
    return vectors, labels, SCENARIO_NAMES


# ---------------------------------------------------------------------------
# Cluster purity helper
# ---------------------------------------------------------------------------

def cluster_purity(cluster_assignments: np.ndarray, true_labels: np.ndarray) -> float:
    """Measure cluster purity: for each cluster, fraction that belongs to the
    majority true label.  Returns weighted average purity in [0, 1]."""
    total = 0
    correct = 0
    for cid in np.unique(cluster_assignments):
        mask = cluster_assignments == cid
        if mask.sum() == 0:
            continue
        counts = Counter(true_labels[mask].tolist())
        majority = counts.most_common(1)[0][1]
        correct += majority
        total += mask.sum()
    return correct / max(1, total)


def n_distinct_clusters(cluster_assignments: np.ndarray, true_labels: np.ndarray) -> int:
    """How many true labels have a unique dominant cluster (i.e., the method
    can discriminate them)?"""
    label_to_dominant = {}
    for label in np.unique(true_labels):
        mask = true_labels == label
        counts = Counter(cluster_assignments[mask].tolist())
        label_to_dominant[label] = counts.most_common(1)[0][0]
    # Count how many labels got a UNIQUE dominant cluster
    dominant_counts = Counter(label_to_dominant.values())
    unique = sum(1 for label, dom in label_to_dominant.items() if dominant_counts[dom] == 1)
    return unique


# ---------------------------------------------------------------------------
# Adaptation speed helper
# ---------------------------------------------------------------------------

def measure_adaptation(
    predict_fn,
    vectors: np.ndarray,
    labels: np.ndarray,
    window: int = 100,
    target_accuracy: float = 0.8,
) -> int:
    """Run predict_fn on vectors in order.  Measure rolling accuracy in windows
    of `window` samples.  Return number of samples until accuracy first exceeds
    target_accuracy, or -1 if it never does.

    predict_fn(vec) -> int label  (called sequentially).
    """
    correct_window: list[bool] = []
    for i in range(len(vectors)):
        pred = predict_fn(vectors[i])
        correct_window.append(pred == labels[i])
        if len(correct_window) > window:
            correct_window.pop(0)
        if len(correct_window) == window:
            acc = sum(correct_window) / window
            if acc >= target_accuracy:
                return i + 1
    return -1


# ---------------------------------------------------------------------------
# Baseline 1: Online k-Means (MiniBatchKMeans)
# ---------------------------------------------------------------------------

def run_kmeans(vectors: np.ndarray, labels: np.ndarray) -> dict:
    """Online k-Means clustering with MiniBatchKMeans."""
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        return {"error": "sklearn not installed. Install with: pip install scikit-learn"}

    print("\n--- Baseline 1: Online k-Means (MiniBatchKMeans) ---")
    t0 = time.time()

    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=100, random_state=42, n_init=3)
    kmeans.fit(vectors)
    assignments = kmeans.labels_

    purity = cluster_purity(assignments, labels)
    distinct = n_distinct_clusters(assignments, labels)
    elapsed = time.time() - t0

    # Adaptation speed: feed samples one at a time via partial_fit, measure
    # when the online model converges.
    kmeans_online = MiniBatchKMeans(n_clusters=4, batch_size=1, random_state=42, n_init=1)
    # We need to use partial_fit in order, build a label mapping after a warm-up
    warm_up = 500
    kmeans_online.partial_fit(vectors[:warm_up])

    # Build label mapping from warm-up data
    warm_assignments = kmeans_online.predict(vectors[:warm_up])
    # For each cluster, find the dominant true label
    cluster_label_map: dict[int, int] = {}
    for cid in range(4):
        mask = warm_assignments == cid
        if mask.sum() > 0:
            counts = Counter(labels[:warm_up][mask].tolist())
            cluster_label_map[cid] = counts.most_common(1)[0][0]
        else:
            cluster_label_map[cid] = -1

    # Now measure adaptation on the remaining data
    remaining_vectors = vectors[warm_up:]
    remaining_labels = labels[warm_up:]

    def kmeans_predict(vec: np.ndarray) -> int:
        pred_cluster = kmeans_online.predict(vec.reshape(1, -1))[0]
        return cluster_label_map.get(pred_cluster, -1)

    adaptation = measure_adaptation(kmeans_predict, remaining_vectors, remaining_labels)
    if adaptation > 0:
        adaptation += warm_up  # account for warm-up samples

    print(f"  Purity:     {purity:.3f}")
    print(f"  Distinct:   {distinct}/4 scenarios separated")
    print(f"  Adaptation: {adaptation if adaptation > 0 else 'never'} samples")
    print(f"  Time:       {elapsed:.2f}s")

    return {
        "method": "online_kmeans",
        "purity": round(purity, 4),
        "distinct_scenarios": distinct,
        "adaptation_samples": adaptation,
        "time_seconds": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Baseline 2: k-NN (supervised)
# ---------------------------------------------------------------------------

def run_knn(vectors: np.ndarray, labels: np.ndarray) -> dict:
    """k-NN classifier: train on first half, test on second half."""
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score
    except ImportError:
        return {"error": "sklearn not installed. Install with: pip install scikit-learn"}

    print("\n--- Baseline 2: k-NN (k=5, supervised) ---")
    t0 = time.time()

    split = len(vectors) // 2
    X_train, y_train = vectors[:split], labels[:split]
    X_test, y_test = vectors[split:], labels[split:]

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Distinct: count how many classes have >50% correct predictions
    distinct = 0
    per_class_acc = {}
    for label_idx, sname in enumerate(SCENARIO_NAMES):
        mask = y_test == label_idx
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == y_test[mask]).mean()
            per_class_acc[sname] = float(class_acc)
            if class_acc > 0.5:
                distinct += 1

    elapsed = time.time() - t0

    # Adaptation speed: train with increasing amounts of data
    adaptation = -1
    window = 100
    for n_train in range(window, split, window):
        knn_partial = KNeighborsClassifier(n_neighbors=min(5, n_train))
        knn_partial.fit(X_train[:n_train], y_train[:n_train])
        y_partial = knn_partial.predict(X_test[:window])
        acc_partial = (y_partial == y_test[:window]).mean()
        if acc_partial >= 0.8:
            adaptation = n_train
            break

    print(f"  Accuracy:   {accuracy:.3f}")
    print(f"  Distinct:   {distinct}/4 scenarios separated")
    print(f"  Per-class:  {per_class_acc}")
    print(f"  Adaptation: {adaptation if adaptation > 0 else 'never'} training samples")
    print(f"  Time:       {elapsed:.2f}s")

    return {
        "method": "knn_k5",
        "accuracy": round(accuracy, 4),
        "distinct_scenarios": distinct,
        "per_class_accuracy": {k: round(v, 4) for k, v in per_class_acc.items()},
        "adaptation_samples": adaptation,
        "time_seconds": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Baseline 3: Threshold Classifier (hand-coded rules)
# ---------------------------------------------------------------------------

def threshold_classify(snap: dict) -> int:
    """Simple rule-based classifier on raw snapshot data.

    Rules:
        keys > 10         -> typing (0)
        mic_rms > 0.05 and keys == 0 -> call (1)
        idle > 60          -> idle (3)
        else               -> other/music (2)
    """
    keys = snap.get("keystroke_rate", {}).get("count", 0)
    mic_rms = snap.get("mic", {}).get("rms", 0.0)
    idle_secs = snap.get("idle", {}).get("seconds", 0.0)

    if keys > 10:
        return 0  # typing
    if mic_rms > 0.05 and keys == 0:
        return 1  # call
    if idle_secs > 60:
        return 3  # idle
    return 2  # other/music


def run_threshold(labels: np.ndarray) -> dict:
    """Threshold classifier: run on raw scenario snapshots."""
    print("\n--- Baseline 3: Threshold Classifier (hand-coded rules) ---")
    t0 = time.time()

    all_preds = []
    all_true = []
    per_scenario_preds: dict[str, list[int]] = defaultdict(list)

    for sname, maker in SCENARIOS.items():
        true_label = LABEL_MAP[sname]
        for _ in range(NUM_SAMPLES):
            snap = maker()
            pred = threshold_classify(snap)
            all_preds.append(pred)
            all_true.append(true_label)
            per_scenario_preds[sname].append(pred)

    all_preds_arr = np.array(all_preds)
    all_true_arr = np.array(all_true)
    accuracy = (all_preds_arr == all_true_arr).mean()

    # Per-class accuracy
    per_class_acc = {}
    distinct = 0
    for sname in SCENARIO_NAMES:
        true_label = LABEL_MAP[sname]
        preds = per_scenario_preds[sname]
        class_acc = sum(1 for p in preds if p == true_label) / len(preds)
        per_class_acc[sname] = class_acc
        if class_acc > 0.5:
            distinct += 1

    elapsed = time.time() - t0

    # Adaptation speed: threshold classifier is instant (no learning)
    adaptation = 1  # works from the first sample

    print(f"  Accuracy:   {accuracy:.3f}")
    print(f"  Distinct:   {distinct}/4 scenarios separated")
    print(f"  Per-class:  {per_class_acc}")
    print(f"  Adaptation: {adaptation} samples (no learning needed)")
    print(f"  Time:       {elapsed:.2f}s")

    return {
        "method": "threshold_classifier",
        "accuracy": round(accuracy, 4),
        "distinct_scenarios": distinct,
        "per_class_accuracy": {k: round(v, 4) for k, v in per_class_acc.items()},
        "adaptation_samples": adaptation,
        "time_seconds": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Baseline 4: SNN (Brain + ConceptTracker)
# ---------------------------------------------------------------------------

def run_snn() -> dict:
    """SNN with Brain: run 5000 ticks per scenario, measure concept clusters."""
    print("\n--- Baseline 4: SNN (Brain + ConceptTracker) ---")
    t0 = time.time()

    brain = Brain(num_sensory=200, num_concept=200)

    # Track which cluster is assigned during each scenario
    scenario_cluster_history: dict[str, list[int]] = defaultdict(list)
    scenario_dominant_cluster: dict[str, int] = {}

    for sname, maker in SCENARIOS.items():
        cluster_counts: Counter = Counter()
        for tick in range(NUM_SAMPLES):
            vec = encode_snapshot(maker())
            brain.tick(vec)
            cid = brain.concept_tracker.current_cluster_id
            if cid >= 0:
                cluster_counts[cid] += 1
            # Record every 200 ticks what cluster we are in
            if tick % 200 == 0 and cid >= 0:
                scenario_cluster_history[sname].append(cid)

        dominant = cluster_counts.most_common(1)[0] if cluster_counts else (-1, 0)
        scenario_dominant_cluster[sname] = dominant[0]
        print(f"  {sname:20s}: dominant=C{dominant[0]} (count={dominant[1]}), "
              f"total_clusters_seen={len(cluster_counts)}")

        # Brief pause between scenarios (zeros) to let clusters settle
        for _ in range(500):
            brain.tick(torch.zeros(200))

    elapsed = time.time() - t0

    # Discrimination: how many scenarios got a UNIQUE dominant cluster?
    dom_counts = Counter(scenario_dominant_cluster.values())
    distinct = sum(1 for sname, dom in scenario_dominant_cluster.items() if dom_counts[dom] == 1)

    # Purity: for each scenario, what fraction of cluster assignments matched
    # the dominant cluster?
    total_correct = 0
    total_assigned = 0
    per_class_purity = {}
    for sname, maker in SCENARIOS.items():
        history = scenario_cluster_history[sname]
        if not history:
            per_class_purity[sname] = 0.0
            continue
        dom = scenario_dominant_cluster[sname]
        correct = sum(1 for c in history if c == dom)
        per_class_purity[sname] = correct / len(history)
        total_correct += correct
        total_assigned += len(history)

    overall_purity = total_correct / max(1, total_assigned)

    # Adaptation speed: how quickly does the SNN settle on a stable cluster?
    # Measure: first tick window (of 200) where cluster == final dominant cluster
    adaptation_per_scenario = {}
    for sname in SCENARIO_NAMES:
        history = scenario_cluster_history[sname]
        dom = scenario_dominant_cluster[sname]
        first_match = -1
        for i, cid in enumerate(history):
            if cid == dom:
                first_match = (i + 1) * 200  # tick number
                break
        adaptation_per_scenario[sname] = first_match

    avg_adaptation = np.mean([v for v in adaptation_per_scenario.values() if v > 0])
    avg_adaptation = int(avg_adaptation) if not np.isnan(avg_adaptation) else -1

    # Concept tracker summary
    tracker_snap = brain.concept_tracker.snapshot()

    print(f"\n  SNN Summary:")
    print(f"  Purity:     {overall_purity:.3f}")
    print(f"  Distinct:   {distinct}/4 scenarios separated")
    print(f"  Per-class purity: {per_class_purity}")
    print(f"  Adaptation (avg): {avg_adaptation} ticks")
    print(f"  Total clusters formed: {tracker_snap['num_clusters']}")
    print(f"  Time:       {elapsed:.2f}s")

    return {
        "method": "snn_brain",
        "purity": round(overall_purity, 4),
        "distinct_scenarios": distinct,
        "per_class_purity": {k: round(v, 4) for k, v in per_class_purity.items()},
        "adaptation_ticks": avg_adaptation,
        "adaptation_per_scenario": adaptation_per_scenario,
        "total_clusters": tracker_snap["num_clusters"],
        "dominant_clusters": {k: v for k, v in scenario_dominant_cluster.items()},
        "time_seconds": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    print("\n")
    print("=" * 80)
    print("  BASELINE COMPARISON: SNN vs Classical Methods")
    print("=" * 80)

    header = f"{'Method':<25} {'Accuracy/Purity':>15} {'Distinct (of 4)':>15} {'Adaptation':>15}"
    print(header)
    print("-" * 80)

    for r in results:
        if "error" in r:
            method_name = r.get("method", "Unknown")
            print(f"  {method_name:<23} ERROR: {r['error']}")
            continue

        method = r["method"]
        acc = r.get("accuracy", r.get("purity", 0.0))
        distinct = r.get("distinct_scenarios", 0)
        adapt = r.get("adaptation_samples", r.get("adaptation_ticks", -1))
        adapt_str = str(adapt) if adapt > 0 else "never"
        time_s = r.get("time_seconds", 0.0)

        print(f"  {method:<23} {acc:>14.3f} {distinct:>14d} {adapt_str:>15}")

    print("-" * 80)
    print()

    # Winner analysis
    valid = [r for r in results if "error" not in r]
    if valid:
        best_acc = max(valid, key=lambda r: r.get("accuracy", r.get("purity", 0.0)))
        best_distinct = max(valid, key=lambda r: r.get("distinct_scenarios", 0))
        adapted = [r for r in valid if r.get("adaptation_samples", r.get("adaptation_ticks", -1)) > 0]
        best_adapt = min(adapted, key=lambda r: r.get("adaptation_samples", r.get("adaptation_ticks", float("inf")))) if adapted else None

        print(f"  Best accuracy/purity:  {best_acc['method']} ({best_acc.get('accuracy', best_acc.get('purity', 0)):.3f})")
        print(f"  Best discrimination:   {best_distinct['method']} ({best_distinct.get('distinct_scenarios', 0)}/4)")
        if best_adapt:
            adapt_val = best_adapt.get("adaptation_samples", best_adapt.get("adaptation_ticks", -1))
            print(f"  Fastest adaptation:    {best_adapt['method']} ({adapt_val} samples/ticks)")

    # SNN comparison note
    snn = next((r for r in results if r.get("method") == "snn_brain"), None)
    if snn and "error" not in snn:
        print()
        print("  NOTE: The SNN is UNSUPERVISED and learns ONLINE with no labels.")
        print("  k-NN uses labeled training data. Threshold uses hand-coded rules.")
        print("  k-Means is unsupervised but has no temporal/adaptive component.")
        print(f"  SNN formed {snn.get('total_clusters', '?')} clusters from raw spiking activity.")

    print()
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 80)
    print("  BASELINE COMPARISON BENCHMARK")
    print("  SNN (Brain + ConceptTracker) vs Classical Methods")
    print("=" * 80)

    # Generate data for classical baselines
    vectors, labels, scenario_order = generate_data()

    results = []

    # 1. Online k-Means
    results.append(run_kmeans(vectors, labels))

    # 2. k-NN
    results.append(run_knn(vectors, labels))

    # 3. Threshold Classifier
    random.seed(42)  # reset seed for consistent snapshot generation
    results.append(run_threshold(labels))

    # 4. SNN
    random.seed(42)
    torch.manual_seed(42)
    results.append(run_snn())

    # Print comparison
    print_comparison_table(results)

    # Save JSON report
    report = {
        "benchmark": "baseline_comparison",
        "timestamp": time.time(),
        "num_samples_per_scenario": NUM_SAMPLES,
        "scenarios": SCENARIO_NAMES,
        "results": results,
    }

    report_dir = Path("benchmark/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "baseline_comparison.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  Report saved to {report_path}")


if __name__ == "__main__":
    main()
