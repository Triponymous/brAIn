"""Live Pattern Validation Report.

Reads the monitor log and generates a report showing:
1. How many distinct clusters appeared
2. Cluster stability (how long each was active)
3. Transitions (when did patterns change)
4. Sensor correlations per cluster (what sensors were active)

Usage: .venv/bin/python benchmark/validate_live.py
"""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def load_log(path: str = "benchmark/reports/live_monitor.jsonl") -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            try:
                entries.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return entries


def analyze(entries: list[dict]) -> None:
    if not entries:
        print("Keine Log-Einträge gefunden.")
        return

    first = entries[0]
    last = entries[-1]
    duration_min = (datetime.fromisoformat(last["time"]) - datetime.fromisoformat(first["time"])).total_seconds() / 60

    print("=" * 60)
    print("  LIVE PATTERN VALIDATION REPORT")
    print("=" * 60)
    print(f"  Zeitraum: {first['time'][:19]} bis {last['time'][:19]}")
    print(f"  Dauer: {duration_min:.0f} Minuten ({len(entries)} Samples)")
    print()

    # Cluster statistics
    cluster_times = defaultdict(list)
    cluster_sensors = defaultdict(lambda: {"mic": [], "keys": [], "mouse": [], "sensory": [], "idle": []})
    transitions = []
    prev = None

    for e in entries:
        c = e["cluster"]
        cluster_times[c].append(e["time"])
        cluster_sensors[c]["mic"].append(e.get("mic", 0))
        cluster_sensors[c]["keys"].append(e.get("keys", 0))
        cluster_sensors[c]["mouse"].append(e.get("mouse", 0))
        cluster_sensors[c]["sensory"].append(e.get("sensory", 0))
        cluster_sensors[c]["idle"].append(e.get("idle", 0))

        if prev is not None and c != prev:
            transitions.append({
                "time": e["time"][:19],
                "from": prev,
                "to": c,
            })
        prev = c

    # Print cluster summary
    print(f"  Verschiedene Cluster: {len(cluster_times)}")
    print()

    for cid, times in sorted(cluster_times.items(), key=lambda x: -len(x[1])):
        count = len(times)
        pct = count / len(entries) * 100
        sensors = cluster_sensors[cid]
        avg_mic = sum(sensors["mic"]) / max(1, len(sensors["mic"]))
        avg_keys = sum(sensors["keys"]) / max(1, len(sensors["keys"]))
        avg_mouse = sum(sensors["mouse"]) / max(1, len(sensors["mouse"]))
        avg_sensory = sum(sensors["sensory"]) / max(1, len(sensors["sensory"]))
        avg_idle = sum(sensors["idle"]) / max(1, len(sensors["idle"]))

        label = entries[-1].get("label") if cid == entries[-1].get("cluster") else None

        print(f"  Cluster #{cid}: {count} Samples ({pct:.0f}%) {label or ''}")
        print(f"    Mic={avg_mic:.4f} Keys={avg_keys:.0f} Mouse={avg_mouse:.0f} Sensory={avg_sensory:.0f} Idle={avg_idle:.0f}s")

        # Characterize
        if avg_mic > 0.01 and avg_keys < 2 and avg_idle < 30:
            print(f"    → VERMUTUNG: Musik hören / Gespräch")
        elif avg_keys > 5:
            print(f"    → VERMUTUNG: Tippen / Arbeiten")
        elif avg_idle > 100:
            print(f"    → VERMUTUNG: Leon ist weg")
        elif avg_mouse > 10:
            print(f"    → VERMUTUNG: Browsen / Maus-intensiv")
        else:
            print(f"    → VERMUTUNG: Ruhig am Mac")
        print()

    # Transitions
    if transitions:
        print(f"  Cluster-Wechsel: {len(transitions)}")
        for t in transitions[:20]:
            print(f"    {t['time']}: #{t['from']} → #{t['to']}")
    else:
        print(f"  Keine Cluster-Wechsel (stabil!)")

    # Stability score
    if len(cluster_times) <= 3:
        print(f"\n  STABILITÄT: GUT ({len(cluster_times)} Cluster)")
    elif len(cluster_times) <= 6:
        print(f"\n  STABILITÄT: OK ({len(cluster_times)} Cluster)")
    else:
        print(f"\n  STABILITÄT: ZU VIELE ({len(cluster_times)} Cluster)")


if __name__ == "__main__":
    entries = load_log()
    analyze(entries)
