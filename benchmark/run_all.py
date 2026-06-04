"""Run all SNN benchmarks and generate summary report.

Usage: .venv/bin/python benchmark/run_all.py
"""
import json
import time
from pathlib import Path


def main():
    print("=" * 60)
    print("  Brain SNN BENCHMARK SUITE")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = {}

    # Test 1: Stability (500K ticks)
    print("\n\n" + "▓" * 60)
    print("  TEST 1: LONG-TERM STABILITY")
    print("▓" * 60)
    from benchmark.stability_test import run_all as stability
    results["stability"] = stability()

    # Test 2: Replay (concept consistency)
    print("\n\n" + "▓" * 60)
    print("  TEST 2: REPLAY CONSISTENCY")
    print("▓" * 60)
    from benchmark.replay_test import run_all as replay
    results["replay"] = replay()

    # Test 3: Discrimination (4 scenarios)
    print("\n\n" + "▓" * 60)
    print("  TEST 3: DISCRIMINATION")
    print("▓" * 60)
    from benchmark.discrimination_test import run_all as discrimination
    results["discrimination"] = discrimination()

    # Test 4: Modulator reactivity
    print("\n\n" + "▓" * 60)
    print("  TEST 4: MODULATOR REACTIVITY")
    print("▓" * 60)
    from benchmark.modulator_test import run_all as modulator
    results["modulator"] = modulator()

    # Summary
    print("\n\n" + "=" * 60)
    print("  FINAL REPORT")
    print("=" * 60)

    total = len(results)
    passed = sum(1 for v in results.values() if v)

    for name, passed_test in results.items():
        print(f"  [{'PASS' if passed_test else 'FAIL'}] {name}")

    print(f"\n  {passed}/{total} PASSED")
    if passed == total:
        print("  ALL BENCHMARKS PASS — SNN is validated!")
    else:
        print(f"  {total - passed} FAILED — needs work")

    report = {"timestamp": time.time(), "results": {k: v for k, v in results.items()}}
    Path("benchmark/reports/summary.json").write_text(json.dumps(report, indent=2))
    print(f"\n  Full report: benchmark/reports/summary.json")


if __name__ == "__main__":
    main()
