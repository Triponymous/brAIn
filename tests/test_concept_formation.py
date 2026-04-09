"""Validates that the SNN forms distinct concepts from patterned input.

Injects 3 repeating sensor patterns with REALISTIC differentiation
(different apps, different typing/mouse intensity, different audio profiles)
for 100K ticks, checking that concept neurons specialize.

Key insight: patterns must differ in enough sensory dimensions for STDP
to learn the difference. Mic mel, activity level, pause type, and app
identity all contribute to the ~50 active neurons per tick. At least 15-20
neurons should differ between patterns for reliable concept separation.
"""
import random
import torch
import pytest
from collections import defaultdict

from brain.core import Brain
from adapters.mac_desktop.encoding import encode_snapshot


def _make_typing_pattern() -> dict:
    """Heavy typing in VSCode: lots of keys, minimal mouse, keyboard sounds."""
    return {
        "active_app": {"name": "VSCode"},
        "keystroke_rate": {"count": random.randint(15, 35), "variability": random.uniform(0.1, 0.4), "burst": 0.0},
        "mouse_rate": {"count": random.randint(0, 3), "variability": random.uniform(0.0, 0.1), "burst": 0.0},
        "idle": {"seconds": random.uniform(0.1, 0.8)},
        # Keyboard click sounds: energy in 1-4kHz range
        "mic": {
            "mel": [random.uniform(0.001, 0.01)] * 10 + [random.uniform(0.05, 0.15)] * 12 + [random.uniform(0.001, 0.01)] * 10,
            "rms": random.uniform(0.02, 0.06),
        },
        "time_tonic": {"day_phase": [0.5, 0.5, 0.5, 0.5], "week_phase": [0.5, 0.5, 0.5, 0.5]},
    }


def _make_browsing_pattern() -> dict:
    """Browsing in Chrome: lots of mouse, minimal keys, quiet."""
    return {
        "active_app": {"name": "Chrome"},
        "keystroke_rate": {"count": random.randint(0, 2), "variability": random.uniform(0.0, 0.1), "burst": 0.0},
        "mouse_rate": {"count": random.randint(25, 60), "variability": random.uniform(0.2, 0.7), "burst": 0.0},
        "idle": {"seconds": random.uniform(0.1, 1.5)},
        # Quiet browsing: very low audio
        "mic": {
            "mel": [random.uniform(0.001, 0.005)] * 32,
            "rms": random.uniform(0.001, 0.005),
        },
        "time_tonic": {"day_phase": [0.5, 0.5, 0.5, 0.5], "week_phase": [0.5, 0.5, 0.5, 0.5]},
    }


def _make_idle_pattern() -> dict:
    """User away: nothing happening, ambient room noise."""
    return {
        "active_app": {"name": "Finder"},
        "keystroke_rate": {"count": 0},
        "mouse_rate": {"count": 0},
        "idle": {"seconds": random.uniform(30.0, 180.0)},
        # Ambient: low-frequency hum
        "mic": {
            "mel": [random.uniform(0.01, 0.03)] * 5 + [random.uniform(0.001, 0.005)] * 27,
            "rms": random.uniform(0.005, 0.015),
        },
        "time_tonic": {"day_phase": [0.5, 0.5, 0.5, 0.5], "week_phase": [0.5, 0.5, 0.5, 0.5]},
    }


PATTERN_MAKERS = {
    "typing": _make_typing_pattern,
    "browsing": _make_browsing_pattern,
    "idle": _make_idle_pattern,
}


def test_three_distinct_concepts_form():
    """After 100K ticks alternating 3 patterns, concept neurons should
    specialize for different activity patterns."""
    random.seed(42)
    brain = Brain(num_sensory=200, num_concept=200)
    pattern_names = list(PATTERN_MAKERS.keys())

    concept_pattern_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    ticks_per_pattern = 1000  # switch every 1000 ticks (10s at 100Hz)
    total_ticks = 100000

    for tick in range(total_ticks):
        pattern_idx = (tick // ticks_per_pattern) % len(pattern_names)
        pattern_name = pattern_names[pattern_idx]
        snap = PATTERN_MAKERS[pattern_name]()
        vec = encode_snapshot(snap)
        out = brain.tick(vec)

        concept_spikes = out["concept"]
        for cid in (concept_spikes > 0).nonzero(as_tuple=True)[0].tolist():
            concept_pattern_counts[cid][pattern_name] += 1

    # A specialized concept fires >50% for one pattern
    specialized = set()
    for cid, counts in concept_pattern_counts.items():
        total = sum(counts.values())
        if total < 20:
            continue
        for pname, count in counts.items():
            if count / total > 0.5:
                specialized.add(cid)
                break

    print(f"\n=== Concept Formation Results ({total_ticks} ticks) ===")
    print(f"Total concepts that fired: {len(concept_pattern_counts)}")
    print(f"Specialized concepts (>50% one pattern): {len(specialized)}")
    for cid in sorted(specialized)[:20]:
        counts = concept_pattern_counts[cid]
        total = sum(counts.values())
        primary = max(counts, key=counts.get)
        pct = counts[primary] / total * 100
        print(f"  C{cid}: {primary} ({pct:.0f}%, {total} total spikes)")

    # Verify specialization
    assert len(specialized) >= 3, (
        f"Only {len(specialized)} specialized concepts (need >=3). "
        f"Fired: {len(concept_pattern_counts)}."
    )

    # Verify they cover multiple patterns
    primary_patterns = set()
    for cid in specialized:
        counts = concept_pattern_counts[cid]
        primary = max(counts, key=counts.get)
        primary_patterns.add(primary)

    assert len(primary_patterns) >= 2, (
        f"All specialized concepts fire for same pattern: {primary_patterns}"
    )
    print(f"Patterns represented: {primary_patterns}")
    print(f"=== PASS ===\n")
