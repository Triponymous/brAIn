"""Phase 2 deliverable: soak test + concept emergence demonstration.

Runs the Brain on a synthetic 4-pattern input stream for a configurable
duration. Periodically saves the brain to SQLite. At the end, prints which
concept neurons emerged as winners for each input pattern, demonstrating
that the WTA Concept layer forms distinct sparse representations
(unlike Phase 1's collapsed two-region brain).

Run:
    .venv/bin/python scripts/run_soak.py --ticks 30000 --checkpoint-every 5000
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
from collections import defaultdict, Counter

import torch

from brain.core import Brain
from brain.persistence import save_brain, load_brain


PATTERNS = {
    "A": torch.tensor([3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "B": torch.tensor([0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "C": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0]),
    "D": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0]),
}
GAP = torch.zeros(16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=20000)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/brain.sqlite"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from --out if it exists")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    if args.resume and args.out.exists():
        print(f"Resuming from {args.out}")
        brain = load_brain(args.out)
        print(f"Loaded brain at tick {brain.tick_count}")
    else:
        brain = Brain(num_sensory=16, num_concept=8, concept_k=2)
        print(f"Fresh brain at tick 0")

    pattern_names = list(PATTERNS.keys())
    # Track which concept neurons fire on each pattern (last 1000 ticks)
    concept_winners: dict[str, Counter] = defaultdict(Counter)

    start = time.time()
    last_log = start

    for step in range(args.ticks):
        # 5 ticks of pattern, 5 ticks of gap, cycle through A,B,C,D
        pattern_name = pattern_names[(step // 10) % 4]
        in_pattern_phase = (step % 10) < 5
        input_current = PATTERNS[pattern_name] if in_pattern_phase else GAP

        out = brain.tick(input_current)

        # Log concept winners during pattern phase
        if in_pattern_phase:
            spike_idx = (out["concept"] > 0).nonzero(as_tuple=True)[0].tolist()
            for idx in spike_idx:
                concept_winners[pattern_name][idx] += 1

        # Periodic checkpoint + status
        if (step + 1) % args.checkpoint_every == 0:
            save_brain(brain, args.out)
            elapsed = time.time() - start
            ticks_per_sec = (step + 1) / elapsed
            print(f"  tick {brain.tick_count}: checkpoint saved, "
                  f"{ticks_per_sec:.0f} ticks/sec, elapsed {elapsed:.1f}s")

    save_brain(brain, args.out)
    elapsed = time.time() - start
    print(f"\nFinished {args.ticks} ticks in {elapsed:.1f}s "
          f"({args.ticks/elapsed:.0f} ticks/sec)")
    print(f"Brain saved to {args.out} at tick {brain.tick_count}")

    print("\n=== Concept emergence ===")
    print(f"Concept layer has {brain.regions['concept'].num_neurons} neurons, k={brain.regions['concept'].k}")
    print()
    for name in pattern_names:
        winners = concept_winners[name].most_common(5)
        if not winners:
            print(f"  Pattern {name}: no concept activity")
        else:
            top = ", ".join(f"#{idx}({count})" for idx, count in winners)
            print(f"  Pattern {name}: {top}")

    # Specialization check: are different concepts winning for different patterns?
    top_per_pattern = {
        name: counter.most_common(1)[0][0] if counter else None
        for name, counter in concept_winners.items()
    }
    distinct = len(set(v for v in top_per_pattern.values() if v is not None))
    print(f"\nDistinct top concept neurons across {len(pattern_names)} patterns: {distinct}/{len(pattern_names)}")
    if distinct == len(pattern_names):
        print("Phase 2 success: each pattern won by a DIFFERENT concept neuron")
    elif distinct >= 2:
        print("Partial success: at least some concept differentiation")
    else:
        print("Collapse: all patterns map to one concept (tune hyperparameters)")


if __name__ == "__main__":
    main()
