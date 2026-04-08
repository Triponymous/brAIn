# brAIntest — Mini-OSCEN

A persistent neuromorphic brain with LLM bridge.
See `docs/plans/2026-04-08-mini-oscen-design.md` for the design.

## Status
- [x] Phase 1: SNN core foundations (LIF, STDP, 2-region brain, viz)
- [x] Phase 2: Full multi-region brain + WTA + modulators + R-STDP + SQLite persistence
- [ ] Phase 3+: see `docs/plans/` (roadmap pending re-brainstorm)

## Quick start

```bash
uv venv
uv pip install -e ".[dev]"
.venv/bin/pytest -v

# Phase 1 viz (2-region STDP demo)
.venv/bin/python scripts/visualize_two_region.py

# Phase 2 soak test (full brain, concept emergence, save/resume)
.venv/bin/python scripts/run_soak.py --ticks 20000
.venv/bin/python scripts/run_soak.py --ticks 5000 --resume
```

## Known limitations (Phase 1)

The two-region brain in this phase has **no lateral inhibition** between feature
neurons. As a result, when trained on multiple input patterns, both feature neurons
tend to collapse onto the first pattern they see (verified across 20 seeds in
`scripts/visualize_two_region.py`). This is structural, not stochastic, and is
deliberately left to **Phase 2**, which adds a Concept layer with winner-take-all
competition that will produce distinct, sparse concept representations.

The `weights_after.png` plot from Phase 1 still demonstrates that STDP learns
input statistics — the matrix moves from uniform 0.5 to a clearly structured
binary pattern — but the visual story of "two distinct concepts" requires Phase 2.
