# brAIntest — Mini-OSCEN

A persistent neuromorphic brain with LLM bridge.
See `docs/plans/2026-04-08-mini-oscen-design.md` for the design.

## Status
- [x] Phase 1: SNN core foundations (LIF, STDP, 2-region brain, viz)
- [ ] Phase 2: Full multi-region brain + persistence
- [ ] Phase 3: Desktop adapter (webcam + mic)
- [ ] Phase 4: FastAPI + minimal dashboard
- [ ] Phase 5: LLM bridge (Ollama + memory tools)
- [ ] Phase 6: Avatar adapter + polish

## Quick start

```bash
uv venv
uv pip install -e ".[dev]"
.venv/bin/pytest -v
.venv/bin/python scripts/visualize_two_region.py
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
