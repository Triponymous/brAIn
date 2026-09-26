# Contributing to brAIn

brAIn explores personal online learning and inspectable desktop context.
Contributions should make its behaviour easier to reproduce, understand and
control—not just add sensors or imply greater biological realism.

Start with the [documentation index](docs/README.md),
[current architecture](docs/living-desktop-manager.md) and [roadmap](docs/ROADMAP.md).

## Setup and safe development

Use Python 3.11 or 3.12. Follow [RUNNING.md](docs/RUNNING.md) for installation
and the distinction between the opt-in observer and the persistent daemon.
Do not start real sensors, the launch agent or voice recording just to test a PR.
Do not use a personal checkpoint as a test fixture.

The default raw `pytest` run includes native sensor constructors that may touch
macOS input/audio APIs. Use the explicitly scoped commands in
[VERIFICATION.md](docs/VERIFICATION.md). That page also records known environment
limitations; a skipped or failed test is never a passing result.

## Repository map

| Path | Responsibility |
| --- | --- |
| `brain/` | LIF dynamics, plasticity, competitive concepts, memory, persistence |
| `adapters/mac_desktop/` | Sensor acquisition and encoding |
| `bridge/` | Learned labels, episode/experience logs, query tools, LLM and SCP |
| `server/observe.py` | Fresh opt-in observation runner, port 8001 |
| `server/braind.py` | Persistent experimental daemon, port 8000 |
| `server/control.py` | Daemon supervisor; serves the dashboard, port 8900 |
| `server/mcp.py` | Stdio MCP proxy to the daemon query API |
| `docs/dashboard-concepts/` | Observatory, the one UI for both runtimes: fixtures, tour and browser tests |
| `benchmark/` | Offline research experiments and synthetic diagnostics |
| `capabilities/` | Experimental grants and action tools; not a security sandbox |
| `3d/` | Anatomy placement, asset provenance and visual verification |
| `tests/` | Python unit and integration tests |

## Contribution rules

- Keep runtime/UI changes focused and preserve unrelated work.
- Default dashboard text is English. The introduction supports English and
  German: update both copies and their coverage tests together.
- Preserve distinctions between observations, user labels, model inference and
  synthetic examples. Missing data must stay missing.
- New data sources need explicit opt-in, visible status, capture-level stop,
  source timestamps, retention/deletion rules and tests for disabled capture.
- Never submit personal checkpoints, activity exports, recordings, transcripts,
  access tokens or machine-specific paths. Git ignore rules are not encryption.
- Prefer existing code and simple functions over speculative frameworks.
  Validate external boundaries; comments should explain why.
- State scientific claims narrowly. A synthetic test or model variable does not
  demonstrate a human state, consciousness, generalization or causal benefit.
- Preserve third-party licenses and attribution for assets and vendored code.

## Pull requests

1. Update from `main`, then use a descriptive branch such as
   `feature/context-freshness`, `fix/capture-status` or `docs/runtime-guide`.
2. Use concise conventional commit subjects: `feat:`, `fix:`, `refactor:`,
   `docs:`, `test:` or `chore:`. Describe the change, not the editor/session.
3. Describe scope, user-visible behaviour, evidence, limitations and data access.
4. Run relevant tests; report the exact command and passed/failed/skipped counts.
5. Open a focused PR. Remove a merged branch after checking no unique work remains.
   Superseded work should be closed with its replacement identified, not merged
   just to remove a conflict warning.

Use your configured contributor identity; preserve credit for human contributors
and third-party work. History rewrites require explicit coordination and a backup.
Do not merge old pre-rewrite branches into the current history.

## Useful contributions

- Reproducible, privacy-respecting evaluation fixtures and baseline comparisons.
- Tests of capture revocation, stale context and model restart behaviour.
- MCP provenance and per-client disclosure controls.
- Dashboard accessibility and honest explanation of uncertainty.
- A legally usable, source-separated audio evaluation corpus.
- Corrections to documentation with pointers to the actual implementation.

See [SECURITY.md](SECURITY.md) before enabling action tools or reporting
security-sensitive findings. Use repository Issues for ordinary bugs and proposals.
