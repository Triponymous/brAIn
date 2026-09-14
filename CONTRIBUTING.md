# Contributing to brAIn

brAIn is artificial *life*, not artificial intelligence. It's a spiking neural network that runs on your Mac, watches how you work (typing rhythm, mouse, the app in focus, microphone loudness — never *content*), and slowly becomes an individual shaped by your days. No pretraining, no generic personality: a blank slate that learns one human.

This is open research and there's a lot of surface area. Whether you want to tune learning rules, improve the training console, add a sensor, or sharpen the docs — you're welcome here. This guide gets you running and oriented.

## How it fits together

Four layers, bottom to top:

```
sensors  ─→  SNN brain  ─→  emergent modulators  ─→  felt-state self-model  ─→  console / SCP / LLM
(adapters)   (brain/)       (prediction error)        (bridge/felt_state.py)     (train-ui, speech layer)
```

- **`brain/`** — the organism. Leaky-integrate-and-fire neurons, an expansion layer for pattern separation, a winner-take-all concept layer that grows sparse concept neurons via STDP + BCM metaplasticity, and a recurrent working-memory buffer. State (weights, modulators, learned labels) persists to SQLite and survives restarts — delete the checkpoint and the individual is gone.
- **Modulators** — Dopamine, Noradrenaline, Acetylcholine, Serotonin. They are *not* hand-tuned if-statements. They emerge from the brain's own prediction error (active inference, in the spirit of Friston/Schmidhuber): surprise drives arousal, familiarity settles it. They feed back into the SNN's dynamics (thresholds, learning rate, lateral inhibition), so emotion and learning are one loop.
- **`bridge/`** — everything between brain and human. The heart of it is `felt_state.py`: the **learned self-model**. Instead of forcing a generic six-emotion classifier on you, it learns *your* vocabulary of internal states ("flow", "wired", "stuck") from your corrections. SCP (`scp_server.py` / `scp_client.py` / `scp_schema.py`) is the protocol the brain speaks; an LLM is an optional, detachable speech layer on top of it — it reads patterns, never your keystrokes or audio. `experience.py` is the experience log: what the pet did, what you did, the state it happened in and what followed — the evidence and reward layer the README's *Thesis* builds on (facts only, never content).
- **`server/`** — the runtime. `control.py` is a tiny always-on server that starts/stops the brain daemon and serves the console. `braind.py` is the daemon: it loads the brain, runs the sensors, ticks the SNN, auto-saves, and streams state over a WebSocket.
- **`adapters/mac_desktop/`** — the six macOS sensors and the spike encoder that turns raw signals into the brain's input.
- **`train-ui/`** — the single-file training console (`index.html`). This is *the* frontend.

### The active-learning loop (the part that makes it personal)

This is modeled on how a dog learns: association and prediction error, attachment to one human, reading behaviour rather than words.

1. The brain runs continuously and tracks a rolling signature of its modulator trend (its felt "mood").
2. `FeltState` matches that signature against the prototypes you've taught it. If one is close enough, the console shows the label and a confidence; if nothing matches, it shows *unknown*.
3. When your state visibly **shifts** and the brain doesn't have a label for the new one, `FeltStateWatcher` freezes that moment and proactively asks — a banner in the console: *"what was that?"*
4. You answer with a word. That label is bound to the frozen signature (`POST /api/feel`), sharpening an existing prototype or creating a new one.
5. Over time the network is shaped entirely by your life. No two brains end up alike.

That's the whole game: you're not training a classifier on a dataset, you're raising one organism on one life.

## Running it

You'll need macOS on Apple Silicon (developed on an M-series MacBook), Python 3.11+, and [`uv`](https://docs.astral.sh/uv/). For the optional LLM speech layer, [Ollama](https://ollama.com).

```bash
git clone https://github.com/Triponymous/brAIn.git
cd brAIn
uv venv && uv pip install -e ".[dev]"
```

Then the loop is: **start the control server → open the console → start the daemon → train felt-states.**

```bash
# 1. Start the always-on control server (it serves the console and manages the daemon)
.venv/bin/python -m server.control
```

```text
# 2. Open the training console
http://127.0.0.1:8900
```

3. **Start the daemon** with the *Start* button in the console. For development without Accessibility/Microphone permissions, the daemon also runs with mock sensors:

   ```bash
   .venv/bin/python -m server.braind start --mock-sensors
   ```

4. **Train felt-states.** Let the brain run while you work. When it asks *"what was that?"*, type a word for the state and hit *Learn*. You can also label or correct the current state at any time. Each correction nudges the self-model toward *you*.

Full-sensor launch (needs macOS Accessibility + Microphone permissions) is wrapped in `./start.sh`. To actually raise one, install it always-on with `./scripts/install_launchd.sh` (see the README) — the control server then starts at login and supervises the daemon.

## Development setup

- Environment: `uv venv` + `uv pip install -e ".[dev]"` as above. Run Python through `.venv/bin/python` so you get the project's interpreter and deps.
- Tests:

  ```bash
  .venv/bin/python -m pytest
  ```

  The felt-state, brain, and server suites are the core — keep them green. A heads-up: some Mac-sensor and voice tests segfault when run headless (they touch real audio/accessibility APIs). That's a known environment limitation, not your change — scope your runs to the suite you're touching if needed, e.g. `.venv/bin/python -m pytest tests/test_felt_state.py`.
- Touching the SNN, modulators, or persistence? Run the benchmark suite to confirm the brain still actually learns:

  ```bash
  .venv/bin/python -m benchmark.run_all
  ```

## Code conventions

- **The frontend is English.** All user-facing strings in `train-ui/index.html` (labels, banners, hints, buttons) are English. Keep it that way, and keep the clean Neumorphism style.
- **Minimal code wins.** Prefer deleting over adding. Don't introduce an abstraction until a third use-case demands it. Skip defensive checks for inputs that can't occur — validate at the system edges (sensor input, HTTP, external APIs), trust the inner layers.
- **Comment the *why*, not the *what*.** The good comments in this codebase explain neuroscience choices and protocol contracts (see the headers in `brain/core.py` and `bridge/felt_state.py`). Keep those; don't add comments that just restate the line below them.
- **No emoji** in code, comments, or docs.
- Python: type hints throughout; small, focused functions.
- Commits: conventional-commit prefixes (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`). Keep them small and logical.

## Good first issues

Friendly places to start:

- **Training console (`train-ui/index.html`)** — it's one self-contained file. Better state visualization, clearer prompts when the brain asks about a shift, keyboard shortcuts, accessibility.
- **Felt-state model (`bridge/felt_state.py`)** — experiment with the signature, the distance metric, or how prototypes sharpen. This is the most rewarding knob to turn for "does it feel like it's learning *me*?"
- **Sensors (`adapters/mac_desktop/`)** — refine an existing sensor or its spike encoding. (Cross-platform sensors are a bigger, very welcome project.)
- **Benchmarks (`benchmark/`)** — new scenarios that probe whether learning is real and stable.
- **Docs** — if something here or in the README tripped you up, fixing it is a genuinely useful PR.

## Pull request flow

1. Branch off `main`: `git checkout -b feature/your-thing`.
2. Keep PRs focused — one idea per PR is much easier to review.
3. Run the relevant tests (and the benchmark suite if you touched the brain). Say what you ran in the PR description.
4. Explain *what* changed and *why*. Link any related issue.
5. Open it. We'll read it, ask questions, and help land it.

## Community

Be decent. This is a collaborative research project — disagree about ideas freely, never about people. Use Issues for bugs and proposals, Discussions for open questions, and PRs to make things real.

Thanks for helping a small artificial creature become someone's own.
