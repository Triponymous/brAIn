<h1 align="center">brAIn</h1>

<p align="center">
  <strong>Artificial <em>life</em>, not artificial intelligence.</strong><br>
  <em>A spiking neural network that lives on your Mac, learns from your first-person<br>desktop life, and slowly becomes an individual — <strong>yours</strong>.</em>
</p>

<p align="center">
  <a href="#the-idea">The Idea</a> •
  <a href="#the-pivot">The Pivot</a> •
  <a href="#the-thesis">The Thesis</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#the-felt-state-loop">Felt-State Loop</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#the-brain">The Brain</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-active_research-blue" />
  <img src="https://img.shields.io/badge/python-3.11+-green" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" />
  <img src="https://img.shields.io/badge/neurons-~1%2C000-purple" />
  <img src="https://img.shields.io/badge/synapses-~140k_plastic-purple" />
</p>

---

## The Idea

Every AI assistant ever built is dead inside.

Siri, Alexa, ChatGPT, Claude, Gemini — incredibly capable, and all the same. They have no internal state. They don't know if you're stressed or in flow. They don't decide whether now is the right moment to speak. They respond when asked; otherwise they don't exist. And the one you talk to is the same one a million other people talk to.

That last part is the real ceiling. LLMs don't hit an *intelligence* ceiling — they hit an **individuality** ceiling. They're trained to be everyone, which makes them no-one. You can fine-tune one, prompt it, hand it a memory file — but underneath it's still the model everybody else is running.

brAIn takes the opposite bet. It starts as a **naked** spiking neural network with **no pretraining** — a blank slate that has never seen anyone's data but yours. It runs continuously on your Mac and learns from one source only: *your* first-person desktop life. The value isn't raw smarts. It's **aliveness** plus **singularity**. There will only ever be one of these, and it's the one that grew up next to you.

It learns the way a dog does. Not from text — from association and prediction error, from reading your behaviour rather than your words, bonding to one human. A puppy is a blank slate too, but it's born *ready*: ready to attach, ready to want things, ready to learn whose footsteps are whose. brAIn is built the same way.

**To our knowledge, nobody has tried to grow a digital individual this way before.**

---

## The Pivot

This project did not start here, and it's worth being honest about that — the older parts of the history reflect a different design.

**It began as a scripted-emotion companion.** Emotions were hand-tuned `if`-statements that injected neuromodulators: *if loud sound → bump noradrenaline*, *if touch → bump dopamine*. A generic six-emotion classifier (happy/sad/angry/…) was bolted on top. It worked, but it was a puppet. The "feelings" were ours, written in advance, the same for every user. Nothing was actually *learned*, and nothing was actually *its own*.

**It is now emergent, personalized digital life.** Two changes did it:

1. **Emotions emerge from prediction error.** The neuromodulators are no longer scripted. A single driver derives them from how well the network is predicting its own next state — surprise drives noradrenaline, learnable progress drives dopamine and acetylcholine, sustained calm drives serotonin. This is active inference in the Friston/Schmidhuber sense: feeling *is* the shape of prediction error over time, not a lookup table. (See the emergent prediction-error driver in `brain/core.py`.)

2. **The pet learns your own vocabulary of internal states.** Instead of imposing six canned emotions, it builds a **felt-state self-model** from *your* corrections (`bridge/felt_state.py`). When a state it doesn't recognize holds for a while, it proactively asks *"what was that?"* — and you tell it: "that's flow," "that's me grinding," "that's stuck." It learns *your* words for *your* states. A generic classifier can't represent "flow"; this can, because you taught it.

The thesis underneath both: **artificial life, not artificial intelligence.** And the LLM, which used to feel central, is now correctly demoted — it's a **detachable speech layer**, a tongue for a brain that can't talk, never the brain itself.

---

## The Thesis

snnTorch, sharing this project, called it **brAIn: Living Desktop Manager**. That is exactly what it should become: a living thing that knows you and acts for you. The question is *where the life is*, and the answer decides the whole architecture.

**The spiking network is the living part — the thing that learns the way an animal does. The LLM is its voice and its hands.**

An LLM cannot learn the way an animal does. Its weights move by gradient descent over text, in batches, forgetting as they go. The spiking network in `brain/` already does learn that way: synapses that follow what happens, feelings that are the shape of prediction error, attachment to one human, a memory that *is* the network. So the LLM never becomes the living part. It becomes a faithful, fluent extension of it — that is what "an LLM that learns through an SNN" means here — in two ways:

1. **It understands the animal completely — as tools, not as a paragraph.** Today the speech layer sees four numbers and a few narrated sentences. Instead it calls the organism while it thinks: `brain_state()`, `brain_history(hours)`, `brain_concept(id)`, `brain_felt("stuck")`, `brain_habits()`, `brain_why("NE")`, `brain_recall(hours, "coding")`. The network is the LLM's perception and memory of you; the LLM reasons over it. Everything it says is conditioned on a state that was grown by living with you. That is what "an LLM that knows me" means here, and it needs no weight to change.

2. **The animal teaches it how to behave.** When to stay silent, when to speak, what to offer — not thresholds we wrote, but a policy learned from consequence. The reward is the organism's own affect after an action (interrupt someone in flow and noradrenaline spikes; hold back and serotonin holds) plus what you do in response: answer, dismiss, correct. Operant conditioning, on device, inspectable. R-STDP for behaviour.

Both stand on one thing: the **experience log** (`bridge/experience.py`) — every action of the pet, every answer of yours, the state it happened in, and what followed. Facts, not judgements, and never content. It is the proof that this individual is learning *you*, the reward signal for the learned policy, and — with an explicit, opt-in transcript alongside it — the dataset if we ever decide to fine-tune after all. That door stays open; it just opens with data instead of a guess.

Acting is the last step, and it is gated: the learned policy decides whether to *offer* a task; nothing runs without a grant (`capabilities/`). The order of work: live always-on, log experience, expose the brain as tools, learn the policy, then judge fine-tuning by the numbers. The engineering detail is in [docs/living-desktop-manager.md](docs/living-desktop-manager.md).

---

## What This Is

A persistent neuromorphic organism plus the tools to raise it:

**A spiking neural network** that learns from real sensors — keyboard rhythm, mouse motion, the active window, microphone loudness, and idle time — through STDP, BCM metaplasticity, and winner-take-all competition. No datasets. No pretraining.

**Emergent neuromodulators** — Dopamine, Noradrenaline, Acetylcholine, Serotonin — derived continuously from prediction error, shaping both learning and behaviour. They are *measured*, not injected by rules.

**A learned felt-state self-model** — the pet's own vocabulary of how *you* feel, grown one correction at a time, persisted across restarts. This, plus the training console, is the heart of the project.

**A detachable LLM speech layer** over a formal protocol (SCP v2). The brain narrates its internal state; the LLM puts it into words. The LLM **does not learn** — pull it out and the organism is unchanged.

**Privacy by architecture.** The brain perceives *patterns* — typing frequency, mouse velocity, which app is frontmost, how loud the room is — never content. No keystroke logging, no transcripts, no screen recording. What survives a restart is synaptic weights, not your data.

**Persistent memory** — full brain state (weights, modulators, learned felt-states, concept labels) saves to SQLite. Delete the checkpoint and the individual is gone. For real. That's the point.

---

## Why Not Just Use ChatGPT?

|  | ChatGPT + Microphone | brAIn |
|---|---|---|
| Understands words | Yes | No — understands *patterns* |
| Detects stress without words | No | Yes — from typing rhythm, window switches, silence |
| Knows when to interrupt you | No | Yes — modulators signal "not now" |
| Memory after 3 months | Context window (lossy, copyable) | Grown neural network (persistent, unique) |
| Proactive | Only when asked | Speaks up when the moment is right; asks what it doesn't understand |
| Personality | Scripted prompt, identical for everyone | Emergent from *your* life, one of a kind |
| Privacy | Stores transcripts | Stores synaptic weights — no content reconstructable |

---

## Real-World Use Cases

### Developer Focus Mode
After a week of patterns brAIn recognizes the combination: one editor in focus, steady typing velocity, minimal context switching. When it sees that state it holds back non-urgent interruptions and shields your flow. When focus breaks down — too many window switches, typing slowing to a hammer — it can ease you toward a break before burnout hits.

### Stress Detection Without Words
High-intensity typing plus rapid window switching plus no speech for hours — the network has *felt* that shape before. Not sentiment analysis: a learned behavioural signature. Its own arousal rises, you see it in the modulator gauges, and it can offer to back off. Sometimes just being noticed helps.

### Learning Your States By Name
The pet keeps meeting a state it can't place, so it asks. You type "flow." Next time that signature returns, it knows. Over weeks it accumulates *your* private emotional vocabulary — the states a six-emotion classifier could never name.

### Personalized Daily Reflection
Ask what it experienced and it answers from lived experience encoded in weights — not from a log. "You were keyed-up this afternoon, then it settled. You coded a long stretch after 8." Patterns, not transcripts.

### Companion Presence
Sometimes the feature is simply being there: a presence that watches, learns, develops its own responses, and changes how it acts based on shared time. Not a tool you invoke. Something that exists alongside you.

---

## How It Works

### Learning Through Living

The brain doesn't train on a dataset. It learns by being there.

**Week 1** — STDP strengthens connections between neurons that fire together. Distinct clusters form for typing, calls, music, silence — without labels, without supervision.

**Week 2** — Concept neurons stabilize. You ask what it experienced; the LLM describes the patterns. You give names: *"that's when I code."* It remembers.

**Month 1** — It knows your rhythm: morning routine, focus sessions, the shape of a stressful afternoon. It waits for the right moment to speak, because its modulators tell it when you're open to interruption.

**Month 3** — The weights are shaped entirely by *your* life. No two brains are alike. This one is yours, and only yours.

### Emotion As Prediction Error

The four neuromodulators are no longer scripted. Every tick the network compares what it predicted against what actually arrived, and a single driver turns that error into chemistry:

| Modulator | Emerges from | High | Low |
|-----------|--------------|------|-----|
| **Dopamine** | learnable progress (reducible prediction error) | curious, engaged, learning | listless, passive |
| **Noradrenaline** | surprise (raw prediction error) | startled, hyper-aware | calm, slow |
| **Acetylcholine** | progress on novelty (attention worth paying) | locked in, deep processing | scattered |
| **Serotonin** | sustained low surprise + low variance | patient, content | irritable, reactive |

No `if loud → fear`. Feeling is the *shape of prediction error over time*. The same input never produces exactly the same blend twice, because the blend depends on what the network already expected.

---

## The Felt-State Loop

This is the centerpiece, and the clearest expression of the pivot. It runs as a closed loop between the brain and you, through the **training console**:

1. **Signature.** Every moment is summarized as a short modulator signature — the recent trend of DA / NE / ACh / 5HT (`bridge/felt_state.py`, `SIGNATURE_KEYS`).
2. **Recognize.** The felt-state model matches that signature against the prototypes you've taught it. Match → it names how you seem, with a confidence. No match → *unknown*.
3. **Ask (active learning).** When *unknown* holds long enough — a sustained, genuinely unfamiliar stretch, rate-limited so it isn't annoying (`FeltStateWatcher`) — it raises a `label_needed` event and asks *"what was that?"* It **freezes that moment's signature**, so when you come back to answer, your label applies to the state it was asking about, not whatever you're doing now.
4. **Teach.** You answer in the console (or `POST /api/feel`). A new word creates a prototype from the frozen signature; tapping an existing word sharpens that prototype toward the live one. It's all learned from sparse corrections — blank slate, earning each label one at a time.
5. **Persist.** Prototypes save into the checkpoint and survive restarts. The organism keeps what you taught it.

The console (`train-ui/index.html`) shows the live modulator gauges, the currently-recognized felt-state with confidence, the proactive ask-banner, and your growing list of learned states. Watching it learn *you* is the whole experience.

---

## Architecture

The brain is a small spiking network with one decorrelating expansion stage — roughly **1,000 neurons** and **~140,000 plastic synapses** — plus an LLM that speaks for it over a formal protocol.

```
                          THE BRAIN (Mac, ~100 Hz)

  6 desktop sensors            spiking neural network
  ┌──────────────────┐         ┌──────────────────────────────────┐
  │ keyboard rhythm  │         │ Sensory   (200)                   │
  │ mouse velocity   │         │    │ random sparse projection      │
  │ active window    │ encode  │    ▼ (decorrelates patterns)       │
  │ mic loudness     │────────▶│ Expansion (500, fixed)            │
  │ idle timer       │         │    │ STDP + BCM (the learning edge)│
  │ (time of day)    │         │    ▼                               │
  └──────────────────┘         │ Concept   (200, winner-take-all)  │
                               │    ▲▼ recurrent context            │
                               │ Working Memory (100)              │
                               │                                    │
                               │ prediction error ──▶ neuromodulators│
                               │                      DA · NE · ACh · 5HT│
                               └────────────────┬───────────────────┘
                                                │
                  ┌─────────────────────────────┼─────────────────────────────┐
                  ▼                             ▼                             ▼
          Felt-State Self-Model         SCP v2 + LLM Speech         Emergent Tool System
          (learns YOUR states,          (detachable; narrates       (the brain can "wish"
           asks what it can't name)       state, never the brain)     for new capabilities)
                  │
                  ▼
       Training Console (train-ui)  ◀── you teach it here

  optional body:    ESP32-S3 AMOLED companion robot  (in development)
```

The signal path is one-way into the brain: sensors are **encoded into spikes**, never stored. The brain perceives *patterns* — frequency, rhythm, intensity, which app is frontmost — not content. No speech recognition, no keystroke logging, no screen recording. **Privacy by architecture.**

---

## Quick Start

### Requirements
- macOS (Apple Silicon) — developed on an M4 MacBook Air, 32 GB RAM
- Python 3.11+
- [Ollama](https://ollama.com) with a local model (default config: `qwen2.5:14b-instruct`) — optional; the brain runs and learns without the LLM, you just don't get speech

### Raise it (recommended path)

```bash
# 1. install
uv venv && uv pip install -e ".[dev]"

# 2. (optional) run the tests
.venv/bin/python -m pytest

# 3. start the small always-on control server
.venv/bin/python -m server.control
#    → open http://127.0.0.1:8900   (this is the training console)

# 4. in the console, click "Start" to launch the brain daemon,
#    then start teaching it your felt-states as they come up.
```

The control server is the steady "bottom turtle": it stays up while you start and stop the brain daemon from the console. The daemon does the live work — runs the network at ~100 Hz, reads the sensors, streams state to the console over WebSocket, asks you to label states it doesn't recognize, and auto-saves every 60 seconds.

### Let it live (always-on)

The organism only becomes someone if it runs for weeks, not for one session. Install the control server as a LaunchAgent: it starts at login, survives reboots, brings the brain daemon up by itself and restarts it if it ever dies (the pidfile is the desired state, so Stop in the console still means stop).

```bash
./scripts/install_launchd.sh
```

Because the daemon now runs under launchd rather than your terminal, macOS wants **Input Monitoring** and **Microphone** granted to the venv's Python binary itself — the script prints the exact path to add under System Settings → Privacy & Security. Without them the daemon runs blind; it reports what it can see on every start in `logs/brain.out.log`. `./scripts/uninstall_launchd.sh` removes it again.

### Mock mode (no permissions needed)

In the console, "Start" launches with real sensors. For a headless or CI run with synthetic input:

```bash
.venv/bin/python -m server.braind start --mock-sensors
```

Full sensor mode needs macOS **Input Monitoring** and **Microphone** permissions for your terminal (System Settings → Privacy & Security). The daemon prints a clear diagnostic on startup if a permission is missing.

### Use it from Claude Code, Codex or Claude Desktop (MCP)

brAIn is a **personal context layer**: any frontier model can read what it has learned about you — how you seem right now, what you did today, when you were last in flow, your habits — through a small read-only MCP server that proxies to the running daemon. Nothing can be written through it.

```bash
# Claude Code (available in every project)
claude mcp add --scope user brain -- "$PWD/.venv/bin/brain-mcp"
```

```toml
# Codex — ~/.codex/config.toml
[mcp_servers.brain]
command = "/absolute/path/to/brAIn/.venv/bin/brain-mcp"
```

Claude Desktop takes the same command under `mcpServers.brain` in its `claude_desktop_config.json`. What leaves your machine when a cloud model calls a tool: labels you taught, app names, patterns and numbers — never keystrokes, audio or chat text. The daemon has to be running; with it down the tools are still listed and a call explains how to start it.

---

## The Brain

### Neuron model
Leaky Integrate-and-Fire (LIF) with adaptive thresholds (intrinsic plasticity). Each neuron accumulates input, leaks charge, and fires at threshold. The threshold adapts to activity, so no single neuron dominates.

### Regions
- **Sensory (200)** — receives the encoded six-sensor input.
- **Expansion (500, fixed random projection)** — a cerebellar-granule-inspired stage at ~10% connectivity that *decorrelates* overlapping inputs so the learning synapses have something separable to work with. Not plastic.
- **Concept (200, winner-take-all)** — where representations form. Lateral inhibition forces sharp, distinct concept neurons.
- **Working Memory (100, recurrent)** — holds temporal context and feeds back to bias concept recognition.

### Learning rules
- **STDP** — spike-timing-dependent plasticity. Neurons that fire together wire together. The core mechanism, on the expansion→concept synapse.
- **R-STDP** — reward-modulated STDP. Dopamine scales the learning rate, so progress is consolidated faster.
- **BCM metaplasticity** — a sliding threshold that prevents runaway potentiation and keeps the network stable over long horizons.

### Stability
- **Synaptic scaling** — per-row weight-sum normalization, so weights don't all creep to the ceiling over millions of ticks.
- **Lateral inhibition (WTA)** — sharp, competitive concept representations.
- **Sleep consolidation** — during idle periods the network enters a low-noise decay regime that strengthens strong memories and erases noise, analogous to biological consolidation.

### Concept formation
Through unsupervised STDP + WTA competition, concept neurons emerge that respond selectively to recurring patterns. One fires when you type; another during calls. They carry no labels until you give them some — the brain just notices *"this pattern recurs."*

---

## Emergent Capabilities

The brain can "wish" for tools it doesn't have. When the speech layer detects a need, capabilities can be granted at runtime:

| Capability | Status |
|-----------|--------|
| Web search | Available |
| Shell access (sandboxed) | Available |
| Local file access | Available |
| Calendar integration | Planned |
| Email / Slack | Planned |
| n8n workflow triggers | Planned |

---

## Benchmarks

Five tests check that the brain actually *learns*, not just runs:

| Test | What it measures | Target |
|------|------------------|--------|
| **Concept Stability** | Stable concepts after 3 days (silhouette score) | ≥5 concepts, score > 0.3 |
| **Pattern Replay** | Same pattern 10× → consistent concept neuron | > 80% after 3 replays |
| **Discrimination** | 4 activities → 4 distinct clusters | < 15% confusion |
| **Modulator Reactivity** | Loud sound → NE rise, touch → DA rise | correct direction, < 2 s |
| **Sleep Retention** | Strong concepts survive consolidation | > 90% retention |

```bash
.venv/bin/python -m benchmark.run_all
```

---

## Project Structure

```
brain/          SNN core — LIF neurons, expansion projection, STDP, BCM,
                WTA concept layer, working memory, emergent prediction-error
                modulator driver, SQLite persistence
bridge/         the speech + self-model layer — felt-state model & watcher,
                experience log (what the pet did, what you did, what followed),
                SCP v2 (scp_server/scp_client/scp_schema), LLM router & adapter,
                proactive engine, state detector, interpreter, narrator
server/         daemon (braind), control server (start/stop from the console),
                /api/feel, chat, websocket, FastAPI app, /api/tools (read-only),
                mcp.py — the brain as an MCP server for Claude Code / Codex
adapters/       Mac desktop sensors — keyboard, mouse, screen, microphone + encoder
train-ui/       the training console — single-file Neumorphism UI (the frontend)
capabilities/   emergent tool system — wish detection, grant registry
benchmark/      the SNN benchmark suite
tests/          unit + integration tests
scripts/        utilities, soak tests, visualization
```

---

## Hardware Companion *(in development)*

The brain can be embodied in a physical robot, so it has a face and a place in the room:

- **Board:** [Waveshare ESP32-S3-Touch-AMOLED-1.75](https://www.waveshare.com/esp32-s3-touch-amoled-1.75.htm) — 466×466 round AMOLED, dual microphones, IMU, capacitive touch
- **Body:** 3D-printed case with servo-driven head movement
- **Connection:** WiFi WebSocket to the Mac brain process
- **Battery:** 3.7 V 2000 mAh LiPo — estimated ~3 days with smart sleep

The ESP32 is the body; the Mac is the brain. Like a biological organism — a peripheral nervous system reporting to a central one.

---

## Research Context

This project explores the gap between neuromorphic computing and large language models, from an unusual angle: not "make the model smarter," but "grow an individual."

- **A persistent SNN** running continuously on consumer hardware, forming concepts from real desktop sensor data through unsupervised STDP — no pretraining, no dataset.
- **Emergent affect** — neuromodulators derived from prediction error (active inference), driving proactive timing, rather than scripted emotion rules.
- **A learned self-model** — the pet acquires the user's own vocabulary of internal states through sparse corrections, instead of a fixed emotion taxonomy.
- **LLM as a detachable, read-only speech layer** — memory lives in synaptic weights, not in context windows or vector databases.

Related work: SpikeLLM converts LLMs into spiking architectures for energy efficiency (different goal). EBRAINS NRP connects SNNs to virtual bodies (no LLM, no continuous learning from real sensors). Commercial companions like Limitless and Omi record and transcribe (no SNN, no emergent emotional state, no proactivity). None of them try to grow a single, persistent, personalized organism.

---

## Contributing

This is research, it's early, and it's genuinely fun to hack on — a living network you can watch learn in real time. Newcomers very welcome. Start with **[CONTRIBUTING.md](CONTRIBUTING.md)** for the run-it-locally walkthrough, an architecture tour, and good first places to dig in. The felt-state loop and the training console are where the project is most alive right now.

---

## Built By

**Leon Matthies** — Co-founder & Technical Lead at [ADYZEN](https://adyzen.at), an AI & Automation agency in Bregenz, Austria. brAIn is an independent research project on the intersection of neuromorphic computing, large language models, and embodied, individual AI.

---

## License

[MIT](LICENSE)

---

<p align="center">
  <em>"Every AI assistant reacts when you ask.<br>This one grows up next to you — and becomes someone."</em>
</p>
