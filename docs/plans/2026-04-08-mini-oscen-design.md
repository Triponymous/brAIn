# Mini-OSCEN — Design Document

**Date:** 2026-04-08
**Status:** Approved, ready for implementation planning
**Inspiration:** OSCEN (oscen.ai) — neuromorphic AI with 1M spiking neurons, STDP learning, developmental phases. This project is a radically scaled-down, single-developer version that preserves the *spirit* (continuous learning, no datasets, no backprop, persistent memory) and adds an LLM bridge to make the brain conversational.

---

## Vision

Build a **persistent neuromorphic "brain"** that:

1. Lives continuously (24/7 if you want), learning from real sensor input via STDP — no datasets, no training/inference split, no forgetting between sessions.
2. Can be **talked to** via a frozen LLM acting as a "language cortex" that queries the brain's state through tools. The LLM does not learn — the brain does.
3. Can run in two modes via a pluggable adapter pattern:
   - **Desktop mode** — webcam + microphone, "sits on your desk and remembers."
   - **Avatar mode** — small Pygame world, the brain inhabits a body that learns to find food and avoid danger via R-STDP.
4. Has a **live web dashboard** where you can watch the brain think — regions, spikes, neuromodulators, concepts forming.

This is not a product. It is a research/learning/demo system whose value comes from understanding neuromorphic computing deeply, having a striking demo, and exploring a genuinely novel architecture (LLM + persistent SNN) that current AI systems don't offer.

## Non-Goals

- Compete with LLMs at language tasks. The LLM is a frozen tool.
- Match OSCEN's scale (1M neurons). We target ~5–10k.
- Smart-home integration. (User has no devices.) Adapter pattern leaves it open for the future.
- Multi-user, auth, cloud deployment, Docker, CI, plugin discovery, embeddings/RAG, fine-tuning.

## What Mini-OSCEN Will Be Able to Do

- Form **concept neurons** for recurring sensor patterns without labels (unsupervised).
- Build **cross-modal associations** (sight ↔ sound) via STDP.
- Answer questions about its own past via the LLM bridge ("what did you see this morning?", "where did you last see X?").
- Learn a simple Pygame agent task (find food, avoid danger) via R-STDP, visibly, in minutes.
- Persist memory across restarts indefinitely.
- Show its own thinking in real time on a web dashboard.

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                        SNN BRAIN (core)                        │
│   Sensory → Feature → Association → Concept → Motor + Meta    │
│                  + Working Memory + Modulators                 │
└──────┬─────────────────────────────────────────────────┬───────┘
       │ Spike I/O                                       │ State
       │                                                 ▼
┌──────┼─────────────┐                          ┌────────────────┐
│      │             │                          │  LLM BRIDGE    │
│  ┌───▼───┐  ┌──────▼──┐                       │                │
│  │Desktop│  │ Avatar  │                       │ Exporter       │
│  │Adapter│  │ Adapter │                       │ Memory Tools   │
│  │       │  │         │                       │ Ollama/Claude  │
│  │ Web-  │  │ Pygame  │                       │                │
│  │ cam   │  │ World   │                       └────────┬───────┘
│  │ + Mic │  │         │                                │
│  └───────┘  └─────────┘                                │
└────────────────────────────────────────────────────────┼───────┐
                                                         │       │
                ┌────────────────────────────────────────┘       │
                │                                                │
                ▼                                                ▼
       ┌──────────────────┐                          ┌──────────────────┐
       │  FastAPI Server  │  WebSocket (30 Hz state) │  React Dashboard │
       │  /ws  /chat      │ ────────────────────────>│  Brain viz +     │
       │                  │  REST (chat, history)    │  Chat UI         │
       └──────────────────┘ <────────────────────────│                  │
                                                     └──────────────────┘
```

## Component 1: SNN Brain (Core)

**Framework:** Python 3.11 + `snnTorch` (PyTorch-based).

**Size:** ~5,000–10,000 LIF neurons, ~500k–1M synapses. CPU-real-time on a modern Mac.

**Regions:**

| Region | Neurons | Role |
|---|---|---|
| Sensory | 2,000 | Encodes raw input to spike trains (vision/audio/events) |
| Feature | 1,000 | Learns low-level features via STDP (edges, tones, event types) |
| Association | 2,500 | Cross-modal binding; "A goes with B" |
| Concept | 500 | Winner-take-all, sparse. Each Concept neuron = one learned thing |
| Working Memory | 500 | Recurrent short-term context (~seconds) |
| Motor | 500 | Output layer, fires into adapter |
| Meta | 100 | Modulator levels (DA, NE, ACh, 5HT), attention |

**Learning rules:**
- STDP on Sensory→Feature, Feature→Association, Association→Concept
- R-STDP on Motor pathways
- Eligibility traces for consolidation
- No backpropagation, no datasets

**Persistence:** Weights serialized to SQLite (BLOBs) every ~30s. Episode logs to Parquet. On restart: seamless resume. **Brain never forgets across sessions** — the core pitch.

**Tick rate:** 100–1000 Hz simulated time, configurable real-time factor.

## Component 2: LLM Bridge

**Three sub-components:**

### 2a. Brain State Exporter

Background process snapshots brain state ~1 Hz into LLM-readable JSON:

```json
{
  "timestamp": "2026-04-08T22:35:12",
  "active_concepts": [
    {"id": 47, "label": null, "activation": 0.91, "first_seen": "2026-04-05", "times_seen": 312},
    {"id": 12, "label": "morning_light", "activation": 0.68, "first_seen": "2026-04-01", "times_seen": 89}
  ],
  "modulators": {"dopamine": 0.4, "noradrenaline": 0.7, "acetylcholine": 0.5},
  "recent_episode": "high novelty in sensory cortex 30s ago, association→concept binding strengthened",
  "mode": "desktop"
}
```

Concepts are unlabeled until the user labels them via the LLM.

### 2b. LLM Runtime

- **Primary:** Ollama with Llama 3.2 3B or Qwen 2.5 7B (local, free, private).
- **Fallback:** Anthropic SDK (Claude) — config-switchable.
- LLM receives system prompt explaining its role, plus current brain state, plus tool definitions.
- LLM is **never fine-tuned**. It is a frozen tongue for a learning brain.

### 2c. Memory Tools

The LLM queries the brain via tools (not by reading raw weights):

- `query_concepts(filter)` — top concepts active in time range
- `episode_search(time_range)` — what was notable when
- `label_concept(concept_id, label)` — user assigns a name to a concept
- `recall_associations(concept_id)` — what co-activates with concept X
- `current_state()` — full snapshot

**No embeddings, no vector DB.** Memory lives in the SNN — using a vector store would defeat the entire point.

## Component 3: Adapters

Common interface:

```python
class Adapter:
    def read_sensors() -> dict[str, np.ndarray]
    def encode_to_spikes(sensors) -> SpikeBatch
    def decode_motor_spikes(spikes) -> dict
    def apply_motor(action) -> reward_signal
```

### 3a. Desktop Adapter

- **Vision:** OpenCV webcam, 64×64 grayscale, 10 fps, rate-coded
- **Audio:** `sounddevice` mic, 16 kHz mono → 32-band mel-spectrogram, rate-coded
- **Time:** 2 tonic neurons firing slow cycles (day-time proxy)
- **Motor:** Fires into the void in V1 (no actuators). Optionally: an attention crop window controlled by motor output (V2).
- **Reward:** Intrinsic novelty (predictive layer surprise → DA spike). Optional explicit reward via LLM ("that was a person").

**Persona:** A small box on your desk that watches your world and remembers.

### 3b. Avatar Adapter

- **World:** 2D Pygame top-down environment with avatar, food (positive reward), danger (negative reward), neutral objects, walls.
- **Vision:** 32×32 patch around avatar.
- **Proprioception:** 4 neurons reporting velocity/heading.
- **Drive:** A slowly rising "hunger" value motivates food-seeking.
- **Motor:** 4 sub-areas (forward/left/right/stop) → Pygame physics.
- **Reward:** Food touch = DA spike. Danger touch = NE spike (negative).

**Persona:** A creature growing up in a tiny world that you can watch learn — and ask about its experiences.

### Mode Isolation

Each mode has its own brain state file. V1 runs one mode at a time. Mixing would cause catastrophic confusion. Two brains, one codebase.

## Component 4: Dashboard

### Architecture

- **Backend:** FastAPI in the brain process. WebSocket pushes ~30 Hz state. REST endpoints for history + chat proxy.
- **Frontend:** Vite + React 18 + TypeScript + Tailwind + shadcn/ui. Plain Canvas 2D for the brain viz (no chart libs).

### Layout

Three-column:

- **Left — Sensors:** webcam feed, mic mel-spectrogram (live)
- **Center — Brain:** region graph (animated nodes/edges), spike raster (5s window), neuromodulator bars, concept cloud
- **Right — Chat:** conversation with the LLM bridge, inline tool-call indicators

**Time Scrubber** (V1): bottom strip, last 5 minutes of brain activity, scrub to view past state snapshots. Reinforces "this thing exists continuously in time."

### Out of Scope (V1)

No login, no multi-user. No hyperparameter knobs in UI (config file + restart). Dark only. No theming.

## Component 5: Tech Stack Summary

| Layer | Choice |
|---|---|
| SNN Core | Python 3.11 + snnTorch + PyTorch + NumPy |
| Persistence | SQLite (weights) + Parquet (episodes) |
| Server | FastAPI + uvicorn + websockets |
| LLM (local) | Ollama + Llama 3.2 3B / Qwen 2.5 7B |
| LLM (cloud, optional) | Anthropic SDK |
| Sensors | OpenCV, sounddevice, Pygame |
| Frontend | Vite + React 18 + TypeScript + Tailwind + shadcn/ui |
| Brain viz | Plain Canvas 2D |
| Build/test | uv, pnpm, pytest, vitest |

## Repo Layout

```
brAIntest/
├── README.md
├── pyproject.toml
├── docs/plans/
├── brain/
│   ├── neurons.py        # LIF model
│   ├── synapses.py       # STDP, R-STDP, eligibility traces
│   ├── regions.py
│   ├── modulators.py
│   ├── core.py           # tick loop
│   └── persistence.py
├── adapters/
│   ├── base.py
│   ├── desktop.py
│   └── avatar.py
├── bridge/
│   ├── exporter.py
│   ├── tools.py
│   ├── llm_local.py
│   └── llm_remote.py
├── server/
│   ├── main.py
│   ├── ws.py
│   └── chat.py
├── ui/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       ├── components/
│       │   ├── BrainCanvas.tsx
│       │   ├── SpikeRaster.tsx
│       │   ├── ConceptCloud.tsx
│       │   ├── Modulators.tsx
│       │   ├── SensorPanel.tsx
│       │   └── ChatPanel.tsx
│       └── lib/ws.ts
├── tests/
└── scripts/
    ├── run_desktop.sh
    └── run_avatar.sh
```

## Build Phases

Each phase ends with something visibly working. No invisible-build phases.

**Phase 1 — SNN Core foundations**
LIF neurons + STDP. 2 regions, synthetic input, pytest verifies weights change correctly. Sanity plot of weight matrix delta.

**Phase 2 — Full brain + persistence**
All regions, modulators, tick loop, save/load. 1h soak test, restart, verify seamless resume. CLI shows emergent concept formation on synthetic input.

**Phase 3 — Desktop adapter**
Webcam + mic encoding, live loop. Concept neurons consistently activate for the user's face. CLI prints active concepts.

**Phase 4 — FastAPI + minimal dashboard**
WebSocket pusher, React frontend with region graph + spike raster only. **Wow moment 1: you see the brain think live in your browser.**

**Phase 5 — LLM bridge**
Exporter, memory tools, Ollama integration, chat endpoint, chat UI. **Wow moment 2: the brain talks to you about itself, demonstrably from real brain state.**

**Phase 6 — Avatar adapter + polish**
Pygame world, avatar adapter, R-STDP loop. Mode switching. Time scrubber, concept cloud, modulator bars. **Wow moment 3: emergent food-seeking without backprop.**

After Phase 6, MVP is complete.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| **Catastrophic forgetting** as the brain runs for days | Eligibility traces + sparse winner-take-all in concept layer + biologically-motivated decay. Not solved fully — accept as known limitation. |
| **STDP doesn't form useful concepts** with naturalistic input | Start with very controlled synthetic input in Phase 2; only move to real sensors in Phase 3 once we trust the dynamics. |
| **CPU real-time is too slow** at target neuron count | Profile early. Reduce neuron count or move tensor ops to MPS/CUDA. snnTorch supports both. |
| **LLM hallucinates brain state** instead of using tools | Strict system prompt + tool-use enforcement + show tool call traces in UI for debugging. |
| **Pygame avatar never learns** | Start with degenerate environment (1 food, no danger) and simplest possible reward. Add complexity only after baseline works. |
| **Scope creep** (the user keeps wanting "all of it") | This document. Refer back to it. Anything not in it goes to a `FUTURE.md`. |

## Open Questions (deferred to implementation)

- Exact LIF parameters (membrane time constant, threshold) — start with snnTorch defaults, tune empirically.
- STDP window shape (symmetric vs asymmetric, exact tau values) — same.
- Concept layer size: 500 may be too few or too many. Start there, observe.
- Whether to use MPS (Apple Silicon) for tensor ops or stay CPU. Profile in Phase 2.

## Success Criteria for MVP

1. Brain runs continuously for ≥24h without crashing or runaway gradients.
2. Survives restart with no behavioral discontinuity.
3. Forms ≥10 stable concept neurons in desktop mode within 1h of normal use.
4. LLM bridge answers ≥3 questions about brain history with correct, tool-derived data.
5. Avatar mode: agent visibly improves food-finding within 30 minutes of training.
6. Dashboard renders at ≥30 fps with full brain visible.
