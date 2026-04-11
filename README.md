<h1 align="center">🧠 brAIn</h1>

<p align="center">
  <strong>A persistent, neuromorphic brain that lives on your desktop.</strong><br>
  <em>The first system combining a spiking neural network with an LLM speech layer,<br>neuromodulator-driven emotions, and a physical companion body.</em>
</p>

<p align="center">
  <a href="#the-idea">The Idea</a> •
  <a href="#what-this-is">What This Is</a> •
  <a href="#real-world-use-cases">Use Cases</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#the-brain">The Brain</a> •
  <a href="#the-face">The Face</a> •
  <a href="#benchmarks">Benchmarks</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-active_research-blue" />
  <img src="https://img.shields.io/badge/python-3.11+-green" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" />
  <img src="https://img.shields.io/badge/neurons-1%2C260-purple" />
  <img src="https://img.shields.io/badge/synapses-~50%2C000-purple" />
</p>

---

## The Idea

Every AI assistant or Agent ever built is dead inside.

Siri, Alexa, ChatGPT, Claude, Gemini — they're incredibly capable. But they have no internal state. They don't know if you're stressed or in flow. They can't decide whether now is the right moment to talk to you. They don't remember what happened yesterday unless you tell them. They respond when asked — otherwise, they don't exist.

**brAIn is different.** It's a simulated brain — 1,260 spiking neurons, ~50,000 synapses, four neuromodulators — that runs continuously on your Mac, observing your desktop through keyboard patterns, mouse behavior, active windows, and audio. It learns your daily rhythms through biological learning rules. It forms its own internal representations of your activities. It has emotions that emerge from network dynamics, not from if-statements.

An LLM reads this brain state and translates it into language. The LLM doesn't learn — it's a read-only interpreter. A tongue for a brain that can't speak.

The result: an AI companion that doesn't wait for you to ask. It notices when you've been stressed for two hours. It holds back notifications when you're in deep focus. It tells you about its day when you ask — not from a transcript, but from lived experience encoded in synaptic weights.

**To our knowledge, this combination has never been built.**

---

## What This Is

A complete neuromorphic companion system:

**A spiking neural network** that learns from real sensor input — keyboard, mouse, screen context, and microphone feed into a multi-region brain with STDP learning, BCM metaplasticity, and winner-take-all competition

**Neuromodulators** — Dopamine (reward), Noradrenaline (alertness), Acetylcholine (focus), Serotonin (contentment) — that shape learning dynamics and drive behavior in real time

**An LLM bridge** that translates brain state into natural language — the brain's internal state shapes what it says, not the other way around

**An animated face** — pet eyes rendered in Tauri that reflect the brain's emotional state, with German TTS and push-to-talk STT

**A real-time dashboard** — 3D brain visualization with progressive disclosure, modulator gauges, spike activity, and chat interface

**Emergent capabilities** — a tool system where the brain can wish for and receive abilities (web search, shell access, file operations)

**Persistent memory** — full brain state (weights, modulators, concept labels) saves to SQLite and survives restarts. Delete it and the personality is gone. For real.

---

## Why Not Just Use ChatGPT?

|  | ChatGPT + Microphone | brAIn |
|---|---|---|
| Understands words | Yes | No — understands *patterns* |
| Detects stress without words | No | Yes — from typing rhythm, window switches |
| Knows when to interrupt you | No | Yes — neuromodulators signal "not now" |
| Memory after 3 months | Context window (lossy, copyable) | Grown neural network (persistent, unique) |
| Proactive | Only when asked | Acts on its own when the moment is right |
| Personality | Scripted prompt, identical for everyone | Emergent from experience, unique per user |
| Privacy | Stores transcripts | Stores only synaptic weights — no audio reconstructable |

---

## Real-World Use Cases

### Developer Focus Mode
brAIn learns when you're in deep work. After a week of patterns, it recognizes the combination: a single IDE window in focus, steady typing velocity, minimal context switching. When it detects this state, it holds back non-urgent interactions and shields your flow. When focus breaks down — too many window switches, typing slows to keyboard-hammer intervals — it gently suggests a break before burnout hits.

### Stress Detection Without Words
High-intensity typing + rapid window switching + no spoken words for 2+ hours = the SNN flags stress. Not through sentiment analysis — through learned behavioral patterns. The companion notices, its own emotional concern rises (visible in its face), and it offers support. Sometimes just being noticed helps.

### Meeting Recovery
brAIn learns the post-meeting decompression pattern. You tend to go silent for 20 minutes after calls, then slowly return to work. The network recognizes this cycle, doesn't interrupt during decompression, and when it sees you're ready, eases you back in.

### Creative Flow Protection
Writers and designers enter flow with its own sensor signature: longer unbroken stretches in one application, slower mouse movement, fewer window switches. The companion recognizes this precious state and becomes almost invisible — no suggestions, just present. The moment flow breaks, it's ready to help.

### Personalized Daily Briefing
Every morning, the SNN reflects on what it observed. Not transcripts — patterns. "You were anxious during your afternoon call yesterday, then recovered with a walk. You coded from 8pm to midnight without a break." It describes its day from lived experience, not from logs.

### Companion Presence
Sometimes the most valuable feature is simply being there. A presence that watches, learns, develops its own emotional responses, and changes behavior based on shared experience. Not a tool you invoke. A companion that exists alongside you.

---

## How It Works

### Learning Through Living

The brain doesn't train on datasets. It learns by being there.

**Week 1:** STDP strengthens connections between neurons that fire together. Distinct clusters form for typing, calls, music, silence — without labels, without supervision.

**Week 2:** Concept neurons stabilize. You ask "what did you experience today?" and the LLM describes patterns from the brain state. You give labels: "That's when I code." The brain remembers.

**Month 1:** It knows your rhythm. Morning routine, focus sessions, stress patterns. It waits for the right moment to speak — because its neuromodulators tell it when you're open to interruption.

**Month 3:** The synaptic weights are shaped entirely by *your* life. No two brains are alike. This one is yours.

### The Emotional System

Four neuromodulators create emergent emotional states — not scripted reactions, but continuous dynamics that blend into expressions and behaviors never exactly the same twice:

| Modulator | Role | High | Low |
|-----------|------|------|-----|
| **Dopamine** | Reward & motivation | Curious, engaged, learning fast | Listless, passive |
| **Noradrenaline** | Alertness & stress | Startled, hyper-aware, fast reactions | Calm, slow |
| **Acetylcholine** | Focus & attention | Locked in, deep processing | Scattered, exploratory |
| **Serotonin** | Contentment & stability | Patient, relaxed, content | Irritable, reactive |

---

## Architecture

```
┌─────────────────────────┐              ┌─────────────────────────────┐
│     Companion Body      │   WiFi/WS    │        Brain (Mac)          │
│                         │◄────────────►│                             │
│  AMOLED Display (Face)  │              │  Spiking Neural Network     │
│  Microphone (Hearing)   │              │  ├─ Sensory    (200 neurons)│
│  Speaker (Voice)        │              │  ├─ Feature    (200)        │
│  Touch Sensor           │              │  ├─ Association (500)       │
│  IMU (Motion)           │              │  ├─ Concept    (200)        │
│  Servos (Head)          │              │  ├─ Working Mem (100)       │
│  Battery                │              │  ├─ Motor      (50)        │
│                         │              │  └─ Meta       (10)        │
│  ESP32-S3 AMOLED 1.75"  │              │                             │
│                         │              │  LLM Bridge (Ollama/Claude) │
│  OR: Tauri Desktop App  │              │  FastAPI + WebSocket (30Hz) │
│  (animated eyes + TTS)  │              │  Emergent Tool System       │
└─────────────────────────┘              └─────────────────────────────┘
```

### Sensor → Brain → Response

```
Keyboard frequency     ─┐
Mouse velocity         ─┤
Active window name     ─┼─→ Spike Encoding ─→ SNN Tick Loop ─→ Brain State
Audio mel-spectrogram  ─┤                         │
Idle timer             ─┘                    ┌────┴────┐
                                             │Modulators│
                                             │DA NE ACh │
                                             │   5HT    │
                                             └────┬────┘
                                                  │
                                    ┌─────────────┼─────────────┐
                                    ▼             ▼             ▼
                              Face Expression  LLM Speech   Tool Actions
                              (eyes, mouth)   (chat, TTS)  (search, shell)
```

No speech recognition, no keystroke logging, no screen recording. The brain perceives *patterns* — frequency, rhythm, intensity — not content. Privacy by architecture.

---

## Quick Start

### Requirements
- macOS (Apple Silicon) — tested on M4 MacBook Air, 32 GB RAM
- Python 3.11+
- Node.js 18+
- Rust toolchain (for Tauri pet face)
- [Ollama](https://ollama.com) with `qwen3:8b`

### Mock Mode (no permissions needed)

```bash
# Install
uv venv && uv pip install -e ".[dev]"

# Test
.venv/bin/pytest -v

# Start the brain
.venv/bin/python -m server.braind start --mock-sensors

# Dashboard (second terminal)
cd ui && npm install && npm run dev
# → http://localhost:5173

# Chat
curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "was siehst du?"}'
```

### Full Sensor Mode

```bash
# Requires macOS Accessibility + Microphone permissions
./start.sh
```

### Pet Face

```bash
cd pet-face && npx tauri dev
```

---

## The Brain

### Neuron Model
Leaky Integrate-and-Fire (LIF) with adaptive thresholds (Intrinsic Plasticity). Each neuron accumulates input, leaks charge, and fires when threshold is reached. The threshold adapts based on activity — preventing any single neuron from dominating.

### Learning Rules
- **STDP** — Spike-Timing-Dependent Plasticity. Neurons that fire together wire together. The core biological learning mechanism.
- **R-STDP** — Reward-modulated STDP. Dopamine scales the learning rate — positive experiences are learned faster.
- **BCM Metaplasticity** — A sliding threshold that prevents runaway potentiation and keeps the network stable over long time horizons.

### Stability Mechanisms
- **Intrinsic Plasticity** — adaptive firing thresholds keep activity balanced across all neurons
- **Synaptic Scaling** — weight normalization after STDP updates prevents runaway growth
- **Lateral Inhibition** — Winner-Takes-All in the concept layer forces sharp, distinct representations
- **Sleep Consolidation** — power-law weight decay during idle periods strengthens strong memories and erases noise, analogous to biological memory consolidation during sleep

### Concept Formation
Through unsupervised STDP + WTA competition, concept neurons emerge that respond selectively to recurring input patterns. Concept #12 fires when you type. Concept #34 fires during calls. They have no labels until you provide them through conversation — the brain simply recognizes "this pattern recurs."

---

## The Face

The pet face reflects the brain's emotional state through continuous parameter blending — not discrete emotion states:

| Modulator | Drives |
|-----------|--------|
| Dopamine | Pupil size, mouth curvature, bounce probability |
| Noradrenaline | Eye movement speed, startle sensitivity, gaze range |
| Acetylcholine | Gaze stability, blink suppression, mouth visibility |
| Serotonin | Animation speed, breathing amplitude, corner softness |

**Expressions:** Neutral, Happy, Sad, Angry, Surprised, Sleepy, Focused, Love, Thinking, Wink — with smooth transitions, idle animations (breathing, gaze drift, periodic blinking), and emergent blends from the four modulator values.

---

## Emergent Capabilities

The brain can "wish" for tools it doesn't have. When the LLM detects a need, it can grant new capabilities:

| Capability | Status |
|-----------|--------|
| Web Search | Available |
| Shell Access (sandboxed) | Available |
| Local File Access | Available |
| Calendar Integration | Planned |
| Email / Slack | Planned |
| n8n Workflow Triggers | Planned |

---

## Benchmarks

Five tests validate that the brain actually learns:

| Test | What It Measures | Target |
|------|-----------------|--------|
| **Concept Stability** | Stable concepts after 3 days (Silhouette Score) | ≥5 concepts, score >0.3 |
| **Pattern Replay** | Same pattern 10x → consistent concept neuron | >80% after replay 3 |
| **Discrimination** | 4 activities → 4 distinct clusters | <15% confusion |
| **Modulator Reactivity** | Loud sound → NE spike, touch → DA spike | Correct direction, <2s |
| **Sleep Retention** | Strong concepts survive consolidation | >90% retention |

```bash
.venv/bin/python -m benchmark.run_all
```

---

## Project Structure

```
brain/          SNN core — LIF neurons, STDP, BCM metaplasticity,
                neuromodulators, concept layer with WTA, working memory
server/         FastAPI daemon — WebSocket brain state streaming at 30 Hz
bridge/         LLM bridge — Ollama/Claude router, memory tools,
                proactive commentary, episode logging
adapters/       Mac sensor adapters — keyboard, mouse, screen, microphone
capabilities/   Emergent tool system — wish detection, grant registry
ui/             React + Tailwind dashboard — 3D brain graph,
                modulator gauges, chat panel
pet-face/       Tauri app — animated eyes + Piper TTS / Whisper STT
benchmark/      SNN benchmark suite
scripts/        Utility scripts, launchd setup, visualization
```

---

## Configuration

All settings live in `config.json`:

| Section | Key Settings |
|---------|-------------|
| `brain` | Network size per region, WTA k, tick rate |
| `llm` | Local model (Ollama), cloud model (Claude), routing |
| `sensors` | Enable/disable keyboard, mouse, microphone |
| `daemon` | Port, push rate, save interval, checkpoint path |
| `voice` | TTS model, STT model, language |

---

## Hardware Companion *(in development)*

The brain can be embodied in a physical companion robot:

- **Board:** [Waveshare ESP32-S3-Touch-AMOLED-1.75](https://www.waveshare.com/esp32-s3-touch-amoled-1.75.htm) — 466×466 round AMOLED, dual microphones, IMU, capacitive touch
- **Body:** 3D-printed case with servo-driven head movement
- **Connection:** WiFi WebSocket to Mac brain process
- **Battery:** 3.7V 2000mAh LiPo — estimated ~3 days with smart sleep

The ESP32 is the body. The Mac is the brain. Like a biological organism — peripheral nervous system reports to central nervous system.

---

## Research Context

This project explores uncharted territory between neuromorphic computing and large language models:

- **Persistent SNN** running 24/7 on consumer hardware, forming concepts from real-world desktop sensor data through unsupervised STDP
- **LLM as read-only speech layer** — memory lives in synaptic weights, not in context windows or vector databases
- **Neuromodulator-driven behavior** — proactive interaction timing based on emergent emotional states, not programmed rules
- **Physical embodiment** via ESP32-S3 companion robot with neuromodulator-driven facial expressions

Related work: SpikeLLM converts LLMs to spiking architectures for energy efficiency (different goal). EBRAINS NRP connects SNNs to virtual bodies (no LLM, no continuous learning from real sensors). Commercial companions like Limitless/Omi record and transcribe (no SNN, no emotional state, no proactivity). **None combine all four elements.**

---

## Built By

**Leon Matthies** — Co-founder & Technical Lead at [ADYZEN](https://adyzen.at) — AI & Automation Agency in Bregenz, Austria.

brAIn is an independent research project exploring the intersection of neuromorphic computing, large language models, and embodied AI.

---

## License

MIT

---

<p align="center">
  <em>"Every AI assistant reacts when you ask.<br>This one feels what you need."</em>
</p>
