# Brain

A persistent, neuromorphic brain built on spiking neural networks — it observes your desktop, forms concepts through STDP learning, and communicates via an LLM bridge. The SNN runs continuously, developing its own internal representations of your activity patterns while neuromodulators (dopamine, norepinephrine, serotonin, acetylcholine) shape learning in real time.

## What this is

A complete neuromorphic system that:

- **Learns from real sensor input** — keyboard, mouse, screen, microphone feed into a multi-region spiking neural network
- **Forms concepts autonomously** — winner-take-all competition + STDP produce sparse, distinct concept neurons that differentiate activity patterns
- **Remembers across restarts** — full brain state (weights, modulators, labels) persists to SQLite
- **Talks back** — an LLM bridge translates brain state into natural language; the brain's internal state shapes what it says
- **Has a face** — animated pet eyes (Tauri) + German TTS/STT with push-to-talk
- **Grants itself tools** — emergent capability system where the brain can wish for and receive tools (web search, shell, file access)

## Requirements

- **macOS** (Apple Silicon) — tested on MacBook Air M4, 32 GB RAM
- **Python 3.11+**
- **Node.js 18+** (for dashboard and pet face)
- **Rust toolchain** (for Tauri pet face)
- [Ollama](https://ollama.com) with `qwen3:8b` (or configure another local model in `config.json`)

## Quick start

```bash
# 1. Install Python dependencies
uv venv
uv pip install -e ".[dev]"

# 2. Run tests
.venv/bin/pytest -v

# 3. Start the daemon (mock sensors — no permissions needed)
.venv/bin/python -m server.braind start --mock-sensors

# 4. Start the dashboard (in a second terminal)
cd ui && npm install && npm run dev
# Open http://localhost:5173

# 5. Chat with the brain
curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "was siehst du?"}'
```

### Full sensor mode (real desktop observation)

```bash
# Requires macOS Accessibility + Microphone permissions for Terminal.app
./start.sh
```

### Pet face (animated eyes + voice)

```bash
cd pet-face && npx tauri dev
```

## Architecture

```
brain/          Spiking neural network (LIF neurons, STDP, BCM metaplasticity,
                neuromodulators, concept layer with WTA, working memory)
server/         FastAPI daemon — WebSocket streaming of brain state at 30 Hz
bridge/         LLM bridge — hybrid Ollama/Claude router, memory tools,
                proactive commentary, episode logging
adapters/       Mac sensor adapters (keyboard, mouse, screen idle, microphone)
capabilities/   Emergent tool system — wish detection, grant registry,
                tools (web search, shell, local files)
ui/             React + Tailwind dashboard — real-time 3D brain graph,
                modulator gauges, chat panel
pet-face/       Tauri app — animated eyes that reflect brain state + 
                Piper TTS / Whisper STT with push-to-talk
benchmark/      SNN benchmark suite (stability, discrimination, replay,
                scenario tests)
scripts/        Utility scripts (visualization, soak tests, launchd setup)
```

## Configuration

All settings live in `config.json`:

| Section | Key settings |
|---------|-------------|
| `brain` | Network size (sensory, feature, association, concept, WM, motor neurons), WTA k, tick rate |
| `llm` | Local model (Ollama), cloud model (Claude), routing |
| `sensors` | Enable/disable keyboard, mouse, microphone |
| `daemon` | Port, push rate, save interval, checkpoint path |
| `voice` | TTS model, STT model, language |

## License

MIT
