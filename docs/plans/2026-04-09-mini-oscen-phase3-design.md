# Mini-OSCEN Phase 3+ — Pet Lebensraum Design

**Date:** 2026-04-09
**Status:** Approved, ready for Phase 3a implementation planning
**Replaces:** The Phase 3-6 sketch in `2026-04-08-mini-oscen-design.md` (which was webcam-only desktop adapter + dashboard + LLM bridge + Pygame avatar). The user's pet vision (Tag 1 → Woche 1 → Monat 1, animated face, emergent capabilities, eventually ESP32 hardware) replaces that arc.

---

## Vision (in user's words, paraphrased)

A persistent neuromorphic pet that lives on the Mac (and later moves to ESP32 hardware). It perceives the user's real desktop life through system sensors and microphone. It forms concept neurons unsupervised through STDP, develops a unique personality over weeks via its own modulator dynamics, and shows emotion through an animated face. Over time the user labels concepts via chat, and the pet learns to talk about its day in the user's vocabulary.

The differentiator from a normal LLM assistant is **"Taktgefühl"** — the pet knows *when* the user is interruptible, because its brain is continuously perceiving the user's rhythm. It only acts when the brain says "open window", not whenever an external trigger fires.

Tools/agent capability is **emergent**, not pre-loaded. The pet starts with a minimal toolkit (memory, TTS, request_capability). It observes user patterns, and when a stable pattern emerges that a tool could help with, the LLM bridge says "I notice you do X often — can I learn to help with that?" The user grants → a new tool is registered at runtime. Each pet's capability set grows uniquely from its user's actual life.

## Non-Goals for Phase 3+

- ESP32 hardware (deferred to Phase 6+)
- Pre-loaded agent tools (Phase 4 builds the catalog + grant system, individual integrations come later via emergent grants)
- Standing orders (Phase 5)
- Wake-word always-on listening (Phase 7)
- Calendar/Mail/Slack integrations (Phase 5+ via grant system)
- Multi-user, distribution, packaging for others

## Architecture Overview

```
                     MAC (always-on daemon)

  ┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐
  │  SENSORS    │    │     BRAIN        │    │   LLM BRIDGE     │
  │ (Phase 3a)  │    │ (Phase 2 stack)  │    │ (Phase 3b)       │
  │             │    │                  │    │                  │
  │ active app  │───▶│ Sensory→Feature  │───▶│ Brain Exporter   │
  │ keystrokes  │    │ →Association→    │    │ Memory Tools     │
  │ mouse       │    │ Concept(WTA)→    │◀───│ Hybrid Router    │
  │ idle        │    │ WM+Motor+Meta    │    │ (Ollama+Claude)  │
  │ mic         │    │                  │    │                  │
  │ time tonics │    │ Modulators       │    │ TTS/STT          │
  └─────────────┘    │ DA/NE/ACh/5HT    │    │ (Phase 3c)       │
                     │                  │    └────┬─────┬───────┘
                     │ SQLite save 60s  │         │     │
                     └────────┬─────────┘         │     │
                              │                   │     │
  ┌──────────────────────────┴───────────────────┴─────┴───┐
  │             FastAPI server (port 8000)                  │
  │  /ws  brain state @ 30 Hz                               │
  │  /chat  REST                                            │
  │  /tts /stt                                              │
  └──────┬───────────────────┬───────────────────┬──────────┘
         │                   │                   │
         ▼                   ▼                   ▼
  ┌──────────────┐    ┌────────────────┐  ┌──────────────────┐
  │  DASHBOARD   │    │   PET FACE     │  │   PUSH-TO-TALK   │
  │  Vite+React  │    │   Tauri app    │  │   ⌥+Space        │
  │  3 columns   │    │   200×200      │  │                  │
  │  brain viz   │    │   bottom-right │  │  STT→chat→TTS    │
  │  chat        │    │   eyes anim    │  │                  │
  └──────────────┘    └────────────────┘  └──────────────────┘
```

Three Mac processes:
1. **`braind`** — main daemon. Loads brain from SQLite, runs sensor loops (asyncio), runs brain ticks at ~100 Hz, exports state via WebSocket, hosts FastAPI endpoints. Lives 24/7 via launchd or manual start.
2. **Pet Face** — Tauri standalone app. Connects to ws://localhost:8000/ws, animates eyes from modulator levels.
3. **Dashboard** — browser tab. Started on demand. Same WebSocket. Three columns: sensors / brain viz / chat.

## Component 1: Mac Sensor Adapter (Phase 3a)

Six sensor streams as asyncio coroutines feeding a shared `SensorBus` that gets sampled by the brain tick loop.

**Sensor → Sensory neuron mapping (200-dim Sensory layer):**

| Sensor | Neurons | Encoding |
|---|---|---|
| Active App | 0–63 | Hash-mapped one-hot, first 64 distinct apps + "other" bucket |
| Keystroke rate | 64–79 | 16 log-scaled bins (0 to ≥10/s) |
| Mouse rate | 80–95 | 16 log-scaled bins (0 to ≥1000 px/s) |
| Idle time | 96–103 | 8 log-scaled bins (<1s to >1h) |
| Mic mel-spectrum | 104–135 | 32 mel-bands, rate-coded |
| Mic loudness | 136–143 | 8 RMS bins |
| Time tonics | 144–151 | 4 day-cycle phase + 4 week-cycle phase |
| Reserve | 152–199 | Empty for Phase 4+ |

**macOS permissions required:**
- Accessibility (for keystroke/mouse counts; never reads content)
- Microphone (for sensor audio; not recorded to disk)

**Sample-and-hold architecture:** sensors run at independent rates (1-50 Hz). Each writes its latest encoding to the SensorBus. Brain tick reads the current snapshot. Sensors with lower rates simply hold their last value between brain ticks.

**Brain tick rate:** 100 Hz simulated time on a single asyncio loop. Phase-2 soak test demonstrated 8000 ticks/sec on default brain size; 100 Hz leaves 80× headroom.

## Component 2: LLM Bridge (Phase 3b)

**Brain State Exporter:** background coroutine, 1 Hz, produces structured JSON snapshot with modulators, top-12 active concepts (with labels and history), recent episode summary (novelty/loud events/stable concepts), current sensor summary.

**Memory Tools (Phase 3b — read & label only, no agent tools):**
- Read: `current_state()`, `query_concepts(filter)`, `episode_search(time_range)`, `recall_associations(concept_id)`, `historical_concept_pattern(concept_id, days)`
- Write (user-mediated): `label_concept(id, label)`, `merge_concepts(ids, label)`, `note_meta(key, value)`

**No agent tools in Phase 3.** No web search, no shell, no file I/O, no calendar, no mail. Those come via the emergent capability system in Phase 4+.

**Hybrid LLM Routing:**
- Default: local Ollama with `qwen2.5:7b-instruct` (~30 tok/s on M-Mac)
- Cloud (Claude Haiku/Sonnet) when: long question, multi-question, reasoning keywords, NE high (alarmed = "important answer"), or explicit `/cloud` flag
- Token budget cap (default 50k/day cloud); on overflow → local with chat warning
- System prompt explains: "You are the language interface to a persistent neuromorphic brain. You DO NOT learn — the brain learns. Speak in first person AS the pet. Don't invent memories not in the state."

## Component 3: Voice (Phase 3c)

**TTS — Piper TTS** with `de_DE-thorsten-medium` voice (~60MB). ~300ms first audio, natural German voice, fully local. macOS `say` as fallback. Emotion hint from modulator state adjusts pitch/rate (curious=higher, sleepy=slower, alarmed=tense).

**STT — Whisper.cpp** with `small` model (244MB), German hint. ~700ms for 5-sec audio on M-series. Push-to-talk via global hotkey `⌥+Space` (does not collide with Spotlight).

**Two mic streams:** sensor mic (continuous, 50 Hz mel-bands) and voice mic (push-to-talk only, full quality for Whisper). Both share a single `sounddevice` reader that splits frames to two consumers, avoiding "mic in use" conflicts.

**Voice latency budget:** STT ~700ms + LLM local ~1.5s + TTS first audio ~500ms = ~2.7s to first spoken syllable. Slower than Siri, charming for "a thinking creature".

**No wake-word in Phase 3.** Push-to-talk is 99% as good and 10% the work.

## Component 4: Pet Face (Phase 3c)

**Tauri 2 standalone app**, 200×200 borderless transparent window, always-on-top, default position bottom-right, drag-to-move.

**Visual:** two abstract circular eyes on black background, optional 3-dot "mouth" that animates during speech.

**Animation states (driven by modulators, not scripted):**

| State | Trigger | Animation |
|---|---|---|
| Idle / Content | DA mid, NE low, ACh low | Normal eyes, slow natural blink, gentle "breathing" up-down |
| Curious | DA high + ACh high | Eyes widen, faster blinks, follows activity |
| Alarmed | NE sudden spike | Eyes wide, small pupils, frozen 2-3s, slight tremor |
| Sleepy | All mods low + high idle sensor | Half-closed lids, very slow blinks, occasional nod-off |
| Asleep | Very long idle (>15min) | Eyes closed, slow breathing |
| Speaking | TTS active | Mouth dots animate to amplitude |
| Listening | Push-to-talk active | Eyes focus, "!" indicator |

**Critical constraint:** every animation is a *direct reflection* of brain state. No scripted "cuteness", no animation that doesn't trace back to a real spike or modulator level. When the pet looks bored, it's because the brain is actually bored.

**System tray icon** with quick actions: pause/resume brain, open dashboard, quit.

## Component 5: Dashboard (Phase 3b)

**Vite + React 18 + TypeScript + Tailwind + shadcn/ui**, three-column layout:

- **Left — Sensors:** live display of all 6 streams (active app text + bargraph, keystroke/mouse rate bars, mic mel-spectrogram canvas, idle counter, time of day)
- **Center — Brain:** region graph (animated canvas, 7 region nodes, modulator-colored), spike raster (5s window canvas), concept cloud (top-12 with labels, click-to-label dialog)
- **Right — Chat:** message list, markdown rendering, push-to-talk button, inline collapsed tool-call traces

**Time scrubber** below: last 5 minutes of brain activity, scrub to view past snapshots. Reinforces "this thing exists continuously in time".

## Tech Stack Summary

| Layer | Choice |
|---|---|
| Brain | Phase 2 Python modules (unchanged) |
| Daemon | Python 3.11 + asyncio + FastAPI + uvicorn |
| Sensors | pyobjc, pynput, sounddevice, numpy.fft |
| Persistence | SQLite (Phase 2 unchanged), 60s auto-save |
| LLM local | Ollama HTTP + qwen2.5:7b-instruct |
| LLM cloud | Anthropic SDK + claude-haiku-4-5 |
| TTS | Piper TTS + de_DE-thorsten-medium |
| STT | pywhispercpp + Whisper small |
| Dashboard | Vite + React 18 + TS + Tailwind + shadcn/ui + Canvas |
| Pet Face | Tauri 2 + Rust + HTML/Canvas in WebView |
| Global hotkey | tauri-plugin-global-shortcut |
| Deployment | launchd plist for braind, .app bundle for Pet Face |

## Repo Layout (Phase 3 additions)

```
brAIntest/
├── brain/                 (Phase 1+2, unchanged)
├── adapters/              (NEW)
│   ├── base.py
│   └── mac_desktop/
│       ├── adapter.py
│       ├── sensor_app.py
│       ├── sensor_keymouse.py
│       ├── sensor_idle.py
│       ├── sensor_mic.py
│       ├── sensor_time.py
│       └── encoding.py
├── bridge/                (NEW)
│   ├── exporter.py
│   ├── memory_tools.py
│   ├── llm_router.py
│   ├── llm_local.py
│   ├── llm_cloud.py
│   ├── tts.py
│   └── stt.py
├── server/                (NEW)
│   ├── main.py
│   ├── ws.py
│   ├── chat.py
│   ├── voice.py
│   └── braind.py
├── ui/                    (NEW — dashboard, Vite/React)
├── pet-face/              (NEW — Tauri app)
├── scripts/
│   └── run_braind.sh
├── tests/
│   ├── (Phase 1+2 tests, unchanged)
│   ├── test_mac_sensors.py
│   ├── test_bridge_memory.py
│   ├── test_bridge_router.py
│   └── test_server.py
└── docs/plans/
```

## Phase 3 Sub-Phasing (essential — Phase 3 is too big as one chunk)

Phase 3 is split into three tightly-related sub-phases. Each gets its own implementation plan, its own tag, and ends with something visibly working.

### Phase 3a — "Brain sees my desktop life"
**Goal:** Mac sensor adapter complete + daemon skeleton (FastAPI + WebSocket push). No LLM bridge, no UI. End state: `braind start` runs continuously, `curl ws://localhost:8000/ws` shows brain reacting to real activity.

**Builds:**
- `adapters/mac_desktop/` (all 6 sensors + encoding)
- `server/main.py` + WebSocket pusher
- `server/braind.py` CLI
- launchd plist (optional)
- Per-sensor tests with OS-API mocks
- `--mock-sensors` mode for CI

**Tests:** ~10-12 new. Total ~58.

**Tag:** `phase-3a-complete`

### Phase 3b — "Brain talks to me"
**Goal:** LLM bridge + dashboard. End state: open browser → see brain live → ask "what are you seeing?" → real answer derived from brain state.

**Builds:**
- `bridge/` (exporter, memory_tools, router, llm_local, llm_cloud)
- `server/chat.py` endpoint
- Dashboard `ui/` (3 columns, brain canvas viz, chat panel)
- Concept-labeling UI

**Tests:** ~12-15 new. Total ~70.

**Tag:** `phase-3b-complete` — first usable conversational pet.

### Phase 3c — "Pet has a face and a voice"
**Goal:** Tauri pet face + full voice pipeline.

**Builds:**
- `pet-face/` Tauri app (eyes canvas, WebSocket client, modulator→state mapping)
- `bridge/tts.py` Piper wrapper
- `bridge/stt.py` Whisper wrapper
- `server/voice.py` endpoints
- Global hotkey integration
- System tray icon
- E2E voice loop smoke test

**Tests:** ~6-8 new + manual E2E. Total ~78.

**Tag:** `phase-3-complete` (aggregates 3a+3b+3c).

## Future Phases (sketch only — not planned in detail)

- **Phase 4 — Capability Wishlist + Grant System:** capability catalog (web search, shell, files as first 3 entries, all disabled by default). Wish detection coroutine in bridge that spots stable concept patterns and suggests capabilities. Grant flow: user approves → tool registered at runtime → persisted across restarts. *This is the emergent tool story.*
- **Phase 5 — Standing Orders + Agent Tools:** trigger-action engine. More tools in catalog (calendar, mail, n8n webhook, ZenBoard). First real agent capabilities.
- **Phase 6 — ESP32 Hardware Adapter:** port pet from Mac-only to physical ESP32 with AMOLED eyes, IMU, mic, touch, servos, speaker. Brain stays on Mac, communicates via WiFi. Pet Face becomes optional second display.
- **Phase 7 — Wake-word, multi-user, privacy polish, distribution.**

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| macOS Accessibility permission denied → no keyboard/mouse sensors | Daemon detects and runs with reduced sensor set; logs warning; documentation explains how to grant |
| Microphone permission denied | Same — daemon runs without mic, audio sensors silent |
| Brain saturation (Phase 2 known WM attractor) | Acknowledged; revisit with periodic dampening or modulator-gated WM in Phase 4 |
| Ollama not installed | Daemon detects and falls back to cloud-only mode with warning |
| Tauri learning curve (if user hasn't used Rust) | Phase 3c can be downscoped to a browser-based pet face if Tauri proves too painful |
| Whisper.cpp build issues | pywhispercpp ships prebuilt wheels for Apple Silicon |
| Sensor encoding too noisy → no concept emergence | Phase 3a includes a 24h soak test that checks for stable concept formation; if fails, encoding gets tuned before 3b |

## Success Criteria for Phase 3 (aggregate)

1. Daemon runs continuously for ≥48h on real Mac usage without crash or memory leak.
2. Brain forms ≥10 stable concept neurons from real desktop activity within 3 days.
3. Survives daemon restart (save → load → resume) bit-identical (Phase 2 capability extended).
4. User can have a meaningful chat about the pet's day, where the pet's answers are demonstrably tied to actual brain state (verifiable via Dashboard time scrubber).
5. Pet face animates correctly across all 7 documented animation states based on real brain state.
6. Voice loop end-to-end works: ⌥+Space → talk → STT → LLM → TTS → audio out, in <5s for short utterances.
7. Test count grows from 48 to ~78, all passing.

## Open Decisions Deferred to Implementation

- Exact mel-band parameters for mic encoding (frequency range, hop size)
- Exact Ollama model — `qwen2.5:7b-instruct` is the default but if it underperforms, fall back to `qwen2.5:14b` or `llama3.3:70b`
- launchd vs manual start as default — recommend launchd but the CLI must work standalone
- Whether Pet Face uses WebGL or 2D Canvas (probably 2D — eyes are simple)
- WebSocket reconnect policy on Pet Face crash
