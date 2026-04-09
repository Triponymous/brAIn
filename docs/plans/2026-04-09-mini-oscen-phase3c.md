# Mini-OSCEN Phase 3c Implementation Plan — Pet Has a Face and a Voice

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add TTS (Piper, local), STT (Whisper.cpp, local, push-to-talk), voice endpoints to the daemon, a push-to-talk button in the dashboard, and a standalone Tauri pet-face app with animated eyes driven by brain modulator state.

**Architecture:** Voice is Python-side (bridge/tts.py, bridge/stt.py, server/voice.py). TTS uses Piper as a subprocess. STT uses pywhispercpp. Push-to-talk is triggered from the dashboard via `/api/stt/start` + `/api/stt/stop`. The pet face is a Tauri 2 app (Rust shell + HTML/Canvas WebView) that connects to the same `/ws` endpoint as the dashboard and maps modulator levels to eye animations. A global ⌥+Space hotkey in the Tauri app also triggers push-to-talk.

**Tech Stack:** Python (sounddevice for recording, piper-tts CLI, pywhispercpp), Tauri 2 + Rust + HTML Canvas, tauri-plugin-global-shortcut.

**Prerequisites (installed during Task 0):**
- `brew install piper` (or `pip install piper-tts`) + voice model download (~60MB)
- `pip install pywhispercpp` + Whisper `small` model download (~244MB)
- `brew install rustup && rustup-init` (for Tauri)
- `cargo install create-tauri-app` (Tauri CLI)

**Out of scope:** Wake-word detection (Phase 7), emotion-modulated TTS prosody (Phase 4 polish), call-awareness (Phase 4).

**Success criteria:**
1. `POST /api/tts {"text": "Hallo"}` plays audio through Mac speakers via Piper.
2. Push-to-talk in dashboard records, transcribes via Whisper, sends to /api/chat, speaks response.
3. Tauri pet-face window shows two animated eyes that react to modulator state in real time.
4. Global ⌥+Space hotkey triggers push-to-talk from anywhere on Mac.
5. Test count grows from 124 to ~132.

---

## Task 0: Install prerequisites (Piper, Whisper, Rust)

**This is a manual setup task — no TDD, just install + verify.**

**Step 1: Install Piper TTS**

```bash
pip install piper-tts  # into the project venv
cd /Users/leonmatthies/brAIntest
.venv/bin/pip install piper-tts
```

Download the German voice model:
```bash
mkdir -p /Users/leonmatthies/brAIntest/models
cd /Users/leonmatthies/brAIntest/models
# Download Thorsten medium voice
curl -L -o de_DE-thorsten-medium.onnx "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx"
curl -L -o de_DE-thorsten-medium.onnx.json "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json"
```

Verify:
```bash
echo "Hallo, ich bin dein neuromorphes Haustier." | .venv/bin/piper --model models/de_DE-thorsten-medium.onnx --output_file /tmp/test_piper.wav
afplay /tmp/test_piper.wav  # should speak German
```

**Step 2: Install pywhispercpp**

```bash
.venv/bin/pip install pywhispercpp
```

The Whisper model will be downloaded on first use (or pre-download):
```bash
.venv/bin/python -c "
from pywhispercpp.model import Model
m = Model('small', print_progress=True)
print('Whisper model loaded OK')
"
```

This downloads the `small` model (~244MB) on first run. Be patient.

Verify transcription:
```bash
.venv/bin/python -c "
from pywhispercpp.model import Model
m = Model('small')
# Transcribe the piper test file
segments = m.transcribe('/tmp/test_piper.wav')
text = ' '.join(s.text for s in segments)
print(f'Transcribed: {text}')
"
```

Expected: something recognizable as the German test sentence.

**Step 3: Install Rust (for Tauri)**

```bash
brew install rustup
rustup-init -y  # accept defaults
source "$HOME/.cargo/env"
rustc --version  # should print 1.x.x
cargo --version
```

If `brew install rustup` fails, use the official installer:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

**Step 4: Add models/ to .gitignore**

Edit `/Users/leonmatthies/brAIntest/.gitignore`, add:
```
models/
```

**Step 5: Update pyproject.toml**

Add to dependencies:
```toml
    "piper-tts>=1.2",
    "pywhispercpp>=1.2",
```

Re-install:
```bash
cd /Users/leonmatthies/brAIntest
uv pip install -e ".[dev]"
```

**Step 6: Verify all imports + existing tests**

```bash
.venv/bin/python -c "import piper; print('piper OK')"
.venv/bin/python -c "from pywhispercpp.model import Model; print('whisper OK')"
.venv/bin/pytest 2>&1 | tail -3  # 124 pass
```

**Step 7: Commit**

```bash
cd /Users/leonmatthies/brAIntest
git add pyproject.toml .gitignore
git commit -m "chore: add Phase 3c deps (piper-tts, pywhispercpp) + models/ gitignore"
```

---

## Task 1: TTS engine (Piper subprocess wrapper)

**Files:**
- Create: `bridge/tts.py`
- Create: `tests/test_tts.py`

**Step 1: Write failing tests**

```python
"""Tests for TTS engine — wraps Piper as subprocess."""
import asyncio
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock
from bridge.tts import TTSEngine


def test_construction():
    engine = TTSEngine(voice_model=Path("models/de_DE-thorsten-medium.onnx"))
    assert engine.voice_model.name == "de_DE-thorsten-medium.onnx"


def test_construction_with_missing_model_does_not_crash():
    """Engine should not crash on construction even if model doesn't exist."""
    engine = TTSEngine(voice_model=Path("/nonexistent/model.onnx"))
    assert engine.voice_model == Path("/nonexistent/model.onnx")


@pytest.mark.asyncio
async def test_synthesize_returns_wav_bytes():
    """Mock the subprocess to verify the pipeline works without actual Piper."""
    engine = TTSEngine(voice_model=Path("models/test.onnx"))
    fake_wav = b"RIFF" + b"\x00" * 100  # fake WAV header
    with patch("bridge.tts._run_piper", new_callable=AsyncMock, return_value=fake_wav):
        result = await engine.synthesize("Hallo Welt")
    assert result[:4] == b"RIFF"
    assert len(result) > 10
```

**Step 2: Run to fail, then implement.**

`bridge/tts.py` should:
- Have a `TTSEngine(voice_model: Path)` class
- `async def synthesize(text: str) -> bytes` — calls Piper via `asyncio.create_subprocess_exec`, pipes text to stdin, returns WAV bytes from stdout
- `async def speak(text: str)` — calls synthesize then plays via `afplay` subprocess (macOS)
- A module-level `async def _run_piper(model_path, text) -> bytes` that the test can mock

**Step 3: Run tests + full suite, commit**

```bash
.venv/bin/pytest tests/test_tts.py -v
.venv/bin/pytest 2>&1 | tail -3  # expect ~127
git add bridge/tts.py tests/test_tts.py
git commit -m "feat(bridge): TTS engine wrapping Piper subprocess"
```

---

## Task 2: STT engine (Whisper.cpp wrapper)

**Files:**
- Create: `bridge/stt.py`
- Create: `tests/test_stt.py`

**Step 1: Write failing tests**

```python
"""Tests for STT engine — wraps pywhispercpp for local speech-to-text."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from bridge.stt import STTEngine


def test_construction():
    # Don't load the real model in tests — mock it
    with patch("bridge.stt.Model") as MockModel:
        engine = STTEngine(model_name="small")
    assert engine.model_name == "small"


def test_transcribe_with_mock():
    """Feed synthetic audio and verify transcription pipeline."""
    with patch("bridge.stt.Model") as MockModel:
        mock_instance = MagicMock()
        # Mock transcribe to return a fake segment
        mock_segment = MagicMock()
        mock_segment.text = "Hallo Welt"
        mock_instance.transcribe.return_value = [mock_segment]
        MockModel.return_value = mock_instance

        engine = STTEngine(model_name="small")
        # Generate 1 second of silence at 16kHz
        audio = np.zeros(16000, dtype=np.float32)
        result = engine.transcribe(audio)
    assert result == "Hallo Welt"
```

**Step 2: Implement `bridge/stt.py`**

```python
"""STT engine — wraps pywhispercpp for local speech-to-text.

Uses the Whisper small model for German transcription. The model is loaded
lazily on first transcribe() call to avoid blocking daemon startup.
"""
from __future__ import annotations
import numpy as np
from pywhispercpp.model import Model


class STTEngine:
    def __init__(self, model_name: str = "small", language: str = "de") -> None:
        self.model_name = model_name
        self.language = language
        self._model: Model | None = None

    def _ensure_model(self) -> Model:
        if self._model is None:
            self._model = Model(self.model_name)
        return self._model

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a float32 numpy audio array to text."""
        model = self._ensure_model()
        segments = model.transcribe(audio, language=self.language)
        return " ".join(s.text.strip() for s in segments).strip()
```

**Step 3: Run tests + commit**

```bash
.venv/bin/pytest tests/test_stt.py -v
.venv/bin/pytest 2>&1 | tail -3  # expect ~129
git add bridge/stt.py tests/test_stt.py
git commit -m "feat(bridge): STT engine wrapping pywhispercpp"
```

---

## Task 3: Voice endpoints (/api/tts, /api/stt) + dashboard push-to-talk

**Files:**
- Create: `server/voice.py`
- Modify: `server/braind.py` (wire voice router)
- Modify: `ui/src/components/ChatPanel.tsx` (add push-to-talk button)
- Create: `tests/test_voice_endpoints.py`

**Step 1: Write failing tests**

```python
"""Tests for voice endpoints."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from server.voice import build_voice_router


@pytest.mark.asyncio
async def test_tts_endpoint_returns_audio():
    mock_tts = MagicMock()
    mock_tts.synthesize = AsyncMock(return_value=b"RIFF" + b"\x00" * 100)
    router = build_voice_router(tts_engine=mock_tts, stt_engine=MagicMock(), chat_fn=AsyncMock())

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/tts", json={"text": "Hallo"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content[:4] == b"RIFF"
```

**Step 2: Implement `server/voice.py`**

The voice router has:
- `POST /api/tts {"text": "..."}` → returns WAV audio bytes
- `POST /api/stt/record {"duration": 5}` → records from mic for N seconds, transcribes, returns `{"text": "..."}`
- `POST /api/voice-chat {"duration": 5}` → records → STT → chat → TTS → returns audio response

**Step 3: Wire into braind.py**

After the chat_router wiring, add:
```python
    from bridge.tts import TTSEngine
    from bridge.stt import STTEngine
    from server.voice import build_voice_router

    tts = TTSEngine(voice_model=Path("models/de_DE-thorsten-medium.onnx"))
    stt = STTEngine(model_name="small")
    voice_router = build_voice_router(tts_engine=tts, stt_engine=stt, chat_fn=...)
    app.include_router(voice_router)
```

**Step 4: Add push-to-talk to ChatPanel**

In `ui/src/components/ChatPanel.tsx`, add a microphone button next to the Send button. On click:
1. POST to `/api/voice-chat` with duration 5
2. Response comes back as audio — play it via `new Audio(URL.createObjectURL(blob))`
3. Also append the transcription + response text to the chat messages

**Step 5: Run tests + build frontend + commit**

```bash
.venv/bin/pytest tests/test_voice_endpoints.py -v
.venv/bin/pytest 2>&1 | tail -3  # expect ~131
cd ui && npm run build && cd ..
git add server/voice.py server/braind.py ui/src/ tests/test_voice_endpoints.py
git commit -m "feat(voice): TTS/STT endpoints + push-to-talk in dashboard"
```

---

## Task 4: Install Tauri + scaffold pet-face app

**Step 1: Verify Rust is available**

```bash
source "$HOME/.cargo/env"
rustc --version
cargo --version
```

If not available, install per Task 0 Step 3.

**Step 2: Create the Tauri project**

```bash
cd /Users/leonmatthies/brAIntest
cargo install create-tauri-app
cargo create-tauri-app pet-face --template vanilla-ts
cd pet-face
npm install
```

Or manually scaffold:
```bash
mkdir -p pet-face/src pet-face/src-tauri/src
```

Write `pet-face/package.json`, `pet-face/src-tauri/Cargo.toml`, `pet-face/src-tauri/tauri.conf.json`, `pet-face/src-tauri/src/main.rs`.

The Tauri config should set:
- Window: 200×200, borderless, transparent, always-on-top, position bottom-right
- Title: "brAIntest Pet"
- decorations: false
- transparent: true
- alwaysOnTop: true

**Step 3: Build to verify**

```bash
cd /Users/leonmatthies/brAIntest/pet-face
npm run tauri build  # or cargo tauri build
```

Expected: produces a .app bundle.

**Step 4: Commit**

```bash
cd /Users/leonmatthies/brAIntest
git add pet-face/
git commit -m "feat(pet-face): Tauri 2 scaffold (borderless, transparent, always-on-top)"
```

---

## Task 5: Pet face eye animation (HTML Canvas in Tauri WebView)

**Files:**
- Create/edit: `pet-face/src/index.html`
- Create: `pet-face/src/eyes.ts`
- Create: `pet-face/src/ws-client.ts`

**The eyes animation:**

`ws-client.ts` connects to `ws://localhost:8000/ws` and parses brain state.

`eyes.ts` renders on a Canvas:
- Two circles (eyes) that react to modulator levels:
  - **Idle/Content** (all mods low): normal size, slow blink every ~5s
  - **Curious** (DA+ACh high): eyes widen, faster blinks
  - **Alarmed** (NE spike): eyes wide open, small pupils, brief freeze
  - **Sleepy** (all mods low + long idle implicit): half-closed lids
  - **Asleep** (very low activity): eyes closed, gentle breathing motion
- Animation loop at 30fps via `requestAnimationFrame`
- Smooth interpolation between states (don't snap)

The mapping from modulators to eye state:
```typescript
function eyeState(mods: Record<string, number>): EyeState {
  const da = mods.DA ?? 0;
  const ne = mods.NE ?? 0;
  const ach = mods.ACh ?? 0;
  const sht = mods["5HT"] ?? 0;

  if (ne > 0.5) return "alarmed";
  if (da > 0.3 && ach > 0.3) return "curious";
  if (da < 0.05 && ne < 0.05 && ach < 0.05) return "sleepy";
  return "idle";
}
```

**Step 1: Implement the eye canvas + WS client**

**Step 2: Build and test visually**

```bash
cd /Users/leonmatthies/brAIntest/pet-face
npm run tauri dev
```

With the daemon running (`braind start --mock-sensors`), the pet face should connect and show animated eyes.

**Step 3: Commit**

```bash
git add pet-face/src/
git commit -m "feat(pet-face): animated eyes driven by brain modulator state"
```

---

## Task 6: Global hotkey (⌥+Space) for push-to-talk in Tauri

**Files:**
- Modify: `pet-face/src-tauri/Cargo.toml` (add tauri-plugin-global-shortcut)
- Modify: `pet-face/src-tauri/src/main.rs` (register hotkey)
- Modify: `pet-face/src/eyes.ts` (add listening/speaking visual states)

**Step 1: Add the plugin**

```bash
cd /Users/leonmatthies/brAIntest/pet-face
cargo add tauri-plugin-global-shortcut -F tauri-plugin-global-shortcut/global-shortcut
```

**Step 2: Register ⌥+Space in main.rs**

When triggered:
1. POST to `http://localhost:8000/api/voice-chat` with duration 5
2. Play the returned audio
3. Eye state: listening → thinking → speaking → idle

**Step 3: Add visual states to eyes.ts**

- "listening": eyes focus, subtle "!" indicator
- "speaking": mouth dots animate (or eyes pulse)

**Step 4: Build + test**

```bash
cd /Users/leonmatthies/brAIntest/pet-face
npm run tauri build
```

**Step 5: Commit**

```bash
git add pet-face/
git commit -m "feat(pet-face): global ⌥+Space push-to-talk via tauri-plugin-global-shortcut"
```

---

## Task 7: Tag phase-3-complete + README

**Step 1: Run full Python test suite**

```bash
cd /Users/leonmatthies/brAIntest
.venv/bin/pytest -v 2>&1 | tail -5
```

Expected: ~132 tests pass.

**Step 2: Build frontend + pet-face**

```bash
cd ui && npm run build && cd ..
cd pet-face && npm run tauri build && cd ..
```

**Step 3: Update README**

Mark Phase 3c complete in the status section. Add Quick Start entries for:
- Voice: `curl -X POST http://localhost:8000/api/tts -H 'Content-Type: application/json' -d '{"text":"Hallo"}' --output /tmp/hello.wav && afplay /tmp/hello.wav`
- Pet face: `cd pet-face && npm run tauri dev`

**Step 4: Commit + tag**

```bash
git add README.md
git commit -m "docs: mark Phase 3 complete"
git tag phase-3-complete -m "Phase 3: sensors + LLM bridge + dashboard + pet face + voice"
```

---

## Phase 3c Done → Phase 3 Complete

After Phase 3c, the project has:
- A 24/7 brain daemon observing 6 Mac sensors
- LLM bridge with hybrid Ollama/Claude routing and memory tools
- React dashboard with brain canvas viz + chat
- Piper TTS (German, local)
- Whisper STT (push-to-talk, local)
- Tauri pet-face with animated eyes driven by real modulator state
- Global hotkey for voice interaction
- ~132 tests all green
- 5 tags: phase-1, phase-2, phase-3a, phase-3b, phase-3-complete

Next: Phase 4 — Capability Wishlist + Grant System (emergent tool acquisition).
