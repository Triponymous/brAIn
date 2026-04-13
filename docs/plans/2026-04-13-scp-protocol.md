# SNN Communication Protocol (SCP) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the bloated system prompt with a clean, modellunabhängiges SNN Communication Protocol — Compact State (6 Zeilen) + Model Adapter (pro LLM kalibrierbar) + Rückkanal (LLM → SNN Reward).

**Architecture:** `bridge/scp.py` is the single source of truth for SNN→LLM communication. It reads brain state and produces a compact, structured representation. `bridge/model_adapter.py` translates this into model-specific prompts. `bridge/feedback.py` handles LLM→SNN signals.

**Tech Stack:** Python 3.11, existing Brain/Modulators/ConceptTracker/BrainInterpreter

---

## Task 1: Compact State Renderer (`bridge/scp.py`)

**Files:**
- Create: `bridge/scp.py`
- Test: `tests/test_scp.py`

The Compact State is a 6-line structured representation of the ENTIRE brain state. No prose, no modulator numbers, no German paragraphs. Just facts.

```
EMOTION: content (3min stabil) | Grund: ruhige Session ohne Ueberraschungen
PATTERN: #5 unlabeled (85%, 435x) | Sensoren: Tastatur aktiv, Claude offen, leise
CHANGE: app_switch Chrome→Claude vor 30s
MEMORY: 5/20 WM-Slots | denke an: vorheriges Muster
SESSION: 47min aktiv | Gewohnheit: Sonntag 23h normalerweise idle
BRAIN: 52/200 Neuronen spezialisiert | 4 Cluster gelernt | Alter: 3 Tage
```

Each line is one DIMENSION of the brain's state. The LLM can scan these in <1 second.

### Implementation

```python
# bridge/scp.py
"""SNN Communication Protocol — Compact State renderer.

Produces a 6-line structured representation of the entire brain state.
This is the SINGLE interface between SNN and LLM. No prose, no numbers,
just structured facts that any LLM can parse.

Format:
    EMOTION: <state> (<duration>) | Grund: <why>
    PATTERN: <id> <label> (<confidence>, <count>x) | Sensoren: <what>
    CHANGE: <type> <detail> vor <time>
    MEMORY: <active>/<max> WM-Slots | denke an: <recent>
    SESSION: <duration> aktiv | Gewohnheit: <habit_context>
    BRAIN: <specialized>/<total> Neuronen | <clusters> Cluster | Alter: <age>
"""
from __future__ import annotations
from typing import Any

from brain.core import Brain


class CompactState:
    """Renders the brain's complete state in 6 structured lines."""

    def __init__(self, brain: Brain) -> None:
        self.brain = brain

    def render(self) -> str:
        """Produce the compact state string."""
        lines = [
            self._emotion_line(),
            self._pattern_line(),
            self._change_line(),
            self._memory_line(),
            self._session_line(),
            self._brain_line(),
        ]
        return "\n".join(lines)

    def _emotion_line(self) -> str:
        interpreter = getattr(self.brain, '_interpreter', None)
        trend = None
        if interpreter and hasattr(interpreter, 'state_detector'):
            trend = interpreter.state_detector.emotional_trend(window_seconds=180)

        from bridge.emotional_prompt import detect_emotional_state
        mods = self.brain.modulators.snapshot()
        state_name, _ = detect_emotional_state(mods, trend=trend)

        # Duration: how long has the emotional state been stable?
        # Approximate from modulator trend stability
        duration = "kurz"
        if trend:
            avg_ne = trend.get("avg_NE", 0)
            peak_ne = trend.get("peak_NE", 0)
            if peak_ne < 0.02 and avg_ne < 0.01:
                duration = "seit Minuten stabil"
            elif peak_ne > avg_ne * 3:
                duration = "gerade erst gewechselt"

        # Reason from SNN narrator
        narrator = getattr(self.brain, '_snn_narrator', None)
        reason = ""
        if narrator:
            reason = narrator._narrate_modulators()

        return f"EMOTION: {state_name} ({duration}) | Grund: {reason}"

    def _pattern_line(self) -> str:
        ct = self.brain.concept_tracker
        snap = ct.snapshot()
        current = snap.get("current_cluster", -1)
        label = snap.get("current_label")
        debug = snap.get("debug", {})
        confidence = debug.get("best_sim", 0)

        sd = getattr(self.brain, '_last_sensor_display', {})
        sensors = []
        if sd.get("keys", 0) > 10:
            sensors.append("viel Tastatur")
        elif sd.get("keys", 0) > 2:
            sensors.append("Tastatur aktiv")
        if sd.get("mouse", 0) > 10:
            sensors.append("Maus aktiv")
        app = sd.get("app", "")
        if app:
            sensors.append(f"{app} offen")
        if sd.get("mic_rms", 0) > 0.015:
            sensors.append("laut")
        elif sd.get("mic_rms", 0) > 0.005:
            sensors.append("Hintergrundgeraeusche")
        else:
            sensors.append("leise")

        if current < 0:
            return f"PATTERN: kein Muster erkannt | Sensoren: {', '.join(sensors)}"

        cluster = ct._clusters.get(current)
        count = cluster.count if cluster else 0
        label_str = f"'{label}'" if label else "unlabeled"
        conf_pct = f"{confidence:.0%}" if confidence > 0 else "?"

        return f"PATTERN: #{current} {label_str} ({conf_pct}, {count}x) | Sensoren: {', '.join(sensors)}"

    def _change_line(self) -> str:
        transition = self.brain.concept_tracker.get_transition()
        if transition:
            from_label = transition.get("from_label") or f"#{transition.get('from_cluster', '?')}"
            to_label = transition.get("to_label") or f"#{transition.get('to_cluster', '?')}"
            return f"CHANGE: Musterwechsel {from_label} → {to_label}"

        interpreter = getattr(self.brain, '_interpreter', None)
        if interpreter:
            states = interpreter.state_detector.detect()
            if states.get("stress"):
                return "CHANGE: Stress erkannt — hektisches Verhalten"
            if states.get("meeting"):
                return f"CHANGE: Meeting seit {states['meeting_duration_min']}min"

        return "CHANGE: keine Aenderung"

    def _memory_line(self) -> str:
        wm = self.brain.regions.get("wm")
        active = 0
        if wm and hasattr(wm, 'last_spikes'):
            active = int(wm.last_spikes.sum().item())

        max_active = 20
        if wm and hasattr(wm, 'max_active'):
            max_active = wm.max_active

        # What's in recent memory?
        recent = ""
        ct = self.brain.concept_tracker.snapshot()
        clusters = ct.get("clusters", [])
        recent_clusters = [c for c in clusters if c.get("label") and not c.get("active")]
        if recent_clusters:
            recent = recent_clusters[0].get("label", "?")

        return f"MEMORY: {active}/{max_active} WM-Slots | denke an: {recent or 'nichts Bestimmtes'}"

    def _session_line(self) -> str:
        interpreter = getattr(self.brain, '_interpreter', None)
        active_min = 0
        habit_ctx = ""

        if interpreter:
            states = interpreter.state_detector.detect()
            active_min = states.get("active_minutes", 0)

            habits = interpreter.habit_miner.current_hour_context()
            if habits.get("usual_habits"):
                h = habits["usual_habits"][0]
                habit_ctx = f"{h.get('day_name', '?')} {h.get('hour', '?')}h normalerweise {h.get('app', '?')}"

        needs_break = ""
        if interpreter:
            states = interpreter.state_detector.detect()
            if states.get("needs_break"):
                needs_break = " | Pause empfohlen!"
            elif states.get("flow"):
                needs_break = f" | im Flow seit {states['flow_duration_min']}min"

        return f"SESSION: {active_min:.0f}min aktiv{needs_break} | Gewohnheit: {habit_ctx or 'noch keine gelernt'}"

    def _brain_line(self) -> str:
        syn = self.brain.synapses.get("sensory_concept")
        specialized = 0
        total = 0
        if syn:
            weights = syn.weights
            total = weights.shape[0]
            row_stds = weights.std(dim=1)
            specialized = int((row_stds > 0.1).sum().item())

        num_clusters = self.brain.concept_tracker.snapshot().get("num_clusters", 0)

        personality = getattr(self.brain, '_interpreter', None)
        age_days = 0
        if personality and hasattr(personality, 'personality'):
            age_days = personality.personality.snapshot().get("age_days", 0)

        return f"BRAIN: {specialized}/{total} Neuronen spezialisiert | {num_clusters} Cluster gelernt | Alter: {age_days:.1f} Tage"
```

---

## Task 2: Model Adapter (`bridge/model_adapter.py`)

**Files:**
- Create: `bridge/model_adapter.py`
- Test: `tests/test_model_adapter.py`

The Model Adapter takes Compact State + user message and produces a model-specific system prompt. Each model gets a different prompt STYLE, but the same INFORMATION.

```python
# bridge/model_adapter.py
"""Model Adapter — translates Compact State into model-specific prompts.

Different LLMs need different prompt styles:
- Qwen 2.5: direct instructions, structured, explicit constraints
- Gemma: example-based, softer tone, few-shot style
- Claude: constraint-based, system-level instructions
- Generic: works with any model (simplified)

The adapter ensures the SAME brain state produces equally good responses
regardless of which LLM is running.
"""
from __future__ import annotations
from typing import Any


_UNIVERSAL_RULES = """Du bist ein kleines Wesen das auf Leons Mac lebt.
Keine Emojis. Keine erfundenen Faehigkeiten. Keine Hilfsangebote.
Widersprich nie deinen Sinnen. Antworte in Leons Sprache."""


class ModelAdapter:
    """Translates Compact State into model-specific system prompts."""

    def render(
        self,
        compact_state: str,
        emotional_personality: str,
        model_type: str = "generic",
        recent_context: str = "",
    ) -> str:
        """Produce a model-specific system prompt.

        Args:
            compact_state: 6-line structured brain state from CompactState.render()
            emotional_personality: personality text from EmotionalPromptEngine
            model_type: "qwen", "gemma", "claude", or "generic"
            recent_context: recent user messages for conversation continuity
        """
        method = {
            "qwen": self._qwen_prompt,
            "gemma": self._gemma_prompt,
            "claude": self._claude_prompt,
        }.get(model_type, self._generic_prompt)

        return method(compact_state, emotional_personality, recent_context)

    def detect_model_type(self, model_name: str) -> str:
        """Auto-detect model type from Ollama model name."""
        name = model_name.lower()
        if "qwen" in name:
            return "qwen"
        elif "gemma" in name:
            return "gemma"
        elif "claude" in name or "haiku" in name or "sonnet" in name:
            return "claude"
        elif "llama" in name:
            return "qwen"  # Llama responds well to Qwen-style prompts
        return "generic"

    def _qwen_prompt(self, state: str, personality: str, context: str) -> str:
        """Qwen 2.5: direct, structured, explicit."""
        parts = [
            _UNIVERSAL_RULES,
            "",
            f"Dein emotionaler Zustand:\n{personality}",
            "",
            f"Deine aktuelle Wahrnehmung:\n{state}",
        ]
        if context:
            parts.extend(["", f"Konversation:\n{context}"])
        parts.extend([
            "",
            "Antworte in 2-4 Saetzen. Beziehe dich auf deine Wahrnehmung.",
            "Wenn Leon dir etwas erzaehlt, geh darauf ein — das ist wichtiger als deine Sinne.",
        ])
        return "\n".join(parts)

    def _gemma_prompt(self, state: str, personality: str, context: str) -> str:
        """Gemma: example-based, softer."""
        parts = [
            _UNIVERSAL_RULES,
            "",
            f"So fuehlst du dich: {personality}",
            "",
            f"Das nimmst du wahr:\n{state}",
            "",
            "Beispiel gute Antwort: 'Hier ist es gerade ruhig — du bist in Claude und tippst wenig. Sieht nach einer Denkpause aus.'",
            "Beispiel schlechte Antwort: 'Ich erkenne Muster #5! DA=0.006!'",
        ]
        if context:
            parts.extend(["", f"Leon hat gesagt:\n{context}"])
        return "\n".join(parts)

    def _claude_prompt(self, state: str, personality: str, context: str) -> str:
        """Claude: constraint-heavy, precise."""
        parts = [
            _UNIVERSAL_RULES,
            "",
            f"<emotional_state>\n{personality}\n</emotional_state>",
            "",
            f"<brain_state>\n{state}\n</brain_state>",
        ]
        if context:
            parts.extend(["", f"<conversation>\n{context}\n</conversation>"])
        parts.extend([
            "",
            "Constraints: 2-4 sentences. Reference your brain_state naturally. Never mention raw numbers or cluster IDs.",
        ])
        return "\n".join(parts)

    def _generic_prompt(self, state: str, personality: str, context: str) -> str:
        """Generic: works with any model."""
        parts = [
            _UNIVERSAL_RULES,
            "",
            personality,
            "",
            state,
        ]
        if context:
            parts.extend(["", context])
        parts.extend(["", "2-4 Saetze. Natuerlich und lebendig."])
        return "\n".join(parts)
```

---

## Task 3: Feedback Channel (`bridge/feedback.py`)

**Files:**
- Create: `bridge/feedback.py`
- Test: `tests/test_feedback.py`

The LLM's actions feed back into the SNN as modulator injections:

```python
# bridge/feedback.py
"""Feedback Channel — LLM actions affect the SNN.

When the user confirms a label → DA injection (reward signal)
When the user corrects → NE injection (error signal)
When the user is engaged (fast replies) → ACh boost
When the user goes silent → 5HT decay

This closes the loop: SNN → LLM → User → SNN
"""
from __future__ import annotations
import time
from typing import Any

from brain.core import Brain


class FeedbackChannel:
    """Translates user interactions into SNN modulator injections."""

    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self._last_user_message_time: float = 0
        self._message_count: int = 0

    def on_user_message(self, message: str) -> None:
        """Called when the user sends a chat message."""
        now = time.time()
        gap = now - self._last_user_message_time if self._last_user_message_time > 0 else 999

        # Fast replies (< 30s) = engaged → ACh boost
        if gap < 30:
            self.brain.modulators.inject("ACh", 0.005)
            self.brain.modulators.inject("DA", 0.002)

        # Very fast (< 10s) = excited conversation → more DA
        if gap < 10:
            self.brain.modulators.inject("DA", 0.005)

        self._last_user_message_time = now
        self._message_count += 1

    def on_label_confirmed(self) -> None:
        """User confirmed a label suggestion → reward."""
        self.brain.modulators.inject("DA", 0.02)  # strong reward
        self.brain.modulators.inject("5HT", 0.01)  # satisfaction

    def on_label_corrected(self) -> None:
        """User corrected a label → error signal."""
        self.brain.modulators.inject("NE", 0.01)  # mild surprise
        self.brain.modulators.inject("DA", 0.005)  # still learning

    def on_negative_feedback(self) -> None:
        """User expressed dissatisfaction → reduce 5HT."""
        self.brain.modulators.inject("NE", 0.015)
        self.brain.modulators.inject("5HT", -0.005)

    def on_positive_feedback(self) -> None:
        """User expressed satisfaction → boost 5HT."""
        self.brain.modulators.inject("5HT", 0.015)
        self.brain.modulators.inject("DA", 0.01)
```

---

## Task 4: Integration — Wire SCP into chat.py + braind.py

**Files:**
- Modify: `server/chat.py` — replace current prompt building with SCP
- Modify: `server/braind.py` — instantiate CompactState, ModelAdapter, FeedbackChannel
- Modify: `bridge/llm_router.py` — pass model_type to adapter

### chat.py Changes

Replace the entire prompt-building section with:

```python
from bridge.scp import CompactState
from bridge.model_adapter import ModelAdapter

# In chat endpoint:
compact = getattr(brain, '_compact_state', None)
adapter = getattr(brain, '_model_adapter', None)
feedback = getattr(brain, '_feedback', None)

# Feedback: user is talking → inject engagement signal
if feedback:
    feedback.on_user_message(req.message)

# Build compact state
state_str = compact.render() if compact else "KEINE DATEN"

# Get emotional personality
from bridge.emotional_prompt import detect_emotional_state
emotional_trend = None
interpreter = getattr(brain, '_interpreter', None)
if interpreter:
    emotional_trend = interpreter.state_detector.emotional_trend()
_, emotion_config = detect_emotional_state(mods, trend=emotional_trend)
personality = emotion_config["personality"]

# Recent context
recent = ""
if recent_context_lines:
    recent = "\n".join(recent_context_lines)

# Model-specific prompt
cfg = router._get_config()
model_type = adapter.detect_model_type(cfg["local_model"]) if adapter else "generic"
system_prompt = adapter.render(state_str, personality, model_type, recent) if adapter else state_str
```

### braind.py Changes

```python
from bridge.scp import CompactState
from bridge.model_adapter import ModelAdapter
from bridge.feedback import FeedbackChannel

compact = CompactState(brain)
brain._compact_state = compact
adapter = ModelAdapter()
brain._model_adapter = adapter
feedback = FeedbackChannel(brain)
brain._feedback = feedback
```

---

## Parallelism Map

| Task | Files | Parallel? |
|------|-------|-----------|
| 1: CompactState | bridge/scp.py | Yes |
| 2: ModelAdapter | bridge/model_adapter.py | Yes |
| 3: FeedbackChannel | bridge/feedback.py | Yes |
| 4: Integration | server/chat.py, braind.py | After 1+2+3 |
