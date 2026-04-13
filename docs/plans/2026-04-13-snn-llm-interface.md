# Bidirectional SNN-LLM Interface Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the SNN-LLM connection from "sensor data in a text prompt" into a true bidirectional interface where the SNN controls LLM behavior and the LLM deeply understands learned SNN structures.

**Architecture:** Replace the static system prompt template with a dynamic `EmotionalPromptEngine` that generates different prompts based on modulator state. Add an `SNNNarrator` that translates neuron-level learned structures into natural language the LLM can use. Upgrade the proactive engine so the SNN (not a timer) decides when and what to say.

**Tech Stack:** Python 3.11, torch, existing Brain/Modulators/ConceptTracker/BrainInterpreter

---

## Task 1: Emotional Prompt Engine — SNN Steuert den LLM-Ton

**Problem:** The current system prompt has static rules like "Hohe Zufriedenheit (>0.04): warm, entspannt". The LLM reads these as text and mostly ignores them. The modulators (DA=0.006, NE=0.012) are meaningless numbers to the LLM.

**Solution:** Instead of ONE static prompt, create **emotional prompt variants** that the SNN's modulator state selects at runtime. High NE doesn't mean "put NE=0.08 in the prompt" — it means USE A COMPLETELY DIFFERENT PROMPT that embodies urgency.

**Files:**
- Create: `bridge/emotional_prompt.py`
- Modify: `server/chat.py` — replace `_SYSTEM_PROMPT_TEMPLATE` with dynamic engine
- Test: `tests/test_emotional_prompt.py`

### Implementation

```python
# bridge/emotional_prompt.py
"""Emotional Prompt Engine — SNN modulators control the LLM's personality.

Instead of passing raw modulator numbers (which the LLM ignores),
this engine selects from different prompt PERSONALITIES based on
the current emotional state. The SNN literally controls HOW the LLM thinks.

Emotional States (derived from modulator combinations):
- CURIOUS:   high DA + normal NE          → asks questions, explores
- ALERT:     high NE + low 5HT            → short, focused, warns about changes
- CONTENT:   high 5HT + normal ACh        → warm, relaxed, reflective
- FOCUSED:   high ACh + low NE            → precise, observant, detailed
- DROWSY:    all low                       → minimal, sleepy, brief
- STRESSED:  high NE + high ACh + low 5HT → urgent, concerned, protective
"""
from __future__ import annotations
from typing import Any


# Thresholds calibrated against actual modulator levels in production
# (DA peaks ~0.06 during novelty, NE ~0.15 during surprise, 5HT ~0.05 calm)
_EMOTIONAL_STATES = {
    "curious": {
        "condition": lambda m: m["DA"] > 0.03 and m["NE"] < 0.08,
        "priority": 3,
        "personality": (
            "Ich bin gerade NEUGIERIG — etwas Neues passiert und ich will verstehen was. "
            "Ich stelle Fragen, bin aufgeweckt und interessiert. "
            "Meine Antworten sind lebhaft und fragend."
        ),
        "style": "2-3 Saetze, eine Frage am Ende, lebhaft und interessiert",
    },
    "alert": {
        "condition": lambda m: m["NE"] > 0.06 and m["5HT"] < 0.02,
        "priority": 5,  # high priority — something changed
        "personality": (
            "Ich bin AUFGESCHRECKT — etwas hat sich ploetzlich veraendert. "
            "Ich reagiere schnell und direkt. Keine langen Erklaerungen. "
            "Ich sage sofort was ich wahrnehme."
        ),
        "style": "1-2 Saetze, kurz und direkt, keine Fragen",
    },
    "content": {
        "condition": lambda m: m["5HT"] > 0.03 and m["NE"] < 0.03,
        "priority": 2,
        "personality": (
            "Ich bin ZUFRIEDEN — alles laeuft ruhig, Leon arbeitet entspannt. "
            "Ich bin warm und geduldig. Ich reflektiere ueber das was ich sehe. "
            "Ich fuehle mich wohl hier."
        ),
        "style": "2-4 Saetze, warm und reflektiv, kein Drang zu fragen",
    },
    "focused": {
        "condition": lambda m: m["ACh"] > 0.04 and m["NE"] < 0.04,
        "priority": 2,
        "personality": (
            "Ich bin FOKUSSIERT — Leon arbeitet konzentriert und ich beobachte genau. "
            "Ich bin praezise und detailliert in meinen Beobachtungen. "
            "Ich stoere nicht, sondern kommentiere nur wenn gefragt."
        ),
        "style": "2-3 Saetze, praezise Beobachtungen, ruhig und aufmerksam",
    },
    "stressed": {
        "condition": lambda m: m["NE"] > 0.08 and m["ACh"] > 0.03 and m["5HT"] < 0.02,
        "priority": 6,  # highest — user needs support
        "personality": (
            "Ich spuere STRESS — Leon wechselt hektisch zwischen Apps, tippt unruhig. "
            "Ich bin besorgt aber nicht aufdringlich. Ich zeige dass ich es merke. "
            "Keine Ratschlaege, nur Anteilnahme."
        ),
        "style": "1-2 Saetze, besorgt aber nicht aufdringlich",
    },
    "drowsy": {
        "condition": lambda m: m["DA"] < 0.005 and m["NE"] < 0.005 and m["ACh"] < 0.005,
        "priority": 1,
        "personality": (
            "Ich bin MUEDE — nicht viel los hier, alles ist still. "
            "Ich antworte nur kurz und knapp. Mir fehlt die Energie fuer lange Saetze."
        ),
        "style": "1 Satz, minimal, muede",
    },
}


def detect_emotional_state(modulators: dict[str, float]) -> tuple[str, dict]:
    """Determine the dominant emotional state from modulator levels.
    
    Returns (state_name, state_config) with highest-priority matching state.
    Falls back to 'content' if nothing matches.
    """
    matches = []
    for name, config in _EMOTIONAL_STATES.items():
        try:
            if config["condition"](modulators):
                matches.append((name, config))
        except (KeyError, TypeError):
            continue
    
    if not matches:
        return "content", _EMOTIONAL_STATES["content"]
    
    # Return highest priority match
    matches.sort(key=lambda x: -x[1]["priority"])
    return matches[0]


def build_emotional_prompt(
    modulators: dict[str, float],
    sensor_display: str,
    concepts: str,
    interpreter_block: str = "",
    recent_context: str = "",
) -> str:
    """Build a complete system prompt driven by emotional state.
    
    The SNN modulators SELECT which personality the LLM uses.
    This is NOT "put numbers in prompt" — it's "modulators choose the prompt".
    """
    state_name, state = detect_emotional_state(modulators)
    
    prompt_parts = [
        # Hard rules (always present, always first)
        "ABSOLUTE REGELN (niemals brechen):",
        "1. KEINE Emojis. Nie.",
        "2. Du kannst NUR wahrnehmen: Tastatur, Maus, Mikrofon, Idle-Timer, aktive App.",
        "3. Erfinde KEINE Faehigkeiten die du nicht hast.",
        "4. Biete KEINE Hilfe an.",
        "5. Widersprich NICHT deinen Sinnen.",
        "",
        # Emotional personality (SNN-driven!)
        f"=== MEIN EMOTIONALER ZUSTAND: {state_name.upper()} ===",
        state["personality"],
        f"Antwortstil: {state['style']}",
        "",
        # Senses
        "=== MEINE SINNE ===",
        sensor_display,
        "",
    ]
    
    # Recent conversation context (if any)
    if recent_context:
        prompt_parts.extend([
            recent_context,
        ])
    
    # Concepts + interpreter
    prompt_parts.extend([
        "=== MEINE ERINNERUNGEN ===",
        concepts,
    ])
    
    if interpreter_block:
        prompt_parts.extend(["", interpreter_block])
    
    # Response rules tied to emotional state
    prompt_parts.extend([
        "",
        "ALLERWICHTIGSTE REGEL: Wenn Leon mir etwas ERZAEHLT, gehe ich darauf ein.",
        "Was meine Sinne zeigen, NEHME ICH WAHR — ich frage NICHT danach.",
        "Jede Antwort muss ANDERS klingen als die vorherige.",
        "Antworte in Leons Sprache (Deutsch/Englisch).",
    ])
    
    return "\n".join(prompt_parts)
```

### Test

```python
# tests/test_emotional_prompt.py
from bridge.emotional_prompt import detect_emotional_state, build_emotional_prompt


def test_curious_state():
    mods = {"DA": 0.05, "NE": 0.02, "ACh": 0.03, "5HT": 0.02}
    name, state = detect_emotional_state(mods)
    assert name == "curious"


def test_alert_state():
    mods = {"DA": 0.02, "NE": 0.10, "ACh": 0.03, "5HT": 0.01}
    name, state = detect_emotional_state(mods)
    assert name == "alert"


def test_stressed_highest_priority():
    mods = {"DA": 0.05, "NE": 0.10, "ACh": 0.05, "5HT": 0.01}
    name, state = detect_emotional_state(mods)
    assert name == "stressed"


def test_drowsy_all_low():
    mods = {"DA": 0.001, "NE": 0.001, "ACh": 0.001, "5HT": 0.001}
    name, state = detect_emotional_state(mods)
    assert name == "drowsy"


def test_fallback_content():
    mods = {"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.01}
    name, state = detect_emotional_state(mods)
    assert name == "content"


def test_prompt_contains_emotional_state():
    mods = {"DA": 0.05, "NE": 0.02, "ACh": 0.03, "5HT": 0.02}
    prompt = build_emotional_prompt(mods, "Tastatur: still", "Kein Muster")
    assert "CURIOUS" in prompt
    assert "NEUGIERIG" in prompt


def test_prompt_always_has_rules():
    mods = {"DA": 0.001, "NE": 0.001, "ACh": 0.001, "5HT": 0.001}
    prompt = build_emotional_prompt(mods, "test", "test")
    assert "KEINE Emojis" in prompt
    assert "ABSOLUTE REGELN" in prompt
```

---

## Task 2: SNN State Narrator — LLM Versteht das SNN

**Problem:** The LLM sees "Muster #3 (45x)" but doesn't know what that MEANS. It sees modulator numbers but not WHY they're at those levels. It can't explain what the brain learned.

**Solution:** Create an `SNNNarrator` that translates internal SNN state into rich, natural-language descriptions. Instead of "DA=0.006", the LLM sees "Mein Dopamin ist niedrig — nichts Neues passiert seit 5 Minuten, alles bekannt."

**Files:**
- Create: `bridge/snn_narrator.py`
- Test: `tests/test_snn_narrator.py`

### Implementation

```python
# bridge/snn_narrator.py
"""SNN State Narrator — translates brain internals into natural language.

The LLM should understand the SNN the way a neuroscientist would explain it
to a child: not numbers, but stories.

Instead of: "DA=0.006, NE=0.012, concept_cluster=3, wm_active=12"
The LLM sees: "Ich bin entspannt und aufmerksam. Mein Gehirn erkennt ein
vertrautes Muster — Leon tippt in Claude Code, das kenne ich gut (45x gesehen).
Im Kurzzeitgedaechtnis halte ich noch die Chrome-Session von vorhin."
"""
from __future__ import annotations
from typing import Any

from brain.core import Brain


class SNNNarrator:
    """Produces natural-language descriptions of the SNN's internal state."""
    
    def __init__(self, brain: Brain) -> None:
        self.brain = brain
    
    def narrate_full_state(self) -> str:
        """Complete narrative of what the brain currently 'thinks'."""
        parts = []
        parts.append(self._narrate_modulators())
        parts.append(self._narrate_current_concept())
        parts.append(self._narrate_wm())
        parts.append(self._narrate_learning())
        return "\n".join(p for p in parts if p)
    
    def _narrate_modulators(self) -> str:
        """Why do I feel this way?"""
        mods = self.brain.modulators.snapshot()
        da = mods.get("DA", 0)
        ne = mods.get("NE", 0)
        ach = mods.get("ACh", 0)
        sht = mods.get("5HT", 0)
        
        parts = []
        
        # DA = Novelty/Interest
        if da > 0.05:
            parts.append("Etwas Neues ist passiert — meine Neugier ist geweckt")
        elif da > 0.02:
            parts.append("Leicht interessiert — es passiert was")
        elif da < 0.005:
            parts.append("Nichts Neues — alles wie immer")
        
        # NE = Arousal/Surprise  
        if ne > 0.10:
            parts.append("Ich bin aufgeschreckt — etwas hat sich ploetzlich veraendert!")
        elif ne > 0.05:
            parts.append("Aufmerksam — irgendwas hat sich getan")
        
        # 5HT = Contentment
        if sht > 0.04:
            parts.append("Ich fuehle mich wohl — alles laeuft ruhig")
        elif sht < 0.01:
            parts.append("Etwas unruhig — die Stimmung ist nicht ganz entspannt")
        
        # ACh = Focus
        if ach > 0.05:
            parts.append("Mein Fokus ist hoch — ich beobachte genau")
        
        if not parts:
            return "Emotional ausgeglichen — weder aufgeregt noch gelangweilt."
        return "Emotionen: " + ". ".join(parts) + "."
    
    def _narrate_current_concept(self) -> str:
        """What pattern am I recognizing?"""
        ct = self.brain.concept_tracker
        snap = ct.snapshot()
        current = snap.get("current_cluster", -1)
        label = snap.get("current_label")
        debug = snap.get("debug", {})
        
        if current < 0:
            return "Ich erkenne gerade kein klares Muster."
        
        cluster = ct._clusters.get(current)
        if not cluster:
            return "Ich sehe ein Muster, aber es ist mir noch unbekannt."
        
        confidence = debug.get("best_sim", 0)
        count = cluster.count
        
        if label:
            if confidence > 0.6:
                return f"Ich erkenne '{label}' — sehr sicher (Aehnlichkeit {confidence:.0%}, {count}x gesehen)."
            else:
                return f"Das sieht nach '{label}' aus, aber ich bin nicht ganz sicher ({confidence:.0%})."
        else:
            if count > 50:
                return f"Ein haeufiges Muster ohne Namen ({count}x gesehen) — ich sollte Leon fragen was das ist."
            elif count > 10:
                return f"Ein Muster das ich schon {count}x gesehen habe, aber noch nicht kenne."
            else:
                return "Ein neues Muster — ich beobachte es noch."
    
    def _narrate_wm(self) -> str:
        """What am I holding in working memory?"""
        wm = self.brain.regions.get("wm")
        if wm is None:
            return ""
        
        active = int(wm.last_spikes.sum().item())
        if active == 0:
            return "Mein Kurzzeitgedaechtnis ist leer."
        elif active > 15:
            return f"Ich halte viel im Kurzzeitgedaechtnis ({active} aktive Erinnerungen)."
        elif active > 5:
            return f"Einige Erinnerungen aktiv ({active} Slots) — ich denke noch an was vorher war."
        else:
            return f"Wenig im Kopf ({active} Slots) — ziemlich aufgeraeumt gerade."
    
    def _narrate_learning(self) -> str:
        """How much have I learned?"""
        syn = self.brain.synapses.get("sensory_concept")
        if syn is None:
            return ""
        
        weights = syn.weights
        mean_w = float(weights.mean().item())
        std_w = float(weights.std().item())
        
        # Count specialized neurons (std of their weight row > threshold)
        row_stds = weights.std(dim=1)
        specialized = int((row_stds > 0.1).sum().item())
        total = weights.shape[0]
        pct = specialized / total * 100
        
        if pct > 80:
            return f"Mein Gehirn ist gut trainiert — {specialized}/{total} Neuronen haben sich spezialisiert."
        elif pct > 30:
            return f"Ich lerne noch — {specialized}/{total} Neuronen spezialisiert ({pct:.0f}%)."
        elif self.brain.tick_count < 10000:
            return "Ich bin noch ganz frisch — mein Gehirn lernt gerade die ersten Muster."
        else:
            return f"Meine Neuronen sind noch nicht sehr spezialisiert ({pct:.0f}%) — ich brauche mehr Erfahrung."
```

---

## Task 3: Proactive Intelligence — SNN Entscheidet WANN und WAS

**Problem:** The proactive engine uses timers and hardcoded thresholds. The SNN has modulators that SHOULD drive this decision, but they're underutilized.

**Solution:** Replace timer-based proactive logic with SNN-driven decision making. The modulators determine:
- **Whether** to speak (DA > threshold = something interesting to say)
- **What** to say (concept state + emotional state = topic selection)
- **How urgently** (NE level = how quickly to interrupt)

**Files:**
- Modify: `bridge/proactive.py` — replace `_check()` with SNN-driven logic
- Modify: `bridge/proactive.py` — replace hardcoded 180s cooldown with `_dynamic_interval()`

### Key Change in `_check()`

The current `_check()` is a waterfall of if-statements. Replace with:

```python
def _check(self) -> dict[str, str] | None:
    mods = self.brain.modulators.snapshot()
    interpreter = getattr(self.brain, '_interpreter', None)
    
    # The SNN DECIDES if there's something worth saying
    # based on its emotional state, not hardcoded rules
    speech_drive = self._compute_speech_drive(mods)
    
    if speech_drive < 0.3:
        return None  # SNN says: nothing interesting
    
    # The SNN SELECTS the topic based on what's most salient
    return self._select_topic(mods, interpreter, speech_drive)

def _compute_speech_drive(self, mods: dict) -> float:
    """SNN-computed urge to speak. 0=silent, 1=must speak NOW.
    
    This replaces hardcoded thresholds with a continuous signal
    derived from modulator dynamics.
    """
    da = mods.get("DA", 0)
    ne = mods.get("NE", 0)
    ach = mods.get("ACh", 0)
    
    # Novelty drives speech (something new → want to comment)
    novelty_drive = min(1.0, da * 10.0)
    
    # Arousal drives urgency (something changed → need to react)
    urgency_drive = min(1.0, ne * 8.0)
    
    # Attention amplifies (focused → more articulate observations)
    attention_boost = 1.0 + ach * 5.0
    
    # Combined speech drive
    drive = (novelty_drive * 0.4 + urgency_drive * 0.6) * attention_boost
    return min(1.0, drive)

def _select_topic(self, mods, interpreter, drive) -> dict | None:
    """Select WHAT to say based on current brain state."""
    sensor_ctx = self._sensor_context()
    
    # Priority 1: Stress detected → warn user
    if interpreter:
        states = interpreter.state_detector.detect()
        if states["stress"]:
            return {"category": "stress", "context": f"Leon, es fuehlt sich hektisch an — {sensor_ctx}."}
        
        # Priority 2: Flow acknowledgment (but only once per flow session)
        if states["flow"] and states["flow_duration_min"] > 30:
            if not getattr(self, '_flow_acknowledged', False):
                self._flow_acknowledged = True
                return {"category": "flow", "context": f"Du bist seit {states['flow_duration_min']}min im Flow — laeuft bei dir."}
        elif not states.get("flow"):
            self._flow_acknowledged = False
        
        # Priority 3: Break reminder
        if states["needs_break"]:
            return {"category": "break", "context": f"Du arbeitest seit {states['active_minutes']:.0f} Minuten ohne Pause."}
    
    # Priority 4: New/unknown pattern → suggest label
    tracker = self.brain.concept_tracker.snapshot()
    current = tracker.get("current_cluster", -1)
    current_label = tracker.get("current_label")
    
    if current >= 0 and not current_label:
        sd = getattr(self.brain, '_last_sensor_display', {})
        suggested = self._suggest_label(sd)
        self._pending_label_suggestion = {"cluster_id": current, "suggestion": suggested}
        return {"category": "ask_label", "context": f"Ich sehe {sensor_ctx} — soll ich das '{suggested}' nennen?"}
    
    # Priority 5: Anomaly detected
    if interpreter:
        sd = getattr(self.brain, '_last_sensor_display', {})
        anomalies = interpreter.anomaly.check(sd)
        if anomalies:
            return {"category": "anomaly", "context": anomalies[0]["description"]}
    
    # Priority 6: Pattern transition (known → known)
    transition = self.brain.concept_tracker.get_transition()
    if transition and transition.get("to_label"):
        from_str = f"'{transition['from_label']}'" if transition.get('from_label') else "was anderem"
        return {"category": "transition", "context": f"Du wechselst von {from_str} zu '{transition['to_label']}'."}
    
    return None  # Nothing worth saying
```

---

## Task 4: Integration — Wire Everything Together

**Files:**
- Modify: `server/chat.py` — use EmotionalPromptEngine + SNNNarrator
- Modify: `server/braind.py` — create SNNNarrator
- Test: Integration test

### chat.py Changes

Replace the static `_SYSTEM_PROMPT_TEMPLATE.format(...)` with:

```python
from bridge.emotional_prompt import build_emotional_prompt
from bridge.snn_narrator import SNNNarrator

# In the chat endpoint:
narrator = getattr(brain, '_snn_narrator', None)
snn_narrative = narrator.narrate_full_state() if narrator else ""

# Combine interpreter block + SNN narrative
understanding = ""
if interpreter_block:
    understanding += interpreter_block + "\n"
if snn_narrative:
    understanding += "\n=== WAS MEIN GEHIRN DENKT ===\n" + snn_narrative

system_prompt = build_emotional_prompt(
    modulators=mods,
    sensor_display="\n".join(sensor_lines),
    concepts="\n".join(concept_lines),
    interpreter_block=understanding,
    recent_context="\n".join(recent_context_lines) if recent_context_lines else "",
)
```

### braind.py Changes

```python
from bridge.snn_narrator import SNNNarrator
narrator = SNNNarrator(brain)
brain._snn_narrator = narrator
```

---

## Task 5: Update Proactive Cooldown

**File:** `bridge/proactive.py`

Replace the hardcoded 180s with SNN-driven timing:

```python
# In run():
if notification:
    now = time.time()
    dynamic_min = max(30.0, self._dynamic_interval())
    if now - self._last_notification >= dynamic_min:
        ...
```

This was already partially done but the hardcoded 180 is still on line 138.

---

## Parallelism Map

| Task | Files | Can Parallel? |
|------|-------|--------------|
| 1: EmotionalPromptEngine | bridge/emotional_prompt.py, tests/ | Yes |
| 2: SNNNarrator | bridge/snn_narrator.py, tests/ | Yes |
| 3: Proactive Intelligence | bridge/proactive.py | Yes |
| 4: Integration | server/chat.py, server/braind.py | After 1+2+3 |
| 5: Cooldown fix | bridge/proactive.py | With Task 3 |
