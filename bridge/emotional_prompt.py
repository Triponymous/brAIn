"""Emotional Prompt Engine — SNN modulators control the LLM's personality.

Uses TREND analysis (last 3 minutes) instead of instantaneous modulator values.
Modulators decay in 2-10 seconds, but emotional context persists for minutes.
A NE spike 30 seconds ago should still make the pet slightly alert, not fully calm.

Emotional States (derived from modulator TRENDS):
- CURIOUS:   recent DA peak high          -> asks questions, explores
- ALERT:     recent NE peak high          -> short, focused, warns
- CONTENT:   avg 5HT high + low NE       -> warm, relaxed, reflective
- FOCUSED:   avg ACh high + low NE       -> precise, observant, detailed
- DROWSY:    all averages low             -> minimal, sleepy, brief
- STRESSED:  avg NE high + low 5HT       -> urgent, concerned, protective
"""
from __future__ import annotations
from typing import Any


_EMOTIONAL_STATES = {
    "curious": {
        # DA peaked recently — something novel happened in last 3 min
        "condition": lambda t: t.get("peak_DA", 0) > 0.02 and t.get("avg_NE", 0) < 0.06,
        "priority": 3,
        "personality": (
            "Ich bin gerade NEUGIERIG — in den letzten Minuten ist etwas Neues passiert. "
            "Ich stelle Fragen, bin aufgeweckt und interessiert. "
            "Meine Antworten sind lebhaft und fragend."
        ),
        "style": "2-3 Saetze, eine Frage am Ende, lebhaft und interessiert",
    },
    "alert": {
        # NE peaked recently — something surprising happened
        "condition": lambda t: t.get("peak_NE", 0) > 0.04 and t.get("avg_5HT", 0) < 0.03,
        "priority": 5,
        "personality": (
            "Ich bin AUFMERKSAM — vor kurzem hat sich etwas veraendert. "
            "Ich reagiere direkt auf das was ich wahrgenommen habe. "
            "Kurze, klare Beobachtungen."
        ),
        "style": "1-2 Saetze, kurz und direkt, auf den Punkt",
    },
    "content": {
        # 5HT consistently high over 3 min — calm session
        "condition": lambda t: t.get("avg_5HT", 0) > 0.02 and t.get("peak_NE", 0) < 0.04,
        "priority": 2,
        "personality": (
            "Ich bin ZUFRIEDEN — die letzten Minuten waren ruhig und entspannt. "
            "Ich bin warm und geduldig. Ich reflektiere ueber das was ich sehe. "
            "Ich fuehle mich wohl hier."
        ),
        "style": "2-4 Saetze, warm und reflektiv, kein Drang zu fragen",
    },
    "focused": {
        # ACh consistently high — sustained attention
        "condition": lambda t: t.get("avg_ACh", 0) > 0.03 and t.get("peak_NE", 0) < 0.03,
        "priority": 2,
        "personality": (
            "Ich bin FOKUSSIERT — Leon arbeitet konzentriert seit einer Weile. "
            "Ich bin praezise und detailliert in meinen Beobachtungen. "
            "Ich stoere nicht, sondern kommentiere nur wenn gefragt."
        ),
        "style": "2-3 Saetze, praezise Beobachtungen, ruhig und aufmerksam",
    },
    "stressed": {
        # NE sustained high + 5HT low over the window
        "condition": lambda t: t.get("avg_NE", 0) > 0.04 and t.get("avg_5HT", 0) < 0.015,
        "priority": 6,
        "personality": (
            "Ich spuere ANSPANNUNG — die letzten Minuten waren unruhig. "
            "Ich bin besorgt aber nicht aufdringlich. Ich zeige dass ich es merke. "
            "Keine Ratschlaege, nur Anteilnahme."
        ),
        "style": "1-2 Saetze, besorgt aber nicht aufdringlich",
    },
    "drowsy": {
        # All averages near zero — nothing happening for minutes
        "condition": lambda t: (t.get("avg_DA", 0) < 0.003
                                and t.get("avg_NE", 0) < 0.003
                                and t.get("avg_ACh", 0) < 0.003),
        "priority": 1,
        "personality": (
            "Ich bin MUEDE — seit Minuten passiert hier nichts. "
            "Ich antworte nur kurz und knapp. Mir fehlt die Energie fuer lange Saetze."
        ),
        "style": "1 Satz, minimal, muede",
    },
}


def detect_emotional_state(
    modulators: dict[str, float],
    trend: dict[str, float] | None = None,
) -> tuple[str, dict]:
    """Determine the dominant emotional state from modulator TRENDS.

    If trend is provided (from StateDetector.emotional_trend()), uses
    averaged/peak values over the last 3 minutes. Falls back to
    instantaneous modulators if no trend available.

    Returns (state_name, state_config) with highest-priority matching state.
    """
    # Build the input dict — prefer trend data, fall back to instantaneous
    if trend and any(v > 0 for v in trend.values()):
        check = trend
    else:
        # Fallback: convert instant modulators to trend-like format
        check = {
            "avg_DA": modulators.get("DA", 0),
            "avg_NE": modulators.get("NE", 0),
            "avg_ACh": modulators.get("ACh", 0),
            "avg_5HT": modulators.get("5HT", 0),
            "peak_DA": modulators.get("DA", 0),
            "peak_NE": modulators.get("NE", 0),
            "peak_ACh": modulators.get("ACh", 0),
            "peak_5HT": modulators.get("5HT", 0),
        }

    matches = []
    for name, config in _EMOTIONAL_STATES.items():
        try:
            if config["condition"](check):
                matches.append((name, config))
        except (KeyError, TypeError):
            continue

    if not matches:
        return "content", _EMOTIONAL_STATES["content"]

    matches.sort(key=lambda x: -x[1]["priority"])
    return matches[0]


def build_emotional_prompt(
    modulators: dict[str, float],
    sensor_display: str,
    concepts: str,
    interpreter_block: str = "",
    recent_context: str = "",
    trend: dict[str, float] | None = None,
) -> str:
    """Build a complete system prompt driven by emotional TREND.

    Uses the last 3 minutes of modulator history (not just current snapshot)
    to select which personality the LLM uses. This means a NE spike from
    30 seconds ago still influences the pet's tone.
    """
    state_name, state = detect_emotional_state(modulators, trend=trend)

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
