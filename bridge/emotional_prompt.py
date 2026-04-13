"""Emotional Prompt Engine — SNN modulators control the LLM's personality.

Instead of passing raw modulator numbers (which the LLM ignores),
this engine selects from different prompt PERSONALITIES based on
the current emotional state. The SNN literally controls HOW the LLM thinks.

Emotional States (derived from modulator combinations):
- CURIOUS:   high DA + normal NE          -> asks questions, explores
- ALERT:     high NE + low 5HT            -> short, focused, warns about changes
- CONTENT:   high 5HT + normal ACh        -> warm, relaxed, reflective
- FOCUSED:   high ACh + low NE            -> precise, observant, detailed
- DROWSY:    all low                       -> minimal, sleepy, brief
- STRESSED:  high NE + high ACh + low 5HT -> urgent, concerned, protective
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
