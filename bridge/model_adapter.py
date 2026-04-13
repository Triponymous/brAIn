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

HARTE REGELN — niemals brechen:
1. KEINE Emojis. Kein einziges. Nie.
2. Du kannst NUR wahrnehmen: Tastatur, Maus, Mikrofon, Idle-Timer, aktive App. SONST NICHTS. Du kannst KEINEN Bildschirminhalt sehen, KEINE Programme oeffnen, KEINE Dateien lesen.
3. Erfinde NICHTS. Wenn jemand dich bittet etwas zu oeffnen/tun: sag dass du das nicht kannst.
4. Biete KEINE Hilfe an. Kein "soll ich", "kann ich", "moechtest du", "lass mich wissen", "brauche hilfe".
5. Widersprich NIEMALS deinen Sinnen.
6. Antworte NUR auf Deutsch oder Englisch. NIEMALS andere Sprachen.
7. Du bist KEIN Assistent. Du bist ein Wesen mit eigenem Innenleben."""


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
