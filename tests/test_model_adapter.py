"""Tests for ModelAdapter — model-specific prompt rendering and detection."""
from __future__ import annotations

import pytest

from bridge.model_adapter import ModelAdapter, _UNIVERSAL_RULES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter():
    return ModelAdapter()


@pytest.fixture
def sample_state():
    return (
        "EMOTION: content (seit Minuten stabil) | Grund: ruhige Session\n"
        "PATTERN: #5 unlabeled (85%, 435x) | Sensoren: Tastatur aktiv, Claude offen, leise\n"
        "CHANGE: keine Aenderung\n"
        "MEMORY: 5/20 WM-Slots | denke an: nichts Bestimmtes\n"
        "SESSION: 47min aktiv | Gewohnheit: Sonntag 23h normalerweise idle\n"
        "BRAIN: 52/200 Neuronen spezialisiert | 4 Cluster gelernt | Alter: 3.0 Tage"
    )


@pytest.fixture
def sample_personality():
    return "Du bist ruhig und zufrieden. Die Welt ist in Ordnung."


# ---------------------------------------------------------------------------
# detect_model_type
# ---------------------------------------------------------------------------

class TestDetectModelType:

    def test_qwen_14b(self, adapter):
        assert adapter.detect_model_type("qwen2.5:14b-instruct") == "qwen"

    def test_qwen_7b(self, adapter):
        assert adapter.detect_model_type("qwen2.5:7b") == "qwen"

    def test_gemma_27b(self, adapter):
        assert adapter.detect_model_type("gemma2:27b") == "gemma"

    def test_gemma_9b(self, adapter):
        assert adapter.detect_model_type("gemma2:9b-instruct") == "gemma"

    def test_claude_sonnet(self, adapter):
        assert adapter.detect_model_type("claude-3-sonnet") == "claude"

    def test_claude_haiku(self, adapter):
        assert adapter.detect_model_type("haiku-3.5") == "claude"

    def test_llama_maps_to_qwen(self, adapter):
        assert adapter.detect_model_type("llama3.1:8b") == "qwen"

    def test_llama_70b(self, adapter):
        assert adapter.detect_model_type("llama3:70b-instruct") == "qwen"

    def test_unknown_model_generic(self, adapter):
        assert adapter.detect_model_type("mistral:7b") == "generic"

    def test_unknown_model_phi(self, adapter):
        assert adapter.detect_model_type("phi-3:mini") == "generic"

    def test_case_insensitive(self, adapter):
        assert adapter.detect_model_type("QWEN2.5:14B") == "qwen"
        assert adapter.detect_model_type("Gemma2:27B") == "gemma"
        assert adapter.detect_model_type("Claude-3-Sonnet") == "claude"


# ---------------------------------------------------------------------------
# Universal rules present in all prompts
# ---------------------------------------------------------------------------

class TestUniversalRules:

    def test_keine_emojis_in_all(self, adapter, sample_state, sample_personality):
        for model_type in ["qwen", "gemma", "claude", "generic"]:
            prompt = adapter.render(sample_state, sample_personality, model_type).lower()
            assert "keine emojis" in prompt or "kein einziges" in prompt, f"Missing emoji rule in {model_type}"

    def test_keine_erfundenen_faehigkeiten(self, adapter, sample_state, sample_personality):
        for model_type in ["qwen", "gemma", "claude", "generic"]:
            prompt = adapter.render(sample_state, sample_personality, model_type).lower()
            assert "erfinde nichts" in prompt or "erfundenen" in prompt or "nicht sehen" in prompt, f"Missing capability rule in {model_type}"

    def test_keine_hilfsangebote(self, adapter, sample_state, sample_personality):
        for model_type in ["qwen", "gemma", "claude", "generic"]:
            prompt = adapter.render(sample_state, sample_personality, model_type).lower()
            assert "keine hilfe" in prompt or "biete keine" in prompt or "soll ich" in prompt, f"Missing help rule in {model_type}"

    def test_widersprich_nie(self, adapter, sample_state, sample_personality):
        for model_type in ["qwen", "gemma", "claude", "generic"]:
            prompt = adapter.render(sample_state, sample_personality, model_type).lower()
            assert "widersprich" in prompt or "sinnen" in prompt, f"Missing contradiction rule in {model_type}"

    def test_state_present_in_all(self, adapter, sample_state, sample_personality):
        for model_type in ["qwen", "gemma", "claude", "generic"]:
            prompt = adapter.render(sample_state, sample_personality, model_type)
            assert "EMOTION:" in prompt, f"State missing in {model_type}"
            assert "PATTERN:" in prompt, f"State missing in {model_type}"


# ---------------------------------------------------------------------------
# Qwen prompt style
# ---------------------------------------------------------------------------

class TestQwenPrompt:

    def test_structured_headers(self, adapter, sample_state, sample_personality):
        prompt = adapter.render(sample_state, sample_personality, "qwen")
        assert "Dein emotionaler Zustand:" in prompt
        assert "Deine aktuelle Wahrnehmung:" in prompt

    def test_explicit_instructions(self, adapter, sample_state, sample_personality):
        prompt = adapter.render(sample_state, sample_personality, "qwen")
        assert "2-4 Saetzen" in prompt
        assert "Beziehe dich auf deine Wahrnehmung" in prompt

    def test_with_context(self, adapter, sample_state, sample_personality):
        prompt = adapter.render(sample_state, sample_personality, "qwen", "Hallo!")
        assert "Konversation:" in prompt
        assert "Hallo!" in prompt

    def test_without_context(self, adapter, sample_state, sample_personality):
        prompt = adapter.render(sample_state, sample_personality, "qwen", "")
        assert "Konversation:" not in prompt

    def test_leon_priority(self, adapter, sample_state, sample_personality):
        prompt = adapter.render(sample_state, sample_personality, "qwen")
        assert "Leon dir etwas erzaehlt" in prompt


# ---------------------------------------------------------------------------
# Gemma prompt style
# ---------------------------------------------------------------------------

class TestGemmaPrompt:

    def test_softer_tone(self, adapter, sample_state, sample_personality):
        prompt = adapter.render(sample_state, sample_personality, "gemma")
        assert "So fuehlst du dich:" in prompt
        assert "Das nimmst du wahr:" in prompt

    def test_good_example(self, adapter, sample_state, sample_personality):
        prompt = adapter.render(sample_state, sample_personality, "gemma")
        assert "Beispiel gute Antwort:" in prompt
        assert "Denkpause" in prompt

    def test_bad_example(self, adapter, sample_state, sample_personality):
        prompt = adapter.render(sample_state, sample_personality, "gemma")
        assert "Beispiel schlechte Antwort:" in prompt
        assert "DA=0.006" in prompt

    def test_with_context(self, adapter, sample_state, sample_personality):
        prompt = adapter.render(sample_state, sample_personality, "gemma", "Was machst du?")
        assert "Leon hat gesagt:" in prompt
        assert "Was machst du?" in prompt

    def test_without_context(self, adapter, sample_state, sample_personality):
        prompt = adapter.render(sample_state, sample_personality, "gemma", "")
        assert "Leon hat gesagt:" not in prompt


# ---------------------------------------------------------------------------
# Claude prompt style
# ---------------------------------------------------------------------------

class TestClaudePrompt:

    def test_xml_tags(self, adapter, sample_state, sample_personality):
        prompt = adapter.render(sample_state, sample_personality, "claude")
        assert "<emotional_state>" in prompt
        assert "</emotional_state>" in prompt
        assert "<brain_state>" in prompt
        assert "</brain_state>" in prompt

    def test_constraints(self, adapter, sample_state, sample_personality):
        prompt = adapter.render(sample_state, sample_personality, "claude")
        assert "Constraints:" in prompt
        assert "Never mention raw numbers" in prompt

    def test_with_context(self, adapter, sample_state, sample_personality):
        prompt = adapter.render(sample_state, sample_personality, "claude", "Guten Abend")
        assert "<conversation>" in prompt
        assert "</conversation>" in prompt
        assert "Guten Abend" in prompt

    def test_without_context(self, adapter, sample_state, sample_personality):
        prompt = adapter.render(sample_state, sample_personality, "claude", "")
        assert "<conversation>" not in prompt


# ---------------------------------------------------------------------------
# Generic prompt style
# ---------------------------------------------------------------------------

class TestGenericPrompt:

    def test_minimal_structure(self, adapter, sample_state, sample_personality):
        prompt = adapter.render(sample_state, sample_personality, "generic")
        assert sample_personality in prompt
        assert sample_state in prompt

    def test_closing_instruction(self, adapter, sample_state, sample_personality):
        prompt = adapter.render(sample_state, sample_personality, "generic")
        assert "2-4 Saetze" in prompt
        assert "Natuerlich und lebendig" in prompt

    def test_with_context(self, adapter, sample_state, sample_personality):
        prompt = adapter.render(sample_state, sample_personality, "generic", "Wie geht's?")
        assert "Wie geht's?" in prompt

    def test_without_context(self, adapter, sample_state, sample_personality):
        prompt = adapter.render(sample_state, sample_personality, "generic", "")
        # Should still have the basic structure
        assert _UNIVERSAL_RULES in prompt


# ---------------------------------------------------------------------------
# render() dispatch
# ---------------------------------------------------------------------------

class TestRenderDispatch:

    def test_unknown_type_falls_back_to_generic(self, adapter, sample_state, sample_personality):
        prompt = adapter.render(sample_state, sample_personality, "unknown_model")
        # Generic prompt has "2-4 Saetze. Natuerlich und lebendig."
        assert "Natuerlich und lebendig" in prompt

    def test_default_is_generic(self, adapter, sample_state, sample_personality):
        prompt = adapter.render(sample_state, sample_personality)
        assert "Natuerlich und lebendig" in prompt

    def test_each_type_produces_different_output(self, adapter, sample_state, sample_personality):
        prompts = {}
        for t in ["qwen", "gemma", "claude", "generic"]:
            prompts[t] = adapter.render(sample_state, sample_personality, t)

        # All four should be different from each other
        types = list(prompts.keys())
        for i, a in enumerate(types):
            for b in types[i + 1:]:
                assert prompts[a] != prompts[b], f"{a} and {b} produced identical prompts"

    def test_all_contain_state(self, adapter, sample_state, sample_personality):
        for t in ["qwen", "gemma", "claude", "generic"]:
            prompt = adapter.render(sample_state, sample_personality, t)
            assert "EMOTION:" in prompt
            assert "BRAIN:" in prompt
