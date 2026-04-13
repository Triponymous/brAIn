"""Tests for SCP Client -- LLM Client, BrainServer, and model adapters.

Tests cover:
1. BrainServer queries return valid SCP response schemas.
2. BrainServer actions dispatch correctly and affect modulators.
3. BrainServer events can be pushed and polled.
4. SCPClient.build_prompt produces valid output for each model type.
5. SCPClient.send_feedback dispatches to BrainServer correctly.
6. SCPClient.detect_model_type maps model names properly.
7. Each adapter produces model-appropriate formatting.
"""
from __future__ import annotations

import torch
from unittest.mock import MagicMock

import pytest

from bridge.scp_client import (
    BrainServer,
    SCPClient,
    QwenAdapter,
    GemmaAdapter,
    ClaudeAdapter,
    GenericAdapter,
)


# ---------------------------------------------------------------------------
# Mock helpers -- reuse patterns from test_scp.py
# ---------------------------------------------------------------------------

def _make_cluster(
    label: str | None = None,
    count: int = 1,
    protected: bool = False,
) -> MagicMock:
    cluster = MagicMock()
    cluster.label = label
    cluster.count = count
    cluster.protected = protected
    cluster.centroid = torch.zeros(200)
    return cluster


def _make_brain(
    *,
    modulators: dict[str, float] | None = None,
    current_cluster: int = -1,
    current_label: str | None = None,
    clusters: dict | None = None,
    debug: dict | None = None,
    cluster_list: list | None = None,
    wm_active: int = 5,
    wm_max_active: int = 20,
    weights: torch.Tensor | None = None,
    sensor_display: dict | None = None,
    has_interpreter: bool = False,
    interpreter_states: dict | None = None,
    interpreter_habits: dict | None = None,
    interpreter_trend: dict | None = None,
    interpreter_age_days: float = 0.0,
    interpreter_anomalies: list | None = None,
    has_narrator: bool = False,
    narrator_text: str = "",
    transition: dict | None = None,
    tick_count: int = 10000,
) -> MagicMock:
    """Build a MagicMock Brain with all attributes the server reads."""
    brain = MagicMock()

    # -- modulators --
    mods = modulators or {"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03}
    brain.modulators.snapshot.return_value = mods
    brain.modulators.level.side_effect = lambda name: mods.get(name, 0)

    # -- concept_tracker --
    if clusters is None:
        clusters = {}
    brain.concept_tracker._clusters = clusters
    brain.concept_tracker.get_transition.return_value = transition

    snap = {
        "current_cluster": current_cluster,
        "current_label": current_label,
        "num_clusters": len(clusters),
        "clusters": cluster_list if cluster_list is not None else [],
        "debug": debug or {},
    }
    brain.concept_tracker.snapshot.return_value = snap

    # -- regions["wm"] --
    wm_mock = MagicMock()
    wm_spikes = torch.zeros(100)
    if wm_active > 0:
        wm_spikes[:wm_active] = 1.0
    wm_mock.last_spikes = wm_spikes
    wm_mock.max_active = wm_max_active
    brain.regions = {"wm": wm_mock}

    # -- synapses["sensory_concept"] --
    syn_mock = MagicMock()
    if weights is None:
        weights = torch.randn(200, 500) * 0.2 + 0.5
    syn_mock.weights = weights
    brain.synapses = {"sensory_concept": syn_mock}

    # -- sensor display --
    brain._last_sensor_display = sensor_display or {}

    # -- tick count --
    brain.tick_count = tick_count

    # -- interpreter (optional) --
    if has_interpreter:
        interp = MagicMock()
        states = interpreter_states or {
            "flow": False, "flow_duration_min": 0, "flow_app": None,
            "stress": False, "meeting": False, "meeting_duration_min": 0,
            "needs_break": False, "active_minutes": 0,
        }
        interp.state_detector.detect.return_value = states
        trend = interpreter_trend or {
            "avg_DA": 0, "avg_NE": 0, "avg_ACh": 0, "avg_5HT": 0,
            "peak_DA": 0, "peak_NE": 0, "peak_ACh": 0, "peak_5HT": 0,
            "recent_DA": 0, "recent_NE": 0, "recent_ACh": 0, "recent_5HT": 0,
        }
        interp.state_detector.emotional_trend.return_value = trend
        habits = interpreter_habits or {"usual_habits": []}
        interp.habit_miner.current_hour_context.return_value = habits
        interp.personality.snapshot.return_value = {"age_days": interpreter_age_days}
        anomalies = interpreter_anomalies or []
        interp.anomaly.check.return_value = anomalies
        brain._interpreter = interp
    else:
        brain._interpreter = None

    # -- narrator (optional) --
    if has_narrator:
        narrator = MagicMock()
        narrator._narrate_modulators.return_value = narrator_text
        brain._snn_narrator = narrator
    else:
        brain._snn_narrator = None

    return brain


def _make_server(**kwargs) -> BrainServer:
    """Create a BrainServer with a mock brain."""
    brain = _make_brain(**kwargs)
    return BrainServer(brain)


def _make_client(model_type: str = "generic", **kwargs) -> SCPClient:
    """Create an SCPClient with a mock brain server."""
    server = _make_server(**kwargs)
    return SCPClient(server, model_type=model_type)


# ═══════════════════════════════════════════════════════════════════
# BrainServer -- Query Tests
# ═══════════════════════════════════════════════════════════════════


class TestBrainServerQueries:
    """BrainServer.query() returns valid SCP response schemas."""

    def test_query_emotion_returns_valid_schema(self):
        server = _make_server(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
        )
        resp = server.query("brain.emotion")
        assert resp["type"] == "response"
        assert "id" in resp
        result = resp["result"]
        assert "state" in result
        assert "confidence" in result
        assert "duration_seconds" in result
        assert "trend" in result
        assert "reason" in result

    def test_query_emotion_detects_content(self):
        server = _make_server(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
        )
        result = server.query("brain.emotion")["result"]
        assert result["state"] == "content"

    def test_query_emotion_with_narrator(self):
        server = _make_server(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
            has_narrator=True,
            narrator_text="ruhige Session",
        )
        result = server.query("brain.emotion")["result"]
        assert result["reason"] == "ruhige Session"

    def test_query_emotion_with_trend(self):
        server = _make_server(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
            has_interpreter=True,
            interpreter_trend={
                "avg_DA": 0.005, "avg_NE": 0.005, "avg_ACh": 0.01, "avg_5HT": 0.03,
                "peak_DA": 0.01, "peak_NE": 0.01, "peak_ACh": 0.01, "peak_5HT": 0.04,
                "recent_DA": 0, "recent_NE": 0, "recent_ACh": 0, "recent_5HT": 0,
            },
        )
        result = server.query("brain.emotion")["result"]
        assert result["trend"]["window_seconds"] == 180
        assert "avg" in result["trend"]
        assert "peak" in result["trend"]
        assert result["trend"]["direction"] == "stable"

    def test_query_pattern_returns_valid_schema(self):
        cluster = _make_cluster(label="Coding", count=200)
        server = _make_server(
            current_cluster=5,
            current_label="Coding",
            clusters={5: cluster},
            debug={"best_sim": 0.85},
            sensor_display={"keys": 15, "app": "Claude", "mic_rms": 0.001},
        )
        result = server.query("brain.pattern")["result"]
        assert result["cluster_id"] == 5
        assert result["label"] == "Coding"
        assert result["confidence"] == 0.85
        assert result["observation_count"] == 200
        assert "sensors" in result
        assert result["sensors"]["app"] == "Claude"
        assert result["sensors"]["keyboard"] == "viel"

    def test_query_pattern_no_cluster(self):
        server = _make_server(current_cluster=-1)
        result = server.query("brain.pattern")["result"]
        assert result["cluster_id"] == -1
        assert result["label"] is None

    def test_query_pattern_keyboard_levels(self):
        """Keyboard sensor mapping: 0-2=still, 3-10=aktiv, 11+=viel."""
        for keys, expected in [(0, "still"), (5, "aktiv"), (15, "viel")]:
            server = _make_server(
                sensor_display={"keys": keys, "app": "", "mic_rms": 0.001},
            )
            result = server.query("brain.pattern")["result"]
            assert result["sensors"]["keyboard"] == expected, f"keys={keys}"

    def test_query_pattern_mic_levels(self):
        """Mic sensor mapping: 0-0.005=still, 0.005-0.015=bg, 0.015+=laut."""
        for mic, expected in [(0.001, "still"), (0.008, "Hintergrundgeraeusche"), (0.02, "laut")]:
            server = _make_server(
                sensor_display={"keys": 0, "app": "", "mic_rms": mic},
            )
            result = server.query("brain.pattern")["result"]
            assert result["sensors"]["mic"] == expected, f"mic_rms={mic}"

    def test_query_memory_returns_valid_schema(self):
        server = _make_server(wm_active=8, wm_max_active=20)
        result = server.query("brain.memory")["result"]
        assert result["wm_active"] == 8
        assert result["wm_capacity"] == 20
        assert "recent_patterns" in result

    def test_query_memory_recent_patterns(self):
        server = _make_server(
            cluster_list=[
                {"id": 5, "label": "Coding", "count": 200, "active": True},
                {"id": 2, "label": "Browsing", "count": 80, "active": False},
            ],
        )
        result = server.query("brain.memory")["result"]
        assert len(result["recent_patterns"]) >= 1
        assert result["recent_patterns"][0]["label"] == "Browsing"

    def test_query_session_returns_valid_schema(self):
        server = _make_server(
            has_interpreter=True,
            interpreter_states={
                "flow": False, "flow_duration_min": 0, "flow_app": None,
                "stress": False, "meeting": False, "meeting_duration_min": 0,
                "needs_break": False, "active_minutes": 47.3,
            },
        )
        result = server.query("brain.session")["result"]
        assert result["active_minutes"] == 47.3
        assert result["needs_break"] is False
        assert result["in_flow"] is False
        assert result["in_meeting"] is False

    def test_query_session_with_habit(self):
        server = _make_server(
            has_interpreter=True,
            interpreter_habits={
                "usual_habits": [
                    {"day_name": "Sonntag", "hour": 23, "app": "idle"},
                ],
            },
        )
        result = server.query("brain.session")["result"]
        assert "Sonntag" in result["habit_context"]
        assert "idle" in result["habit_context"]

    def test_query_session_without_interpreter(self):
        server = _make_server(has_interpreter=False)
        result = server.query("brain.session")["result"]
        assert result["active_minutes"] == 0.0
        assert result["habit_context"] == ""

    def test_query_learning_returns_valid_schema(self):
        server = _make_server()
        result = server.query("brain.learning")["result"]
        assert "specialized_neurons" in result
        assert "total_neurons" in result
        assert "specialization_pct" in result
        assert "cluster_count" in result
        assert "age_ticks" in result
        assert "age_days" in result

    def test_query_learning_with_age(self):
        server = _make_server(
            has_interpreter=True,
            interpreter_age_days=3.2,
        )
        result = server.query("brain.learning")["result"]
        assert result["age_days"] == 3.2

    def test_query_full_contains_all_domains(self):
        server = _make_server(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
            sensor_display={"keys": 5, "app": "Claude", "mic_rms": 0.001},
        )
        result = server.query("brain.full")["result"]
        assert "emotion" in result
        assert "pattern" in result
        assert "memory" in result
        assert "session" in result
        assert "learning" in result

    def test_query_unknown_method_returns_error(self):
        server = _make_server()
        resp = server.query("brain.nonexistent")
        assert "error" in resp


# ═══════════════════════════════════════════════════════════════════
# BrainServer -- Action Tests
# ═══════════════════════════════════════════════════════════════════


class TestBrainServerActions:
    """BrainServer.action() dispatches correctly and affects modulators."""

    def test_reward_injects_da_and_sht(self):
        server = _make_server()
        result = server.action("brain.reward")
        assert result["ok"] is True
        server.brain.modulators.inject.assert_any_call("DA", 0.02)
        server.brain.modulators.inject.assert_any_call("5HT", 0.01)

    def test_correct_injects_ne_and_da(self):
        server = _make_server()
        result = server.action("brain.correct")
        assert result["ok"] is True
        server.brain.modulators.inject.assert_any_call("NE", 0.01)
        server.brain.modulators.inject.assert_any_call("DA", 0.005)

    def test_label_sets_cluster_label(self):
        cluster = _make_cluster(label=None, count=50)
        server = _make_server(clusters={3: cluster})
        result = server.action("brain.label", {"cluster_id": 3, "label": "Coding"})
        assert result["ok"] is True
        assert result["label"] == "Coding"
        server.brain.concept_tracker.set_label.assert_called_once_with(3, "Coding")

    def test_label_requires_params(self):
        server = _make_server()
        result = server.action("brain.label", {})
        assert result["ok"] is False
        assert "error" in result

    def test_engage_injects_ach_and_da(self):
        server = _make_server()
        result = server.action("brain.engage")
        assert result["ok"] is True
        server.brain.modulators.inject.assert_any_call("ACh", 0.005)
        server.brain.modulators.inject.assert_any_call("DA", 0.002)

    def test_disengage_reduces_sht(self):
        server = _make_server()
        result = server.action("brain.disengage")
        assert result["ok"] is True
        server.brain.modulators.inject.assert_any_call("5HT", -0.005)

    def test_unknown_action_returns_error(self):
        server = _make_server()
        result = server.action("brain.unknown")
        assert result["ok"] is False
        assert "error" in result


# ═══════════════════════════════════════════════════════════════════
# BrainServer -- Events
# ═══════════════════════════════════════════════════════════════════


class TestBrainServerEvents:
    """Events can be pushed and polled."""

    def test_push_and_poll_events(self):
        server = _make_server()
        server.push_event("brain.pattern_changed", {
            "from": {"cluster_id": 3, "label": "Coding"},
            "to": {"cluster_id": 5, "label": None},
        })
        events = server.poll_events()
        assert len(events) == 1
        assert events[0]["method"] == "brain.pattern_changed"
        assert events[0]["type"] == "event"

    def test_poll_clears_events(self):
        server = _make_server()
        server.push_event("brain.emotion_changed", {"state": "alert"})
        server.poll_events()
        events = server.poll_events()
        assert len(events) == 0

    def test_events_bounded(self):
        server = _make_server()
        for i in range(120):
            server.push_event("brain.tick", {"i": i})
        # Should be bounded (not grow unbounded)
        events = server.poll_events()
        assert len(events) <= 100

    def test_event_has_timestamp(self):
        server = _make_server()
        server.push_event("brain.stress_detected", {})
        events = server.poll_events()
        assert "timestamp" in events[0]


# ═══════════════════════════════════════════════════════════════════
# BrainServer -- Personality
# ═══════════════════════════════════════════════════════════════════


class TestBrainServerPersonality:
    """Personality mode reflects emotional state."""

    def test_personality_returns_valid_schema(self):
        server = _make_server(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
        )
        p = server.personality()
        assert p["type"] == "personality"
        params = p["params"]
        assert "mode" in params
        assert "style" in params
        assert "constraints" in params
        assert "priority_topic" in params

    def test_personality_content_mode(self):
        server = _make_server(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
        )
        mode = server.personality()["params"]["mode"]
        assert mode == "content"

    def test_personality_style_varies_with_mode(self):
        """Different emotional states produce different style configs."""
        # Content: warm, 2-4 Saetze
        server_content = _make_server(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
        )
        style_content = server_content.personality()["params"]["style"]
        assert "warm" in style_content["tone"]

    def test_personality_has_constraints(self):
        server = _make_server()
        constraints = server.personality()["params"]["constraints"]
        assert "keine Emojis" in constraints
        assert "keine Hilfe anbieten" in constraints


# ═══════════════════════════════════════════════════════════════════
# SCPClient -- build_prompt Tests
# ═══════════════════════════════════════════════════════════════════


class TestSCPClientBuildPrompt:
    """SCPClient.build_prompt produces valid output for each model type."""

    def test_build_prompt_returns_nonempty_string(self):
        client = _make_client(
            model_type="generic",
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
            sensor_display={"keys": 5, "app": "Claude", "mic_rms": 0.001},
        )
        prompt = client.build_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_build_prompt_all_model_types(self):
        """All four model types produce valid, non-empty prompts."""
        for model_type in ["qwen", "gemma", "claude", "generic"]:
            client = _make_client(
                model_type=model_type,
                modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
                sensor_display={"keys": 5, "app": "Claude", "mic_rms": 0.001},
            )
            prompt = client.build_prompt(user_message="Hallo!")
            assert len(prompt) > 50, f"{model_type} produced too-short prompt"

    def test_build_prompt_each_type_different(self):
        """Each model type produces a different prompt from the same state."""
        prompts = {}
        for model_type in ["qwen", "gemma", "claude", "generic"]:
            client = _make_client(
                model_type=model_type,
                modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
                sensor_display={"keys": 5, "app": "Claude", "mic_rms": 0.001},
            )
            prompts[model_type] = client.build_prompt()

        types = list(prompts.keys())
        for i, a in enumerate(types):
            for b in types[i + 1:]:
                assert prompts[a] != prompts[b], f"{a} and {b} identical"

    def test_build_prompt_includes_universal_rules(self):
        """All model types include the universal rules."""
        for model_type in ["qwen", "gemma", "claude", "generic"]:
            client = _make_client(model_type=model_type)
            prompt = client.build_prompt().lower()
            assert "keine emojis" in prompt or "kein einziges" in prompt

    def test_build_prompt_with_history(self):
        client = _make_client(model_type="qwen")
        history = [
            {"role": "user", "content": "Was siehst du?"},
            {"role": "assistant", "content": "Ich sehe Claude offen."},
        ]
        prompt = client.build_prompt(user_message="Und jetzt?", history=history)
        assert "Was siehst du?" in prompt
        assert "Und jetzt?" in prompt

    def test_build_prompt_without_history(self):
        client = _make_client(model_type="generic")
        prompt = client.build_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_build_prompt_with_empty_history(self):
        client = _make_client(model_type="generic")
        prompt = client.build_prompt(history=[])
        assert isinstance(prompt, str)

    def test_build_prompt_no_raw_numbers(self):
        """Prompts should not expose raw modulator values."""
        client = _make_client(
            model_type="generic",
            modulators={"DA": 0.035, "NE": 0.072, "ACh": 0.06, "5HT": 0.045},
        )
        prompt = client.build_prompt()
        assert "DA=" not in prompt
        assert "NE=" not in prompt
        assert "0.035" not in prompt


# ═══════════════════════════════════════════════════════════════════
# SCPClient -- send_feedback Tests
# ═══════════════════════════════════════════════════════════════════


class TestSCPClientFeedback:
    """SCPClient.send_feedback dispatches correctly."""

    def test_feedback_reward(self):
        client = _make_client()
        result = client.send_feedback("reward")
        assert result["ok"] is True

    def test_feedback_correct(self):
        client = _make_client()
        result = client.send_feedback("correct")
        assert result["ok"] is True

    def test_feedback_label(self):
        cluster = _make_cluster(label=None, count=50)
        client = _make_client(clusters={3: cluster})
        result = client.send_feedback("label", cluster_id=3, label="Coding")
        assert result["ok"] is True

    def test_feedback_engage(self):
        client = _make_client()
        result = client.send_feedback("engage")
        assert result["ok"] is True

    def test_feedback_disengage(self):
        client = _make_client()
        result = client.send_feedback("disengage")
        assert result["ok"] is True

    def test_feedback_unknown_type(self):
        client = _make_client()
        result = client.send_feedback("nonexistent")
        assert result["ok"] is False


# ═══════════════════════════════════════════════════════════════════
# SCPClient -- get_events Tests
# ═══════════════════════════════════════════════════════════════════


class TestSCPClientEvents:
    """SCPClient.get_events proxies to server."""

    def test_get_events_empty(self):
        client = _make_client()
        events = client.get_events()
        assert events == []

    def test_get_events_returns_pushed(self):
        client = _make_client()
        client.server.push_event("brain.pattern_changed", {"test": True})
        events = client.get_events()
        assert len(events) == 1
        assert events[0]["method"] == "brain.pattern_changed"

    def test_get_events_clears_after_poll(self):
        client = _make_client()
        client.server.push_event("brain.emotion_changed", {})
        client.get_events()
        assert client.get_events() == []


# ═══════════════════════════════════════════════════════════════════
# SCPClient -- detect_model_type
# ═══════════════════════════════════════════════════════════════════


class TestDetectModelType:
    """Static model detection maps names to adapter types."""

    def test_qwen_variants(self):
        assert SCPClient.detect_model_type("qwen2.5:14b-instruct") == "qwen"
        assert SCPClient.detect_model_type("qwen2.5:7b") == "qwen"

    def test_gemma_variants(self):
        assert SCPClient.detect_model_type("gemma2:27b") == "gemma"
        assert SCPClient.detect_model_type("gemma2:9b-instruct") == "gemma"

    def test_claude_variants(self):
        assert SCPClient.detect_model_type("claude-3-sonnet") == "claude"
        assert SCPClient.detect_model_type("haiku-3.5") == "claude"

    def test_llama_maps_to_qwen(self):
        assert SCPClient.detect_model_type("llama3.1:8b") == "qwen"

    def test_unknown_is_generic(self):
        assert SCPClient.detect_model_type("mistral:7b") == "generic"
        assert SCPClient.detect_model_type("phi-3:mini") == "generic"

    def test_case_insensitive(self):
        assert SCPClient.detect_model_type("QWEN2.5:14B") == "qwen"
        assert SCPClient.detect_model_type("Gemma2:27B") == "gemma"
        assert SCPClient.detect_model_type("Claude-3-Sonnet") == "claude"


# ═══════════════════════════════════════════════════════════════════
# Adapter-specific Tests
# ═══════════════════════════════════════════════════════════════════


def _sample_full_state() -> dict:
    return {
        "emotion": {
            "state": "content",
            "confidence": 0.33,
            "duration_seconds": 180,
            "trend": {},
            "reason": "ruhige Session",
        },
        "pattern": {
            "cluster_id": 5,
            "label": "Coding",
            "confidence": 0.85,
            "observation_count": 435,
            "sensors": {
                "app": "Claude",
                "keyboard": "aktiv",
                "mouse": "gelegentlich",
                "mic": "still",
                "idle_seconds": 0,
            },
            "suggested_label": None,
        },
        "memory": {
            "wm_active": 5,
            "wm_capacity": 20,
            "recent_patterns": [
                {"cluster_id": 2, "label": "Browsing", "ago_seconds": 120},
            ],
        },
        "session": {
            "active_minutes": 47.3,
            "needs_break": False,
            "in_flow": False,
            "in_meeting": False,
            "habit_context": "Sonntag 23h: normalerweise idle",
            "anomalies": [],
        },
        "learning": {
            "specialized_neurons": 52,
            "total_neurons": 200,
            "specialization_pct": 26.0,
            "cluster_count": 5,
            "age_ticks": 10000000,
            "age_days": 1.2,
        },
    }


def _sample_personality() -> dict:
    return {
        "type": "personality",
        "params": {
            "mode": "content",
            "personality_text": (
                "Ich bin ZUFRIEDEN -- die letzten Minuten waren ruhig und entspannt. "
                "Ich bin warm und geduldig."
            ),
            "style": {
                "tone": "warm, reflektiv",
                "length": "2-4 Saetze",
                "questions": False,
                "urgency": "low",
            },
            "constraints": [
                "keine Emojis",
                "keine Hilfe anbieten",
                "keine erfundenen Faehigkeiten",
                "nicht den Sinnen widersprechen",
            ],
            "priority_topic": None,
        },
    }


class TestQwenAdapter:
    """Qwen: structured headers, explicit instructions."""

    def test_structured_headers(self):
        adapter = QwenAdapter()
        prompt = adapter.render(_sample_full_state(), _sample_personality())
        assert "Dein emotionaler Zustand:" in prompt
        assert "Deine aktuelle Wahrnehmung:" in prompt

    def test_perception_data(self):
        adapter = QwenAdapter()
        prompt = adapter.render(_sample_full_state(), _sample_personality())
        assert "Stimmung:" in prompt
        assert "Muster:" in prompt
        assert "Session:" in prompt

    def test_explicit_instructions(self):
        adapter = QwenAdapter()
        prompt = adapter.render(_sample_full_state(), _sample_personality())
        assert "Beziehe dich auf deine Wahrnehmung" in prompt

    def test_with_context(self):
        adapter = QwenAdapter()
        prompt = adapter.render(_sample_full_state(), _sample_personality(), "Hallo!")
        assert "Konversation:" in prompt
        assert "Hallo!" in prompt

    def test_without_context(self):
        adapter = QwenAdapter()
        prompt = adapter.render(_sample_full_state(), _sample_personality(), "")
        assert "Konversation:" not in prompt


class TestGemmaAdapter:
    """Gemma: example-based, softer."""

    def test_softer_tone(self):
        adapter = GemmaAdapter()
        prompt = adapter.render(_sample_full_state(), _sample_personality())
        assert "So fuehlst du dich:" in prompt
        assert "Das nimmst du wahr:" in prompt

    def test_examples(self):
        adapter = GemmaAdapter()
        prompt = adapter.render(_sample_full_state(), _sample_personality())
        assert "Beispiel gute Antwort:" in prompt
        assert "Beispiel schlechte Antwort:" in prompt
        assert "DA=0.006" in prompt

    def test_with_context(self):
        adapter = GemmaAdapter()
        prompt = adapter.render(_sample_full_state(), _sample_personality(), "Was machst du?")
        assert "Leon hat gesagt:" in prompt
        assert "Was machst du?" in prompt

    def test_without_context(self):
        adapter = GemmaAdapter()
        prompt = adapter.render(_sample_full_state(), _sample_personality(), "")
        assert "Leon hat gesagt:" not in prompt


class TestClaudeAdapter:
    """Claude: XML tags, constraints."""

    def test_xml_tags(self):
        adapter = ClaudeAdapter()
        prompt = adapter.render(_sample_full_state(), _sample_personality())
        assert "<emotional_state>" in prompt
        assert "</emotional_state>" in prompt
        assert "<brain_state>" in prompt
        assert "</brain_state>" in prompt

    def test_constraints_block(self):
        adapter = ClaudeAdapter()
        prompt = adapter.render(_sample_full_state(), _sample_personality())
        assert "Constraints:" in prompt
        assert "Never mention raw numbers" in prompt

    def test_with_context(self):
        adapter = ClaudeAdapter()
        prompt = adapter.render(_sample_full_state(), _sample_personality(), "Guten Abend")
        assert "<conversation>" in prompt
        assert "</conversation>" in prompt
        assert "Guten Abend" in prompt

    def test_without_context(self):
        adapter = ClaudeAdapter()
        prompt = adapter.render(_sample_full_state(), _sample_personality(), "")
        assert "<conversation>" not in prompt

    def test_brain_data_in_brain_state(self):
        adapter = ClaudeAdapter()
        prompt = adapter.render(_sample_full_state(), _sample_personality())
        assert "52/200 spezialisiert" in prompt
        assert "5 Cluster" in prompt


class TestGenericAdapter:
    """Generic: minimal, works with any model."""

    def test_minimal_structure(self):
        adapter = GenericAdapter()
        prompt = adapter.render(_sample_full_state(), _sample_personality())
        assert "Stimmung:" in prompt
        assert "Muster:" in prompt

    def test_closing_instruction(self):
        adapter = GenericAdapter()
        prompt = adapter.render(_sample_full_state(), _sample_personality())
        assert "Natuerlich und lebendig" in prompt

    def test_with_context(self):
        adapter = GenericAdapter()
        prompt = adapter.render(_sample_full_state(), _sample_personality(), "Wie geht's?")
        assert "Wie geht's?" in prompt

    def test_personality_text_present(self):
        adapter = GenericAdapter()
        prompt = adapter.render(_sample_full_state(), _sample_personality())
        assert "ZUFRIEDEN" in prompt


# ═══════════════════════════════════════════════════════════════════
# Integration: Full round-trip
# ═══════════════════════════════════════════════════════════════════


class TestIntegration:
    """End-to-end: create server + client, build prompt, send feedback, poll events."""

    def test_full_round_trip(self):
        cluster = _make_cluster(label="Coding", count=200)
        brain = _make_brain(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
            current_cluster=5,
            current_label="Coding",
            clusters={5: cluster},
            debug={"best_sim": 0.85},
            sensor_display={"keys": 15, "app": "Claude", "mic_rms": 0.001},
            has_interpreter=True,
            interpreter_states={
                "flow": False, "flow_duration_min": 0, "flow_app": None,
                "stress": False, "meeting": False, "meeting_duration_min": 0,
                "needs_break": False, "active_minutes": 47.3,
            },
            interpreter_habits={
                "usual_habits": [
                    {"day_name": "Sonntag", "hour": 23, "app": "idle"},
                ],
            },
            interpreter_age_days=3.0,
            has_narrator=True,
            narrator_text="ruhige Session",
        )

        server = BrainServer(brain)
        client = SCPClient(server, model_type="claude")

        # 1. Build prompt
        prompt = client.build_prompt(
            user_message="Was siehst du?",
            history=[{"role": "assistant", "content": "Hallo Leon!"}],
        )
        assert "<brain_state>" in prompt
        assert "Was siehst du?" in prompt
        assert len(prompt) > 200

        # 2. Send feedback
        result = client.send_feedback("reward")
        assert result["ok"] is True

        # 3. Push + poll events
        server.push_event("brain.pattern_changed", {
            "from": {"cluster_id": 3},
            "to": {"cluster_id": 5},
        })
        events = client.get_events()
        assert len(events) == 1

    def test_unknown_model_type_falls_back_to_generic(self):
        server = _make_server(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
        )
        client = SCPClient(server, model_type="unknown_model")
        prompt = client.build_prompt()
        assert "Natuerlich und lebendig" in prompt
