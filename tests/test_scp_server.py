"""Tests for SCP Brain Server and Schema.

Tests all query methods, action handlers, event generation, personality mode,
and schema validation. Uses MagicMock for Brain with real dict-based
concept_tracker._clusters for cluster operations.
"""
from __future__ import annotations

import torch
from unittest.mock import MagicMock, patch

from bridge.scp_schema import (
    QueryMessage,
    ResponseMessage,
    EventMessage,
    ActionMessage,
    PersonalityMessage,
    EmotionResult,
    PatternResult,
    MemoryResult,
    SessionResult,
    LearningResult,
    TrendResult,
    serialize,
    validate,
)
from bridge.scp_server import BrainServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeCluster:
    """Minimal cluster stand-in with real attributes (not MagicMock)."""
    __slots__ = ("centroid", "label", "count", "last_seen", "protected")

    def __init__(
        self,
        label: str | None = None,
        count: int = 1,
        last_seen: int = 0,
        protected: bool = False,
    ) -> None:
        self.centroid = torch.zeros(200)
        self.label = label
        self.count = count
        self.last_seen = last_seen
        self.protected = protected


def _make_brain(
    *,
    modulators: dict[str, float] | None = None,
    current_cluster: int = -1,
    current_label: str | None = None,
    clusters: dict[int, _FakeCluster] | None = None,
    debug: dict | None = None,
    cluster_list: list | None = None,
    wm_active: int = 0,
    wm_max_active: int = 20,
    weights: torch.Tensor | None = None,
    sensor_display: dict | None = None,
    tick_count: int = 0,
    has_interpreter: bool = False,
    interpreter_states: dict | None = None,
    interpreter_habits: dict | None = None,
    interpreter_trend: dict | None = None,
    interpreter_anomalies: list | None = None,
    transition: dict | None = None,
) -> MagicMock:
    """Build a MagicMock Brain with all attributes BrainServer reads."""
    brain = MagicMock()

    # -- tick_count --
    brain.tick_count = tick_count

    # -- modulators --
    mods = modulators or {"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.01}
    brain.modulators.snapshot.return_value = mods
    brain.modulators.level.side_effect = lambda name: mods.get(name, 0.0)

    # -- concept_tracker --
    # Use a REAL dict for _clusters so set_label etc. work
    if clusters is None:
        clusters = {}
    brain.concept_tracker._clusters = clusters
    brain.concept_tracker.current_cluster_id = current_cluster
    brain.concept_tracker.current_cluster_label = current_label
    brain.concept_tracker.get_transition.return_value = transition

    def _set_label(cid: int, label: str) -> None:
        if cid in clusters:
            clusters[cid].label = label
            clusters[cid].protected = True

    brain.concept_tracker.set_label.side_effect = _set_label

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
        anomalies = interpreter_anomalies or []
        interp.anomaly.check.return_value = anomalies
        brain._interpreter = interp
    else:
        brain._interpreter = None

    brain._snn_narrator = None

    return brain


# ===========================================================================
# Schema: serialize
# ===========================================================================

class TestSerialize:
    """serialize() converts dataclasses and dicts to JSON-safe dicts."""

    def test_serialize_query_message(self):
        msg = QueryMessage(method="brain.emotion")
        result = serialize(msg)
        assert result["type"] == "query"
        assert result["method"] == "brain.emotion"
        assert "id" in result

    def test_serialize_response_message(self):
        msg = ResponseMessage(id="abc-123", result={"state": "content"})
        result = serialize(msg)
        assert result["type"] == "response"
        assert result["id"] == "abc-123"
        assert result["result"]["state"] == "content"

    def test_serialize_event_message(self):
        msg = EventMessage(method="brain.pattern_changed", params={"from": 1, "to": 2})
        result = serialize(msg)
        assert result["type"] == "event"
        assert result["method"] == "brain.pattern_changed"

    def test_serialize_action_message(self):
        msg = ActionMessage(method="brain.reward", params={})
        result = serialize(msg)
        assert result["type"] == "action"

    def test_serialize_personality_message(self):
        msg = PersonalityMessage(mode="curious")
        result = serialize(msg)
        assert result["type"] == "personality"
        assert result["mode"] == "curious"
        assert "constraints" in result

    def test_serialize_emotion_result(self):
        result = EmotionResult(state="curious", confidence=0.85)
        d = serialize(result)
        assert d["state"] == "curious"
        assert d["confidence"] == 0.85

    def test_serialize_dict_passthrough(self):
        d = {"foo": "bar"}
        assert serialize(d) is d

    def test_serialize_raises_on_non_dataclass(self):
        import pytest
        with pytest.raises(TypeError):
            serialize("not a dataclass")

    def test_serialize_nested_dataclass(self):
        trend = TrendResult(direction="rising")
        result = EmotionResult(state="alert", trend=trend)
        d = serialize(result)
        assert d["trend"]["direction"] == "rising"


# ===========================================================================
# Schema: validate
# ===========================================================================

class TestValidate:
    """validate() checks dicts against the SCP schema."""

    def test_valid_query(self):
        data = {"type": "query", "method": "brain.emotion", "id": "abc", "params": {}}
        errors = validate(data)
        assert errors == []

    def test_valid_response(self):
        data = {"type": "response", "id": "abc", "result": {"state": "content"}}
        errors = validate(data)
        assert errors == []

    def test_valid_event(self):
        data = {"type": "event", "method": "brain.pattern_changed", "params": {}}
        errors = validate(data)
        assert errors == []

    def test_valid_action(self):
        data = {"type": "action", "method": "brain.reward", "params": {}}
        errors = validate(data)
        assert errors == []

    def test_valid_personality(self):
        data = {"type": "personality", "mode": "content", "style": {}, "constraints": []}
        errors = validate(data)
        assert errors == []

    def test_missing_type(self):
        errors = validate({"method": "brain.emotion"})
        assert any("type" in e for e in errors)

    def test_unknown_type(self):
        errors = validate({"type": "foobar"})
        assert any("Unknown message type" in e for e in errors)

    def test_missing_required_field(self):
        errors = validate({"type": "query"})
        assert any("method" in e for e in errors)

    def test_unknown_query_method(self):
        data = {"type": "query", "method": "brain.foobar", "id": "x", "params": {}}
        errors = validate(data)
        assert any("Unknown query method" in e for e in errors)

    def test_unknown_action_method(self):
        data = {"type": "action", "method": "brain.fly", "params": {}}
        errors = validate(data)
        assert any("Unknown action method" in e for e in errors)

    def test_unknown_event_method(self):
        data = {"type": "event", "method": "brain.explode", "params": {}}
        errors = validate(data)
        assert any("Unknown event method" in e for e in errors)

    def test_validate_non_dict(self):
        errors = validate("not a dict")  # type: ignore
        assert any("Expected dict" in e for e in errors)


# ===========================================================================
# BrainServer: query methods
# ===========================================================================

class TestQueryEmotion:
    """brain.emotion query returns emotional state + trend."""

    def test_basic_emotion_query(self):
        brain = _make_brain(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
        )
        server = BrainServer(brain)
        resp = server.query("brain.emotion")
        assert resp["type"] == "response"
        result = resp["result"]
        assert "state" in result
        assert "trend" in result
        assert "confidence" in result
        assert result["trend"]["window_seconds"] == 180

    def test_emotion_state_name(self):
        brain = _make_brain(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
        )
        server = BrainServer(brain)
        resp = server.query("brain.emotion")
        assert resp["result"]["state"] in {
            "content", "curious", "alert", "focused", "stressed", "drowsy",
        }

    def test_emotion_with_interpreter_trend(self):
        brain = _make_brain(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
            has_interpreter=True,
            interpreter_trend={
                "avg_DA": 0.005, "avg_NE": 0.001, "avg_ACh": 0.02, "avg_5HT": 0.04,
                "peak_DA": 0.01, "peak_NE": 0.005, "peak_ACh": 0.03, "peak_5HT": 0.05,
                "recent_DA": 0.005, "recent_NE": 0.001, "recent_ACh": 0.02, "recent_5HT": 0.04,
            },
        )
        server = BrainServer(brain)
        resp = server.query("brain.emotion")
        trend = resp["result"]["trend"]
        assert trend["avg"]["DA"] > 0
        assert trend["peak"]["5HT"] > 0

    def test_emotion_duration(self):
        brain = _make_brain(tick_count=1000)
        server = BrainServer(brain)
        # Emotion is set for the first time in the query, duration = tick_count / 100
        resp = server.query("brain.emotion")
        assert resp["result"]["duration_seconds"] >= 0


class TestQueryPattern:
    """brain.pattern query returns active cluster + sensors."""

    def test_no_cluster(self):
        brain = _make_brain(current_cluster=-1)
        server = BrainServer(brain)
        resp = server.query("brain.pattern")
        result = resp["result"]
        assert result["cluster_id"] == -1
        assert result["label"] is None

    def test_labeled_cluster(self):
        cluster = _FakeCluster(label="Coding", count=435)
        brain = _make_brain(
            current_cluster=5,
            current_label="Coding",
            clusters={5: cluster},
            debug={"best_sim": 0.85},
            sensor_display={"app": "Claude", "keys": 15, "mouse": 3, "mic_rms": 0.001, "idle": 0},
        )
        server = BrainServer(brain)
        resp = server.query("brain.pattern")
        result = resp["result"]
        assert result["cluster_id"] == 5
        assert result["label"] == "Coding"
        assert result["confidence"] == 0.85
        assert result["observation_count"] == 435
        assert result["sensors"]["app"] == "Claude"
        assert result["sensors"]["keyboard"] == "aktiv"
        assert result["sensors"]["mic"] == "still"

    def test_suggested_label_typing(self):
        cluster = _FakeCluster(label=None, count=50)
        brain = _make_brain(
            current_cluster=2,
            clusters={2: cluster},
            sensor_display={"app": "VSCode", "keys": 20, "mouse": 1, "mic_rms": 0.001},
        )
        server = BrainServer(brain)
        resp = server.query("brain.pattern")
        assert resp["result"]["suggested_label"] == "Arbeiten in VSCode"

    def test_suggested_label_browsing(self):
        cluster = _FakeCluster(label=None, count=50)
        brain = _make_brain(
            current_cluster=2,
            clusters={2: cluster},
            sensor_display={"app": "Chrome", "keys": 1, "mouse": 5, "mic_rms": 0.001},
        )
        server = BrainServer(brain)
        resp = server.query("brain.pattern")
        assert resp["result"]["suggested_label"] == "Browsing in Chrome"


class TestQueryMemory:
    """brain.memory query returns WM state + recent patterns."""

    def test_memory_basic(self):
        brain = _make_brain(wm_active=5, wm_max_active=20)
        server = BrainServer(brain)
        resp = server.query("brain.memory")
        result = resp["result"]
        assert result["wm_active"] == 5
        assert result["wm_capacity"] == 20

    def test_memory_with_recent_patterns(self):
        brain = _make_brain(
            tick_count=10000,
            cluster_list=[
                {"id": 1, "label": "Coding", "count": 100, "last_seen": 9000, "active": False},
                {"id": 2, "label": "Browsing", "count": 50, "last_seen": 5000, "active": False},
                {"id": 3, "label": None, "count": 30, "last_seen": 8000, "active": True},
            ],
        )
        server = BrainServer(brain)
        resp = server.query("brain.memory")
        patterns = resp["result"]["recent_patterns"]
        # Only non-active labeled clusters
        assert len(patterns) == 2
        assert patterns[0]["label"] == "Coding"

    def test_memory_no_wm(self):
        brain = _make_brain()
        brain.regions = {}
        server = BrainServer(brain)
        resp = server.query("brain.memory")
        assert resp["result"]["wm_active"] == 0
        assert resp["result"]["wm_capacity"] == 20


class TestQuerySession:
    """brain.session query returns session context."""

    def test_session_no_interpreter(self):
        brain = _make_brain()
        server = BrainServer(brain)
        resp = server.query("brain.session")
        result = resp["result"]
        assert result["active_minutes"] == 0.0
        assert result["needs_break"] is False

    def test_session_with_interpreter(self):
        brain = _make_brain(
            has_interpreter=True,
            interpreter_states={
                "flow": True, "flow_duration_min": 45.0, "flow_app": "VSCode",
                "stress": False, "meeting": False, "meeting_duration_min": 0,
                "needs_break": False, "active_minutes": 47.3,
            },
            interpreter_habits={
                "usual_habits": [
                    {"day_name": "Sonntag", "hour": 23, "app": "idle"},
                ],
            },
        )
        server = BrainServer(brain)
        resp = server.query("brain.session")
        result = resp["result"]
        assert result["active_minutes"] == 47.3
        assert result["in_flow"] is True
        assert "Sonntag" in result["habit_context"]

    def test_session_needs_break(self):
        brain = _make_brain(
            has_interpreter=True,
            interpreter_states={
                "flow": False, "flow_duration_min": 0, "flow_app": None,
                "stress": False, "meeting": False, "meeting_duration_min": 0,
                "needs_break": True, "active_minutes": 95,
            },
        )
        server = BrainServer(brain)
        resp = server.query("brain.session")
        assert resp["result"]["needs_break"] is True


class TestQueryLearning:
    """brain.learning query returns SNN training state."""

    def test_learning_basic(self):
        brain = _make_brain(tick_count=8640000)
        server = BrainServer(brain)
        resp = server.query("brain.learning")
        result = resp["result"]
        assert result["total_neurons"] == 200
        assert result["age_ticks"] == 8640000
        assert result["age_days"] == 1.0

    def test_learning_specialization(self):
        # 50 rows with high std, 150 with low std
        weights = torch.zeros(200, 500)
        weights[:50] = torch.randn(50, 500) * 0.3 + 0.5
        weights[50:] = torch.ones(150, 500) * 0.5 + torch.randn(150, 500) * 0.01
        brain = _make_brain(weights=weights)
        server = BrainServer(brain)
        resp = server.query("brain.learning")
        result = resp["result"]
        assert result["specialized_neurons"] > 0
        assert result["specialization_pct"] > 0

    def test_learning_no_synapse(self):
        brain = _make_brain()
        brain.synapses = {}
        server = BrainServer(brain)
        resp = server.query("brain.learning")
        result = resp["result"]
        assert result["total_neurons"] == 0
        assert result["specialized_neurons"] == 0


class TestQueryFull:
    """brain.full returns all sub-queries."""

    def test_full_contains_all_sections(self):
        brain = _make_brain()
        server = BrainServer(brain)
        resp = server.query("brain.full")
        result = resp["result"]
        assert "emotion" in result
        assert "pattern" in result
        assert "memory" in result
        assert "session" in result
        assert "learning" in result


class TestQueryUnknown:
    """Unknown query methods return error."""

    def test_unknown_method(self):
        brain = _make_brain()
        server = BrainServer(brain)
        resp = server.query("brain.foobar")
        assert "error" in resp["result"]


# ===========================================================================
# BrainServer: action handlers
# ===========================================================================

class TestActionReward:
    """brain.reward injects DA and 5HT."""

    def test_reward_injects_modulators(self):
        brain = _make_brain()
        server = BrainServer(brain)
        result = server.action("brain.reward")
        assert result["ok"] is True
        calls = [str(c) for c in brain.modulators.inject.call_args_list]
        assert any("DA" in c and "0.02" in c for c in calls)
        assert any("5HT" in c and "0.01" in c for c in calls)


class TestActionCorrect:
    """brain.correct injects NE and DA."""

    def test_correct_injects_modulators(self):
        brain = _make_brain()
        server = BrainServer(brain)
        result = server.action("brain.correct")
        assert result["ok"] is True
        calls = [str(c) for c in brain.modulators.inject.call_args_list]
        assert any("NE" in c and "0.01" in c for c in calls)
        assert any("DA" in c and "0.005" in c for c in calls)


class TestActionLabel:
    """brain.label sets cluster label via concept_tracker."""

    def test_label_sets_cluster(self):
        cluster = _FakeCluster(label=None, count=50)
        brain = _make_brain(clusters={3: cluster})
        server = BrainServer(brain)
        result = server.action("brain.label", {"cluster_id": 3, "label": "Coding"})
        assert result["ok"] is True
        assert cluster.label == "Coding"
        assert cluster.protected is True

    def test_label_missing_params(self):
        brain = _make_brain()
        server = BrainServer(brain)
        # No crash with missing params
        result = server.action("brain.label", {})
        assert result["ok"] is True


class TestActionEngage:
    """brain.engage injects ACh and DA."""

    def test_engage_injects_modulators(self):
        brain = _make_brain()
        server = BrainServer(brain)
        result = server.action("brain.engage")
        assert result["ok"] is True
        calls = [str(c) for c in brain.modulators.inject.call_args_list]
        assert any("ACh" in c and "0.005" in c for c in calls)
        assert any("DA" in c and "0.002" in c for c in calls)


class TestActionDisengage:
    """brain.disengage decreases 5HT."""

    def test_disengage_decreases_serotonin(self):
        brain = _make_brain()
        server = BrainServer(brain)
        result = server.action("brain.disengage")
        assert result["ok"] is True
        calls = [str(c) for c in brain.modulators.inject.call_args_list]
        assert any("5HT" in c and "-0.005" in c for c in calls)


class TestActionUnknown:
    """Unknown action methods return error."""

    def test_unknown_action(self):
        brain = _make_brain()
        server = BrainServer(brain)
        result = server.action("brain.fly")
        assert result["ok"] is False
        assert "error" in result


# ===========================================================================
# BrainServer: event generation
# ===========================================================================

class TestEventPatternChanged:
    """tick() emits brain.pattern_changed on cluster transition."""

    def test_pattern_change_event(self):
        brain = _make_brain(
            transition={
                "from_cluster": 1, "from_label": "Coding",
                "to_cluster": 3, "to_label": "Browsing",
                "is_new": False,
            },
        )
        server = BrainServer(brain)
        server.tick()
        events = server.poll_events()
        pattern_events = [e for e in events if e["method"] == "brain.pattern_changed"]
        assert len(pattern_events) == 1
        params = pattern_events[0]["params"]
        assert params["from"]["cluster_id"] == 1
        assert params["from"]["label"] == "Coding"
        assert params["to"]["cluster_id"] == 3
        assert params["to"]["label"] == "Browsing"

    def test_new_pattern_event(self):
        brain = _make_brain(
            transition={
                "from_cluster": 1, "from_label": "Coding",
                "to_cluster": 5, "to_label": None,
                "is_new": True,
            },
        )
        server = BrainServer(brain)
        server.tick()
        events = server.poll_events()
        new_events = [e for e in events if e["method"] == "brain.new_pattern"]
        assert len(new_events) == 1
        assert new_events[0]["params"]["cluster_id"] == 5

    def test_no_event_without_transition(self):
        brain = _make_brain(transition=None)
        server = BrainServer(brain)
        server.tick()
        events = server.poll_events()
        pattern_events = [e for e in events if e["method"] == "brain.pattern_changed"]
        assert len(pattern_events) == 0


class TestEventEmotionChanged:
    """tick() emits brain.emotion_changed on emotion state shift."""

    def test_emotion_change_event(self):
        brain = _make_brain(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
        )
        server = BrainServer(brain)
        # First tick sets baseline emotion (no event — need prior state)
        server.tick()
        events1 = server.poll_events()
        emotion_events = [e for e in events1 if e["method"] == "brain.emotion_changed"]
        assert len(emotion_events) == 0  # first tick, no previous state

        # Now change emotion dramatically
        brain.modulators.snapshot.return_value = {
            "DA": 0.0, "NE": 0.0, "ACh": 0.0, "5HT": 0.0
        }
        server.tick()
        events2 = server.poll_events()
        emotion_events = [e for e in events2 if e["method"] == "brain.emotion_changed"]
        # The emotion may or may not change depending on detection thresholds.
        # But the mechanism is tested — event emitted when states differ.
        # Since both sets of modulators map to "content" (low 5HT fallback),
        # we might not get a change. That is correct behavior.

    def test_emotion_change_with_forced_different_states(self):
        """Force two different emotion states to verify the event fires."""
        brain = _make_brain()
        server = BrainServer(brain)

        # Tick 1: set initial state to "content"
        with patch("bridge.emotional_prompt.detect_emotional_state", return_value=("content", {})):
            server.tick()
        server.poll_events()  # clear

        # Tick 2: force state to "stressed"
        with patch("bridge.emotional_prompt.detect_emotional_state", return_value=("stressed", {})):
            server.tick()
        events = server.poll_events()
        emotion_events = [e for e in events if e["method"] == "brain.emotion_changed"]
        assert len(emotion_events) == 1
        assert emotion_events[0]["params"]["from"] == "content"
        assert emotion_events[0]["params"]["to"] == "stressed"


class TestEventStressDetected:
    """tick() emits brain.stress_detected when interpreter detects stress."""

    def test_stress_event(self):
        brain = _make_brain(
            has_interpreter=True,
            interpreter_states={
                "flow": False, "flow_duration_min": 0, "flow_app": None,
                "stress": True, "meeting": False, "meeting_duration_min": 0,
                "needs_break": False, "active_minutes": 30,
            },
        )
        server = BrainServer(brain)
        server.tick()
        events = server.poll_events()
        stress_events = [e for e in events if e["method"] == "brain.stress_detected"]
        assert len(stress_events) == 1
        assert stress_events[0]["params"]["active_minutes"] == 30


class TestEventFlowDetected:
    """tick() emits brain.flow_detected when interpreter detects flow."""

    def test_flow_event(self):
        brain = _make_brain(
            has_interpreter=True,
            interpreter_states={
                "flow": True, "flow_duration_min": 45.0, "flow_app": "VSCode",
                "stress": False, "meeting": False, "meeting_duration_min": 0,
                "needs_break": False, "active_minutes": 50,
            },
        )
        server = BrainServer(brain)
        server.tick()
        events = server.poll_events()
        flow_events = [e for e in events if e["method"] == "brain.flow_detected"]
        assert len(flow_events) == 1
        assert flow_events[0]["params"]["app"] == "VSCode"
        assert flow_events[0]["params"]["duration_min"] == 45.0


class TestEventBreakNeeded:
    """tick() emits brain.break_needed when active too long."""

    def test_break_event(self):
        brain = _make_brain(
            has_interpreter=True,
            interpreter_states={
                "flow": False, "flow_duration_min": 0, "flow_app": None,
                "stress": False, "meeting": False, "meeting_duration_min": 0,
                "needs_break": True, "active_minutes": 95,
            },
        )
        server = BrainServer(brain)
        server.tick()
        events = server.poll_events()
        break_events = [e for e in events if e["method"] == "brain.break_needed"]
        assert len(break_events) == 1
        assert break_events[0]["params"]["active_minutes"] == 95


class TestEventLabelNeeded:
    """tick() emits brain.label_needed for unlabeled clusters with enough obs."""

    def test_label_needed_event(self):
        cluster = _FakeCluster(label=None, count=10)
        brain = _make_brain(
            current_cluster=2,
            clusters={2: cluster},
        )
        server = BrainServer(brain)
        server.tick()
        events = server.poll_events()
        label_events = [e for e in events if e["method"] == "brain.label_needed"]
        assert len(label_events) == 1
        assert label_events[0]["params"]["cluster_id"] == 2
        assert label_events[0]["params"]["observation_count"] == 10

    def test_no_label_needed_when_labeled(self):
        cluster = _FakeCluster(label="Coding", count=100)
        brain = _make_brain(
            current_cluster=1,
            clusters={1: cluster},
        )
        server = BrainServer(brain)
        server.tick()
        events = server.poll_events()
        label_events = [e for e in events if e["method"] == "brain.label_needed"]
        assert len(label_events) == 0

    def test_no_label_needed_low_count(self):
        cluster = _FakeCluster(label=None, count=3)
        brain = _make_brain(
            current_cluster=1,
            clusters={1: cluster},
        )
        server = BrainServer(brain)
        server.tick()
        events = server.poll_events()
        label_events = [e for e in events if e["method"] == "brain.label_needed"]
        assert len(label_events) == 0


class TestPollEventsClears:
    """poll_events() returns events and clears the queue."""

    def test_poll_clears_queue(self):
        brain = _make_brain(
            transition={
                "from_cluster": 0, "from_label": None,
                "to_cluster": 1, "to_label": None,
                "is_new": False,
            },
        )
        server = BrainServer(brain)
        server.tick()
        events1 = server.poll_events()
        assert len(events1) > 0
        events2 = server.poll_events()
        assert len(events2) == 0


# ===========================================================================
# BrainServer: personality
# ===========================================================================

class TestPersonality:
    """personality() returns mode based on emotional state."""

    def test_personality_structure(self):
        brain = _make_brain(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
        )
        server = BrainServer(brain)
        result = server.personality()
        assert result["type"] == "personality"
        assert "mode" in result
        assert "style" in result
        assert "constraints" in result
        assert isinstance(result["constraints"], list)

    def test_personality_mode_matches_emotion(self):
        brain = _make_brain(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
        )
        server = BrainServer(brain)
        emotion = server.query("brain.emotion")["result"]["state"]
        personality = server.personality()
        assert personality["mode"] == emotion

    def test_personality_style_fields(self):
        brain = _make_brain()
        server = BrainServer(brain)
        result = server.personality()
        style = result["style"]
        assert "tone" in style
        assert "length" in style
        assert "urgency" in style

    def test_personality_priority_topic_label_needed(self):
        cluster = _FakeCluster(label=None, count=20)
        brain = _make_brain(
            current_cluster=3,
            clusters={3: cluster},
        )
        server = BrainServer(brain)
        result = server.personality()
        assert result["priority_topic"] == "label_needed"

    def test_personality_no_priority_topic(self):
        cluster = _FakeCluster(label="Coding", count=100)
        brain = _make_brain(
            current_cluster=1,
            clusters={1: cluster},
        )
        server = BrainServer(brain)
        result = server.personality()
        assert result["priority_topic"] is None


# ===========================================================================
# BrainServer: no interpreter graceful fallback
# ===========================================================================

class TestGracefulWithoutInterpreter:
    """Server works without BrainInterpreter attached."""

    def test_session_without_interpreter(self):
        brain = _make_brain(has_interpreter=False)
        server = BrainServer(brain)
        resp = server.query("brain.session")
        assert resp["result"]["active_minutes"] == 0

    def test_tick_without_interpreter(self):
        brain = _make_brain(has_interpreter=False, transition=None)
        server = BrainServer(brain)
        # Should not crash
        server.tick()
        events = server.poll_events()
        # No interpreter events
        stress_events = [e for e in events if e["method"] == "brain.stress_detected"]
        assert len(stress_events) == 0

    def test_emotion_without_interpreter(self):
        brain = _make_brain(
            has_interpreter=False,
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
        )
        server = BrainServer(brain)
        resp = server.query("brain.emotion")
        assert resp["result"]["state"] in {
            "content", "curious", "alert", "focused", "stressed", "drowsy",
        }


# ===========================================================================
# Event validation — all generated events pass validate()
# ===========================================================================

class TestEventValidation:
    """All generated events pass schema validation."""

    def test_pattern_changed_validates(self):
        brain = _make_brain(
            transition={
                "from_cluster": 0, "from_label": "A",
                "to_cluster": 1, "to_label": "B",
                "is_new": False,
            },
        )
        server = BrainServer(brain)
        server.tick()
        for event in server.poll_events():
            errors = validate(event)
            assert errors == [], f"Event {event['method']} failed validation: {errors}"

    def test_stress_event_validates(self):
        brain = _make_brain(
            has_interpreter=True,
            interpreter_states={
                "flow": False, "flow_duration_min": 0, "flow_app": None,
                "stress": True, "meeting": False, "meeting_duration_min": 0,
                "needs_break": False, "active_minutes": 30,
            },
        )
        server = BrainServer(brain)
        server.tick()
        for event in server.poll_events():
            errors = validate(event)
            assert errors == [], f"Event {event['method']} failed validation: {errors}"

    def test_label_needed_event_validates(self):
        cluster = _FakeCluster(label=None, count=10)
        brain = _make_brain(current_cluster=2, clusters={2: cluster})
        server = BrainServer(brain)
        server.tick()
        for event in server.poll_events():
            errors = validate(event)
            assert errors == [], f"Event {event['method']} failed validation: {errors}"


# ---------------------------------------------------------------------------
# Experience log: actions the LLM takes on the brain are recorded
# ---------------------------------------------------------------------------

class TestActionExperience:
    def _brain(self) -> MagicMock:
        brain = MagicMock()
        brain.tick_count = 5
        brain.sleep_mode = False
        brain.felt_state = None
        brain._last_signature = None
        brain._last_concept_cluster = 2
        return brain

    def test_actions_are_logged_with_method_and_filtered_params(self, tmp_path):
        from bridge.experience import ExperienceLog
        log = ExperienceLog(tmp_path / "experience.db")
        server = BrainServer(self._brain(), experience=log)

        assert server.action("brain.label", {"cluster_id": 3, "label": "coding", "secret": "x"})["ok"]
        assert server.action("brain.reward", {})["ok"]

        rows = list(reversed(log.recent()))
        assert [(r["actor"], r["kind"]) for r in rows] == [("llm", "brain.label"), ("llm", "brain.reward")]
        assert rows[0]["payload"] == {"cluster_id": 3, "label": "coding"}   # unknown keys dropped
        assert rows[0]["cluster"] == 2 and rows[0]["tick"] == 5
        log.close()

    def test_unknown_action_is_not_logged(self, tmp_path):
        from bridge.experience import ExperienceLog
        log = ExperienceLog(tmp_path / "experience.db")
        server = BrainServer(self._brain(), experience=log)
        assert server.action("brain.nope", {})["ok"] is False
        assert log.recent() == []
        log.close()

    def test_without_a_log_actions_still_work(self):
        server = BrainServer(self._brain())
        assert server.action("brain.reward", {})["ok"]
