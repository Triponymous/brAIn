"""Tests for SCP Compact State renderer.

Each of the 6 line renderers is tested independently with controlled mock data,
then render() is tested for integration. Uses MagicMock for Brain so tests run
without the full SNN stack.
"""
from __future__ import annotations

import torch
from unittest.mock import MagicMock, patch

from bridge.scp import CompactState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cluster(
    label: str | None = None,
    count: int = 1,
    protected: bool = False,
) -> MagicMock:
    """Create a mock _Cluster object."""
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
    wm_active: int = 0,
    wm_max_active: int = 20,
    weights: torch.Tensor | None = None,
    sensor_display: dict | None = None,
    has_interpreter: bool = False,
    interpreter_states: dict | None = None,
    interpreter_habits: dict | None = None,
    interpreter_trend: dict | None = None,
    interpreter_age_days: float = 0.0,
    has_narrator: bool = False,
    narrator_text: str = "",
    transition: dict | None = None,
) -> MagicMock:
    """Build a MagicMock Brain with all attributes CompactState reads."""
    brain = MagicMock()

    # -- modulators --
    mods = modulators or {"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.01}
    brain.modulators.snapshot.return_value = mods

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
        }
        interp.state_detector.emotional_trend.return_value = trend
        habits = interpreter_habits or {"usual_habits": []}
        interp.habit_miner.current_hour_context.return_value = habits
        interp.personality.snapshot.return_value = {"age_days": interpreter_age_days}
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


# ===========================================================================
# _emotion_line
# ===========================================================================

class TestEmotionLine:
    """EMOTION line: state, duration estimate, reason."""

    def test_basic_emotion_without_interpreter(self):
        brain = _make_brain(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
        )
        cs = CompactState(brain)
        line = cs._emotion_line()
        assert line.startswith("EMOTION:")
        assert "content" in line
        assert "(kurz)" in line

    def test_emotion_with_narrator_reason(self):
        brain = _make_brain(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
            has_narrator=True,
            narrator_text="ruhige Session ohne Ueberraschungen",
        )
        cs = CompactState(brain)
        line = cs._emotion_line()
        assert "ruhige Session" in line
        assert "Grund:" in line

    def test_emotion_stable_duration(self):
        """Low NE peak + low NE avg -> 'seit Minuten stabil'."""
        brain = _make_brain(
            modulators={"DA": 0.01, "NE": 0.001, "ACh": 0.01, "5HT": 0.03},
            has_interpreter=True,
            interpreter_trend={
                "avg_DA": 0.005, "avg_NE": 0.005, "avg_ACh": 0.01, "avg_5HT": 0.03,
                "peak_DA": 0.01, "peak_NE": 0.01, "peak_ACh": 0.01, "peak_5HT": 0.04,
            },
        )
        cs = CompactState(brain)
        line = cs._emotion_line()
        assert "seit Minuten stabil" in line

    def test_emotion_just_changed(self):
        """High peak_NE relative to avg_NE -> 'gerade erst gewechselt'."""
        brain = _make_brain(
            modulators={"DA": 0.01, "NE": 0.1, "ACh": 0.01, "5HT": 0.01},
            has_interpreter=True,
            interpreter_trend={
                "avg_DA": 0.005, "avg_NE": 0.02, "avg_ACh": 0.01, "avg_5HT": 0.01,
                "peak_DA": 0.01, "peak_NE": 0.15, "peak_ACh": 0.01, "peak_5HT": 0.01,
            },
        )
        cs = CompactState(brain)
        line = cs._emotion_line()
        assert "gerade erst gewechselt" in line

    def test_emotion_no_narrator_empty_reason(self):
        brain = _make_brain(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
        )
        cs = CompactState(brain)
        line = cs._emotion_line()
        assert "Grund: " in line  # present but empty after colon


# ===========================================================================
# _pattern_line
# ===========================================================================

class TestPatternLine:
    """PATTERN line: cluster ID, label, confidence, sensors."""

    def test_no_cluster(self):
        brain = _make_brain(
            current_cluster=-1,
            sensor_display={"keys": 5, "app": "Claude", "mic_rms": 0.001},
        )
        cs = CompactState(brain)
        line = cs._pattern_line()
        assert "kein Muster erkannt" in line
        assert "Sensoren:" in line
        assert "Claude offen" in line
        assert "leise" in line

    def test_labeled_cluster(self):
        cluster = _make_cluster(label="Coding", count=435)
        brain = _make_brain(
            current_cluster=5,
            current_label="Coding",
            clusters={5: cluster},
            debug={"best_sim": 0.85},
            sensor_display={"keys": 15, "app": "Claude", "mic_rms": 0.001},
        )
        cs = CompactState(brain)
        line = cs._pattern_line()
        assert "#5" in line
        assert "'Coding'" in line
        assert "85%" in line
        assert "435x" in line
        assert "viel Tastatur" in line

    def test_unlabeled_cluster(self):
        cluster = _make_cluster(label=None, count=50)
        brain = _make_brain(
            current_cluster=2,
            current_label=None,
            clusters={2: cluster},
            debug={"best_sim": 0.60},
        )
        cs = CompactState(brain)
        line = cs._pattern_line()
        assert "unlabeled" in line
        assert "60%" in line
        assert "50x" in line

    def test_sensor_keyboard_active(self):
        brain = _make_brain(
            current_cluster=-1,
            sensor_display={"keys": 5, "mouse": 0, "app": "", "mic_rms": 0.001},
        )
        cs = CompactState(brain)
        line = cs._pattern_line()
        assert "Tastatur aktiv" in line

    def test_sensor_mouse_active(self):
        brain = _make_brain(
            current_cluster=-1,
            sensor_display={"keys": 0, "mouse": 15, "app": "", "mic_rms": 0.001},
        )
        cs = CompactState(brain)
        line = cs._pattern_line()
        assert "Maus aktiv" in line

    def test_sensor_loud_mic(self):
        brain = _make_brain(
            current_cluster=-1,
            sensor_display={"keys": 0, "mouse": 0, "app": "", "mic_rms": 0.02},
        )
        cs = CompactState(brain)
        line = cs._pattern_line()
        assert "laut" in line

    def test_sensor_background_noise(self):
        brain = _make_brain(
            current_cluster=-1,
            sensor_display={"keys": 0, "mouse": 0, "app": "", "mic_rms": 0.008},
        )
        cs = CompactState(brain)
        line = cs._pattern_line()
        assert "Hintergrundgeraeusche" in line

    def test_zero_confidence_shows_question_mark(self):
        cluster = _make_cluster(label=None, count=10)
        brain = _make_brain(
            current_cluster=1,
            current_label=None,
            clusters={1: cluster},
            debug={"best_sim": 0},
        )
        cs = CompactState(brain)
        line = cs._pattern_line()
        assert "(?, 10x)" in line


# ===========================================================================
# _change_line
# ===========================================================================

class TestChangeLine:
    """CHANGE line: transitions, stress, meeting, or no change."""

    def test_cluster_transition(self):
        brain = _make_brain(
            transition={
                "from_cluster": 1, "from_label": "Coding",
                "to_cluster": 3, "to_label": "Browsing",
            },
        )
        cs = CompactState(brain)
        line = cs._change_line()
        assert "Musterwechsel" in line
        assert "Coding" in line
        assert "Browsing" in line

    def test_cluster_transition_no_labels(self):
        brain = _make_brain(
            transition={
                "from_cluster": 1, "from_label": None,
                "to_cluster": 3, "to_label": None,
            },
        )
        cs = CompactState(brain)
        line = cs._change_line()
        assert "#1" in line
        assert "#3" in line

    def test_stress_detected(self):
        brain = _make_brain(
            has_interpreter=True,
            interpreter_states={
                "flow": False, "flow_duration_min": 0, "flow_app": None,
                "stress": True, "meeting": False, "meeting_duration_min": 0,
                "needs_break": False, "active_minutes": 30,
            },
        )
        cs = CompactState(brain)
        line = cs._change_line()
        assert "Stress" in line

    def test_meeting_detected(self):
        brain = _make_brain(
            has_interpreter=True,
            interpreter_states={
                "flow": False, "flow_duration_min": 0, "flow_app": None,
                "stress": False, "meeting": True, "meeting_duration_min": 15.5,
                "needs_break": False, "active_minutes": 30,
            },
        )
        cs = CompactState(brain)
        line = cs._change_line()
        assert "Meeting" in line
        assert "15.5" in line

    def test_no_change(self):
        brain = _make_brain()
        cs = CompactState(brain)
        line = cs._change_line()
        assert "keine Aenderung" in line

    def test_transition_takes_priority_over_stress(self):
        """Transition should be reported even when stress is also detected."""
        brain = _make_brain(
            transition={
                "from_cluster": 0, "from_label": "Idle",
                "to_cluster": 2, "to_label": "Working",
            },
            has_interpreter=True,
            interpreter_states={
                "flow": False, "flow_duration_min": 0, "flow_app": None,
                "stress": True, "meeting": False, "meeting_duration_min": 0,
                "needs_break": False, "active_minutes": 60,
            },
        )
        cs = CompactState(brain)
        line = cs._change_line()
        assert "Musterwechsel" in line
        assert "Stress" not in line


# ===========================================================================
# _memory_line
# ===========================================================================

class TestMemoryLine:
    """MEMORY line: WM slots and recent memory."""

    def test_basic_memory(self):
        brain = _make_brain(wm_active=5, wm_max_active=20)
        cs = CompactState(brain)
        line = cs._memory_line()
        assert "MEMORY:" in line
        assert "5/20 WM-Slots" in line
        assert "denke an:" in line

    def test_memory_with_recent_cluster(self):
        brain = _make_brain(
            wm_active=8,
            cluster_list=[
                {"id": 1, "label": "Coding", "count": 100, "active": True},
                {"id": 2, "label": "Browsing", "count": 50, "active": False},
                {"id": 3, "label": "Meeting", "count": 20, "active": False},
            ],
        )
        cs = CompactState(brain)
        line = cs._memory_line()
        assert "denke an: Browsing" in line

    def test_memory_no_recent_label(self):
        brain = _make_brain(
            wm_active=3,
            cluster_list=[
                {"id": 1, "label": None, "count": 100, "active": True},
            ],
        )
        cs = CompactState(brain)
        line = cs._memory_line()
        assert "nichts Bestimmtes" in line

    def test_memory_no_wm_region(self):
        brain = _make_brain(wm_active=0)
        brain.regions = {}
        cs = CompactState(brain)
        line = cs._memory_line()
        assert "0/20 WM-Slots" in line

    def test_memory_custom_max_active(self):
        brain = _make_brain(wm_active=10, wm_max_active=50)
        cs = CompactState(brain)
        line = cs._memory_line()
        assert "10/50 WM-Slots" in line


# ===========================================================================
# _session_line
# ===========================================================================

class TestSessionLine:
    """SESSION line: active time, habits, break/flow status."""

    def test_session_no_interpreter(self):
        brain = _make_brain()
        cs = CompactState(brain)
        line = cs._session_line()
        assert "SESSION:" in line
        assert "0min aktiv" in line
        assert "noch keine gelernt" in line

    def test_session_with_active_time(self):
        brain = _make_brain(
            has_interpreter=True,
            interpreter_states={
                "flow": False, "flow_duration_min": 0, "flow_app": None,
                "stress": False, "meeting": False, "meeting_duration_min": 0,
                "needs_break": False, "active_minutes": 47.3,
            },
        )
        cs = CompactState(brain)
        line = cs._session_line()
        assert "47min aktiv" in line

    def test_session_with_habit(self):
        brain = _make_brain(
            has_interpreter=True,
            interpreter_states={
                "flow": False, "flow_duration_min": 0, "flow_app": None,
                "stress": False, "meeting": False, "meeting_duration_min": 0,
                "needs_break": False, "active_minutes": 10,
            },
            interpreter_habits={
                "usual_habits": [
                    {"day_name": "Sonntag", "hour": 23, "app": "idle"},
                ],
            },
        )
        cs = CompactState(brain)
        line = cs._session_line()
        assert "Sonntag" in line
        assert "23h" in line
        assert "idle" in line

    def test_session_needs_break(self):
        brain = _make_brain(
            has_interpreter=True,
            interpreter_states={
                "flow": False, "flow_duration_min": 0, "flow_app": None,
                "stress": False, "meeting": False, "meeting_duration_min": 0,
                "needs_break": True, "active_minutes": 95,
            },
        )
        cs = CompactState(brain)
        line = cs._session_line()
        assert "Pause empfohlen!" in line

    def test_session_in_flow(self):
        brain = _make_brain(
            has_interpreter=True,
            interpreter_states={
                "flow": True, "flow_duration_min": 45.2, "flow_app": "VSCode",
                "stress": False, "meeting": False, "meeting_duration_min": 0,
                "needs_break": False, "active_minutes": 50,
            },
        )
        cs = CompactState(brain)
        line = cs._session_line()
        assert "im Flow seit 45.2min" in line


# ===========================================================================
# _brain_line
# ===========================================================================

class TestBrainLine:
    """BRAIN line: neuron specialization, clusters, age."""

    def test_brain_basic(self):
        brain = _make_brain(
            clusters={0: _make_cluster(count=10), 1: _make_cluster(count=5)},
        )
        # Update snapshot to reflect cluster count
        brain.concept_tracker.snapshot.return_value = {
            "current_cluster": 0,
            "current_label": None,
            "num_clusters": 2,
            "clusters": [],
            "debug": {},
        }
        cs = CompactState(brain)
        line = cs._brain_line()
        assert "BRAIN:" in line
        assert "Neuronen spezialisiert" in line
        assert "2 Cluster gelernt" in line
        assert "Alter:" in line

    def test_brain_with_age(self):
        brain = _make_brain(
            has_interpreter=True,
            interpreter_age_days=3.2,
        )
        brain.concept_tracker.snapshot.return_value = {
            "current_cluster": -1,
            "current_label": None,
            "num_clusters": 4,
            "clusters": [],
            "debug": {},
        }
        cs = CompactState(brain)
        line = cs._brain_line()
        assert "3.2 Tage" in line
        assert "4 Cluster" in line

    def test_brain_no_synapse(self):
        brain = _make_brain()
        brain.synapses = {}
        brain.concept_tracker.snapshot.return_value = {
            "current_cluster": -1,
            "current_label": None,
            "num_clusters": 0,
            "clusters": [],
            "debug": {},
        }
        cs = CompactState(brain)
        line = cs._brain_line()
        assert "0/0 Neuronen" in line

    def test_brain_specialization_count(self):
        """Neurons with row std > 0.1 are 'specialized'."""
        # Create weights: 50 rows with high std, 150 with low std
        weights = torch.zeros(200, 500)
        weights[:50] = torch.randn(50, 500) * 0.3 + 0.5
        weights[50:] = torch.ones(150, 500) * 0.5 + torch.randn(150, 500) * 0.01
        brain = _make_brain(weights=weights)
        brain.concept_tracker.snapshot.return_value = {
            "current_cluster": -1, "current_label": None,
            "num_clusters": 0, "clusters": [], "debug": {},
        }
        cs = CompactState(brain)
        line = cs._brain_line()
        # Should have approximately 50 specialized neurons (not exact due to randomness)
        row_stds = weights.std(dim=1)
        expected = int((row_stds > 0.1).sum().item())
        assert f"{expected}/200" in line


# ===========================================================================
# render (integration)
# ===========================================================================

class TestRender:
    """Integration: render() produces exactly 6 lines, each prefixed."""

    def test_render_produces_six_lines(self):
        brain = _make_brain(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
            sensor_display={"keys": 5, "app": "Claude", "mic_rms": 0.001},
        )
        cs = CompactState(brain)
        result = cs.render()
        lines = result.strip().split("\n")
        assert len(lines) == 6

    def test_render_line_prefixes(self):
        brain = _make_brain(
            sensor_display={"keys": 0, "app": "", "mic_rms": 0.001},
        )
        cs = CompactState(brain)
        result = cs.render()
        lines = result.strip().split("\n")
        assert lines[0].startswith("EMOTION:")
        assert lines[1].startswith("PATTERN:")
        assert lines[2].startswith("CHANGE:")
        assert lines[3].startswith("MEMORY:")
        assert lines[4].startswith("SESSION:")
        assert lines[5].startswith("BRAIN:")

    def test_render_no_raw_modulator_values(self):
        """Compact state must not leak raw modulator numbers."""
        brain = _make_brain(
            modulators={"DA": 0.035, "NE": 0.072, "ACh": 0.06, "5HT": 0.045},
            sensor_display={"keys": 5, "app": "VSCode", "mic_rms": 0.001},
        )
        cs = CompactState(brain)
        result = cs.render()
        assert "DA=" not in result
        assert "NE=" not in result
        assert "0.035" not in result
        assert "0.072" not in result

    def test_render_graceful_without_interpreter(self):
        """CompactState must work even without BrainInterpreter attached."""
        brain = _make_brain(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.03},
            sensor_display={"keys": 0, "app": "", "mic_rms": 0.001},
        )
        cs = CompactState(brain)
        result = cs.render()
        assert "EMOTION:" in result
        assert "SESSION:" in result
        assert "keine Aenderung" in result

    def test_render_full_context(self):
        """Full render with interpreter, narrator, labeled clusters."""
        cluster_active = _make_cluster(label="Coding", count=200)
        cluster_inactive = _make_cluster(label="Browsing", count=80)
        brain = _make_brain(
            modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.04},
            current_cluster=5,
            current_label="Coding",
            clusters={5: cluster_active, 2: cluster_inactive},
            debug={"best_sim": 0.90},
            cluster_list=[
                {"id": 5, "label": "Coding", "count": 200, "active": True},
                {"id": 2, "label": "Browsing", "count": 80, "active": False},
            ],
            wm_active=8,
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
            narrator_text="ruhige Session ohne Ueberraschungen",
        )
        cs = CompactState(brain)
        result = cs.render()
        lines = result.strip().split("\n")
        assert len(lines) == 6
        # Spot-check key content across lines
        assert "ruhige Session" in lines[0]           # EMOTION reason
        assert "'Coding'" in lines[1]                 # PATTERN label
        assert "Claude offen" in lines[1]             # PATTERN sensor
        assert "keine Aenderung" in lines[2]          # CHANGE
        assert "denke an: Browsing" in lines[3]       # MEMORY
        assert "47min aktiv" in lines[4]              # SESSION
        assert "Sonntag" in lines[4]                  # SESSION habit
        assert "3.0 Tage" in lines[5]                 # BRAIN age
