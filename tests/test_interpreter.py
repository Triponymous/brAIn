"""Tests for BrainInterpreter — the orchestrator that ties all modules together."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from bridge.interpreter import BrainInterpreter
from bridge.episode_log import EpisodeLogger


class _FakeBrain:
    """Minimal stand-in for Brain that avoids MagicMock's auto-attribute pitfalls.

    MagicMock returns a mock for any attribute access, so hasattr(_personality_state)
    is always True -- which breaks BrainInterpreter's init logic. This class only
    exposes the attributes we explicitly set.
    """
    def __init__(self, sensor_display, modulators_snap, tick_count=100000):
        self._last_sensor_display = sensor_display
        self.tick_count = tick_count
        self.synapses = {"sensory_concept": MagicMock()}
        self.concept_tracker = MagicMock()
        self.concept_tracker._clusters = {}
        # Modulators mock
        self.modulators = MagicMock()
        self.modulators.snapshot.return_value = modulators_snap


def _mock_brain(app="Claude", keys=10, mouse=5, mic=0.001, idle=0,
                switch_rate=0, ne=0.01, sht=0.03, da=0.02, ach=0.03):
    """Create a fake Brain with sensor display and modulators."""
    return _FakeBrain(
        sensor_display={
            "app": app, "keys": keys, "mouse": mouse,
            "mic_rms": mic, "idle": idle, "switch_rate": switch_rate,
        },
        modulators_snap={
            "DA": da, "NE": ne, "ACh": ach, "5HT": sht,
        },
    )


def test_interpreter_creates_prompt():
    """BrainInterpreter should produce a non-empty prompt string."""
    with tempfile.TemporaryDirectory() as td:
        brain = _mock_brain()
        logger = EpisodeLogger(Path(td) / "test.db")
        interp = BrainInterpreter(brain, logger)
        prompt = interp.format_for_prompt()
        assert "BEWUSSTSEIN" in prompt
        assert len(prompt) > 10
        logger.close()


def test_interpreter_interpret_returns_all_keys():
    """interpret() should return all six interpretation sections."""
    with tempfile.TemporaryDirectory() as td:
        brain = _mock_brain()
        logger = EpisodeLogger(Path(td) / "test.db")
        interp = BrainInterpreter(brain, logger)
        result = interp.interpret()
        assert "states" in result
        assert "personality" in result
        assert "habits" in result
        assert "anomalies" in result
        assert "narrative" in result
        assert "explanations" in result
        logger.close()


def test_interpreter_tick_updates_state():
    """tick() should update the state detector."""
    with tempfile.TemporaryDirectory() as td:
        brain = _mock_brain()
        logger = EpisodeLogger(Path(td) / "test.db")
        interp = BrainInterpreter(brain, logger)
        assert interp.state_detector._snapshot_count == 0
        interp.tick()
        assert interp.state_detector._snapshot_count == 1
        logger.close()


def test_interpreter_personality_persistence():
    """save_personality / load_personality should round-trip."""
    with tempfile.TemporaryDirectory() as td:
        brain = _mock_brain()
        logger = EpisodeLogger(Path(td) / "test.db")
        interp = BrainInterpreter(brain, logger)
        # Feed some data into personality
        for _ in range(100):
            interp.personality.update({"DA": 0.08, "NE": 0.02, "ACh": 0.05, "5HT": 0.04})
        saved = interp.save_personality()
        # Create new interpreter and load the saved state
        interp2 = BrainInterpreter(brain, logger)
        interp2.load_personality(saved)
        snap1 = interp.personality.snapshot()
        snap2 = interp2.personality.snapshot()
        assert abs(snap1["baselines"]["DA"] - snap2["baselines"]["DA"]) < 0.001
        logger.close()


def test_interpreter_loads_personality_from_brain():
    """If brain has _personality_state, interpreter loads it on init."""
    with tempfile.TemporaryDirectory() as td:
        brain = _mock_brain()
        brain._personality_state = {
            "baselines": {"DA": 0.05, "NE": 0.01, "ACh": 0.03, "5HT": 0.02},
            "age_ticks": 500,
        }
        logger = EpisodeLogger(Path(td) / "test.db")
        interp = BrainInterpreter(brain, logger)
        snap = interp.personality.snapshot()
        assert snap["baselines"]["DA"] == 0.05
        assert snap["age_ticks"] == 500
        logger.close()


def test_interpreter_prompt_contains_state():
    """Prompt should contain a Zustand line."""
    with tempfile.TemporaryDirectory() as td:
        brain = _mock_brain()
        logger = EpisodeLogger(Path(td) / "test.db")
        interp = BrainInterpreter(brain, logger)
        prompt = interp.format_for_prompt()
        assert "Zustand:" in prompt
        logger.close()
