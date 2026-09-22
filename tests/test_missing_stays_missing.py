"""An unshared (or not yet observed) source is absent, and nothing downstream
turns that absence into a reading: no "quiet", no "away", no "0 keys"."""
import datetime
from unittest.mock import MagicMock

import pytest

from adapters.mac_desktop.adapter import MacDesktopAdapter
from brain.core import Brain
from bridge.anomaly import AnomalyDetector
from bridge.brain_tools import BrainTools
from bridge.habit_miner import HabitMiner
from bridge.narrative import NarrativeBuilder, _local_midnight_ts
from bridge.scp_client import SCPClient
from bridge.scp_server import BrainServer
from bridge.state_detector import StateDetector


def _brain(sensor_display):
    brain = Brain(num_sensory=8, num_concept=4, num_wm=4)
    brain._last_sensor_display = sensor_display
    return brain


def test_brain_state_tells_unshared_from_unobserved():
    brain = _brain({})
    adapter = MacDesktopAdapter(mock_mode=True, enabled={"idle": True})
    state = BrainTools(brain, adapter=adapter).brain_state()
    assert state["paused"] is False
    assert state["sources"]["idle"] == "waiting" and state["sources"]["mic"] == "disabled"
    assert state["senses"]["mic_rms"] is None and state["senses"]["keys_per_s"] is None


def test_pattern_query_reports_missing_senses_as_none():
    sensors = BrainServer(_brain({"app": "Editor"})).query("brain.pattern")["result"]["sensors"]
    assert sensors == {"app": "Editor", "keyboard": None, "mouse": None, "mic": None, "idle_seconds": None}


@pytest.mark.parametrize("model", ["qwen", "gemma", "claude", "generic"])
def test_every_prompt_says_no_data_instead_of_a_default(model):
    prompt = SCPClient(BrainServer(_brain({"app": "Editor"})), model_type=model).build_prompt("wie ist es?")
    assert "Editor" in prompt and "keine Daten" in prompt
    assert "None" not in prompt and "Tastatur: still" not in prompt and "Tastatur=still" not in prompt


def test_detector_claims_no_flow_or_break_without_keyboard_and_idle():
    detector = StateDetector()
    brain = MagicMock()
    brain._last_sensor_display = {"app": "Editor", "switch_rate": 0.1}
    brain.modulators.snapshot.return_value = {"DA": 0.02, "NE": 0.01, "ACh": 0.03, "5HT": 0.03}
    for _ in range(100 * 60):                                   # 100 minutes in the same app
        detector.update(brain)
    states = detector.detect()
    assert states["flow"] is False and states["needs_break"] is False
    assert states["active_minutes"] is None                     # continuous work is unknown without idle


def test_anomalies_need_the_data_they_compare():
    now = datetime.datetime.now()
    miner = MagicMock()
    miner.current_habits.return_value = [{"hour": now.hour, "day_of_week": now.weekday(), "app": "VSCode",
                                          "confidence": 0.9, "day_name": "heute"}]
    miner.hourly_profile.return_value = {now.hour: {"avg_activity": 25}}
    assert AnomalyDetector(miner).check({}) == []               # not "usually VSCode, today not", not "quiet"


def test_habit_activity_averages_only_observed_hours(tmp_path):
    from bridge.episode_log import EpisodeLogger
    log = EpisodeLogger(tmp_path / "episodes.db")
    for keys in (20, None, None):
        summary = {"app": "Editor"} if keys is None else {"app": "Editor", "keys": keys}
        log.log(tick=1, modulators={}, active_concepts=[], sensor_summary=summary, sleep_mode=False)
    profile = HabitMiner(log).hourly_profile()
    assert profile[datetime.datetime.now().hour]["avg_activity"] == 20.0   # not diluted to 6.7


def test_narrative_does_not_call_an_unobserved_hour_quiet(tmp_path):
    midnight = _local_midnight_ts(offset_days=0)
    hour = max(0, datetime.datetime.now().hour - 1)
    def episode(summary):
        return {"id": 1, "timestamp": midnight + hour * 3600, "tick": 1, "modulators": {},
                "active_concepts": [], "sensor_summary": summary, "sleep_mode": False}
    nb = NarrativeBuilder(MagicMock())
    assert nb._build_narrative([episode({"app": "Editor"})]) == f"{hour}h: in Editor"
    assert nb._build_narrative([episode({})]) == "Keine Daten."      # not "no activity"
    assert nb._build_narrative([episode({"app": "Editor", "keys": 1})]) == f"{hour}h: ruhig in Editor"


@pytest.mark.parametrize("model", ["qwen", "gemma", "claude", "generic"])
def test_prompts_claim_no_session_minutes_without_idle(tmp_path, model):
    from bridge.episode_log import EpisodeLogger
    from bridge.interpreter import BrainInterpreter
    from bridge.scp import CompactState
    brain = _brain({"app": "Editor", "keys": 5})
    brain._interpreter = BrainInterpreter(brain, EpisodeLogger(tmp_path / "episodes.db"))
    for _ in range(600):                                        # ten minutes of typing, idle not shared
        brain._interpreter.state_detector.update(brain)
    prompt = SCPClient(BrainServer(brain), model_type=model).build_prompt("wie ist es?")
    assert "Session: keine Daten" in prompt and "min aktiv" not in prompt
    assert "unbekannt" in CompactState(brain)._session_line()
