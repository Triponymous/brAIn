"""BrainTools — the brain as tools the LLM calls while it reasons."""
import json
import time

import pytest
import torch

from brain.core import Brain
from brain.concept_tracker import _Cluster
from bridge.brain_tools import BrainTools
from bridge.episode_log import EpisodeLogger
from bridge.experience import ExperienceLog
from bridge.felt_state import FeltState, SIGNATURE_KEYS

FLOW = [0.028, 0.027, 0.058, 0.035, 0.060, 0.069]
NOW = time.time()


def _brain() -> Brain:
    brain = Brain(num_sensory=8, num_concept=4, num_wm=4)
    brain.felt_state = FeltState()
    brain.felt_state.label("flow", FLOW, cluster=1)
    brain._last_signature = list(FLOW)
    brain._last_concept_cluster = 1
    brain._last_sensor_display = {"app": "Code", "keys": 12, "mouse": 2, "idle": 0.5,
                                  "mic_rms": 0.003, "switch_rate": 0.5}
    brain.concept_tracker._clusters[1] = _Cluster(centroid=torch.zeros(8), label="coding", count=45, last_seen=0)
    brain.concept_tracker._clusters[2] = _Cluster(centroid=torch.zeros(8), label="browsing", count=7, last_seen=0)
    torch.manual_seed(1)
    for _ in range(5):
        brain.tick(torch.rand(8) * 2.0)
    return brain


def _episodes(tmp_path) -> EpisodeLogger:
    """The last four hours, one sample every 10 s:
    0-120 min coding in Code, 120-150 browsing in Chrome, 150-240 coding in Code."""
    lg = EpisodeLogger(tmp_path / "episodes.db")
    t0 = NOW - 4 * 3600
    rows = []
    for start, end, cid, label, app in [(0, 120, 1, "coding", "Code"), (120, 150, 2, "browsing", "Chrome"),
                                        (150, 240, 1, "coding", "Code")]:
        for sec in range(start * 60, end * 60, 10):
            rows.append((t0 + sec, sec * 10, json.dumps({"DA": 0.02, "NE": 0.01, "ACh": 0.03, "5HT": 0.04}),
                         json.dumps([cid]), json.dumps({"app": app, "keys": 10, "cluster_id": cid, "cluster_label": label}), 0))
    lg._conn.executemany(
        "INSERT INTO episodes(timestamp, tick, modulators, active_concepts, sensor_summary, sleep_mode) "
        "VALUES (?, ?, ?, ?, ?, ?)", rows)
    lg._conn.commit()
    return lg


@pytest.fixture
def tools(tmp_path):
    brain = _brain()
    episodes = _episodes(tmp_path)
    experience = ExperienceLog(tmp_path / "experience.db")
    yield BrainTools(brain, episodes=episodes, experience=experience)
    episodes.close()
    experience.close()


def test_state_reads_the_whole_organism(tools):
    s = tools.brain_state()
    assert s["felt"]["label"] == "flow" and s["felt"]["confidence"] > 0.9 and s["felt"]["known_labels"] == ["flow"]
    assert set(s["modulators"]) == {"DA", "NE", "ACh", "5HT"}
    assert set(s["trend_180s"]) == set(SIGNATURE_KEYS)
    assert set(s["prediction_error"]) == {"fast", "slow", "variance", "surprise", "progress"}
    assert s["senses"]["app"] == "Code" and s["senses"]["keys_per_s"] == 12
    assert s["sleeping"] is False and s["age"]["ticks"] == 5


def test_history_buckets_by_step(tools):
    h = tools.brain_history(hours=4, step_minutes=60)
    assert h["step_minutes"] == 60 and len(h["buckets"]) == 4
    assert h["buckets"][0]["pattern"] == "coding" and h["buckets"][0]["apps"] == ["Code"]
    assert "browsing" in h["buckets"][2]["patterns"]
    assert h["buckets"][3]["pattern"] == "coding"
    assert h["buckets"][0]["modulators"]["5HT"] == 0.04 and h["buckets"][0]["asleep"] == 0.0


def test_recall_finds_stretches(tools):
    r = tools.brain_recall(hours=4)
    assert [s["pattern"] for s in r["spans"]] == ["coding", "browsing", "coding"]
    assert r["longest"]["pattern"] == "coding" and 119 <= r["longest"]["minutes"] <= 121
    assert r["spans"][1]["apps"] == ["Chrome"]

    only_coding = tools.brain_recall(hours=4, label="cod")
    assert [s["pattern"] for s in only_coding["spans"]] == ["coding", "coding"]

    long_only = tools.brain_recall(hours=4, min_minutes=100)
    assert len(long_only["spans"]) == 1


def test_concept_profile_and_unknown(tools):
    c = tools.brain_concept(1)
    assert c["label"] == "coding" and c["times_seen"] == 45 and c["active_now"] is False
    assert c["minutes_last_7_days"] == 210.0        # 120 + 90 minutes of samples
    assert "senses" not in c                        # no interpreter attached here

    unknown = tools.brain_concept(99)
    assert "error" in unknown and {k["label"] for k in unknown["known"]} == {"coding", "browsing"}


def test_felt_without_and_with_label(tools):
    tools.experience.record("human", "teach_felt", {"label": "flow"}, state={}, now=NOW - 60)
    all_states = tools.brain_felt()
    assert all_states["now"]["label"] == "flow"
    assert all_states["known"] == [{"label": "flow", "times_taught": 1, "cluster": 1}]

    one = tools.brain_felt("FLOW")                  # case does not matter
    assert one["label"] == "flow" and one["recognized_now"] is True
    assert set(one["signature"]) == set(SIGNATURE_KEYS) and len(one["taught_at"]) == 1

    assert "error" in tools.brain_felt("stuck") and tools.brain_felt("stuck")["known"] == ["flow"]


def test_why_one_modulator_and_all(tools):
    ne = tools.brain_why("ne")
    assert ne["modulator"] == "NE" and isinstance(ne["level"], float) and "surprise" in ne["mechanism"]
    assert ne["prediction_error"] is not None
    everything = tools.brain_why()
    assert set(everything["modulators"]) == {"DA", "NE", "ACh", "5HT"}
    assert "error" in tools.brain_why("cortisol")


def test_experience_tool(tools):
    tools.experience.record("pet", "ask_label", {}, state={}, now=NOW - 30)
    tools.experience.record("human", "teach_felt", {"label": "flow"}, state={}, now=NOW - 20)
    e = tools.brain_experience(hours=1)
    assert e["summary"]["events"] == 2 and len(e["recent"]) == 2
    assert e["recent"][0]["kind"] == "teach_felt"   # most recent first


def test_tools_needing_the_interpreter_say_so(tools):
    assert "error" in tools.brain_habits()
    assert "error" in tools.brain_anomalies()


def test_with_a_real_interpreter(tmp_path):
    from bridge.interpreter import BrainInterpreter
    brain = _brain()
    episodes = _episodes(tmp_path)
    tools = BrainTools(brain, episodes=episodes, interpreter=BrainInterpreter(brain, episodes))
    habits = tools.brain_habits()
    assert {"now", "habits"} <= set(habits) and habits["habits"], "four hours of one app is a habit"
    assert habits["habits"][0]["app"] in {"Code", "Chrome"}
    assert isinstance(tools.brain_anomalies()["anomalies"], list)
    assert "senses" in tools.brain_concept(1)
    assert tools.brain_why("da")["reading"]
    episodes.close()


def test_execute_rejects_unknown_tools_and_bad_arguments(tools):
    assert "error" in tools.execute("brain_nope", {})
    assert "bad arguments" in tools.execute("brain_history", {"bogus": 1})["error"]
    assert tools.execute("brain_state", None)["age"]["ticks"] == 5


def test_definitions_are_callable_identifiers(tools):
    defs = tools.tool_definitions()
    names = [d["name"] for d in defs]
    assert set(names) == BrainTools.NAMES and len(names) == len(set(names))
    for d in defs:
        assert d["name"].isidentifier() and d["description"] and d["input_schema"]["type"] == "object"
