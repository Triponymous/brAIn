"""ExperienceLog — the (state, action, consequence, response) record."""
import pytest

from brain.core import Brain
from bridge.experience import ExperienceLog, state_of
from bridge.felt_state import FeltState, SIGNATURE_KEYS

FLOW = [0.028, 0.027, 0.058, 0.035, 0.060, 0.069]
CALM = [0.010, 0.005, 0.030, 0.045, 0.020, 0.010]


def _state(sig=FLOW, **over):
    s = {"tick": 100, "felt_label": None, "felt_conf": 0.0, "cluster": 3, "sleep": False, "signature": sig}
    s.update(over)
    return s


@pytest.fixture
def log(tmp_path):
    lg = ExperienceLog(tmp_path / "experience.db", settle_after=120.0)
    yield lg
    lg.close()


def test_record_and_read_back(log):
    eid = log.record("pet", "ask_label", {"message": "what was that?"}, state=_state(), now=1000.0)
    rows = log.recent(limit=5)
    assert len(rows) == 1
    r = rows[0]
    assert r["id"] == eid and r["ts"] == 1000.0 and r["actor"] == "pet" and r["kind"] == "ask_label"
    assert r["payload"] == {"message": "what was that?"}
    assert r["cluster"] == 3 and r["sleep"] is False and r["signature"] == FLOW
    assert r["after_signature"] is None and r["response_kind"] is None


def test_respond_links_the_human_event(log):
    ask = log.record("pet", "ask_label", {}, state=_state(), now=1000.0)
    teach = log.record("human", "teach_felt", {"label": "flow"}, state=_state(), now=1030.0)
    log.respond(ask, "answered", teach)
    r = next(x for x in log.recent() if x["id"] == ask)
    assert r["response_kind"] == "answered" and r["response_id"] == teach


def test_settle_writes_the_consequence_after_the_window(log):
    log.record("pet", "notify", {"category": "break"}, state=_state(FLOW), now=1000.0)
    assert log.settle(now=1060.0, signature=CALM) == 0          # too early
    assert log.settle(now=1130.0, signature=CALM) == 1
    r = log.recent()[0]
    assert r["after_signature"] == CALM and r["after_ts"] == 1130.0
    assert log.settle(now=1200.0, signature=FLOW) == 0          # settled exactly once


def test_settle_gives_stale_events_no_consequence(log):
    """The daemon was down: today's mood is not what followed an action hours ago."""
    log.record("pet", "notify", {}, state=_state(), now=1000.0)
    assert log.settle(now=1000.0 + 5000, signature=CALM) == 1
    r = log.recent()[0]
    assert r["after_signature"] is None and r["after_ts"] == 6000.0


def test_settle_without_a_signature_only_expires_stale_rows(log):
    log.record("pet", "notify", {}, state=_state(), now=1000.0)
    assert log.settle(now=1130.0, signature=None) == 0          # due, but nothing to write yet
    assert log.settle(now=1300.0, signature=None) == 1          # stale → expired


def test_summary_counts_asks_and_consequence(log):
    a1 = log.record("pet", "ask_label", {}, state=_state(FLOW), now=1000.0)
    a2 = log.record("pet", "ask_label", {}, state=_state(FLOW), now=2000.0)
    log.record("pet", "ask_label", {}, state=_state(FLOW), now=3000.0)
    t = log.record("human", "teach_felt", {"label": "flow"}, state=_state(), now=1020.0)
    d = log.record("human", "dismiss_ask", {}, state=_state(), now=2010.0)
    log.respond(a1, "answered", t)
    log.respond(a2, "dismissed", d)
    log.settle(now=1130.0, signature=CALM)                       # settles a1 only

    s = log.summary(since_hours=24, now=3600.0)
    assert s["events"] == 5
    assert s["counts"] == {"pet": {"ask_label": 3}, "human": {"teach_felt": 1, "dismiss_ask": 1}}
    assert s["asks"] == {"asked": 3, "answered": 1, "dismissed": 1, "unanswered": 1, "answer_rate": 0.333}
    c = s["consequence"]["ask_label"]
    assert c["n"] == 1
    assert c["delta"] == {k: round(after - before, 5) for k, after, before in zip(SIGNATURE_KEYS, CALM, FLOW)}


def test_summary_window_excludes_old_events(log):
    log.record("pet", "notify", {}, state=_state(), now=0.0)
    log.record("pet", "notify", {}, state=_state(), now=90_000.0)
    s = log.summary(since_hours=24, now=100_000.0)
    assert s["events"] == 1 and s["asks"]["answer_rate"] is None


def test_state_of_reads_the_organism():
    brain = Brain(num_sensory=8, num_concept=4, num_wm=4)
    brain.felt_state = FeltState()
    brain.felt_state.label("flow", FLOW, cluster=2)
    brain._last_signature = list(FLOW)
    brain._last_concept_cluster = 2
    s = state_of(brain)
    assert s["felt_label"] == "flow" and s["felt_conf"] > 0.9
    assert s["cluster"] == 2 and s["sleep"] is False and s["signature"] == FLOW and s["tick"] == 0


def test_state_of_before_any_signature():
    brain = Brain(num_sensory=8, num_concept=4, num_wm=4)
    s = state_of(brain)
    assert s["felt_label"] is None and s["signature"] is None and s["cluster"] is None
