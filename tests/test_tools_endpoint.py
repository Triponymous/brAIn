"""/api/tools — the brain's read-only tools over HTTP (what the MCP server proxies)."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from brain.core import Brain
from bridge.brain_tools import BrainTools
from bridge.experience import ExperienceLog
from bridge.felt_state import FeltState
from server.tools import build_tools_router

FLOW = [0.028, 0.027, 0.058, 0.035, 0.060, 0.069]


def _client(tmp_path):
    brain = Brain(num_sensory=8, num_concept=4, num_wm=4)
    brain.felt_state = FeltState()
    brain.felt_state.label("flow", FLOW)
    brain._last_signature = list(FLOW)
    brain._experience = ExperienceLog(tmp_path / "experience.db")
    app = FastAPI()
    app.include_router(build_tools_router(brain, BrainTools(brain, experience=brain._experience)))
    return TestClient(app), brain


def test_lists_exactly_the_read_only_brain_tools(tmp_path):
    c, _ = _client(tmp_path)
    tools = c.get("/api/tools").json()["tools"]
    assert {t["name"] for t in tools} == BrainTools.NAMES
    assert all(t["input_schema"]["type"] == "object" for t in tools)


def test_calls_a_tool_and_records_the_experience(tmp_path):
    c, brain = _client(tmp_path)
    body = c.post("/api/tools/brain_state").json()
    assert body["tool"] == "brain_state" and body["result"]["felt"]["label"] == "flow"

    body = c.post("/api/tools/brain_recall", json={"hours": 2}, headers={"X-Brain-Via": "mcp"}).json()
    assert body["result"]["error"] == "no episode log"   # honest about what is not attached here

    rows = list(reversed(brain._experience.recent()))
    assert [(r["actor"], r["kind"], r["payload"]["tool"], r["payload"]["via"]) for r in rows] == [
        ("llm", "tool_call", "brain_state", "api"),
        ("llm", "tool_call", "brain_recall", "mcp"),
    ]
    assert rows[1]["payload"]["args"] == {"hours": 2}


def test_only_brain_tools_are_reachable(tmp_path):
    c, _ = _client(tmp_path)
    assert c.post("/api/tools/label_concept", json={"concept_id": 1, "label": "x"}).status_code == 404
    assert c.post("/api/tools/shell", json={"command": "rm -rf /"}).status_code == 404
    assert c.post("/api/tools/current_state").status_code == 404


def test_bad_arguments_are_an_answer_not_a_crash(tmp_path):
    c, _ = _client(tmp_path)
    assert "bad arguments" in c.post("/api/tools/brain_history", json={"bogus": 1}).json()["result"]["error"]
    assert c.post("/api/tools/brain_state", json=[1, 2]).status_code == 422   # arguments must be an object
