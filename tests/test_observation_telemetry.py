"""No OS collectors, microphones, checkpoints, cloud calls or personal data."""
import json
from types import SimpleNamespace

import httpx
import pytest
import torch

from adapters.mac_desktop.metadata import INPUTS, DesktopMetadata, app_category, encode_metadata
from brain.core import Brain
from server.telemetry import TelemetryBuffer, build_telemetry_app


def sensor_fixture():
    data = {key: {"status": "unavailable", "value": None, "unit": None,
                  "observed_at": None, "window_s": None, "source": "test"}
            for key in ("keystroke_rate", "mouse_rate", "idle", "active_app", "microphone", "wearable")}
    data["microphone"]["status"] = "disabled"
    data["wearable"]["status"] = "not_connected"
    data["active_app"].update(status="available", value="development", unit="category", observed_at=1700000000)
    return data


@pytest.fixture
def observed():
    torch.manual_seed(17)
    brain = Brain()
    buffer = TelemetryBuffer(brain, input_kind="test_fixture", seed=17, capacity=4)
    return brain, buffer


def step(brain, buffer, sensors=None):
    sensors = sensors or sensor_fixture()
    output = brain.tick(encode_metadata(sensors))
    buffer.record(brain, output, sensors)
    return output


def test_exact_outputs_and_real_post_step_membrane(observed):
    brain, buffer = observed
    output = step(brain, buffer)
    payload = buffer.read()
    frame = payload["frames"][0]
    assert frame["regions"]["c"]["membrane"] == brain.regions["concept"].membrane.tolist()
    assert frame["regions"]["e"]["output"] == output["expansion"].tolist()
    assert frame["regions"]["e"]["membrane"] is None
    for key, name in (("s", "sensory"), ("c", "concept"), ("w", "wm")):
        assert frame["regions"][key]["output"] == output[name].tolist()
    assert payload["source"]["input_kind"] == "test_fixture"
    assert len(payload["source"]["code_sha256"]) == 64
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_record_and_read_are_immutable_and_do_not_tick(observed):
    brain, buffer = observed
    sensors = sensor_fixture()
    step(brain, buffer, sensors)
    sensors["active_app"]["value"] = "changed"
    frame = buffer.read()["frames"][0]
    frame["regions"]["c"]["output"][0] = 22
    assert buffer.read()["frames"][0]["sensors"]["active_app"]["value"] == "development"
    assert buffer.read()["frames"][0]["regions"]["c"]["output"][0] in (0, 1)
    assert brain.tick_count == 1


def test_ring_cursor_restart_gap_and_stale(observed):
    brain, buffer = observed
    assert buffer.read()["status"] == "waiting"
    for _ in range(8):
        step(brain, buffer)
    assert [f["tick"] for f in buffer.read()["frames"]] == [5, 6, 7, 8]
    assert buffer.read(1, buffer.session_id)["gap"] is True
    assert [f["tick"] for f in buffer.read(6, buffer.session_id)["frames"]] == [7, 8]
    assert not buffer.read(8, buffer.session_id)["frames"]
    assert buffer.read(9999, "old-session")["reset"] is True
    assert buffer.read(now=buffer.start + 10000)["status"] == "stale"


def test_invalid_model_values_stop_instead_of_fabricating(observed):
    brain, buffer = observed
    output = brain.tick(torch.zeros(200))
    brain.regions["concept"].membrane[0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        buffer.record(brain, output, sensor_fixture())
    assert not buffer.frames


def test_missing_counts_are_not_idle_or_steady_typing():
    sensors = sensor_fixture()
    vector = encode_metadata(sensors)
    assert not vector[40:].any()
    assert vector[:40].any()
    sensors["keystroke_rate"].update(status="available", value=10)
    vector = encode_metadata(sensors)
    assert vector[60:76].any()
    assert not vector[76:80].any()
    assert not vector[112:148].any()
    assert not vector[160:164].any()


def test_permission_denial_does_not_read_input_counters():
    collector = DesktopMetadata.__new__(DesktopMetadata)
    collector.cg = SimpleNamespace(CGPreflightListenEventAccess=lambda: False)
    collector.workspace = SimpleNamespace(frontmostApplication=lambda: None)
    collector.keys, collector.mouse = [1], [2]
    collector.reset()
    result = collector.sample(dict.fromkeys(INPUTS, True))
    for key in ("keystroke_rate", "mouse_rate", "idle"):
        assert result[key]["status"] == "permission_required"
        assert result[key]["value"] is None
    assert result["active_app"]["status"] == "unavailable"
    assert result["microphone"]["status"] == "disabled"


def test_categories_never_return_app_names():
    assert app_category("Visual Studio Code") == "development"
    assert app_category("Sensitive custom client application") == "other"


@pytest.mark.asyncio
async def test_endpoint_read_only_local_host_and_origin(observed):
    brain, buffer = observed
    step(brain, buffer)
    app = build_telemetry_app(buffer)
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 8888))
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8001") as client:
        response = await client.get("/api/telemetry", headers={"Origin": "http://127.0.0.1:4178"})
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["Access-Control-Allow-Origin"] == "http://127.0.0.1:4178"
        assert (await client.post("/api/telemetry")).status_code == 405
        assert (await client.get("/api/telemetry?after=-1")).status_code == 422
        assert (await client.get("/api/telemetry", headers={"Origin": "https://attacker.example"})).status_code == 403
        assert (await client.get("/api/telemetry", headers={"Host": "attacker.example"})).status_code == 400
        assert (await client.get("/api/telemetry", headers={"Origin": "null"})).status_code == 403
    remote = httpx.ASGITransport(app=app, client=("192.168.1.10", 8888))
    async with httpx.AsyncClient(transport=remote, base_url="http://127.0.0.1:8001") as client:
        assert (await client.get("/api/telemetry")).status_code == 403
    assert brain.tick_count == 1
