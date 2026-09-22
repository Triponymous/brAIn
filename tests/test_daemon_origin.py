"""The daemon answers loopback scripts and the training console, never another website."""
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from server.main import build_app

CONSOLE = "http://127.0.0.1:8900"
EVIL = "https://evil.example"
WS = "ws://127.0.0.1:8000/ws"  # the test client would otherwise send host "testserver"


@pytest.fixture
def client():
    return TestClient(build_app(brain=None, adapter=None, pusher=None), base_url="http://127.0.0.1:8000")


def test_scripts_without_origin_and_the_console_pass(client):
    assert client.get("/healthz").status_code == 200                      # curl, MCP proxy
    r = client.get("/healthz", headers={"Origin": CONSOLE})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == CONSOLE
    assert client.get("/healthz", headers={"Origin": "http://localhost:8900"}).status_code == 200


def test_another_website_is_refused(client):
    assert client.get("/healthz", headers={"Origin": EVIL}).status_code == 403
    r = client.post("/api/wishes/grant", headers={"Origin": EVIL},
                    json={"wish_id": 1, "tool_name": "shell"})
    assert r.status_code == 403                                              # refused before any route
    preflight = client.options("/api/wishes/grant", headers={
        "Origin": EVIL, "Access-Control-Request-Method": "POST"})
    assert preflight.status_code == 403


def test_console_preflight_is_allowed(client):
    r = client.options("/api/feel", headers={
        "Origin": CONSOLE, "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type"})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == CONSOLE


def test_dns_rebinding_host_is_refused():
    """A page on a name that resolves to 127.0.0.1 is same-origin to itself and sends no Origin on GET."""
    app = build_app(brain=None, adapter=None, pusher=None)
    assert TestClient(app, base_url="http://evil.example:8000").get("/healthz").status_code == 400


def test_websocket_from_another_website_is_closed(client):
    with pytest.raises(WebSocketDisconnect) as refused:
        with client.websocket_connect(WS, headers={"Origin": EVIL}):
            pass
    assert refused.value.code == 1008                                        # closed before accept: never streamed
    with client.websocket_connect(WS, headers={"Origin": CONSOLE}):
        pass                                                                 # the console streams


def test_a_configured_origin_may_call_the_daemon():
    """daemon.allowed_origins in config.json, e.g. a new dashboard on its own port."""
    from server.main import CONSOLE_ORIGINS
    app = build_app(brain=None, adapter=None, pusher=None, origins=CONSOLE_ORIGINS + ("http://localhost:4321",))
    c = TestClient(app, base_url="http://127.0.0.1:8000")
    r = c.get("/healthz", headers={"Origin": "http://localhost:4321"})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "http://localhost:4321"
    refused = c.get("/healthz", headers={"Origin": EVIL})
    assert refused.status_code == 403 and "allowed_origins" in refused.json()["detail"]
    with c.websocket_connect(WS, headers={"Origin": "http://localhost:4321"}):
        pass
