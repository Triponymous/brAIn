"""Tests for the FastAPI server skeleton.

We test the WSPusher state-broadcast logic and the /healthz endpoint.
The full daemon main loop is tested via the E2E smoke test in Task 11.
"""
import asyncio
import json
import pytest
from httpx import AsyncClient, ASGITransport
from server.main import build_app
from server.ws import WSPusher


@pytest.mark.asyncio
async def test_healthz_endpoint():
    app = build_app(brain=None, adapter=None, pusher=None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
        response = await client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_wspusher_construction():
    pusher = WSPusher(rate_hz=30.0)
    assert pusher.rate_hz == 30.0
    assert len(pusher.clients) == 0


@pytest.mark.asyncio
async def test_wspusher_register_unregister():
    pusher = WSPusher()
    fake_ws = object()
    await pusher.register(fake_ws)
    assert fake_ws in pusher.clients
    await pusher.unregister(fake_ws)
    assert fake_ws not in pusher.clients


@pytest.mark.asyncio
async def test_wspusher_broadcast_to_registered_clients():
    """Broadcasting a state dict should call send_text on each client."""
    pusher = WSPusher()

    class FakeWS:
        def __init__(self):
            self.received = []
        async def send_text(self, text):
            self.received.append(text)

    a, b = FakeWS(), FakeWS()
    await pusher.register(a)
    await pusher.register(b)
    await pusher.broadcast({"tick": 42, "modulators": {"DA": 0.1}})
    assert len(a.received) == 1
    assert len(b.received) == 1
    msg_a = json.loads(a.received[0])
    assert msg_a["tick"] == 42


@pytest.mark.asyncio
async def test_wspusher_failed_send_removes_client():
    """A client that throws on send_text gets removed."""
    pusher = WSPusher()

    class BrokenWS:
        async def send_text(self, text):
            raise RuntimeError("connection closed")

    bad = BrokenWS()
    await pusher.register(bad)
    await pusher.broadcast({"x": 1})
    assert bad not in pusher.clients
