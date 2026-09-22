"""End-to-end smoke test: spin up a real daemon (mock sensors), connect a
WebSocket client, verify state messages flow.

Uses uvicorn in-process via lifespan + httpx websockets. This is the
single most important integration test in Phase 3a — it proves all the
pieces actually fit together.
"""
import asyncio
import json
import pytest
import websockets
from contextlib import asynccontextmanager
from pathlib import Path
import tempfile

from brain.core import Brain
from adapters.mac_desktop.adapter import SOURCES, MacDesktopAdapter
from server.main import build_app, brain_tick_loop, push_loop
from server.ws import WSPusher


@asynccontextmanager
async def _running_server(port: int):
    """Start a daemon on the given port for the duration of the context."""
    import uvicorn

    brain = Brain(num_sensory=200)  # default size
    adapter = MacDesktopAdapter(mock_mode=True, enabled=dict.fromkeys(SOURCES, True))  # synthetic data only
    pusher = WSPusher(rate_hz=30.0)
    app = build_app(brain=brain, adapter=adapter, pusher=pusher)

    sensor_task = asyncio.create_task(adapter.run())
    tick_task = asyncio.create_task(brain_tick_loop(brain, adapter, hz=100.0))
    push_task = asyncio.create_task(push_loop(brain, pusher))

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    # Wait for the server to actually be up
    for _ in range(50):
        if server.started:
            break
        await asyncio.sleep(0.1)

    try:
        yield brain
    finally:
        server.should_exit = True
        for t in (sensor_task, tick_task, push_task):
            t.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        adapter.stop()


@pytest.mark.asyncio
async def test_e2e_daemon_starts_and_streams():
    """Start daemon, connect WebSocket, receive ≥1 state message."""
    port = 8765  # avoid conflict with default 8000
    async with _running_server(port) as brain:
        url = f"ws://127.0.0.1:{port}/ws"
        async with websockets.connect(url) as ws:
            # Receive a few messages
            messages = []
            for _ in range(3):
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                messages.append(json.loads(msg))
        # Verify messages have the expected shape
        assert len(messages) >= 1
        for m in messages:
            assert "tick" in m
            assert "modulators" in m
            assert "DA" in m["modulators"]
        # Brain should have advanced
        assert brain.tick_count > 0


@pytest.mark.asyncio
async def test_e2e_healthz_endpoint():
    """Daemon's /healthz returns ok status with tick count."""
    import httpx
    port = 8766
    async with _running_server(port) as brain:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://127.0.0.1:{port}/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["adapter_running"] is True
