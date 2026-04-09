"""FastAPI app builder + main daemon loop entry point.

build_app(brain, adapter, pusher) returns a FastAPI instance with:
    GET /healthz                  liveness check
    WS  /ws                       brain state stream

The actual brain tick loop and the periodic WSPusher broadcast loop are
started by run_daemon() which is called from the braind CLI.
"""
from __future__ import annotations
import asyncio
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from server.ws import WSPusher


def build_app(
    brain: Optional[Any],
    adapter: Optional[Any],
    pusher: Optional[WSPusher],
) -> FastAPI:
    app = FastAPI(title="braind", version="3.0.0")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "status": "ok",
            "brain_tick_count": brain.tick_count if brain is not None else None,
            "adapter_running": adapter is not None,
            "ws_clients": len(pusher.clients) if pusher is not None else 0,
        }

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        if pusher is not None:
            await pusher.register(ws)
        try:
            # Keep the connection open by awaiting client messages
            # (we don't actually need them; this just blocks)
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            if pusher is not None:
                await pusher.unregister(ws)

    return app


async def brain_tick_loop(brain: Any, adapter: Any, hz: float = 100.0, exporter: Any = None) -> None:
    """Run the brain tick loop in a thread to avoid uvicorn event-loop starvation."""
    import threading
    import time

    stop_event = threading.Event()

    def _tick_thread():
        period = 1.0 / hz
        while not stop_event.is_set():
            vec = adapter.encode()
            out = brain.tick(vec)
            # Record concept spikes for the exporter's accumulator
            if exporter is not None and "concept" in out:
                exporter.record_spikes(out["concept"])
            time.sleep(period)

    thread = threading.Thread(target=_tick_thread, daemon=True)
    thread.start()
    try:
        # Keep the coroutine alive so it can be cancelled
        while True:
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        stop_event.set()
        thread.join(timeout=2.0)
        return


async def push_loop(brain: Any, pusher: WSPusher, exporter: Any = None) -> None:
    """Periodically broadcast brain state to all WS clients."""
    period = 1.0 / pusher.rate_hz
    try:
        while True:
            # Use spike accumulator for concept activity (membrane is always ~0 after WTA reset)
            concept_activity = exporter._spike_counts.tolist() if exporter else brain.regions["concept"].membrane.tolist()
            state = {
                "tick": brain.tick_count,
                "modulators": brain.modulators.snapshot(),
                "concept_membrane": concept_activity,
                "wm_membrane": brain.regions["wm"].membrane.tolist(),
            }
            await pusher.broadcast(state)
            await asyncio.sleep(period)
    except asyncio.CancelledError:
        return


async def persistence_loop(brain: Any, save_path: str, period_sec: float = 60.0) -> None:
    """Periodically save the brain to SQLite."""
    from brain.persistence import save_brain
    from pathlib import Path
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            await asyncio.sleep(period_sec)
            try:
                save_brain(brain, p)
            except Exception as e:
                print(f"[persistence] save failed: {e}")
    except asyncio.CancelledError:
        return
