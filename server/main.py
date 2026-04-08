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


async def brain_tick_loop(brain: Any, adapter: Any, hz: float = 100.0) -> None:
    """Run the brain tick loop indefinitely. Cancellable."""
    period = 1.0 / hz
    try:
        while True:
            vec = adapter.encode()
            brain.tick(vec)
            await asyncio.sleep(period)
    except asyncio.CancelledError:
        return


async def push_loop(brain: Any, pusher: WSPusher) -> None:
    """Periodically broadcast brain state to all WS clients."""
    period = 1.0 / pusher.rate_hz
    try:
        while True:
            state = {
                "tick": brain.tick_count,
                "modulators": brain.modulators.snapshot(),
                # Phase 3b will add: active_concepts, sensors, etc.
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
