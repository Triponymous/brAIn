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


async def push_loop(brain: Any, pusher: WSPusher, exporter: Any = None, adapter: Any = None) -> None:
    """Periodically broadcast brain state to all WS clients."""
    period = 1.0 / pusher.rate_hz
    try:
        while True:
            # Sensor snapshot for dashboard display
            sensor_snap = adapter.bus.snapshot() if adapter else {}
            sensor_display = {}
            if "active_app" in sensor_snap:
                sensor_display["app"] = sensor_snap["active_app"].get("name", "?")
            if "keystroke_rate" in sensor_snap:
                sensor_display["keys"] = sensor_snap["keystroke_rate"].get("count", 0)
            if "mouse_rate" in sensor_snap:
                sensor_display["mouse"] = sensor_snap["mouse_rate"].get("count", 0)
            if "idle" in sensor_snap:
                sensor_display["idle"] = round(sensor_snap["idle"].get("seconds", 0), 1)
            if "mic" in sensor_snap:
                sensor_display["mic_rms"] = round(sensor_snap["mic"].get("rms", 0), 4)

            state = {
                "tick": brain.tick_count,
                "modulators": brain.modulators.snapshot(),
                "concept_membrane": brain.concept_spike_accum.tolist(),
                "wm_membrane": brain.regions["wm"].membrane.tolist(),
                "sensors": sensor_display,
                "spike_counts": {
                    "sensory": int(brain._last_sensory_spikes) if hasattr(brain, '_last_sensory_spikes') else 0,
                    "feature": int(brain._last_feature_spikes) if hasattr(brain, '_last_feature_spikes') else 0,
                    "concept": int(brain._last_concept_spikes) if hasattr(brain, '_last_concept_spikes') else 0,
                },
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
