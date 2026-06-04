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
from fastapi.middleware.cors import CORSMiddleware

from server.ws import WSPusher


def build_app(
    brain: Optional[Any],
    adapter: Optional[Any],
    pusher: Optional[WSPusher],
) -> FastAPI:
    app = FastAPI(title="braind", version="3.0.0")

    # Allow browser WebSocket connections from Vite dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
            while True:
                raw = await ws.receive_text()
                # Forward subscription messages to pusher
                if pusher is not None:
                    pusher.handle_client_message(ws, raw)
        except WebSocketDisconnect:
            pass
        finally:
            if pusher is not None:
                await pusher.unregister(ws)

    return app


async def brain_tick_loop(brain: Any, adapter: Any, hz: float = 100.0, exporter: Any = None, episode_logger: Any = None) -> None:
    """Run the brain tick loop in a thread to avoid uvicorn event-loop starvation."""
    import threading
    import time

    stop_event = threading.Event()

    def _tick_thread():
        period = 1.0 / hz
        sleep_idle_threshold = 300.0  # 5 minutes idle → enter sleep
        while not stop_event.is_set():
            vec = adapter.encode()

            # Auto sleep/wake based on user idle time
            idle_snap = adapter.bus.snapshot().get("idle", {})
            idle_secs = idle_snap.get("seconds", 0) if idle_snap else 0
            if idle_secs > sleep_idle_threshold and not brain.sleep_mode:
                brain.enter_sleep()
            elif idle_secs < sleep_idle_threshold and brain.sleep_mode:
                brain.exit_sleep()

            out = brain.tick(vec)
            # Record concept spikes WITH sensor context for auto-correlation
            if exporter is not None and "concept" in out:
                sensor_snap = adapter.bus.snapshot() if adapter else {}
                exporter.record_spikes_with_context(out["concept"], sensor_snap)

            # Episode logging every 1000 ticks (~10s at 100Hz)
            if episode_logger is not None and brain.tick_count % 1000 == 0:
                try:
                    snap = exporter.snapshot() if exporter else {}
                    sensor_snap_ep = adapter.bus.snapshot() if adapter else {}
                    sensor_display = {}
                    if "active_app" in sensor_snap_ep:
                        sensor_display["app"] = sensor_snap_ep["active_app"].get("name", "?")
                    if "keystroke_rate" in sensor_snap_ep:
                        sensor_display["keys"] = sensor_snap_ep["keystroke_rate"].get("count", 0)
                    # Include ConceptTracker cluster info
                    tracker = brain.concept_tracker.snapshot()
                    sensor_display["cluster_id"] = tracker.get("current_cluster", -1)
                    sensor_display["cluster_label"] = tracker.get("current_label")
                    episode_logger.log(
                        tick=brain.tick_count,
                        modulators=brain.modulators.snapshot(),
                        active_concepts=[tracker.get("current_cluster", -1)],
                        sensor_summary=sensor_display,
                        sleep_mode=brain.sleep_mode,
                    )
                except Exception as e:
                    print(f"[episode_log] failed: {e}")

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


async def push_loop(brain: Any, pusher: WSPusher, exporter: Any = None, adapter: Any = None, detector: Any = None) -> None:
    """Periodically broadcast brain state to all WS clients."""
    period = 1.0 / pusher.rate_hz

    # Smoothed sensor values for dashboard display.
    # Raw sensors sample at 5Hz but we push at 30Hz. Without smoothing,
    # the dashboard flickers between 0 and real values.
    _smooth_keys = 0.0
    _smooth_mouse = 0.0
    _smooth_mic = 0.0
    _decay = 0.85  # exponential smoothing: keeps ~1s of history
    _felt_tick = 0

    try:
        while True:
            # Sensor snapshot for dashboard display
            sensor_snap = adapter.bus.snapshot() if adapter else {}
            sensor_display = {}
            if "active_app" in sensor_snap:
                app_data = sensor_snap["active_app"]
                sensor_display["app"] = app_data.get("name", "?")
                sensor_display["background_apps"] = app_data.get("background_apps", [])
                sensor_display["app_count"] = app_data.get("app_count", 1)
                sensor_display["app_switched"] = app_data.get("switched", False)
                sensor_display["switch_rate"] = app_data.get("switch_rate", 0.0)
            if "keystroke_rate" in sensor_snap:
                raw_keys = sensor_snap["keystroke_rate"].get("count", 0)
                # Weighted average smoothing (keeps ~1s of history)
                _smooth_keys = _smooth_keys * _decay + raw_keys * (1 - _decay)
                sensor_display["keys"] = round(max(raw_keys, _smooth_keys))
            if "mouse_rate" in sensor_snap:
                raw_mouse = sensor_snap["mouse_rate"].get("count", 0)
                _smooth_mouse = _smooth_mouse * _decay + raw_mouse * (1 - _decay)
                sensor_display["mouse"] = round(max(raw_mouse, _smooth_mouse))
            if "idle" in sensor_snap:
                sensor_display["idle"] = round(sensor_snap["idle"].get("seconds", 0), 1)
            if "mic" in sensor_snap:
                raw_mic = sensor_snap["mic"].get("rms", 0)
                # Don't smooth with max() — it amplifies noise into fake signal
                _smooth_mic = _smooth_mic * _decay + raw_mic * (1 - _decay)
                sensor_display["mic_rms"] = round(_smooth_mic, 4)
                sensor_display["mic_rms_raw"] = round(raw_mic, 6)  # debug: show unsmoothed

            # ConceptTracker: stable cluster info for dashboard + LLM
            tracker_snap = brain.concept_tracker.snapshot()

            # Store latest sensor display on brain for the chat endpoint to read
            # This is the SMOOTHED data that matches what the dashboard shows
            brain._last_sensor_display = dict(sensor_display)

            base_state = {
                "tick": brain.tick_count,
                "sleep_mode": brain.sleep_mode,
                "modulators": brain.modulators.snapshot(),
                "concept_membrane": brain.concept_spike_accum.tolist(),
                "wm_membrane": brain.regions["wm"].membrane.tolist(),
                "sensors": sensor_display,
                "concepts": tracker_snap,  # stable cluster IDs!
                "spike_counts": {
                    "sensory": int(brain._last_sensory_spikes) if hasattr(brain, '_last_sensory_spikes') else 0,
                    "concept": int(brain._last_concept_spikes) if hasattr(brain, '_last_concept_spikes') else 0,
                    "wm": int(brain._last_wm_spikes) if hasattr(brain, '_last_wm_spikes') else 0,
                },
            }

            # Felt-state: update the trend detector ~1Hz, recognize the learned state,
            # ride it on the push so the training console shows it live.
            if detector is not None and getattr(brain, "felt_state", None) is not None:
                from bridge.felt_state import signature_from_trend
                _felt_tick += 1
                if _felt_tick == 1 or _felt_tick % max(1, int(pusher.rate_hz)) == 0:
                    detector.update(brain)
                sig = signature_from_trend(detector.emotional_trend())
                brain._last_signature = sig
                cluster = brain.concept_tracker.current_cluster if getattr(brain, "concept_tracker", None) is not None else -1
                brain._last_concept_cluster = cluster
                fname, fconf = brain.felt_state.recognize(sig, cluster)
                base_state["felt"] = {"label": fname, "confidence": fconf,
                                      "signature": sig,
                                      "known_labels": brain.felt_state.known_labels()}

            # Gather detail data only if clients need it (meso/micro subscriptions)
            detail_state = None
            needed_regions = pusher.get_needed_regions()
            if needed_regions:
                region_spikes = {}
                for region_name in needed_regions:
                    region = brain.regions.get(region_name)
                    if region and hasattr(region, 'membrane'):
                        # Detect recent spikes: membrane near 0 after reset = just spiked
                        # For WTA layers, use the spike accumulator if available
                        if hasattr(brain, f'_last_{region_name}_spike_vec'):
                            region_spikes[region_name] = getattr(brain, f'_last_{region_name}_spike_vec').tolist()
                        else:
                            # Fallback: threshold detection on membrane
                            spikes_vec = (region.membrane < 0.1).float()
                            region_spikes[region_name] = spikes_vec.tolist()
                detail_state = {"region_spikes": region_spikes}

            await pusher.broadcast(base_state, detail_state)
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
