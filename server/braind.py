"""braind — Brain background daemon CLI.

Usage:
    braind start [--mock-sensors] [--port 8000] [--checkpoint PATH]
    braind status
    braind --help

`start` runs the daemon in the foreground. The daemon:
- Loads the brain from the checkpoint file (or creates fresh if missing)
- Starts the MacDesktopAdapter with all 6 sensors
- Runs brain.tick() at ~100 Hz on incoming sensor data
- Pushes brain state to /ws clients at ~30 Hz
- Auto-saves the brain to checkpoint every 60s
- Listens on the configured port
"""
from __future__ import annotations
import argparse
import asyncio
from pathlib import Path
import sys

import uvicorn

from brain.core import Brain
from brain.persistence import load_brain, save_brain
from adapters.mac_desktop.adapter import MacDesktopAdapter
from server.main import build_app, brain_tick_loop, push_loop, persistence_loop
from server.ws import WSPusher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="braind")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="Start the daemon")
    start.add_argument("--mock-sensors", action="store_true",
                       help="Run with mock sensors (CI/dev mode)")
    start.add_argument("--port", type=int, default=8000)
    start.add_argument("--checkpoint", type=str, default="checkpoints/braind.sqlite")
    start.add_argument("--tick-hz", type=float, default=100.0)
    start.add_argument("--push-hz", type=float, default=30.0)

    sub.add_parser("status", help="Print daemon status")

    return parser


def _check_permissions() -> None:
    """Check macOS privacy permissions and print clear warnings."""
    import sys
    if sys.platform != "darwin":
        return

    print("\n╔══════════════════════════════════════════╗")
    print("║       brAIn Permission Diagnostics       ║")
    print("╚══════════════════════════════════════════╝")

    # 1. Input Monitoring (keyboard, mouse, idle)
    try:
        import Quartz
        import time

        before = Quartz.CGEventSourceCounterForEventType(
            Quartz.kCGEventSourceStateHIDSystemState, 10)  # keydown
        time.sleep(0.1)
        after = Quartz.CGEventSourceCounterForEventType(
            Quartz.kCGEventSourceStateHIDSystemState, 10)

        idle = Quartz.CGEventSourceSecondsSinceLastEventType(
            Quartz.kCGEventSourceStateHIDSystemState, int(0xFFFFFFFF))

        # If idle > 300s AND counters didn't change, permission is missing
        # (user was presumably just typing to start this daemon)
        if idle > 120:
            print("[WARN] INPUT MONITORING: NOT GRANTED")
            print("   → Keyboard, mouse, and idle sensors will NOT work!")
            print("   → Fix: System Settings → Privacy & Security → Input Monitoring")
            print("   → Add Terminal.app (or your terminal) and RESTART this daemon")
            print()
        else:
            print("[OK] Input Monitoring: OK")
    except ImportError:
        print("[WARN] Quartz framework not available")

    # 2. Microphone
    try:
        import sounddevice as sd
        rec = sd.rec(int(0.1 * 16000), samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        rms = float((rec ** 2).mean() ** 0.5)
        if rms < 0.0001:
            print("[WARN] MICROPHONE: May not be granted (RMS=0)")
            print("   → Fix: System Settings → Privacy & Security → Microphone")
        else:
            print(f"[OK] Microphone: OK (RMS={rms:.6f})")
    except Exception as e:
        print(f"[WARN] Microphone: Error ({e})")

    # 3. Active app (no permission needed)
    try:
        from AppKit import NSWorkspace
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        print(f"[OK] Active App: OK (currently: {app.localizedName()})")
    except Exception:
        print("[WARN] Active App: NSWorkspace unavailable")

    print()


async def _run_daemon(args: argparse.Namespace) -> None:
    # Load or create brain
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        print(f"Loading brain from {checkpoint}")
        brain = load_brain(checkpoint)
        print(f"Resumed at tick {brain.tick_count}")
    else:
        print("Creating fresh brain")
        brain = Brain()

    # ═══ PERMISSION CHECK ═══
    # Without Input Monitoring, keyboard/mouse/idle sensors are blind.
    if not args.mock_sensors:
        _check_permissions()

    # Create adapter — store on brain for chat endpoint access
    adapter = MacDesktopAdapter(mock_mode=args.mock_sensors)
    brain._adapter = adapter  # chat endpoint reads live sensor bus from this
    pusher = WSPusher(rate_hz=args.push_hz)

    # Bridge setup
    from bridge.exporter import BrainStateExporter
    from bridge.llm_router import HybridLLMRouter
    from server.chat import build_chat_router

    # Capability system
    from capabilities.grants import GrantStore
    from capabilities.registry import ToolRegistry
    from server.grants import build_grants_router

    from bridge.episode_log import EpisodeLogger
    episode_logger = EpisodeLogger(checkpoint.parent / "episodes.db")

    # BrainInterpreter — consciousness layer (must be created before proactive/chat)
    from bridge.interpreter import BrainInterpreter
    interpreter = BrainInterpreter(brain, episode_logger)
    brain._interpreter = interpreter  # accessible from chat endpoint + proactive

    # SNNNarrator — translates brain internals into natural language for the LLM
    from bridge.snn_narrator import SNNNarrator
    narrator = SNNNarrator(brain)
    brain._snn_narrator = narrator

    # SCP v2: formal protocol for SNN-LLM communication
    from bridge.scp_server import BrainServer
    from bridge.scp_client import SCPClient

    scp_server = BrainServer(brain)
    brain._scp_server = scp_server

    # Detect model type from config
    from server.config import get
    local_model = get("llm", "local_model", "qwen2.5:14b-instruct")
    model_type = SCPClient.detect_model_type(local_model)
    scp_client = SCPClient(scp_server, model_type=model_type)
    brain._scp_client = scp_client

    exporter = BrainStateExporter(brain)
    grant_store = GrantStore(checkpoint.parent / "grants.sqlite")
    tool_registry = ToolRegistry(brain, exporter, grant_store)
    llm_router = HybridLLMRouter()
    chat_router = build_chat_router(brain, exporter, tool_registry, llm_router)
    grants_router = build_grants_router(grant_store, refresh_fn=tool_registry.refresh_grants)

    # Felt-state self-model (learned emotional states) + the trend detector that feeds it
    from bridge.felt_state import FeltState
    from bridge.state_detector import StateDetector
    from server.feel import build_feel_router
    if getattr(brain, "felt_state", None) is None:
        brain.felt_state = FeltState()
    state_detector = StateDetector()
    feel_router = build_feel_router(brain, state_detector)

    # Config API
    from server.config import load_config, build_config_router
    load_config()
    config_router = build_config_router()

    # Build FastAPI app
    app = build_app(brain=brain, adapter=adapter, pusher=pusher)
    app.include_router(chat_router)
    app.include_router(grants_router)
    app.include_router(config_router)
    app.include_router(feel_router)

    # Voice setup
    from bridge.tts import TTSEngine
    from bridge.stt import STTEngine
    from server.voice import build_voice_router

    tts_engine = TTSEngine(voice_model=Path("models/de_DE-thorsten-medium.onnx"))
    stt_engine = STTEngine(model_name="small")

    async def _chat_fn(message: str) -> dict:
        """Adapter: routes a voice message through the same /api/chat pipeline."""
        import httpx
        # Reuse the full chat endpoint so sensor data is included
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"http://127.0.0.1:{args.port}/api/chat",
                json={"message": message},
            )
            return resp.json()

    voice_router = build_voice_router(
        tts_engine=tts_engine,
        stt_engine=stt_engine,
        chat_fn=_chat_fn,
    )
    app.include_router(voice_router)

    # Schedule background tasks — start sensors first, wait for bus to fill
    sensor_task = asyncio.create_task(adapter.run())
    await asyncio.sleep(2.0)  # give sensors time to populate the bus
    print(f"Sensor bus keys: {list(adapter.bus.snapshot().keys())}")
    tick_task = asyncio.create_task(brain_tick_loop(brain, adapter, hz=args.tick_hz, exporter=exporter, episode_logger=episode_logger))
    push_task = asyncio.create_task(push_loop(brain, pusher, exporter=exporter, adapter=adapter, detector=state_detector))
    persist_task = asyncio.create_task(persistence_loop(brain, str(checkpoint)))

    # Proactive notifications — pet speaks up when something interesting happens
    from bridge.proactive import ProactiveEngine
    proactive = ProactiveEngine(brain, exporter, pusher, router=llm_router)
    brain._proactive_engine = proactive  # accessible from chat endpoint for label suggestions
    proactive_task = asyncio.create_task(proactive.run(check_interval=10.0))

    # Interpreter tick loop — updates state detector + personality at 1 Hz
    async def interpreter_tick_loop():
        while True:
            interpreter.tick()
            scp_server.tick()  # check for events (pattern changes, etc.)
            await asyncio.sleep(1.0)
    interpreter_task = asyncio.create_task(interpreter_tick_loop())

    # Run uvicorn in the same loop
    config = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="info", ws="wsproto")
    server = uvicorn.Server(config)
    try:
        await server.serve()
    finally:
        for t in (sensor_task, tick_task, push_task, persist_task, proactive_task, interpreter_task):
            t.cancel()
        adapter.stop()
        # Final save
        try:
            save_brain(brain, checkpoint)
            print(f"Final save to {checkpoint} at tick {brain.tick_count}")
        except Exception as e:
            print(f"Final save failed: {e}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "start":
        try:
            asyncio.run(_run_daemon(args))
        except KeyboardInterrupt:
            print("\nbraind stopped by user")
        return 0
    elif args.command == "status":
        # Phase 3a: just print a stub. Real status check via /healthz comes later.
        print("braind status — use `curl http://localhost:8000/healthz` for live status")
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
