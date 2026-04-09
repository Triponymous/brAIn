"""braind — Mini-OSCEN background daemon CLI.

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

    # Create adapter
    adapter = MacDesktopAdapter(mock_mode=args.mock_sensors)
    pusher = WSPusher(rate_hz=args.push_hz)

    # Bridge setup
    from bridge.exporter import BrainStateExporter
    from bridge.llm_router import HybridLLMRouter
    from server.chat import build_chat_router

    # Capability system
    from capabilities.grants import GrantStore
    from capabilities.registry import ToolRegistry
    from server.grants import build_grants_router

    exporter = BrainStateExporter(brain)
    grant_store = GrantStore(checkpoint.parent / "grants.sqlite")
    tool_registry = ToolRegistry(brain, exporter, grant_store)
    llm_router = HybridLLMRouter()
    chat_router = build_chat_router(brain, exporter, tool_registry, llm_router)
    grants_router = build_grants_router(grant_store, refresh_fn=tool_registry.refresh_grants)

    # Build FastAPI app
    app = build_app(brain=brain, adapter=adapter, pusher=pusher)
    app.include_router(chat_router)
    app.include_router(grants_router)

    # Voice setup
    from bridge.tts import TTSEngine
    from bridge.stt import STTEngine
    from server.voice import build_voice_router

    tts_engine = TTSEngine(voice_model=Path("models/de_DE-thorsten-medium.onnx"))
    stt_engine = STTEngine(model_name="small")

    async def _chat_fn(message: str) -> dict:
        """Adapter: routes a voice message through the same chat pipeline."""
        import json
        snap = exporter.snapshot()
        labels = exporter.all_labels()
        from server.chat import _SYSTEM_PROMPT_TEMPLATE
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            brain_state=json.dumps(snap, indent=2, default=str),
            labels=json.dumps(labels, default=str) if labels else "None yet.",
        )
        return await llm_router.chat(
            user_message=message,
            system_prompt=system_prompt,
            brain_state=snap,
            tools=tool_registry.tool_definitions(),
        )

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
    tick_task = asyncio.create_task(brain_tick_loop(brain, adapter, hz=args.tick_hz, exporter=exporter))
    push_task = asyncio.create_task(push_loop(brain, pusher, exporter=exporter, adapter=adapter))
    persist_task = asyncio.create_task(persistence_loop(brain, str(checkpoint)))

    # Proactive notifications — pet speaks up when something interesting happens
    from bridge.proactive import ProactiveEngine
    proactive = ProactiveEngine(brain, exporter, pusher)
    proactive_task = asyncio.create_task(proactive.run(check_interval=10.0))

    # Run uvicorn in the same loop
    config = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="info")
    server = uvicorn.Server(config)
    try:
        await server.serve()
    finally:
        for t in (sensor_task, tick_task, push_task, persist_task, proactive_task):
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
