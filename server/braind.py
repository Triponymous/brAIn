"""braind — Brain background daemon CLI.

Usage:
    braind start [--mock-sensors] [--port 8000] [--checkpoint PATH]
    braind status
    braind erase [--checkpoint PATH] [--port 8000] [--yes]
    braind --help

`start` runs the daemon in the foreground. The daemon:
- Loads the brain from the checkpoint file (or creates fresh if missing)
- Starts the MacDesktopAdapter with the sources the user has shared
  (consent.json next to the checkpoint; none on first start, see server/consent.py)
- Runs brain.tick() at ~100 Hz on incoming sensor data, paused while nothing is shared
- Pushes brain state to /ws clients at ~30 Hz
- Auto-saves the brain to checkpoint every 60s
- Listens on the configured port
"""
from __future__ import annotations
import argparse
import asyncio
from pathlib import Path
import signal
import sys
import time

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

    erase = sub.add_parser("erase", help="Delete everything the daemon stored (lists first; --yes deletes)")
    erase.add_argument("--checkpoint", type=str, default="checkpoints/braind.sqlite")
    erase.add_argument("--port", type=int, default=8000, help="port to check for a running daemon")
    erase.add_argument("--yes", action="store_true", help="actually delete the listed files")

    return parser


_PROJ = Path(__file__).resolve().parents[1]
_SQLITE_FILES = ("", "-wal", "-shm", "-journal")  # a database is its main file plus these


def lock_checkpoint(checkpoint: Path):
    """Hold an exclusive lock on <checkpoint>.lock, or return None if a process already does.

    One daemon per checkpoint, and the way erase knows a daemon is using these
    files whatever port it serves. The kernel drops the lock when the holder
    exits, crash included. Keep the returned file open for as long as needed.
    """
    import fcntl
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    f = open(checkpoint.with_name(checkpoint.name + ".lock"), "a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return None
    return f


def erase_plan(checkpoint: Path) -> list[Path]:
    """Every file the daemon keeps about its user next to this checkpoint.

    The stores listed in docs/PRIVACY.md with their SQLite side files, the
    daily backups, the consent choices and leftover temp files. For the default
    checkpoints/ directory also the pidfile (removed first, so the control
    server does not respawn the daemon mid-erase) and the login service's logs.
    """
    from brain.persistence import backup_pattern
    d = checkpoint.parent
    default_dir = d.resolve() == (_PROJ / "checkpoints").resolve()
    plan: list[Path] = [d / "braind.pid"] if default_dir else []
    for name in (checkpoint.name, "episodes.db", "experience.db", "grants.sqlite"):
        plan += [d / (name + side) for side in _SQLITE_FILES]
    plan += [d / (checkpoint.name + ".tmp")]
    plan += [d / (c + t) for c in ("consent.json", "consent-mock.json") for t in ("", ".tmp")]
    plan += sorted((d / "backups").glob(backup_pattern(checkpoint)))
    if default_dir:
        plan += [_PROJ / "logs" / "brain.out.log", _PROJ / "logs" / "brain.err.log"]
    return [f for f in plan if f.is_file()]


def _default_daemon_up(port: int) -> bool:
    """A daemon from before the checkpoint lock existed: the pidfile or the port tells."""
    from server.control import _running_pid
    if _running_pid() is not None:
        return True
    import urllib.request
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1).close()
        return True
    except OSError:
        return False


def erase(checkpoint: Path, port: int = 8000, yes: bool = False) -> int:
    """List (and with yes=True delete) what the daemon stored. Refuses while one uses it."""
    checkpoint = Path(checkpoint)
    if not checkpoint.parent.is_dir():
        print(f"Nothing to erase next to {checkpoint}.")
        return 0
    lock_path = checkpoint.with_name(checkpoint.name + ".lock")
    had_lock_file = lock_path.exists()
    lock = lock_checkpoint(checkpoint)  # held while deleting: a daemon starting meanwhile gives up
    default_dir = checkpoint.parent.resolve() == (_PROJ / "checkpoints").resolve()
    if lock is None or (default_dir and _default_daemon_up(port)):
        if lock is not None:
            lock.close()
        print("A daemon is using these files. Stop it first (training console, or Ctrl-C); "
              "a running daemon would keep writing what you are deleting.")
        return 1
    try:
        plan = erase_plan(checkpoint)
        if not plan:
            print(f"Nothing to erase next to {checkpoint}.")
        for f in plan:
            print(f"  {f}  ({f.stat().st_size:,} bytes)")
        if plan and not yes:
            print(f"{len(plan)} files would be deleted. Run again with --yes to delete them. "
                  "Explicit exports and copies made by other tools are not included.")
        if not (plan and yes):
            if not had_lock_file:
                lock_path.unlink(missing_ok=True)  # a look should leave nothing behind
            return 0
        for f in plan:
            f.unlink(missing_ok=True)
        backups = checkpoint.parent / "backups"
        if backups.is_dir() and not any(backups.iterdir()):
            backups.rmdir()
        lock_path.unlink(missing_ok=True)  # last: every data file is already gone
        print(f"Deleted {len(plan)} files. The next start begins with a fresh brain that shares nothing.")
        return 0
    finally:
        lock.close()


def _check_permissions(enabled: dict[str, bool]) -> None:
    """Check macOS privacy permissions of the shared sources and print clear warnings.

    Unshared sources are not probed: the microphone check records 0.1 s of audio.
    """
    import sys
    if sys.platform != "darwin":
        return

    print("\n╔══════════════════════════════════════════╗")
    print("║       brAIn Permission Diagnostics       ║")
    print("╚══════════════════════════════════════════╝")
    if not any(enabled.values()):
        print("No source is shared yet: nothing is captured and the brain waits.")
        print("   → Switch sources on in the training console (http://127.0.0.1:8900)\n")
        return

    # 1. Input Monitoring (keyboard, mouse, idle)
    if any(enabled[s] for s in ("keystroke_rate", "mouse_rate", "idle")):
        try:
            import Quartz
            idle = Quartz.CGEventSourceSecondsSinceLastEventType(
                Quartz.kCGEventSourceStateHIDSystemState, int(0xFFFFFFFF))
            # A long idle right after the user started this daemon means the
            # HID counters are hidden from this process.
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
    if enabled["mic"]:
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
    if enabled["active_app"]:
        try:
            from AppKit import NSWorkspace
            app = NSWorkspace.sharedWorkspace().frontmostApplication()
            print(f"[OK] Active App: OK (currently: {app.localizedName()})")
        except Exception:
            print("[WARN] Active App: NSWorkspace unavailable")

    print()


def build_uvicorn_config(app, port: int) -> uvicorn.Config:
    """Build the daemon's uvicorn config.

    The websocket backend is deliberately left on uvicorn's "auto" default. With
    websockets installed (it ships with uvicorn[standard]) that resolves to the
    sansio implementation, which accepts browser upgrades including an Origin
    header.

    Do not pin ws="wsproto" here: wsproto is not a declared dependency, so a
    clean install of this project crashes on startup with
    ModuleNotFoundError: No module named 'wsproto'. The pin was a workaround for
    an older uvicorn whose legacy websockets backend rejected browser upgrades
    with 400 Bad Request; the sansio backend handles them correctly.
    """
    return uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info")


async def _run_daemon(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    # One daemon per checkpoint: two would overwrite each other's saves, and
    # erase needs to see that these files are in use. Held until the process ends.
    lock = lock_checkpoint(checkpoint)
    if lock is None:
        print(f"Another daemon is already using {checkpoint}; not starting a second one.")
        raise SystemExit(1)

    # Load or create brain
    if checkpoint.exists():
        print(f"Loading brain from {checkpoint}")
        brain = load_brain(checkpoint)
        print(f"Resumed at tick {brain.tick_count}")
    else:
        print("Creating fresh brain")
        brain = Brain()

    # Consent decides what is captured; nothing is, until the user shares a source.
    from server.consent import ConsentStore, build_consent_router
    # Mock mode keeps its own file: agreeing to synthetic data is not agreeing
    # to real capture when the same directory later runs a real daemon.
    consent = ConsentStore(checkpoint.parent / ("consent-mock.json" if args.mock_sensors else "consent.json"))
    shared = [name for name, on in consent.enabled().items() if on]
    print(f"Shared sources: {', '.join(shared) if shared else 'none (the brain waits)'}")

    # ═══ PERMISSION CHECK ═══
    # Without Input Monitoring, keyboard/mouse/idle sensors are blind.
    if not args.mock_sensors:
        _check_permissions(consent.enabled())

    # Create adapter — store on brain for chat endpoint access
    adapter = MacDesktopAdapter(mock_mode=args.mock_sensors, enabled=consent.enabled())
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

    # Experience log — (state, action, consequence, response); see README, The Thesis
    from bridge.experience import ExperienceLog
    from server.experience import build_experience_router
    experience = ExperienceLog(checkpoint.parent / "experience.db")
    brain._experience = experience  # proactive engine, /api/feel, chat and SCP actions record here

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

    scp_server = BrainServer(brain, experience=experience)
    brain._scp_server = scp_server

    # Detect model type from config
    from server.config import get
    local_model = get("llm", "local_model", "qwen2.5:14b-instruct")
    model_type = SCPClient.detect_model_type(local_model)
    scp_client = SCPClient(scp_server, model_type=model_type)
    brain._scp_client = scp_client

    exporter = BrainStateExporter(brain)
    grant_store = GrantStore(checkpoint.parent / "grants.sqlite")
    # The brain as tools: what the LLM calls while it reasons (README, The Thesis)
    from bridge.brain_tools import BrainTools
    brain_tools = BrainTools(brain, episodes=episode_logger, experience=experience,
                             interpreter=interpreter, narrator=narrator, adapter=adapter)
    tool_registry = ToolRegistry(brain, exporter, grant_store, brain_tools=brain_tools)
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

    # Build FastAPI app. Web pages other than the training console (say, a new
    # dashboard on its own port) must be listed explicitly to call the daemon.
    from server.main import CONSOLE_ORIGINS
    extra = get("daemon", "allowed_origins", [])
    extra = tuple(o for o in extra if isinstance(o, str)) if isinstance(extra, list) else ()
    if extra:
        print(f"Also allowed to call the daemon: {', '.join(extra)}")
    app = build_app(brain=brain, adapter=adapter, pusher=pusher, origins=CONSOLE_ORIGINS + extra)
    app.include_router(chat_router)
    app.include_router(grants_router)
    app.include_router(config_router)
    app.include_router(feel_router)
    app.include_router(build_experience_router(experience))
    app.include_router(build_consent_router(consent, adapter))
    from server.tools import build_tools_router
    app.include_router(build_tools_router(brain, brain_tools))  # read-only, for server/mcp.py and scripts

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
        mic_shared=lambda: adapter.enabled["mic"] and not adapter.mock_mode,  # the synthetic mic is not a real one
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
            # Paused: frozen modulators must not move the persisted personality
            # or raise pattern events; nothing new was observed.
            if adapter.acquiring:
                interpreter.tick()
                scp_server.tick()  # check for events (pattern changes, etc.)
            # Paused (nothing shared): no new observation, so no consequence to
            # write; due events expire without one instead of getting a frozen state.
            experience.settle(time.time(), getattr(brain, "_last_signature", None) if adapter.acquiring else None)
            await asyncio.sleep(1.0)
    interpreter_task = asyncio.create_task(interpreter_tick_loop())

    # Run uvicorn in the same loop
    config = build_uvicorn_config(app, args.port)
    server = uvicorn.Server(config)
    # uvicorn captures SIGTERM for a graceful stop, then restores the ORIGINAL
    # handler and re-raises the signal (Server.capture_signals). The default
    # SIGTERM action terminates the process on the spot, inside serve(), so the
    # final save in the finally below never ran. launchd, the console Stop
    # button and system shutdown all send SIGTERM, so everything learned since
    # the last 60 s autosave (a felt-state label taught a moment ago) was lost
    # on every stop. A Python-level handler is what uvicorn restores; the
    # re-raised signal lands here harmlessly and serve() returns. Ctrl+C was
    # never affected: the default SIGINT action raises KeyboardInterrupt, which
    # unwinds through the finally.
    signal.signal(signal.SIGTERM, lambda *_: setattr(server, "should_exit", True))
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
    elif args.command == "erase":
        return erase(Path(args.checkpoint), port=args.port, yes=args.yes)
    elif args.command == "status":
        # Phase 3a: just print a stub. Real status check via /healthz comes later.
        print("braind status — use `curl http://localhost:8000/healthz` for live status")
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
