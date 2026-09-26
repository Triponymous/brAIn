"""Tiny always-on control server: start/stop/status the brain daemon from the UI.

Run once:  .venv/bin/python -m server.control   (default 127.0.0.1:8900)

The brain daemon (braind) is the thing started/stopped/restarted often; this
small server stays up, is the bottom turtle the dashboard talks to, and serves it.
It is also the daemon's supervisor: the pidfile is the DESIRED state. While it
exists the daemon is meant to be running, so a crashed daemon is respawned;
Stop removes it, so a stopped daemon stays stopped. launchd keeps only this
process alive (scripts/install_launchd.sh) — never the daemon directly, or
launchd would fight the Stop button.
"""
from __future__ import annotations
import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from server.main import DASHBOARD_ORIGINS, LOCAL_HOSTS

_PROJ = Path(__file__).resolve().parents[1]
_PIDFILE = _PROJ / "checkpoints" / "braind.pid"
_DASHBOARD = _PROJ / "docs" / "dashboard-concepts"


class StartRequest(BaseModel):
    mock: bool | None = None  # None: the mode this server was started with, or the last one chosen


def _alive(pid: int) -> bool:
    """Whether pid is a live process.

    The daemon is normally our own child. A crashed child lingers as a zombie
    until reaped, and a zombie still answers the kill-0 probe as alive, so
    reap first: waitpid tells us it exited (and clears it) or that it is still
    running. Only a daemon spawned by an earlier control server is not our
    child; for that one the probe is all there is.
    """
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        return reaped != pid
    except ChildProcessError:
        pass  # not our child
    try:
        os.kill(pid, 0)  # existence probe — signal 0 does not kill
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _running_pid() -> int | None:
    """The brain daemon's pid if a live process matches the pidfile, else None."""
    if not _PIDFILE.exists():
        return None
    try:
        pid = int(_PIDFILE.read_text().strip())
    except ValueError:
        return None
    return pid if _alive(pid) else None


def _spawn_daemon(mock: bool) -> int:
    args = [sys.executable, "-m", "server.braind", "start"]
    if mock:
        args.append("--mock-sensors")
    # Own session: launchd kills a job's whole process group when the job
    # exits, so a control-server restart (crash, code update) must not take
    # the brain down with it. The pidfile lets the next control server adopt
    # the running daemon.
    return subprocess.Popen(args, cwd=_PROJ, start_new_session=True).pid


def _start(spawn, mock: bool) -> int:
    pid = spawn(mock)
    _PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    _PIDFILE.write_text(str(pid))
    return pid


def respawn_if_down(spawn, mock: bool) -> int | None:
    """Respawn the daemon if it is wanted (pidfile exists) but not alive.

    Returns the new pid, or None when nothing had to be done.
    """
    if not _PIDFILE.exists() or _running_pid() is not None:
        return None
    pid = _start(spawn, mock)
    print(f"[control] daemon was down, restarted as pid {pid}", flush=True)
    return pid


def build_control_app(spawn=None, killer=None, *, autostart: bool = False,
                      mock: bool = False, supervise_every: float = 10.0,
                      stop_wait: float = 30.0) -> FastAPI:
    """`spawn(mock)->pid` and `killer(pid, sig)` are injectable for tests.

    autostart: bring the daemon up when this server starts (the launchd path).
    mock: sensor mode for autostart, respawns and starts until the API chooses one.
    stop_wait: how long Stop waits for the daemon to save and exit before answering.
    """
    spawn = spawn or _spawn_daemon
    killer = killer or os.kill
    wanted = {"mock": mock}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if autostart and _running_pid() is None:
            print(f"[control] autostart: daemon pid {_start(spawn, wanted['mock'])}", flush=True)

        async def supervise() -> None:
            while True:
                await asyncio.sleep(supervise_every)
                respawn_if_down(spawn, wanted["mock"])

        task = asyncio.create_task(supervise())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(title="braind-control", lifespan=lifespan)
    # The dashboard is served from here (same origin) or by its static preview
    # on 4178. Allowing every origin let any website start or stop the daemon.
    # The host check keeps a DNS-rebinding page from posing as same-origin.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(LOCAL_HOSTS))
    app.add_middleware(CORSMiddleware, allow_origins=list(DASHBOARD_ORIGINS),
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.middleware("http")
    async def dashboard_only(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin is not None and origin not in (f"http://{request.headers.get('host', '')}", *DASHBOARD_ORIGINS):
            return JSONResponse({"detail": "Only the local dashboard may control the daemon"},
                                status_code=403)
        return await call_next(request)

    @app.get("/daemon/status")
    async def status() -> dict:
        pid = _running_pid()
        return {"running": pid is not None, "pid": pid}

    @app.post("/daemon/start")
    async def start(req: StartRequest) -> dict:
        pid = _running_pid()
        if pid is not None:
            return {"running": True, "pid": pid, "already": True}
        if req.mock is not None:
            wanted["mock"] = req.mock
        return {"running": True, "pid": _start(spawn, wanted["mock"])}

    @app.post("/daemon/stop")
    async def stop() -> dict:
        pid = _running_pid()
        if pid is not None:
            try:
                killer(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        _PIDFILE.unlink(missing_ok=True)  # no pidfile = not wanted = no respawn
        # Answer once it has exited. Until then it saves and holds the checkpoint
        # lock, and a Start clicked meanwhile would find the checkpoint taken.
        deadline = time.monotonic() + stop_wait
        while pid is not None and _alive(pid) and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        return {"running": False}

    @app.get("/")
    async def dashboard() -> RedirectResponse:
        return RedirectResponse("/observatory.html")

    # The dashboard's static files; the routes above take precedence.
    app.mount("/", StaticFiles(directory=_DASHBOARD), name="dashboard")

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="server.control")
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument("--autostart", action="store_true",
                        help="start the brain daemon on launch and keep it alive")
    parser.add_argument("--mock", action="store_true",
                        help="autostart with mock sensors (dev/CI)")
    return parser


if __name__ == "__main__":
    import uvicorn
    args = build_parser().parse_args()
    # The dashboard polls /daemon/status every 3 s while its Live page is open;
    # an access log line each would fill the launchd log for nothing.
    uvicorn.run(build_control_app(autostart=args.autostart, mock=args.mock),
                host="127.0.0.1", port=args.port, access_log=False)
