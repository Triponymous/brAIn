"""Tiny always-on control server: start/stop/status the brain daemon from the UI.

Run once:  .venv/bin/python -m server.control   (default 127.0.0.1:8900)

The brain daemon (braind) is the thing started/stopped/restarted often; this
small server stays up and is the bottom turtle the training console talks to.
"""
from __future__ import annotations
import os
import signal
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

_PIDFILE = Path("checkpoints/braind.pid")
_CONSOLE = Path(__file__).resolve().parents[1] / "train-ui" / "index.html"


class StartRequest(BaseModel):
    mock: bool = False


def _running_pid() -> int | None:
    """The brain daemon's pid if a live process matches the pidfile, else None."""
    if not _PIDFILE.exists():
        return None
    try:
        pid = int(_PIDFILE.read_text().strip())
        os.kill(pid, 0)  # existence probe — signal 0 does not kill
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None


def _spawn_daemon(mock: bool) -> int:
    args = [sys.executable, "-m", "server.braind", "start"]
    if mock:
        args.append("--mock-sensors")
    return subprocess.Popen(args).pid


def build_control_app(spawn=None, killer=None) -> FastAPI:
    """`spawn(mock)->pid` and `killer(pid, sig)` are injectable for tests."""
    spawn = spawn or _spawn_daemon
    killer = killer or os.kill
    app = FastAPI(title="braind-control")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.get("/daemon/status")
    async def status() -> dict:
        pid = _running_pid()
        return {"running": pid is not None, "pid": pid}

    @app.post("/daemon/start")
    async def start(req: StartRequest) -> dict:
        pid = _running_pid()
        if pid is not None:
            return {"running": True, "pid": pid, "already": True}
        pid = spawn(req.mock)
        _PIDFILE.parent.mkdir(parents=True, exist_ok=True)
        _PIDFILE.write_text(str(pid))
        return {"running": True, "pid": pid}

    @app.post("/daemon/stop")
    async def stop() -> dict:
        pid = _running_pid()
        if pid is not None:
            try:
                killer(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            _PIDFILE.unlink(missing_ok=True)
        return {"running": False}

    @app.get("/")
    async def console() -> FileResponse:
        return FileResponse(_CONSOLE)

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(build_control_app(), host="127.0.0.1", port=8900)
