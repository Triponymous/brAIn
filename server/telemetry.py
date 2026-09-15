"""Coherent, bounded observations taken AFTER a model tick, never inferred spikes."""
from __future__ import annotations

import copy
import hashlib
import math
import platform
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from server.capture import CaptureControl, build_capture_router

SCHEMA = "brain.telemetry.v1"
REGIONS = {"s": "sensory", "e": "expansion", "c": "concept", "w": "wm"}
ORIGINS = ["http://127.0.0.1:4178", "http://localhost:4178"]


class TelemetryBuffer:
    def __init__(self, brain, *, input_kind="disabled", seed=4271, capacity=512):
        self.session_id = str(uuid4())
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.start = time.monotonic()
        self.frames = deque(maxlen=capacity)
        self.lock = threading.Lock()
        self.state = "waiting"
        self.architecture = {
            key: brain.num_expansion if key == "e" else brain.regions[name].num_neurons
            for key, name in REGIONS.items()
        }
        self.source = {
            "model": "brain.core.Brain", "input_kind": input_kind,
            "initialization": "fresh", "seed": seed, "checkpoint": None,
            "encoder": "desktop-metadata-v1", "model_dt": 1.0,
            "model_time_unit": "Euler integration unit; not wall-clock seconds",
            "membrane_phase": "post-step / after reset and inhibition",
            "connectivity": "not exported", "positions": "illustrative",
            "storage": "bounded process memory; no automatic disk persistence",
            "wearables": "not connected", "llm": "disabled", "microphone": "disabled",
            "runtime": {"torch": torch.__version__, "python": platform.python_version(), "device": "cpu"},
        }
        root = Path(__file__).resolve().parent.parent
        files = sorted((root / "brain").glob("*.py")) + [root / "adapters/mac_desktop/encoding.py",
                    root / "adapters/mac_desktop/metadata.py", root / "server/observe.py",
                    root / "server/capture.py", Path(__file__)]
        self.source["code_sha256"] = hashlib.sha256(b"".join(
            str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() for path in files)).hexdigest()

    def record(self, brain, outputs, sensors, *, monotonic=None, captured_at=None, capture_revision=None):
        stamp = time.monotonic() if monotonic is None else monotonic
        regions = {}
        for key, name in REGIONS.items():
            output = outputs[name].detach().cpu().tolist()
            membrane = None if key == "e" else brain.regions[name].membrane.detach().cpu().tolist()
            if any(value not in (0, 1) for value in output):
                raise ValueError("Non-binary model output")
            if membrane is not None and not all(math.isfinite(v) for v in membrane):
                raise ValueError("Non-finite model membrane")
            regions[key] = {"output": output, "membrane": membrane}
        frame = {
            "tick": brain.tick_count, "elapsed_s": stamp - self.start,
            "captured_at": captured_at or datetime.now(timezone.utc).isoformat(),
            "regions": regions, "modulators": brain.modulators.snapshot(),
            "sensors": copy.deepcopy(sensors), "sleep_mode": brain.sleep_mode,
            "capture_revision": capture_revision,
        }
        with self.lock:
            self.frames.append(frame)
            self.state = "streaming"

    def read(self, after=0, session=None, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            frames = list(self.frames)
            age = None if not frames else max(0, now - self.start - frames[-1]["elapsed_s"])
            reset = session != self.session_id
            gap = bool(frames and not reset and after and after < frames[0]["tick"] - 1)
            selected = frames[-200:] if reset or gap else [f for f in frames if f["tick"] > after][:200]
            return copy.deepcopy({
                "schema": SCHEMA, "session_id": self.session_id,
                "started_at": self.started_at, "source": self.source,
                "architecture": self.architecture, "buffer_capacity": self.frames.maxlen,
                "status": "stale" if self.state == "streaming" and age > 2 else self.state,
                "age_s": age, "reset": reset, "gap": gap, "frames": selected,
            })


def build_telemetry_app(buffer, capture=None):
    capture = capture or CaptureControl(buffer.session_id)
    app = FastAPI(title="brAIn local observation", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])
    app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "X-Brain-Control"], allow_credentials=False)

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        origin = request.headers.get("origin")
        local = request.client and request.client.host in ("127.0.0.1", "::1")
        same_origin = f"http://{request.headers.get('host', '')}"
        if not local or (origin and origin not in [*ORIGINS, same_origin]):
            return JSONResponse({"detail": "Local observation only"}, status_code=403)
        if request.method == "POST" and request.url.path == "/api/capture" and (
                origin not in ORIGINS or request.headers.get("x-brain-control") != "capture-v1" or
                request.headers.get("content-type", "").split(";")[0] != "application/json"):
            return JSONResponse({"detail": "Capture changes require the local dashboard"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/api/telemetry")
    async def telemetry(after: int = Query(0, ge=0), session: str | None = Query(None, max_length=64)):
        with capture.lock:
            data = buffer.read(after, session)
            data["capture"] = capture.read()
            if buffer.source["input_kind"] == "desktop_metadata":
                if capture.failed:
                    data["status"] = "error"
                elif not any(capture.enabled.values()):
                    data["status"] = "paused"
                elif capture.vector is None or not any(status == "available" for status in capture.status.values()):
                    data["status"] = "waiting"
            return data

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "service": "brain-observation", "schema": SCHEMA}

    app.include_router(build_capture_router(capture))
    return app
