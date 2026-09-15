"""Session-scoped capture policy. Disabling waits for any in-flight model step."""
from __future__ import annotations

import threading
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from adapters.mac_desktop.metadata import INPUTS, encode_metadata

InputName = Literal["keystroke_rate", "mouse_rate", "idle", "active_app"]


class CaptureUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=1, max_length=64, strict=True)
    revision: int = Field(ge=0, strict=True)
    enabled: dict[InputName, StrictBool] = Field(min_length=1, max_length=4)


class CaptureControl:
    def __init__(self, session_id, *, armed=False):
        self.session_id, self.armed = session_id, armed
        self.lock = threading.RLock()
        self.revision = 0
        self.enabled = dict.fromkeys(INPUTS, False)
        self.status = dict.fromkeys(INPUTS, "disabled")
        self.sample_revision = -1
        self.next_sample = 0
        self.sensors = self.vector = None
        self.failed = False

    def read(self):
        with self.lock:
            return {"schema": "brain.capture.v1", "session_id": self.session_id,
                    "revision": self.revision, "armed": self.armed, "failed": self.failed,
                    "scope": "observation_runner", "persistence": "session_only",
                    "inputs": {key: {"enabled": self.enabled[key], "status": self.status[key]}
                               for key in INPUTS}}

    def update(self, change):
        with self.lock:
            if change.session_id != self.session_id:
                raise HTTPException(409, "Session changed; reload capture settings")
            enabling = any(change.enabled.values())
            if change.revision > self.revision or (enabling and change.revision != self.revision):
                raise HTTPException(409, "Settings changed; reload before enabling an input")
            if enabling and (not self.armed or self.failed):
                raise HTTPException(403, "Start an opted-in, healthy observation runner first")
            # A stop wins over stale clients, and invalidates pending enable requests.
            self.enabled.update(change.enabled)
            self.revision += 1
            self.status = {key: "waiting" if enabled else "disabled"
                           for key, enabled in self.enabled.items()}
            self.sensors = self.vector = None
            return self.read()

    def sample(self, collector, now):
        with self.lock:
            if self.sample_revision != self.revision:
                collector.reset()
                self.sample_revision = self.revision
                self.next_sample = 0
            if self.failed or not any(self.enabled.values()) or now < self.next_sample:
                return
            # Called on the main thread: NSWorkspace must not move into the tick worker.
            self.sensors = collector.sample(self.enabled)
            self.vector = encode_metadata(self.sensors)
            self.status = {key: self.sensors[key]["status"] for key in INPUTS}
            self.next_sample = now + 1

    def step(self, brain, buffer):
        with self.lock:
            if self.vector is None or self.failed or not any(
                    self.sensors[key]["status"] == "available" for key in INPUTS):
                return False
            output = brain.tick(self.vector)
            buffer.record(brain, output, self.sensors, capture_revision=self.revision)
            return True

    def fail(self):
        with self.lock:
            self.failed = True
            self.sensors = self.vector = None
            self.status = {key: "error" if enabled else "disabled"
                           for key, enabled in self.enabled.items()}


def build_capture_router(control):
    router = APIRouter()

    @router.get("/api/capture")
    async def read_capture():
        return control.read()

    @router.post("/api/capture")
    async def update_capture(request: Request):
        body = b""
        async for chunk in request.stream():
            body += chunk
            if len(body) > 2048:
                raise HTTPException(413, "Capture settings payload too large")
        try:
            change = CaptureUpdate.model_validate_json(body)
        except ValidationError:
            raise HTTPException(422, "Invalid capture settings") from None
        return control.update(change)

    return router
