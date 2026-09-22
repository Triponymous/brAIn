"""Per-source consent for the persistent daemon.

The daemon acquires nothing from a source until the user switches it on, and
a restart restores exactly the last recorded choice. A missing, unreadable or
unknown-format file means every source is off, and a source added in a later
version starts off: restarting never widens consent. Switching a source on
needs the current revision; a stop always wins, even from a client holding an
outdated view, and holds for the running daemon even if it cannot be saved.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from adapters.mac_desktop.adapter import SOURCES

SCHEMA = "brain.consent.v1"
SourceName = Literal["keystroke_rate", "mouse_rate", "idle", "active_app", "mic"]


class ConsentUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=0, strict=True)
    enabled: dict[SourceName, StrictBool] = Field(min_length=1, max_length=len(SOURCES))


def _all_off() -> dict[str, dict[str, Any]]:
    return {name: {"enabled": False, "changed_at": None} for name in SOURCES}


class ConsentStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.revision, self.sources = self._load()

    def _load(self) -> tuple[int, dict[str, dict[str, Any]]]:
        sources = _all_off()
        try:
            data = json.loads(self.path.read_text())
            if data["schema"] != SCHEMA:
                return 0, _all_off()
            for name, entry in data["sources"].items():
                if name in sources:
                    at = entry["changed_at"]
                    sources[name] = {"enabled": entry["enabled"] is True,
                                     "changed_at": None if at is None else float(at)}
            return max(0, int(data["revision"])), sources
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return 0, _all_off()  # unreadable counts as "never agreed"

    def _save(self, revision: int, sources: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump({"schema": SCHEMA, "revision": revision, "sources": sources}, f, indent=2)
            f.flush()
            os.fsync(f.fileno())  # a stop must survive a power cut, not come back as the old "on"
        os.replace(tmp, self.path)

    def enabled(self) -> dict[str, bool]:
        with self._lock:
            return {name: entry["enabled"] for name, entry in self.sources.items()}

    def read(self) -> dict[str, Any]:
        with self._lock:
            return {"schema": SCHEMA, "revision": self.revision,
                    "sources": {name: dict(entry) for name, entry in self.sources.items()}}

    def update(self, change: ConsentUpdate, now: float) -> None:
        with self._lock:
            enabling = any(change.enabled.values())
            if change.revision > self.revision or (enabling and change.revision != self.revision):
                raise HTTPException(409, "Consent changed elsewhere; reload before switching a source on")
            sources = {name: dict(entry) for name, entry in self.sources.items()}
            for name, on in change.enabled.items():
                if sources[name]["enabled"] != on:
                    sources[name] = {"enabled": on, "changed_at": now}
            try:
                self._save(self.revision + 1, sources)
            except OSError:
                for name, on in change.enabled.items():
                    if not on:
                        self.sources[name] = sources[name]  # unsaved, but stopped for this run
                raise
            self.revision, self.sources = self.revision + 1, sources


def build_consent_router(store: ConsentStore, adapter: Any) -> APIRouter:
    router = APIRouter()

    def state() -> dict[str, Any]:
        out = store.read()
        status = adapter.status()
        for name, entry in out["sources"].items():
            entry["status"] = status[name]
        out["paused"] = not adapter.acquiring  # nothing shared: the brain does not step
        return out

    @router.get("/api/consent")
    async def read_consent() -> dict[str, Any]:
        return state()

    # async on purpose: adapter.apply() must run on the event-loop (main) thread,
    # where macOS delivers input events to the listeners it starts.
    @router.post("/api/consent")
    async def update_consent(request: Request) -> dict[str, Any]:
        body = b""
        async for chunk in request.stream():
            body += chunk
            if len(body) > 2048:
                raise HTTPException(413, "Consent payload too large")
        try:
            change = ConsentUpdate.model_validate_json(body)
        except ValidationError:
            raise HTTPException(422, "Invalid consent settings") from None
        try:
            store.update(change, time.time())
        except OSError:
            adapter.apply(store.enabled())
            raise HTTPException(500, "Could not save the choice. Sources switched off stay off "
                                     "until the daemon restarts; the others are unchanged.") from None
        adapter.apply(store.enabled())
        return state()

    return router
