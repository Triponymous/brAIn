"""WSPusher — broadcasts brain state snapshots to all connected WebSocket clients.

Holds a set of clients (FastAPI WebSocket objects). Each broadcast serializes
the state dict to JSON and sends to every client. Failed sends silently
unregister the client.
"""
from __future__ import annotations
import asyncio
import json
from typing import Any


class WSPusher:
    def __init__(self, rate_hz: float = 30.0) -> None:
        self.rate_hz = rate_hz
        self.clients: set[Any] = set()
        self._lock = asyncio.Lock()

    async def register(self, client: Any) -> None:
        async with self._lock:
            self.clients.add(client)

    async def unregister(self, client: Any) -> None:
        async with self._lock:
            self.clients.discard(client)

    async def broadcast(self, state: dict[str, Any]) -> None:
        text = json.dumps(state, default=_json_default)
        async with self._lock:
            failed = []
            for client in list(self.clients):
                try:
                    await client.send_text(text)
                except Exception:
                    failed.append(client)
            for c in failed:
                self.clients.discard(c)


def _json_default(o: Any) -> Any:
    """Fallback JSON serializer for tensors etc."""
    try:
        import torch
        if isinstance(o, torch.Tensor):
            return o.tolist()
    except ImportError:
        pass
    if hasattr(o, "__iter__"):
        return list(o)
    return str(o)
