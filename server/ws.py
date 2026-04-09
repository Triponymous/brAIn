"""WSPusher — broadcasts brain state to WebSocket clients with subscription support.

Each client can subscribe to different detail levels:
- "macro" (default): compact summary only
- "meso": adds region_spikes for the subscribed region
- "micro": adds synapse_activity for the subscribed neuron
"""
from __future__ import annotations
import asyncio
import json
from typing import Any


class ClientSubscription:
    __slots__ = ("level", "region", "neuron_id")

    def __init__(self) -> None:
        self.level: str = "macro"
        self.region: str | None = None
        self.neuron_id: int | None = None


class WSPusher:
    def __init__(self, rate_hz: float = 30.0) -> None:
        self.rate_hz = rate_hz
        self.clients: dict[Any, ClientSubscription] = {}
        self._lock = asyncio.Lock()

    async def register(self, client: Any) -> None:
        async with self._lock:
            self.clients[client] = ClientSubscription()

    async def unregister(self, client: Any) -> None:
        async with self._lock:
            self.clients.pop(client, None)

    def handle_client_message(self, client: Any, raw: str) -> None:
        """Process subscription messages from clients."""
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        sub = self.clients.get(client)
        if not sub:
            return
        if "subscribe" in msg:
            sub.level = msg["subscribe"]
            sub.region = msg.get("region")
            nid = msg.get("neuron_id")
            sub.neuron_id = int(nid) if nid is not None else None

    def get_needed_regions(self) -> set[str]:
        """Return which regions need detail data (for meso/micro clients)."""
        regions: set[str] = set()
        for sub in self.clients.values():
            if sub.level in ("meso", "micro") and sub.region:
                regions.add(sub.region)
        return regions

    async def broadcast(self, base_state: dict[str, Any], detail_state: dict[str, Any] | None = None) -> None:
        """Send state to all clients.

        Macro clients get base_state only.
        Meso/micro clients get base_state merged with detail_state.
        """
        base_text = json.dumps(base_state, default=_json_default)

        detail_text: str | None = None
        if detail_state:
            merged = {**base_state, **detail_state}
            detail_text = json.dumps(merged, default=_json_default)

        async with self._lock:
            failed = []
            for client, sub in list(self.clients.items()):
                try:
                    if sub.level in ("meso", "micro") and detail_text:
                        await client.send_text(detail_text)
                    else:
                        await client.send_text(base_text)
                except Exception:
                    failed.append(client)
            for c in failed:
                self.clients.pop(c, None)


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
