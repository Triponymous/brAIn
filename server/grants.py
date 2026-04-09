"""Grant API endpoints -- /api/wishes, /api/grants, /api/wishes/grant, /api/wishes/deny."""
from __future__ import annotations
from typing import Any, Callable
from fastapi import APIRouter
from pydantic import BaseModel
from capabilities.grants import GrantStore


class GrantRequest(BaseModel):
    wish_id: int
    tool_name: str


class DenyRequest(BaseModel):
    wish_id: int
    tool_name: str


def build_grants_router(store: GrantStore, refresh_fn: Callable[[], None]) -> APIRouter:
    api = APIRouter()

    @api.get("/api/wishes")
    async def list_wishes() -> dict[str, Any]:
        return {"wishes": store.list_wishes(status="pending")}

    @api.get("/api/grants")
    async def list_grants() -> dict[str, Any]:
        return {"grants": store.list_grants()}

    @api.post("/api/wishes/grant")
    async def grant_wish(req: GrantRequest) -> dict[str, str]:
        store.grant(req.tool_name)
        store.resolve_wish(req.wish_id, "granted")
        refresh_fn()
        return {"status": "granted", "tool_name": req.tool_name}

    @api.post("/api/wishes/deny")
    async def deny_wish(req: DenyRequest) -> dict[str, str]:
        store.deny(req.tool_name, cooldown_days=7)
        store.resolve_wish(req.wish_id, "denied")
        return {"status": "denied", "tool_name": req.tool_name}

    return api
