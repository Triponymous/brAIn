"""/api/experience — read the experience log: recent events and a summary."""
from __future__ import annotations
from typing import Any

from fastapi import APIRouter

from bridge.experience import ExperienceLog


def build_experience_router(log: ExperienceLog) -> APIRouter:
    api = APIRouter()

    @api.get("/api/experience")
    async def get_experience(limit: int = 50, hours: float = 168.0) -> dict[str, Any]:
        return {"recent": log.recent(limit=limit), "summary": log.summary(since_hours=hours)}

    return api
