"""/api/feel — read the recognized felt-state, or teach/correct it.

The recognized felt-state also rides on the /ws push (added in push_loop).
This router is the explicit correction channel: POST a free-form label and the
current live signature becomes a prototype for it (FeltState).
"""
from __future__ import annotations
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from bridge.felt_state import signature_from_trend, SIGNATURE_KEYS


class FeelRequest(BaseModel):
    label: str


def build_feel_router(brain, detector) -> APIRouter:
    api = APIRouter()

    def _current_sig() -> list[float]:
        sig = getattr(brain, "_last_signature", None)
        if sig:
            return sig
        return signature_from_trend(detector.emotional_trend())

    @api.get("/api/feel")
    async def get_feel() -> dict[str, Any]:
        sig = _current_sig()
        name, conf = brain.felt_state.recognize(sig)
        return {"recognized": name, "confidence": conf,
                "signature": dict(zip(SIGNATURE_KEYS, sig)),
                "known_labels": brain.felt_state.known_labels()}

    @api.post("/api/feel")
    async def post_feel(req: FeelRequest) -> dict[str, Any]:
        sig = _current_sig()
        brain.felt_state.label(req.label, sig)
        name, conf = brain.felt_state.recognize(sig)
        return {"labeled": req.label, "recognized": name, "confidence": conf,
                "known_labels": brain.felt_state.known_labels()}

    return api
