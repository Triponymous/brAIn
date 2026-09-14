"""/api/feel — read the recognized felt-state, or teach/correct it.

The recognized felt-state also rides on the /ws push (added in push_loop).
This router is the explicit correction channel: POST a free-form label and the
current live signature becomes a prototype for it (FeltState).
"""
from __future__ import annotations
import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from bridge.experience import state_of
from bridge.felt_state import signature_from_trend, SIGNATURE_KEYS

# A pending "what was that?" ask stays answerable for an hour (covers real breaks).
_PENDING_TTL = 3600.0


class FeelRequest(BaseModel):
    label: str


def build_feel_router(brain, detector) -> APIRouter:
    api = APIRouter()

    def _current_sig() -> list[float]:
        sig = getattr(brain, "_last_signature", None)
        if sig:
            return sig
        return signature_from_trend(detector.emotional_trend())

    def _current_cluster() -> int:
        c = getattr(brain, "_last_concept_cluster", -1)
        return int(c) if c is not None else -1

    @api.get("/api/feel")
    async def get_feel() -> dict[str, Any]:
        sig = _current_sig()
        name, conf = brain.felt_state.recognize(sig, _current_cluster())
        return {"recognized": name, "confidence": conf,
                "signature": dict(zip(SIGNATURE_KEYS, sig)),
                "known_labels": brain.felt_state.known_labels()}

    @api.post("/api/feel")
    async def post_feel(req: FeelRequest) -> dict[str, Any]:
        # If the pet asked about a state-change, label the FROZEN moment's signature
        # (the anomaly) — not whatever Leon is doing now that he is back to answer.
        pending = getattr(brain, "_pending_ask", None)
        answered = None
        if pending and pending.get("signature") and (time.time() - pending["at"]) < _PENDING_TTL:
            sig = pending["signature"]
            pc = pending.get("cluster")
            cluster = int(pc) if pc is not None else -1
            brain._pending_ask = None
            answered = pending
        else:
            sig = _current_sig()
            cluster = _current_cluster()
        brain.felt_state.label(req.label, sig, cluster)
        log = getattr(brain, "_experience", None)
        if log is not None:
            eid = log.record("human", "teach_felt",
                             {"label": req.label, "answered_ask": answered is not None},
                             state=state_of(brain))
            if answered is not None and answered.get("event_id") is not None:
                log.respond(answered["event_id"], "answered", eid)
        live = _current_sig()
        name, conf = brain.felt_state.recognize(live, _current_cluster())
        return {"labeled": req.label, "recognized": name, "confidence": conf,
                "known_labels": brain.felt_state.known_labels()}

    @api.post("/api/feel/dismiss")
    async def dismiss_feel() -> dict[str, Any]:
        """"Later" in the console. Releases the frozen moment: a label taught
        afterwards applies to the live state, not to an ask that was waved
        away. The dismissal itself is an experience — how often the pet asks
        at the wrong moment is exactly what a learned policy needs to know."""
        pending = getattr(brain, "_pending_ask", None)
        brain._pending_ask = None
        log = getattr(brain, "_experience", None)
        if log is not None and pending is not None:
            eid = log.record("human", "dismiss_ask", {}, state=state_of(brain))
            if pending.get("event_id") is not None:
                log.respond(pending["event_id"], "dismissed", eid)
        return {"dismissed": pending is not None}

    return api
