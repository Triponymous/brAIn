"""SCP Schema — typed message definitions for the Spiking Communication Protocol.

Defines dataclasses for ALL SCP message types (Query, Response, Event, Action,
Personality) and their result payloads. Each message is a plain Python dataclass
that can be serialized to JSON via ``serialize()`` and validated via ``validate()``.

This module is the single source of truth for SCP message shapes. Both
BrainServer (producer) and LLM Client (consumer) import from here.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, fields, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Result payloads — one per query method
# ---------------------------------------------------------------------------

@dataclass
class TrendResult:
    """Modulator trend over a time window."""
    window_seconds: int = 180
    avg: dict[str, float] = field(default_factory=lambda: {"DA": 0.0, "NE": 0.0, "ACh": 0.0, "5HT": 0.0})
    peak: dict[str, float] = field(default_factory=lambda: {"DA": 0.0, "NE": 0.0, "ACh": 0.0, "5HT": 0.0})
    direction: str = "stable"  # stable | rising | falling


@dataclass
class EmotionResult:
    """Response payload for brain.emotion."""
    state: str = "content"
    confidence: float = 0.0
    duration_seconds: int = 0
    trend: TrendResult = field(default_factory=TrendResult)
    reason: str = ""


@dataclass
class PatternResult:
    """Response payload for brain.pattern."""
    cluster_id: int = -1
    label: str | None = None
    confidence: float = 0.0
    observation_count: int = 0
    sensors: dict[str, Any] = field(default_factory=dict)
    suggested_label: str | None = None


@dataclass
class MemoryResult:
    """Response payload for brain.memory."""
    wm_active: int = 0
    wm_capacity: int = 20
    recent_patterns: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SessionResult:
    """Response payload for brain.session."""
    active_minutes: float = 0.0
    needs_break: bool = False
    in_flow: bool = False
    in_meeting: bool = False
    habit_context: str = ""
    anomalies: list[str] = field(default_factory=list)


@dataclass
class LearningResult:
    """Response payload for brain.learning."""
    specialized_neurons: int = 0
    total_neurons: int = 0
    specialization_pct: float = 0.0
    cluster_count: int = 0
    age_ticks: int = 0
    age_days: float = 0.0


# ---------------------------------------------------------------------------
# Top-level SCP messages
# ---------------------------------------------------------------------------

@dataclass
class QueryMessage:
    """Client -> Server: request neural state."""
    method: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    params: dict[str, Any] = field(default_factory=dict)
    type: str = field(default="query", init=False)


@dataclass
class ResponseMessage:
    """Server -> Client: answer to a query."""
    id: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    type: str = field(default="response", init=False)


@dataclass
class EventMessage:
    """Server -> Client: pushed state change."""
    method: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    type: str = field(default="event", init=False)


@dataclass
class ActionMessage:
    """Client -> Server: feedback that affects SNN."""
    method: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    type: str = field(default="action", init=False)


@dataclass
class PersonalityMessage:
    """Server -> Client: personality mode specification."""
    mode: str = "content"
    style: dict[str, Any] = field(default_factory=lambda: {
        "tone": "warm, reflektiv",
        "length": "2-4 Saetze",
        "questions": False,
        "urgency": "low",
    })
    constraints: list[str] = field(default_factory=lambda: [
        "keine Emojis",
        "keine Hilfe anbieten",
        "keine erfundenen Faehigkeiten",
        "nicht den Sinnen widersprechen",
    ])
    priority_topic: str | None = None
    type: str = field(default="personality", init=False)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def serialize(obj: Any) -> dict[str, Any]:
    """Convert any SCP dataclass (or plain dict) to a JSON-serializable dict.

    Handles nested dataclasses, filtering out None values in nested results.
    """
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Cannot serialize {type(obj).__name__} — expected dataclass or dict")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Allowed message types and their required fields
_SCHEMAS: dict[str, set[str]] = {
    "query": {"type", "method", "id"},
    "response": {"type", "id", "result"},
    "event": {"type", "method", "params"},
    "action": {"type", "method", "params"},
    "personality": {"type", "mode", "style", "constraints"},
}

# Allowed query methods
_QUERY_METHODS = {
    "brain.emotion",
    "brain.pattern",
    "brain.memory",
    "brain.session",
    "brain.learning",
    "brain.full",
}

# Allowed action methods
_ACTION_METHODS = {
    "brain.reward",
    "brain.correct",
    "brain.label",
    "brain.engage",
    "brain.disengage",
}

# Allowed event methods
_EVENT_METHODS = {
    "brain.pattern_changed",
    "brain.emotion_changed",
    "brain.stress_detected",
    "brain.flow_detected",
    "brain.break_needed",
    "brain.new_pattern",
    "brain.label_needed",
}


def validate(data: dict[str, Any]) -> list[str]:
    """Validate a dict against the SCP schema.

    Returns a list of error strings. Empty list means valid.
    """
    errors: list[str] = []

    if not isinstance(data, dict):
        return [f"Expected dict, got {type(data).__name__}"]

    msg_type = data.get("type")
    if msg_type is None:
        errors.append("Missing required field: 'type'")
        return errors

    if msg_type not in _SCHEMAS:
        errors.append(f"Unknown message type: '{msg_type}'")
        return errors

    # Check required fields
    required = _SCHEMAS[msg_type]
    for f in required:
        if f not in data:
            errors.append(f"Missing required field: '{f}' (type={msg_type})")

    # Validate method names for typed messages
    method = data.get("method", "")
    if msg_type == "query" and method and method not in _QUERY_METHODS:
        errors.append(f"Unknown query method: '{method}'")
    if msg_type == "action" and method and method not in _ACTION_METHODS:
        errors.append(f"Unknown action method: '{method}'")
    if msg_type == "event" and method and method not in _EVENT_METHODS:
        errors.append(f"Unknown event method: '{method}'")

    return errors
