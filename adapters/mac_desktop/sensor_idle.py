"""IdleSensor — reports seconds since the last user input event.

Uses Quartz CGEventSourceSecondsSinceLastEventType which is a free macOS
API (no permission required). Returns 0 immediately after any input.

Mock mode returns whatever is in `_mock_idle_seconds` (default 0.0).
Without Quartz (not macOS, or the import failed) the sample is None: an
unknown idle time must not read as "the user is active".
"""
from __future__ import annotations
import sys
from typing import Any

from adapters.base import Sensor


class IdleSensor(Sensor):
    name = "idle"
    rate_hz = 1.0

    def __init__(self, mock_mode: bool = False) -> None:
        super().__init__(mock_mode=mock_mode)
        self._mock_idle_seconds: float = 0.0
        self._cg = None
        if not mock_mode and sys.platform == "darwin":
            try:
                import Quartz  # type: ignore
                self._cg = Quartz
            except ImportError:
                pass

    async def sample(self) -> dict[str, float] | None:
        if self.mock_mode:
            return {"seconds": float(self._mock_idle_seconds)}
        if self._cg is None:
            return None
        # kCGAnyInputEventType = ~ -1 (all event types)
        seconds = self._cg.CGEventSourceSecondsSinceLastEventType(
            self._cg.kCGEventSourceStateHIDSystemState,
            int(0xFFFFFFFF),  # any event type
        )
        return {"seconds": float(seconds)}
