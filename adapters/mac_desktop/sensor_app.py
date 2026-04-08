"""ActiveAppSensor — reads the frontmost macOS application.

Uses NSWorkspace.sharedWorkspace.frontmostApplication via pyobjc. No
permission required. In mock_mode (used by tests), returns a deterministic
rotating set of fake app names so the test suite runs without macOS bindings.
"""
from __future__ import annotations
import sys
from typing import Any

from adapters.base import Sensor


_MOCK_APPS = ["VSCode", "Chrome", "Slack", "Terminal", "Notes"]


class ActiveAppSensor(Sensor):
    name = "active_app"
    rate_hz = 1.0

    def __init__(self, mock_mode: bool = False) -> None:
        super().__init__(mock_mode=mock_mode)
        self._mock_idx = 0
        self._workspace = None
        if not mock_mode and sys.platform == "darwin":
            try:
                from AppKit import NSWorkspace  # type: ignore
                self._workspace = NSWorkspace.sharedWorkspace()
            except ImportError:
                # pyobjc not installed; degrade to mock
                self.mock_mode = True

    async def sample(self) -> dict[str, Any]:
        if self.mock_mode:
            name = _MOCK_APPS[self._mock_idx % len(_MOCK_APPS)]
            self._mock_idx += 1
            return {"name": name}
        # Real macOS path
        app = self._workspace.frontmostApplication()
        if app is None:
            return {"name": "<unknown>"}
        return {"name": str(app.localizedName())}
