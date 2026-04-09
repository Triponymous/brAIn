"""ActiveAppSensor — reads frontmost app + background apps + switch rate.

Tracks:
- Foreground app name
- Background app names (all regular-policy apps)
- App switch detection (did the foreground change?)
- Switch RATE: how many switches in the last 30 seconds (rolling window)
  → High switch rate = multitasking/stress. Low = focused deep work.
"""
from __future__ import annotations
import sys
import time
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
        self._prev_app: str = ""
        # Rolling switch timestamps for switch-rate calculation
        self._switch_times: list[float] = []
        self._switch_window = 30.0  # seconds
        if not mock_mode and sys.platform == "darwin":
            try:
                from AppKit import NSWorkspace, NSApplicationActivationPolicyRegular  # type: ignore
                self._workspace = NSWorkspace.sharedWorkspace()
                self._regular_policy = NSApplicationActivationPolicyRegular
            except ImportError:
                self.mock_mode = True

    def _update_switch_rate(self, switched: bool) -> float:
        """Track switch timestamps, return switches per 30s window."""
        now = time.monotonic()
        if switched:
            self._switch_times.append(now)
        # Prune old entries
        cutoff = now - self._switch_window
        self._switch_times = [t for t in self._switch_times if t > cutoff]
        return len(self._switch_times) / (self._switch_window / 60.0)  # switches per minute

    async def sample(self) -> dict[str, Any]:
        if self.mock_mode:
            name = _MOCK_APPS[self._mock_idx % len(_MOCK_APPS)]
            self._mock_idx += 1
            switched = name != self._prev_app
            self._prev_app = name
            switch_rate = self._update_switch_rate(switched)
            return {
                "name": name,
                "background_apps": [],
                "app_count": 1,
                "switched": switched,
                "switch_rate": round(switch_rate, 1),
            }

        # Real macOS path
        app = self._workspace.frontmostApplication()
        front_name = str(app.localizedName()) if app else "<unknown>"

        # Detect app switch
        switched = front_name != self._prev_app and self._prev_app != ""
        self._prev_app = front_name
        switch_rate = self._update_switch_rate(switched)

        # Get ALL running regular apps (not daemons, not menu bar items)
        running = self._workspace.runningApplications()
        background_apps = []
        for a in running:
            try:
                if a.activationPolicy() == self._regular_policy:
                    name = str(a.localizedName())
                    if name != front_name:
                        background_apps.append(name)
            except Exception:
                continue

        return {
            "name": front_name,
            "background_apps": background_apps[:10],
            "app_count": len(background_apps) + 1,
            "switched": switched,
            "switch_rate": round(switch_rate, 1),  # switches per minute
        }
