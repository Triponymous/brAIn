"""ActiveAppSensor — reads frontmost app + background apps + window/space changes.

Tracks:
- Foreground app name
- Background app names (all regular-policy apps)
- Window/context switch detection:
  - App switch (different frontmost app)
  - Window switch (same app but different window — detected via window title/ID)
  - Space switch (macOS Spaces/Desktops — detected via frontmost window change)
- Switch RATE: how many context switches in the last 30 seconds

Without AppKit (not macOS, or the import failed) the sample is None. It used
to fall back to mock mode, which reported a rotating list of invented apps.
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
        self._prev_window_id: int = 0  # track window changes within same app
        self._switch_times: list[float] = []
        self._switch_window = 30.0
        self._cg = None
        if not mock_mode and sys.platform == "darwin":
            try:
                from AppKit import NSWorkspace, NSApplicationActivationPolicyRegular
                self._workspace = NSWorkspace.sharedWorkspace()
                self._regular_policy = NSApplicationActivationPolicyRegular
            except ImportError:
                pass
            try:
                import Quartz
                self._cg = Quartz
            except ImportError:
                pass

    def _get_frontmost_window_id(self) -> int:
        """Get the window ID of the frontmost normal window.
        Filters to layer=0 (normal windows, not menubar/statusbar/overlays).
        This detects Space switches even when the app name doesn't change."""
        if self._cg is None:
            return 0
        try:
            windows = self._cg.CGWindowListCopyWindowInfo(
                self._cg.kCGWindowListOptionOnScreenOnly | self._cg.kCGWindowListExcludeDesktopElements,
                self._cg.kCGNullWindowID,
            )
            if windows:
                for w in windows:
                    # Layer 0 = normal windows (not menubar, statusbar, overlays)
                    if w.get("kCGWindowLayer", -1) == 0:
                        return int(w.get("kCGWindowNumber", 0))
        except Exception:
            pass
        return 0

    def _update_switch_rate(self, switched: bool) -> float:
        now = time.monotonic()
        if switched:
            self._switch_times.append(now)
        cutoff = now - self._switch_window
        self._switch_times = [t for t in self._switch_times if t > cutoff]
        return len(self._switch_times) / (self._switch_window / 60.0)

    async def sample(self) -> dict[str, Any] | None:
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

        if self._workspace is None:
            return None

        # Real macOS path — use CGWindowList instead of NSWorkspace.
        # NSWorkspace.frontmostApplication() returns the daemon's own app (Terminal)
        # when called from a background thread. CGWindowList always returns the
        # actual frontmost window regardless of which process asks.
        front_name = "<unknown>"
        window_id = 0
        if self._cg:
            try:
                windows = self._cg.CGWindowListCopyWindowInfo(
                    self._cg.kCGWindowListOptionOnScreenOnly | self._cg.kCGWindowListExcludeDesktopElements,
                    self._cg.kCGNullWindowID,
                )
                if windows:
                    for w in windows:
                        if w.get("kCGWindowLayer", -1) == 0:
                            front_name = str(w.get("kCGWindowOwnerName", "<unknown>"))
                            window_id = int(w.get("kCGWindowNumber", 0))
                            break
            except Exception:
                pass

        # Fallback to NSWorkspace if CGWindowList failed
        if front_name == "<unknown>" and self._workspace:
            app = self._workspace.frontmostApplication()
            front_name = str(app.localizedName()) if app else "<unknown>"

        # Detect context switch: app change OR window change
        app_switched = front_name != self._prev_app and self._prev_app != ""
        window_switched = window_id != self._prev_window_id and self._prev_window_id != 0
        switched = app_switched or window_switched

        self._prev_app = front_name
        self._prev_window_id = window_id
        switch_rate = self._update_switch_rate(switched)

        # Get ALL running regular apps
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
            "switch_rate": round(switch_rate, 1),
        }
