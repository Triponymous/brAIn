"""Opt-in desktop counters. No event hooks, key contents, audio or window titles."""
from __future__ import annotations

import math
import time

from adapters.mac_desktop.encoding import encode_snapshot

INPUTS = ("keystroke_rate", "mouse_rate", "idle", "active_app")

CATEGORIES = {
    "development": ("code", "xcode", "terminal", "iterm", "pycharm", "cursor", "codex"),
    "browser": ("safari", "chrome", "firefox", "edge", "brave", "arc"),
    "communication": ("slack", "teams", "zoom", "mail", "messages", "discord"),
    "writing": ("pages", "word", "notes", "obsidian", "notion"),
    "media": ("spotify", "music", "vlc", "quicktime"),
    "design": ("figma", "sketch", "photoshop", "illustrator", "blender"),
}


def app_category(name):
    normalized = name.casefold()
    return next((category for category, names in CATEGORIES.items()
                 if any(token in normalized for token in names)), "other")


def encode_metadata(sensors):
    """Missing counts are NOT silence; unavailable rhythm/composite slots stay masked."""
    snap = {}
    for field in ("keystroke_rate", "mouse_rate", "idle", "active_app"):
        item = sensors[field]
        if item["status"] != "available":
            continue
        if field == "active_app":
            snap[field] = {"name": item["value"]}
        else:
            snap[field] = {"seconds" if field == "idle" else "count": item["value"]}
    vector = encode_snapshot(snap)
    # Counter polling cannot measure within-window rhythm or app-switch rate.
    vector[76:80] = 0
    vector[96:100] = 0
    vector[156:160] = 0
    if not all(sensors[key]["status"] == "available" for key in ("keystroke_rate", "mouse_rate", "idle")):
        vector[160:164] = 0
    return vector


class DesktopMetadata:
    def __init__(self):
        import Quartz
        from AppKit import NSWorkspace
        self.cg = Quartz
        self.workspace = NSWorkspace.sharedWorkspace()
        self.reset()
        self.keys = [Quartz.kCGEventKeyDown]
        self.mouse = [Quartz.kCGEventMouseMoved, Quartz.kCGEventLeftMouseDown,
                      Quartz.kCGEventRightMouseDown, Quartz.kCGEventScrollWheel,
                      Quartz.kCGEventLeftMouseDragged, Quartz.kCGEventRightMouseDragged]

    def reset(self):
        self.previous = {}

    def _rate(self, key, events, now):
        count = sum(self.cg.CGEventSourceCounterForEventType(
            self.cg.kCGEventSourceStateHIDSystemState, event) for event in events)
        previous = self.previous.get(key)
        self.previous[key] = (count, now)
        if previous and now > previous[1] and count >= previous[0]:
            return (count - previous[0]) / (now - previous[1]), now - previous[1]
        return None

    def sample(self, enabled):
        now = time.monotonic()
        observed_at = time.time()
        def item(status, value=None, unit=None, window_s=None):
            return {"status": status, "value": value, "unit": unit,
                    "observed_at": observed_at if status == "available" else None,
                    "window_s": window_s, "source": "macos-metadata"}
        result = {key: item("unavailable" if enabled[key] else "disabled") for key in INPUTS}
        # Preflight does not trigger a macOS permission dialog or change a permission.
        input_keys = ("keystroke_rate", "mouse_rate", "idle")
        permitted = any(enabled[key] for key in input_keys) and self.cg.CGPreflightListenEventAccess()
        for key, events in (("keystroke_rate", self.keys), ("mouse_rate", self.mouse)):
            if not enabled[key] or not permitted:
                self.previous.pop(key, None)
                continue
            try:
                rate = self._rate(key, events, now)
                if rate is not None:
                    result[key] = item("available", rate[0], "events/s", rate[1])
            except Exception:
                self.previous.pop(key, None)
        if enabled["idle"] and permitted:
            try:
                idle = float(self.cg.CGEventSourceSecondsSinceLastEventType(
                    self.cg.kCGEventSourceStateHIDSystemState, 0xFFFFFFFF))
                if math.isfinite(idle) and 0 <= idle < 1e9:
                    result["idle"] = item("available", idle, "s")
            except Exception:
                pass
        for key in input_keys:
            if enabled[key] and not permitted:
                result[key] = item("permission_required")
        if enabled["active_app"]:
            try:
                app = self.workspace.frontmostApplication()
                if app:
                    result["active_app"] = item("available", app_category(str(app.localizedName())), "category")
            except Exception:
                pass
        result["microphone"] = item("disabled")
        result["wearable"] = item("not_connected")
        return result
