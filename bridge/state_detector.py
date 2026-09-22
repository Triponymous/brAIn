"""State detection: Flow, Stress, Meeting, Break-needed.

Reads live sensor data from brain._last_sensor_display and modulator levels.
Maintains rolling history to detect sustained states (not momentary spikes).

Usage:
    detector = StateDetector()
    # In your tick loop (called once per second):
    detector.update(brain)
    # When you need the current state:
    state = detector.detect()
    # state = {flow, flow_duration_min, flow_app, stress,
    #          meeting, meeting_duration_min, needs_break, active_minutes}
"""
from __future__ import annotations
from collections import deque
from typing import Any


_MEETING_APPS = {"zoom", "teams", "meet", "webex", "skype", "facetime", "discord"}


class StateDetector:
    """Detects sustained user states from rolling sensor + modulator snapshots.

    All thresholds are configurable via constructor kwargs for testing
    and per-user tuning.
    """

    def __init__(
        self,
        flow_min_minutes: float = 30.0,
        flow_max_switch_rate: float = 1.0,
        flow_min_keys: float = 3.0,
        stress_min_seconds: float = 120.0,
        meeting_min_seconds: float = 30.0,
        break_after_minutes: float = 90.0,
        break_reset_idle: float = 300.0,  # 5 min idle resets break timer
    ) -> None:
        self.flow_min_minutes = flow_min_minutes
        self.flow_max_switch_rate = flow_max_switch_rate
        self.flow_min_keys = flow_min_keys
        self.stress_min_seconds = stress_min_seconds
        self.meeting_min_seconds = meeting_min_seconds
        self.break_after_minutes = break_after_minutes
        self.break_reset_idle = break_reset_idle

        # Rolling state
        self._history: deque[dict] = deque(maxlen=7200)  # 2h at 1/sec
        self._last_app: str = ""
        self._same_app_since: int = 0   # snapshot index when app last changed
        self._active_since: int = 0     # snapshot index of last idle reset
        self._snapshot_count: int = 0

    def update(self, brain: Any) -> None:
        """Call once per second (or per snapshot interval).

        Reads brain._last_sensor_display and brain.modulators.snapshot()
        to build a rolling history of sensor + modulator state.
        """
        sd = getattr(brain, '_last_sensor_display', {})
        mods = brain.modulators.snapshot()

        # None = source not shared (or not observed yet). Each state below needs
        # the sources it reasons about; it is never inferred from a default 0.
        snap = {
            "app": sd.get("app", ""),
            "keys": sd.get("keys"),
            "mouse": sd.get("mouse"),
            "mic_rms": sd.get("mic_rms"),
            "idle": sd.get("idle"),
            "switch_rate": sd.get("switch_rate"),
            "da": mods.get("DA", 0),
            "ne": mods.get("NE", 0),
            "ach": mods.get("ACh", 0),
            "sht": mods.get("5HT", 0),
        }
        self._history.append(snap)
        self._snapshot_count += 1

        # Track app continuity — record the snapshot index where the
        # new app started (current count minus 1 = this snapshot's index)
        if snap["app"] != self._last_app:
            self._last_app = snap["app"]
            self._same_app_since = self._snapshot_count - 1

        # Track active time (reset on long idle). Without the idle source a
        # continuous stretch of work cannot be known, so it never accumulates.
        if snap["idle"] is None or snap["idle"] > self.break_reset_idle:
            self._active_since = self._snapshot_count

    def detect(self) -> dict[str, Any]:
        """Return current detected states.

        Returns dict with keys:
            flow            - bool: user is in deep work flow
            flow_duration_min - float: minutes in current flow (0 if not in flow)
            flow_app        - str|None: app being used in flow
            stress          - bool: sustained stress indicators detected
            meeting         - bool: user appears to be in a video call
            meeting_duration_min - float: minutes in current meeting (0 if not)
            needs_break     - bool: user has been active >90min without a break
            active_minutes  - float: minutes since last idle reset
        """
        if not self._history:
            return self._empty()

        recent = self._history
        latest = recent[-1]

        # --- Flow: same app for 30+ min, steady keys, low switching ---
        same_app_secs = self._snapshot_count - self._same_app_since
        same_app_min = same_app_secs / 60.0
        flow = (
            latest["keys"] is not None and latest["switch_rate"] is not None
            and same_app_min >= self.flow_min_minutes
            and latest["switch_rate"] <= self.flow_max_switch_rate
            and latest["keys"] >= self.flow_min_keys
        )

        # --- Stress: high NE + low 5HT + high switch rate for >2 min ---
        stress_window = list(recent)[-int(self.stress_min_seconds):]
        stress = False
        if len(stress_window) >= self.stress_min_seconds and all(
                s["switch_rate"] is not None for s in stress_window):
            avg_ne = sum(s["ne"] for s in stress_window) / len(stress_window)
            avg_sht = sum(s["sht"] for s in stress_window) / len(stress_window)
            avg_switch = sum(s["switch_rate"] for s in stress_window) / len(stress_window)
            stress = avg_ne > 0.05 and avg_sht < 0.02 and avg_switch > 3.0

        # --- Meeting: meeting app + mic active + no typing for >30s ---
        app_lower = latest["app"].lower() if latest["app"] else ""
        is_meeting_app = any(m in app_lower for m in _MEETING_APPS)
        meeting_secs = 0
        if is_meeting_app:
            for s in reversed(list(recent)):
                s_app = (s["app"] or "").lower()
                if any(m in s_app for m in _MEETING_APPS) and (s["mic_rms"] or 0) > 0.005:
                    meeting_secs += 1
                else:
                    break
        meeting = (
            is_meeting_app
            and (latest["mic_rms"] or 0) > 0.005
            and meeting_secs >= self.meeting_min_seconds
        )

        # --- Break needed: active for 90+ min without 5-min idle pause ---
        active_secs = self._snapshot_count - self._active_since
        active_min = active_secs / 60.0
        needs_break = (active_min >= self.break_after_minutes
                       and latest["idle"] is not None and latest["idle"] < 30)

        return {
            "flow": flow,
            "flow_duration_min": round(same_app_min, 1) if flow else 0,
            "flow_app": latest["app"] if flow else None,
            "stress": stress,
            "meeting": meeting,
            "meeting_duration_min": round(meeting_secs / 60, 1) if meeting else 0,
            "needs_break": needs_break,
            "active_minutes": round(active_min, 1),
        }

    def emotional_trend(self, window_seconds: int = 180) -> dict[str, float]:
        """Analyze modulator TRENDS over the last N seconds.

        Returns averaged + peak modulator values. This is what the
        EmotionalPromptEngine should use instead of the instantaneous snapshot,
        because modulators decay in 2-10 seconds but emotional context
        persists for minutes.

        Example: NE spiked 30 seconds ago due to a sudden noise.
        Current NE = 0.001 (decayed). But trend_peak_NE = 0.12 (it WAS high).
        The pet should still be slightly alert, not fully calm.
        """
        if not self._history:
            return {"avg_DA": 0, "avg_NE": 0, "avg_ACh": 0, "avg_5HT": 0,
                    "peak_DA": 0, "peak_NE": 0, "peak_ACh": 0, "peak_5HT": 0,
                    "recent_DA": 0, "recent_NE": 0, "recent_ACh": 0, "recent_5HT": 0}

        window = list(self._history)[-window_seconds:]
        if not window:
            window = list(self._history)

        # Average over window
        avg = {
            "avg_DA": sum(s.get("da", 0) for s in window) / len(window),
            "avg_NE": sum(s.get("ne", 0) for s in window) / len(window),
            "avg_ACh": sum(s.get("ach", 0) for s in window) / len(window),
            "avg_5HT": sum(s.get("sht", 0) for s in window) / len(window),
        }

        # Peak in window (captures spikes that have since decayed)
        peaks = {
            "peak_DA": max(s.get("da", 0) for s in window),
            "peak_NE": max(s.get("ne", 0) for s in window),
            "peak_ACh": max(s.get("ach", 0) for s in window),
            "peak_5HT": max(s.get("sht", 0) for s in window),
        }

        # Recent (last 30s) — what's happening NOW vs the trend
        recent_window = window[-30:] if len(window) > 30 else window
        recent = {
            "recent_DA": sum(s.get("da", 0) for s in recent_window) / len(recent_window),
            "recent_NE": sum(s.get("ne", 0) for s in recent_window) / len(recent_window),
            "recent_ACh": sum(s.get("ach", 0) for s in recent_window) / len(recent_window),
            "recent_5HT": sum(s.get("sht", 0) for s in recent_window) / len(recent_window),
        }

        return {**avg, **peaks, **recent}

    def _empty(self) -> dict[str, Any]:
        """Return default state when no history is available."""
        return {
            "flow": False, "flow_duration_min": 0, "flow_app": None,
            "stress": False, "meeting": False, "meeting_duration_min": 0,
            "needs_break": False, "active_minutes": 0,
        }
