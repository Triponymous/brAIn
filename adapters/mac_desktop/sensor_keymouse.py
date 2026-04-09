"""KeystrokeRateSensor and MouseRateSensor — Quartz counter-based.

Uses CGEventSourceCounterForEventType which reads system-wide HID event
counters. This is the SAME low-level API that IdleSensor uses and does NOT
require the pynput library. However, it DOES require the host terminal app
to have "Input Monitoring" permission in macOS System Settings.

The sensor polls the counter each sample() call and computes the delta
since the last call. Also tracks typing RHYTHM features (burst detection,
inter-event variability) that let the SNN distinguish "calm typing" from
"hektisches Tippen" — critical for stress detection.
"""
from __future__ import annotations
import sys
import time
from typing import Any

from adapters.base import Sensor


class _QuartzCounterSensor(Sensor):
    """Base class: reads a Quartz HID counter and reports delta per sample.

    Reports:
    - count: raw event count in this window
    - variability: coefficient of variation of inter-sample counts (0=steady, >1=erratic)
    - burst: 1 if current count >> recent average (sudden activity spike)

    FALLBACK: If Input Monitoring permission is not granted, Quartz counters
    freeze at their boot-time values and always return delta=0. After 20
    consecutive zero-delta samples, we print a warning.
    """

    name = "_quartz_counter"
    rate_hz = 5.0  # 5 Hz = 200ms windows

    # Subclasses set this to the Quartz event type constant
    _event_types: list[int] = []

    def __init__(self, mock_mode: bool = False) -> None:
        super().__init__(mock_mode=mock_mode)
        self._cg = None
        self._last_counts: list[int] = []
        self._zero_streak = 0
        self._using_idle_proxy = False
        self._mock_count: int = 0
        # Rolling window for rhythm detection (last 10 samples = 2 seconds)
        self._recent_counts: list[int] = []
        self._window_size = 10
        if not mock_mode and sys.platform == "darwin":
            try:
                import Quartz  # type: ignore
                self._cg = Quartz
                # Read initial counter values
                self._last_counts = [
                    self._cg.CGEventSourceCounterForEventType(
                        self._cg.kCGEventSourceStateHIDSystemState, et
                    )
                    for et in self._event_types
                ]
            except ImportError:
                self.mock_mode = True

    def _inject_event(self) -> None:
        """Test hook: simulate an input event in mock mode."""
        self._mock_count += 1

    def _compute_rhythm(self, count: int) -> dict[str, float]:
        """Compute typing rhythm features from rolling window."""
        self._recent_counts.append(count)
        if len(self._recent_counts) > self._window_size:
            self._recent_counts.pop(0)

        window = self._recent_counts
        n = len(window)
        if n < 3:
            return {"variability": 0.0, "burst": 0.0}

        mean = sum(window) / n
        if mean < 0.5:
            return {"variability": 0.0, "burst": 0.0}

        # Coefficient of variation: std/mean. High = erratic typing
        variance = sum((x - mean) ** 2 for x in window) / n
        std = variance ** 0.5
        variability = min(2.0, std / max(0.1, mean))

        # Burst detection: current count > 2x recent average
        recent_avg = sum(window[:-1]) / max(1, n - 1) if n > 1 else mean
        burst = 1.0 if count > recent_avg * 2 and count > 3 else 0.0

        return {"variability": round(variability, 3), "burst": burst}

    async def sample(self) -> dict[str, Any]:
        if self.mock_mode:
            count = self._mock_count
            self._mock_count = 0
            rhythm = self._compute_rhythm(count)
            return {"count": count, **rhythm}

        # Try Quartz counters first (requires Input Monitoring permission)
        if not self._using_idle_proxy:
            total_delta = 0
            new_counts = []
            for i, et in enumerate(self._event_types):
                current = self._cg.CGEventSourceCounterForEventType(
                    self._cg.kCGEventSourceStateHIDSystemState, et
                )
                new_counts.append(current)
                if i < len(self._last_counts):
                    delta = current - self._last_counts[i]
                    if delta > 0:
                        total_delta += delta
            self._last_counts = new_counts

            if total_delta > 0:
                self._zero_streak = 0
                rhythm = self._compute_rhythm(total_delta)
                return {"count": total_delta, **rhythm}

            self._zero_streak += 1
            if self._zero_streak >= 20:
                self._using_idle_proxy = True
                print(f"[{self.name}] Quartz counters frozen — switching to idle-time proxy "
                      f"(grant Input Monitoring permission to fix)")
            rhythm = self._compute_rhythm(0)
            return {"count": 0, **rhythm}

        rhythm = self._compute_rhythm(0)
        return {"count": 0, **rhythm}

    def stop(self) -> None:
        pass


class KeystrokeRateSensor(_QuartzCounterSensor):
    """Counts key-down events via Quartz HID counter + typing rhythm."""
    name = "keystroke_rate"

    def __init__(self, mock_mode: bool = False) -> None:
        # kCGEventKeyDown = 10
        self._event_types = [10]
        super().__init__(mock_mode=mock_mode)


class MouseRateSensor(_QuartzCounterSensor):
    """Counts mouse moves + clicks via Quartz HID counter + movement rhythm."""
    name = "mouse_rate"

    def __init__(self, mock_mode: bool = False) -> None:
        # kCGEventMouseMoved = 5, kCGEventLeftMouseDown = 1,
        # kCGEventRightMouseDown = 3, kCGEventScrollWheel = 22
        self._event_types = [5, 1, 3, 22]
        super().__init__(mock_mode=mock_mode)
