"""KeystrokeRateSensor and MouseRateSensor.

Two strategies for input detection:
1. Quartz HID counters (preferred — no background threads, monotonic counters)
2. pynput listeners (fallback — background thread hooks global events)

The sensor tries Quartz first. If counters freeze (20 consecutive zeros =
permission issue), it falls back to pynput. pynput requires Accessibility
permission but is more reliable on macOS Sequoia.

Both strategies report: count per sample + rhythm features (variability, burst).
"""
from __future__ import annotations
import sys
import threading
from typing import Any

from adapters.base import Sensor


class _InputCounterSensor(Sensor):
    """Base class for keyboard/mouse rate sensors with dual-strategy detection."""

    name = "_input_counter"
    rate_hz = 5.0  # 5 Hz = 200ms windows

    # Subclasses set these
    _event_types: list[int] = []  # Quartz event type constants
    _pynput_type: str = ""  # "keyboard" or "mouse"

    def __init__(self, mock_mode: bool = False) -> None:
        super().__init__(mock_mode=mock_mode)
        self._cg = None
        self._last_counts: list[int] = []
        self._zero_streak = 0
        self._mock_count: int = 0
        self._strategy = "quartz"  # "quartz" or "pynput"

        # pynput fallback state
        self._pynput_count = 0
        self._pynput_lock = threading.Lock()
        self._pynput_listener = None

        # Rolling window for rhythm detection
        self._recent_counts: list[int] = []
        self._window_size = 10

        if not mock_mode and sys.platform == "darwin":
            try:
                import Quartz
                self._cg = Quartz
                self._last_counts = [
                    self._cg.CGEventSourceCounterForEventType(
                        self._cg.kCGEventSourceStateHIDSystemState, et
                    )
                    for et in self._event_types
                ]
            except ImportError:
                self._strategy = "pynput"
                self._start_pynput()

    def _start_pynput(self) -> None:
        """Start pynput listener as fallback."""
        if self._pynput_listener is not None:
            return
        try:
            if self._pynput_type == "keyboard":
                from pynput.keyboard import Listener
                self._pynput_listener = Listener(on_press=self._on_pynput_event)
            else:
                from pynput.mouse import Listener
                self._pynput_listener = Listener(
                    on_move=self._on_pynput_event,
                    on_click=self._on_pynput_event,
                    on_scroll=self._on_pynput_event,
                )
            self._pynput_listener.daemon = True
            self._pynput_listener.start()
            print(f"[{self.name}] Using pynput fallback (Quartz counters frozen)")
        except Exception as e:
            print(f"[{self.name}] pynput fallback failed: {e}")

    def _on_pynput_event(self, *args) -> None:
        with self._pynput_lock:
            self._pynput_count += 1

    def _inject_event(self) -> None:
        """Test hook: simulate an input event in mock mode."""
        self._mock_count += 1

    def _compute_rhythm(self, count: int) -> dict[str, float]:
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

        variance = sum((x - mean) ** 2 for x in window) / n
        std = variance ** 0.5
        variability = min(2.0, std / max(0.1, mean))

        recent_avg = sum(window[:-1]) / max(1, n - 1) if n > 1 else mean
        burst = 1.0 if count > recent_avg * 2 and count > 3 else 0.0

        return {"variability": round(variability, 3), "burst": burst}

    async def sample(self) -> dict[str, Any]:
        if self.mock_mode:
            count = self._mock_count
            self._mock_count = 0
            rhythm = self._compute_rhythm(count)
            return {"count": count, **rhythm}

        # Strategy 1: Quartz counters
        if self._strategy == "quartz":
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
                # Quartz frozen — switch to pynput permanently
                self._strategy = "pynput"
                self._start_pynput()

            rhythm = self._compute_rhythm(0)
            return {"count": 0, **rhythm}

        # Strategy 2: pynput listener
        with self._pynput_lock:
            count = self._pynput_count
            self._pynput_count = 0
        rhythm = self._compute_rhythm(count)
        return {"count": count, **rhythm}

    def stop(self) -> None:
        if self._pynput_listener is not None:
            try:
                self._pynput_listener.stop()
            except Exception:
                pass
            self._pynput_listener = None


class KeystrokeRateSensor(_InputCounterSensor):
    """Counts key-down events. Falls back to pynput if Quartz is frozen."""
    name = "keystroke_rate"

    def __init__(self, mock_mode: bool = False) -> None:
        self._event_types = [10]  # kCGEventKeyDown
        self._pynput_type = "keyboard"
        super().__init__(mock_mode=mock_mode)


class MouseRateSensor(_InputCounterSensor):
    """Counts mouse events. Falls back to pynput if Quartz is frozen."""
    name = "mouse_rate"

    def __init__(self, mock_mode: bool = False) -> None:
        self._event_types = [5, 1, 3, 22]  # move, left-click, right-click, scroll
        self._pynput_type = "mouse"
        super().__init__(mock_mode=mock_mode)
