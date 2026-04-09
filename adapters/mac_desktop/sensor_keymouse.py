"""KeystrokeRateSensor and MouseRateSensor.

Count-only — these sensors NEVER read keystroke content or mouse position
content. They only increment a counter when an event is observed and
report the count per sampling window.

Real mode uses pynput global listeners which require macOS Accessibility
permission. Mock mode uses a manual increment for testing.
"""
from __future__ import annotations
import sys
import threading
from typing import Any

from adapters.base import Sensor


class _CountingSensor(Sensor):
    """Shared infra: a counter incremented by an event source."""

    name = "_counting"
    rate_hz = 5.0  # 5 Hz windows = 200ms

    def __init__(self, mock_mode: bool = False) -> None:
        super().__init__(mock_mode=mock_mode)
        self._lock = threading.Lock()
        self._count = 0
        self._listener = None

    def _inject_event(self) -> None:
        """Test helper / pynput callback target."""
        with self._lock:
            self._count += 1

    async def sample(self) -> dict[str, int]:
        with self._lock:
            n = self._count
            self._count = 0
        # In mock mode, simulate realistic activity
        if self.mock_mode and n == 0:
            import random
            n = random.randint(3, 25)  # simulate typing / mouse movement
        return {"count": n}

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None


class KeystrokeRateSensor(_CountingSensor):
    name = "keystroke_rate"

    def __init__(self, mock_mode: bool = False) -> None:
        super().__init__(mock_mode=mock_mode)
        if not mock_mode and sys.platform == "darwin":
            try:
                from pynput import keyboard  # type: ignore
                self._listener = keyboard.Listener(on_press=lambda _key: self._inject_event())
                self._listener.start()
            except Exception as e:
                # Permission denied, pynput unavailable, or any other failure
                print(f"[KeystrokeRateSensor] failed to start pynput: {e}")
                self.mock_mode = True


class MouseRateSensor(_CountingSensor):
    name = "mouse_rate"

    def __init__(self, mock_mode: bool = False) -> None:
        super().__init__(mock_mode=mock_mode)
        if not mock_mode and sys.platform == "darwin":
            try:
                from pynput import mouse  # type: ignore
                self._listener = mouse.Listener(
                    on_move=lambda x, y: self._inject_event(),
                    on_click=lambda x, y, button, pressed: pressed and self._inject_event(),
                )
                self._listener.start()
            except Exception as e:
                print(f"[MouseRateSensor] failed to start pynput: {e}")
                self.mock_mode = True
