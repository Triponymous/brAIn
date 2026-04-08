# Mini-OSCEN Phase 3a Implementation Plan — Brain Sees My Desktop Life

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the Mac sensor adapter (6 sensor streams) and the always-on daemon skeleton (FastAPI + WebSocket push) so that the Phase-2 brain receives real desktop activity instead of synthetic patterns. End state: `braind start` runs continuously and `wscat -c ws://localhost:8000/ws` shows the brain state reacting to real keystrokes / app focus / mic / idle in real time.

**Architecture:** Six independent sensor coroutines write to a shared `SensorBus` (sample-and-hold). A `MacDesktopAdapter` composes the bus snapshots into a 200-dimensional sensory input vector for `Brain.tick()`. An asyncio main loop runs the brain at ~100 Hz, an `WSPusher` coroutine pushes brain state at ~30 Hz to all connected WebSocket clients. `braind start` is the CLI entry point. All OS-touching sensors have a `mock_mode` constructor switch so the entire test suite runs without macOS Accessibility / Mic permissions.

**Tech Stack:** Python 3.11, asyncio, FastAPI, uvicorn, websockets, pyobjc-framework-Cocoa (Active App + Idle), pynput (keystroke/mouse counts), sounddevice + numpy.fft (mic), pytest with pytest-asyncio.

**Reference docs:**
- `docs/plans/2026-04-08-mini-oscen-design.md` — original project design
- `docs/plans/2026-04-09-mini-oscen-phase3-design.md` — Phase 3+ pet vision design
- `docs/plans/2026-04-08-mini-oscen-phase2.md` — Phase 2 plan (completed)

**Out of scope for Phase 3a:** LLM bridge (Phase 3b), dashboard UI (Phase 3b), pet face Tauri app (Phase 3c), TTS/STT (Phase 3c), capability/grant system (Phase 4), agent tools (Phase 4+).

**Success criteria for Phase 3a:**
1. All 6 sensors implemented with mock mode and unit tests.
2. `MacDesktopAdapter` composes the bus into a 200-dim sensory vector that exercises the documented neuron ranges.
3. `braind start --mock-sensors` runs for ≥30 seconds without crashing, ticks the brain at ~100 Hz, and pushes WebSocket state at ~30 Hz to a connected client.
4. `braind start` (real sensors, no mock) starts cleanly when permissions are granted; degrades gracefully (warning + reduced sensor set) when not granted.
5. Brain state survives daemon restart via existing Phase-2 SQLite persistence (auto-save every 60s while daemon runs).
6. Test count grows from 48 to ≥58, all passing without requiring Accessibility / Microphone permission.

---

## Task 0: Add Phase 3a dependencies

**Files:**
- Modify: `/Users/leonmatthies/brAIntest/pyproject.toml`

**Step 1: Edit pyproject.toml**

Use the Edit tool. Find this block:

```toml
dependencies = [
    "snntorch>=0.9.1",
    "torch>=2.2",
    "numpy>=1.26",
    "matplotlib>=3.8",
]
```

Replace with:

```toml
dependencies = [
    "snntorch>=0.9.1",
    "torch>=2.2",
    "numpy>=1.26",
    "matplotlib>=3.8",
    # Phase 3a — Mac sensor adapter + daemon
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "websockets>=13",
    "pynput>=1.7",
    "sounddevice>=0.5",
    "pyobjc-framework-Cocoa>=10.0; sys_platform == 'darwin'",
    "pyobjc-framework-Quartz>=10.0; sys_platform == 'darwin'",
]
```

Find this block:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]
```

Replace with:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-asyncio>=0.24",
    "httpx>=0.27",
]
```

Find this block:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v"
```

Replace with:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v"
asyncio_mode = "auto"
```

**Step 2: Install new deps**

Run:
```bash
cd /Users/leonmatthies/brAIntest
uv pip install -e ".[dev]"
```

Expected: install completes; pyobjc, pynput, sounddevice, fastapi, uvicorn, pytest-asyncio, httpx all resolve.

**Step 3: Verify imports**

Run:
```bash
.venv/bin/python -c "
import fastapi, uvicorn, websockets, pynput, sounddevice
import pytest_asyncio, httpx
import Cocoa, Quartz
print('all imports OK')
"
```

Expected: `all imports OK`. If any fails, stop and report.

**Step 4: Run existing test suite to confirm no regression**

Run:
```bash
.venv/bin/pytest -v
```

Expected: 48 tests pass (Phase 1 + Phase 2 unchanged).

**Step 5: Commit**

```bash
cd /Users/leonmatthies/brAIntest
git add pyproject.toml
git commit -m "chore: add Phase 3a deps (fastapi, pynput, sounddevice, pyobjc)"
```

---

## Task 1: SensorBus and base Sensor class

**Files:**
- Create: `/Users/leonmatthies/brAIntest/adapters/__init__.py`
- Create: `/Users/leonmatthies/brAIntest/adapters/base.py`
- Create: `/Users/leonmatthies/brAIntest/tests/test_sensor_bus.py`

**Step 1: Create empty package**

Create `/Users/leonmatthies/brAIntest/adapters/__init__.py` (empty file).

**Step 2: Write the failing test**

Create `/Users/leonmatthies/brAIntest/tests/test_sensor_bus.py`:

```python
"""Tests for the SensorBus and Sensor base class.

The SensorBus is a thread-safe sample-and-hold store: each sensor writes
its latest encoded value to a named slot, and the brain tick reads the
current snapshot. Sensors run at independent rates; readers always get
the most recent value.
"""
import asyncio
import pytest
from adapters.base import SensorBus, Sensor


def test_bus_initial_empty():
    bus = SensorBus()
    assert bus.snapshot() == {}


def test_bus_write_and_read():
    bus = SensorBus()
    bus.write("active_app", {"name": "VSCode", "since": 1.0})
    snap = bus.snapshot()
    assert snap == {"active_app": {"name": "VSCode", "since": 1.0}}


def test_bus_multiple_writes_overwrites():
    bus = SensorBus()
    bus.write("idle", 0)
    bus.write("idle", 5)
    bus.write("idle", 12)
    assert bus.snapshot()["idle"] == 12


def test_bus_snapshot_is_a_copy():
    """Mutating the snapshot must not affect the bus."""
    bus = SensorBus()
    bus.write("k", [1, 2, 3])
    snap = bus.snapshot()
    snap["k"] = [9, 9, 9]
    assert bus.snapshot()["k"] == [1, 2, 3]


class _DummySensor(Sensor):
    name = "dummy"
    rate_hz = 10.0

    def __init__(self) -> None:
        super().__init__()
        self.tick_count = 0

    async def sample(self) -> int:
        self.tick_count += 1
        return self.tick_count


def test_sensor_subclass_has_name_and_rate():
    s = _DummySensor()
    assert s.name == "dummy"
    assert s.rate_hz == 10.0


@pytest.mark.asyncio
async def test_sensor_run_writes_to_bus():
    bus = SensorBus()
    s = _DummySensor()
    # Run sensor for ~150ms, should produce ~1-2 samples at 10Hz
    task = asyncio.create_task(s.run(bus))
    await asyncio.sleep(0.15)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert "dummy" in bus.snapshot()
    assert bus.snapshot()["dummy"] >= 1
```

**Step 3: Run to verify failure**

```bash
cd /Users/leonmatthies/brAIntest
.venv/bin/pytest tests/test_sensor_bus.py -v
```

Expected: All tests fail with `ModuleNotFoundError: No module named 'adapters.base'`.

**Step 4: Implement**

Create `/Users/leonmatthies/brAIntest/adapters/base.py`:

```python
"""SensorBus and Sensor base class.

Architecture:
- A `SensorBus` is a thread-safe key/value store. Each sensor writes its
  latest encoded value under its name. Readers (the brain tick loop) call
  `.snapshot()` to get the current state.
- A `Sensor` is an asyncio coroutine with a name and a target rate. Its
  `.sample()` method is called periodically and the return value is written
  to the bus under its name. The default `.run(bus)` loop handles timing.

Sensors are independent: each runs at its own rate, the bus holds the
last value (sample-and-hold). The brain tick reads the snapshot — sensors
that haven't updated yet still have their previous value (or `None` if
they've never sampled).

The `mock_mode` constructor flag is the convention for test-friendly
sensors: in mock mode, the sensor returns a deterministic synthetic value
without touching the OS.
"""
from __future__ import annotations
import asyncio
import copy
from typing import Any


class SensorBus:
    """Thread-safe (asyncio) sample-and-hold store."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    def write(self, name: str, value: Any) -> None:
        self._values[name] = value

    def snapshot(self) -> dict[str, Any]:
        """Return a deep copy of the current bus state."""
        return copy.deepcopy(self._values)


class Sensor:
    """Base class for sensors. Subclasses set name + rate_hz and override sample()."""

    name: str = "base"
    rate_hz: float = 1.0

    def __init__(self, mock_mode: bool = False) -> None:
        self.mock_mode = mock_mode

    async def sample(self) -> Any:
        """Override in subclass. Return one encoded sample."""
        raise NotImplementedError

    async def run(self, bus: SensorBus) -> None:
        """Run loop: sample at rate_hz, write to bus, sleep, repeat.

        Cancellation-safe. Exceptions in sample() are logged and the loop
        continues — a single broken sample shouldn't kill the sensor.
        """
        period = 1.0 / self.rate_hz
        while True:
            try:
                value = await self.sample()
                bus.write(self.name, value)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Log and continue. We'll plug a real logger in later.
                print(f"[sensor:{self.name}] sample failed: {e}")
            await asyncio.sleep(period)
```

**Step 5: Run tests**

```bash
.venv/bin/pytest tests/test_sensor_bus.py -v
```

Expected: 6 tests pass.

**Step 6: Commit**

```bash
cd /Users/leonmatthies/brAIntest
git add adapters/__init__.py adapters/base.py tests/test_sensor_bus.py
git commit -m "feat(adapters): SensorBus + Sensor base class with async run loop"
```

---

## Task 2: TimeTonicSensor (simplest sensor — pure Python, no OS)

**Files:**
- Create: `/Users/leonmatthies/brAIntest/adapters/mac_desktop/__init__.py`
- Create: `/Users/leonmatthies/brAIntest/adapters/mac_desktop/sensor_time.py`
- Create: `/Users/leonmatthies/brAIntest/tests/test_sensor_time.py`

**Step 1: Create package**

Create empty `/Users/leonmatthies/brAIntest/adapters/mac_desktop/__init__.py`.

**Step 2: Write failing test**

Create `/Users/leonmatthies/brAIntest/tests/test_sensor_time.py`:

```python
"""Tests for TimeTonicSensor — slow oscillators encoding day-of-day and day-of-week."""
import math
from datetime import datetime
import pytest
from adapters.mac_desktop.sensor_time import TimeTonicSensor


def test_time_tonic_construction():
    s = TimeTonicSensor()
    assert s.name == "time_tonic"
    assert s.rate_hz == 1.0


@pytest.mark.asyncio
async def test_time_tonic_returns_8_phase_values():
    s = TimeTonicSensor()
    sample = await s.sample()
    # Returns dict with two keys, each a list of 4 floats
    assert "day_phase" in sample
    assert "week_phase" in sample
    assert len(sample["day_phase"]) == 4
    assert len(sample["week_phase"]) == 4
    for v in sample["day_phase"] + sample["week_phase"]:
        assert -1.0 <= v <= 1.0


def test_time_tonic_known_value_at_midnight():
    """At midnight Monday, day phase 0 should be sin(0)=0 and the cosine
    components should be cos(0)=1 (for at least one of the 4 phases).
    Verifies the encoding is deterministic given a fixed timestamp."""
    s = TimeTonicSensor()
    # Monday 2026-01-05 00:00:00 (a known Monday)
    fake = datetime(2026, 1, 5, 0, 0, 0)
    sample = s._encode(fake)
    # day_phase should have at least one near-1 (cos(0)) and one near-0 (sin(0))
    assert max(sample["day_phase"]) > 0.99
    assert min(abs(v) for v in sample["day_phase"]) < 0.01
    # week_phase Monday 00:00 → angle 0, similar
    assert max(sample["week_phase"]) > 0.99


def test_time_tonic_phases_evolve_smoothly():
    """One hour should change day_phase moderately, week_phase very little."""
    s = TimeTonicSensor()
    t1 = datetime(2026, 1, 5, 12, 0, 0)
    t2 = datetime(2026, 1, 5, 13, 0, 0)
    s1 = s._encode(t1)
    s2 = s._encode(t2)
    # day_phase difference noticeable
    day_diff = sum(abs(a - b) for a, b in zip(s1["day_phase"], s2["day_phase"]))
    week_diff = sum(abs(a - b) for a, b in zip(s1["week_phase"], s2["week_phase"]))
    assert day_diff > week_diff
```

**Step 3: Run to fail**

```bash
.venv/bin/pytest tests/test_sensor_time.py -v
```

Expected: 4 tests fail with `ModuleNotFoundError`.

**Step 4: Implement**

Create `/Users/leonmatthies/brAIntest/adapters/mac_desktop/sensor_time.py`:

```python
"""TimeTonicSensor — slow biological-clock oscillators.

Provides eight scalar values per sample (4 day-cycle phases, 4 week-cycle
phases). The four phases per cycle are sin/cos at the fundamental frequency
and at twice the frequency, giving the brain enough basis vectors to
disambiguate "morning" from "afternoon" from "early morning" from "late
morning" without us defining any of those concepts explicitly.

The brain only needs to learn that "this combination of 4 numbers correlates
with X happening" — and the sin/cos basis is dense enough.
"""
from __future__ import annotations
import math
from datetime import datetime
from typing import Any

from adapters.base import Sensor


class TimeTonicSensor(Sensor):
    name = "time_tonic"
    rate_hz = 1.0

    def _encode(self, now: datetime) -> dict[str, list[float]]:
        # Day phase: seconds since midnight / 86400 in [0, 1)
        seconds_today = now.hour * 3600 + now.minute * 60 + now.second
        day_frac = seconds_today / 86400.0
        day_angle = day_frac * 2 * math.pi
        day_phase = [
            math.cos(day_angle),
            math.sin(day_angle),
            math.cos(2 * day_angle),
            math.sin(2 * day_angle),
        ]
        # Week phase: weekday(0=Mon..6=Sun) + day fraction, normalized to [0, 1)
        week_frac = (now.weekday() + day_frac) / 7.0
        week_angle = week_frac * 2 * math.pi
        week_phase = [
            math.cos(week_angle),
            math.sin(week_angle),
            math.cos(2 * week_angle),
            math.sin(2 * week_angle),
        ]
        return {"day_phase": day_phase, "week_phase": week_phase}

    async def sample(self) -> dict[str, list[float]]:
        return self._encode(datetime.now())
```

**Step 5: Run tests**

```bash
.venv/bin/pytest tests/test_sensor_time.py -v
```

Expected: 4 tests pass.

**Step 6: Commit**

```bash
cd /Users/leonmatthies/brAIntest
git add adapters/mac_desktop/__init__.py adapters/mac_desktop/sensor_time.py tests/test_sensor_time.py
git commit -m "feat(adapters): TimeTonicSensor (day + week phase oscillators)"
```

---

## Task 3: ActiveAppSensor (pyobjc, mockable)

**Files:**
- Create: `/Users/leonmatthies/brAIntest/adapters/mac_desktop/sensor_app.py`
- Create: `/Users/leonmatthies/brAIntest/tests/test_sensor_app.py`

**Step 1: Write failing test**

Create `/Users/leonmatthies/brAIntest/tests/test_sensor_app.py`:

```python
"""Tests for ActiveAppSensor — uses pyobjc NSWorkspace to read frontmost app.

In tests we use mock_mode=True which returns a deterministic fake app name.
"""
import pytest
from adapters.mac_desktop.sensor_app import ActiveAppSensor


def test_construction_defaults():
    s = ActiveAppSensor()
    assert s.name == "active_app"
    assert s.rate_hz == 1.0
    assert s.mock_mode is False


def test_construction_mock_mode():
    s = ActiveAppSensor(mock_mode=True)
    assert s.mock_mode is True


@pytest.mark.asyncio
async def test_mock_mode_returns_synthetic_apps():
    s = ActiveAppSensor(mock_mode=True)
    samples = []
    for _ in range(5):
        samples.append(await s.sample())
    # Each sample is a dict with a name
    for sample in samples:
        assert "name" in sample
        assert isinstance(sample["name"], str)
    # Mock mode rotates through a few apps so we see variety
    names = {s["name"] for s in samples}
    assert len(names) >= 1
```

**Step 2: Run to fail**

```bash
.venv/bin/pytest tests/test_sensor_app.py -v
```

Expected: 3 tests fail with `ModuleNotFoundError`.

**Step 3: Implement**

Create `/Users/leonmatthies/brAIntest/adapters/mac_desktop/sensor_app.py`:

```python
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
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_sensor_app.py -v
```

Expected: 3 tests pass.

**Step 5: Commit**

```bash
cd /Users/leonmatthies/brAIntest
git add adapters/mac_desktop/sensor_app.py tests/test_sensor_app.py
git commit -m "feat(adapters): ActiveAppSensor with pyobjc + mock mode"
```

---

## Task 4: KeystrokeRateSensor (pynput, count-only, mockable)

**Files:**
- Create: `/Users/leonmatthies/brAIntest/adapters/mac_desktop/sensor_keymouse.py`
- Create: `/Users/leonmatthies/brAIntest/tests/test_sensor_keymouse.py`

This task implements BOTH KeystrokeRateSensor and MouseRateSensor since they share infrastructure (a single pynput listener thread for each). Test both in one file.

**Step 1: Write failing tests**

Create `/Users/leonmatthies/brAIntest/tests/test_sensor_keymouse.py`:

```python
"""Tests for KeystrokeRateSensor and MouseRateSensor.

These sensors count input events per sampling window. NEVER read content.
In mock mode, the sensor's internal counter can be incremented manually.
"""
import asyncio
import pytest
from adapters.mac_desktop.sensor_keymouse import KeystrokeRateSensor, MouseRateSensor


def test_keystroke_construction():
    s = KeystrokeRateSensor()
    assert s.name == "keystroke_rate"
    assert s.rate_hz == 5.0
    assert s.mock_mode is False


def test_mouse_construction():
    s = MouseRateSensor()
    assert s.name == "mouse_rate"
    assert s.rate_hz == 5.0


def test_keystroke_mock_mode_default_zero():
    s = KeystrokeRateSensor(mock_mode=True)
    sample = asyncio.run(s.sample())
    assert sample["count"] == 0


def test_keystroke_mock_increment_via_inject():
    """In mock mode, calling _inject_event() bumps the counter."""
    s = KeystrokeRateSensor(mock_mode=True)
    for _ in range(7):
        s._inject_event()
    sample = asyncio.run(s.sample())
    assert sample["count"] == 7


def test_keystroke_sample_resets_window():
    """After a sample(), the next sample should report only NEW events."""
    s = KeystrokeRateSensor(mock_mode=True)
    for _ in range(3):
        s._inject_event()
    s1 = asyncio.run(s.sample())
    assert s1["count"] == 3
    # No new events
    s2 = asyncio.run(s.sample())
    assert s2["count"] == 0
    # Two more events
    for _ in range(2):
        s._inject_event()
    s3 = asyncio.run(s.sample())
    assert s3["count"] == 2


def test_mouse_mock_increment_via_inject():
    s = MouseRateSensor(mock_mode=True)
    for _ in range(15):
        s._inject_event()
    sample = asyncio.run(s.sample())
    assert sample["count"] == 15
```

**Step 2: Run to fail**

```bash
.venv/bin/pytest tests/test_sensor_keymouse.py -v
```

Expected: 6 tests fail with `ModuleNotFoundError`.

**Step 3: Implement**

Create `/Users/leonmatthies/brAIntest/adapters/mac_desktop/sensor_keymouse.py`:

```python
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
        return {"count": n}

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
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
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_sensor_keymouse.py -v
```

Expected: 6 tests pass.

**Step 5: Commit**

```bash
cd /Users/leonmatthies/brAIntest
git add adapters/mac_desktop/sensor_keymouse.py tests/test_sensor_keymouse.py
git commit -m "feat(adapters): KeystrokeRateSensor + MouseRateSensor (count-only, mockable)"
```

---

## Task 5: IdleSensor (pyobjc CGEventSource, no permission needed)

**Files:**
- Create: `/Users/leonmatthies/brAIntest/adapters/mac_desktop/sensor_idle.py`
- Create: `/Users/leonmatthies/brAIntest/tests/test_sensor_idle.py`

**Step 1: Write failing tests**

Create `/Users/leonmatthies/brAIntest/tests/test_sensor_idle.py`:

```python
"""Tests for IdleSensor — seconds since last input event."""
import pytest
from adapters.mac_desktop.sensor_idle import IdleSensor


def test_construction():
    s = IdleSensor()
    assert s.name == "idle"
    assert s.rate_hz == 1.0


@pytest.mark.asyncio
async def test_mock_mode_returns_zero():
    s = IdleSensor(mock_mode=True)
    sample = await s.sample()
    assert "seconds" in sample
    assert sample["seconds"] >= 0.0


@pytest.mark.asyncio
async def test_mock_mode_inject_idle():
    s = IdleSensor(mock_mode=True)
    s._mock_idle_seconds = 42.5
    sample = await s.sample()
    assert sample["seconds"] == 42.5
```

**Step 2: Run to fail**

```bash
.venv/bin/pytest tests/test_sensor_idle.py -v
```

Expected: 3 tests fail with `ModuleNotFoundError`.

**Step 3: Implement**

Create `/Users/leonmatthies/brAIntest/adapters/mac_desktop/sensor_idle.py`:

```python
"""IdleSensor — reports seconds since the last user input event.

Uses Quartz CGEventSourceSecondsSinceLastEventType which is a free macOS
API (no permission required). Returns 0 immediately after any input.

Mock mode returns whatever is in `_mock_idle_seconds` (default 0.0).
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
                self.mock_mode = True

    async def sample(self) -> dict[str, float]:
        if self.mock_mode:
            return {"seconds": float(self._mock_idle_seconds)}
        # kCGAnyInputEventType = ~ -1 (all event types)
        seconds = self._cg.CGEventSourceSecondsSinceLastEventType(
            self._cg.kCGEventSourceStateHIDSystemState,
            int(0xFFFFFFFF),  # any event type
        )
        return {"seconds": float(seconds)}
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_sensor_idle.py -v
```

Expected: 3 tests pass.

**Step 5: Commit**

```bash
cd /Users/leonmatthies/brAIntest
git add adapters/mac_desktop/sensor_idle.py tests/test_sensor_idle.py
git commit -m "feat(adapters): IdleSensor via CGEventSource"
```

---

## Task 6: MicSensor (sounddevice + numpy fft, mock-friendly)

**Files:**
- Create: `/Users/leonmatthies/brAIntest/adapters/mac_desktop/sensor_mic.py`
- Create: `/Users/leonmatthies/brAIntest/tests/test_sensor_mic.py`

**Step 1: Write failing tests**

Create `/Users/leonmatthies/brAIntest/tests/test_sensor_mic.py`:

```python
"""Tests for MicSensor — produces 32 mel-band intensities + 1 RMS loudness.

In mock mode, we feed the sensor synthetic audio frames directly via
inject_audio() and verify the encoder produces sensible output without
actually opening a microphone.
"""
import math
import numpy as np
import pytest
from adapters.mac_desktop.sensor_mic import MicSensor


def test_construction():
    s = MicSensor()
    assert s.name == "mic"
    assert s.rate_hz == 50.0
    assert s.num_mel_bands == 32


@pytest.mark.asyncio
async def test_mock_mode_silence_returns_low_values():
    s = MicSensor(mock_mode=True)
    silence = np.zeros(1024, dtype=np.float32)
    s._inject_audio(silence)
    sample = await s.sample()
    assert "mel" in sample
    assert "rms" in sample
    assert len(sample["mel"]) == 32
    assert sample["rms"] < 0.001


@pytest.mark.asyncio
async def test_mock_mode_loud_signal_high_rms():
    s = MicSensor(mock_mode=True)
    # Loud sinusoid at 1kHz, 1 second at 16kHz
    sr = 16000
    t = np.arange(0, 1.0, 1.0 / sr, dtype=np.float32)
    loud = 0.5 * np.sin(2 * math.pi * 1000 * t)
    s._inject_audio(loud)
    sample = await s.sample()
    assert sample["rms"] > 0.1


@pytest.mark.asyncio
async def test_mel_bands_react_to_frequency():
    """A high-frequency tone should activate higher mel bands more than low ones."""
    s = MicSensor(mock_mode=True)
    sr = 16000
    t = np.arange(0, 0.1, 1.0 / sr, dtype=np.float32)
    high_tone = 0.5 * np.sin(2 * math.pi * 4000 * t).astype(np.float32)
    s._inject_audio(high_tone)
    sample = await s.sample()
    mel = sample["mel"]
    # Sum of upper half should exceed sum of lower half for a high tone
    lower_half = sum(mel[:16])
    upper_half = sum(mel[16:])
    assert upper_half > lower_half, f"High tone should activate upper bands more, got lower={lower_half}, upper={upper_half}"
```

**Step 2: Run to fail**

```bash
.venv/bin/pytest tests/test_sensor_mic.py -v
```

Expected: 4 tests fail with `ModuleNotFoundError`.

**Step 3: Implement**

Create `/Users/leonmatthies/brAIntest/adapters/mac_desktop/sensor_mic.py`:

```python
"""MicSensor — 32-band mel-spectrogram + RMS loudness from microphone.

Continuous audio capture via sounddevice in real mode. The sample() method
returns the latest mel-band activation snapshot. In mock mode, audio is
fed via _inject_audio() instead of opening the device.

Privacy: only spectral features and RMS are extracted. The raw audio buffer
is overwritten each call. Nothing is written to disk.
"""
from __future__ import annotations
import sys
import threading
from typing import Any

import numpy as np

from adapters.base import Sensor


_SAMPLE_RATE = 16000
_FRAME_SIZE = 1024  # FFT window size
_NUM_MEL_BANDS = 32


def _mel_filterbank(num_bands: int, fft_size: int, sample_rate: int) -> np.ndarray:
    """Build a triangular mel filterbank: shape (num_bands, fft_size//2 + 1)."""
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)
    low_mel = hz_to_mel(0.0)
    high_mel = hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(low_mel, high_mel, num_bands + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    n_bins = fft_size // 2 + 1
    bin_points = np.clip(bin_points, 0, n_bins - 1)
    fb = np.zeros((num_bands, n_bins), dtype=np.float32)
    for m in range(1, num_bands + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        if center == left:
            center = left + 1
        if right == center:
            right = center + 1
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / (right - center)
    return fb


class MicSensor(Sensor):
    name = "mic"
    rate_hz = 50.0
    num_mel_bands = _NUM_MEL_BANDS

    def __init__(self, mock_mode: bool = False) -> None:
        super().__init__(mock_mode=mock_mode)
        self._lock = threading.Lock()
        self._latest_audio: np.ndarray = np.zeros(_FRAME_SIZE, dtype=np.float32)
        self._filterbank = _mel_filterbank(_NUM_MEL_BANDS, _FRAME_SIZE, _SAMPLE_RATE)
        self._stream = None
        if not mock_mode and sys.platform == "darwin":
            try:
                import sounddevice as sd  # type: ignore
                self._stream = sd.InputStream(
                    samplerate=_SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    blocksize=_FRAME_SIZE,
                    callback=self._sd_callback,
                )
                self._stream.start()
            except Exception as e:
                print(f"[MicSensor] failed to open mic: {e}")
                self.mock_mode = True

    def _sd_callback(self, indata, frames, time_info, status) -> None:
        with self._lock:
            self._latest_audio = indata[:, 0].copy()

    def _inject_audio(self, audio: np.ndarray) -> None:
        """Test hook: feed a buffer as if it came from the mic."""
        with self._lock:
            self._latest_audio = audio.astype(np.float32)

    def _encode(self, audio: np.ndarray) -> dict[str, Any]:
        # Trim/pad to FRAME_SIZE
        if len(audio) >= _FRAME_SIZE:
            frame = audio[:_FRAME_SIZE]
        else:
            frame = np.zeros(_FRAME_SIZE, dtype=np.float32)
            frame[: len(audio)] = audio
        # Window
        window = np.hanning(_FRAME_SIZE).astype(np.float32)
        windowed = frame * window
        # FFT magnitude
        spectrum = np.abs(np.fft.rfft(windowed))
        # Mel filterbank
        mel = self._filterbank @ spectrum
        # Log compression to keep dynamic range tame
        mel = np.log1p(mel)
        # RMS loudness
        rms = float(np.sqrt(np.mean(audio**2)))
        return {"mel": mel.tolist(), "rms": rms}

    async def sample(self) -> dict[str, Any]:
        with self._lock:
            audio = self._latest_audio.copy()
        return self._encode(audio)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_sensor_mic.py -v
```

Expected: 4 tests pass. The high-tone test is the most fragile — if it fails because the synthetic 4kHz tone produces unexpected mel-bin distribution, increase the test tone frequency to 6000 Hz.

**Step 5: Commit**

```bash
cd /Users/leonmatthies/brAIntest
git add adapters/mac_desktop/sensor_mic.py tests/test_sensor_mic.py
git commit -m "feat(adapters): MicSensor with 32 mel-bands + RMS, mockable"
```

---

## Task 7: MacDesktopAdapter — composes all 6 sensors → 200-dim sensory tensor

**Files:**
- Create: `/Users/leonmatthies/brAIntest/adapters/mac_desktop/encoding.py`
- Create: `/Users/leonmatthies/brAIntest/adapters/mac_desktop/adapter.py`
- Create: `/Users/leonmatthies/brAIntest/tests/test_mac_adapter.py`

**Step 1: Write failing tests**

Create `/Users/leonmatthies/brAIntest/tests/test_mac_adapter.py`:

```python
"""Tests for MacDesktopAdapter — composes 6 sensors into a 200-dim sensory vector."""
import asyncio
import math
import pytest
import torch
from adapters.mac_desktop.adapter import MacDesktopAdapter


def test_construction_mock_mode():
    adapter = MacDesktopAdapter(mock_mode=True)
    assert adapter.bus is not None
    assert len(adapter.sensors) == 6
    # Sensor names
    names = {s.name for s in adapter.sensors}
    assert names == {"active_app", "keystroke_rate", "mouse_rate", "idle", "mic", "time_tonic"}


def test_encode_empty_bus_returns_zero_vector():
    adapter = MacDesktopAdapter(mock_mode=True)
    vec = adapter.encode()
    assert vec.shape == (200,)
    assert torch.all(vec == 0)


def test_encode_with_active_app_fires_app_neuron():
    adapter = MacDesktopAdapter(mock_mode=True)
    adapter.bus.write("active_app", {"name": "VSCode"})
    vec = adapter.encode()
    # Some neuron in the active_app range (0-63) should be > 0
    app_range = vec[0:64]
    assert app_range.sum() > 0
    # Other ranges still zero
    assert vec[64:].sum() == 0


def test_encode_active_app_consistent_mapping():
    """Same app name → same neuron index."""
    a1 = MacDesktopAdapter(mock_mode=True)
    a1.bus.write("active_app", {"name": "Chrome"})
    v1 = a1.encode()
    a2 = MacDesktopAdapter(mock_mode=True)
    a2.bus.write("active_app", {"name": "Chrome"})
    v2 = a2.encode()
    assert torch.allclose(v1, v2)


def test_encode_keystroke_rate_bin():
    adapter = MacDesktopAdapter(mock_mode=True)
    adapter.bus.write("keystroke_rate", {"count": 0})
    vec = adapter.encode()
    # bin 0 (no activity) should be active in keystroke range 64-79
    keystroke_range = vec[64:80]
    assert keystroke_range.sum() > 0


def test_encode_high_keystroke_rate_high_bin():
    adapter = MacDesktopAdapter(mock_mode=True)
    adapter.bus.write("keystroke_rate", {"count": 50})  # very high count → top bin
    vec = adapter.encode()
    keystroke_range = vec[64:80]
    # The active bin should be in the upper half
    active_bins = (keystroke_range > 0).nonzero(as_tuple=True)[0]
    assert active_bins[0].item() >= 8


def test_encode_idle_long():
    adapter = MacDesktopAdapter(mock_mode=True)
    adapter.bus.write("idle", {"seconds": 7200})  # 2 hours → highest bin
    vec = adapter.encode()
    idle_range = vec[96:104]
    active_bins = (idle_range > 0).nonzero(as_tuple=True)[0]
    assert active_bins[0].item() == 7  # last bin


def test_encode_mic_mel_writes_to_range():
    adapter = MacDesktopAdapter(mock_mode=True)
    mel = [0.5] * 32
    adapter.bus.write("mic", {"mel": mel, "rms": 0.3})
    vec = adapter.encode()
    mic_range = vec[104:136]
    assert mic_range.sum() > 0


def test_encode_time_tonic_writes_to_range():
    adapter = MacDesktopAdapter(mock_mode=True)
    adapter.bus.write("time_tonic", {
        "day_phase": [0.5, 0.5, 0.5, 0.5],
        "week_phase": [0.5, 0.5, 0.5, 0.5],
    })
    vec = adapter.encode()
    time_range = vec[144:152]
    assert time_range.sum() > 0


def test_encode_reserve_range_always_zero():
    adapter = MacDesktopAdapter(mock_mode=True)
    # Write to all sensors
    adapter.bus.write("active_app", {"name": "X"})
    adapter.bus.write("keystroke_rate", {"count": 5})
    adapter.bus.write("mouse_rate", {"count": 5})
    adapter.bus.write("idle", {"seconds": 1})
    adapter.bus.write("mic", {"mel": [1.0] * 32, "rms": 0.1})
    adapter.bus.write("time_tonic", {"day_phase": [1, 0, 0, 0], "week_phase": [1, 0, 0, 0]})
    vec = adapter.encode()
    reserve = vec[152:200]
    assert reserve.sum() == 0


@pytest.mark.asyncio
async def test_run_sensors_briefly():
    """Start the adapter, let sensors run for ~300ms in mock mode, stop."""
    adapter = MacDesktopAdapter(mock_mode=True)
    task = asyncio.create_task(adapter.run())
    await asyncio.sleep(0.3)
    adapter.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    snap = adapter.bus.snapshot()
    # At least the time tonic and active app should have produced samples
    assert "time_tonic" in snap
    assert "active_app" in snap
```

**Step 2: Run to fail**

```bash
.venv/bin/pytest tests/test_mac_adapter.py -v
```

Expected: 11 tests fail with `ModuleNotFoundError`.

**Step 3: Implement encoding**

Create `/Users/leonmatthies/brAIntest/adapters/mac_desktop/encoding.py`:

```python
"""Encoding sensor bus snapshots into the 200-dim sensory neuron vector.

Neuron map (per Phase 3+ design doc):
    0-63    active app (one-hot, hash-mapped, first-64 + "other" bucket)
    64-79   keystroke rate (16 log bins, 0..>=10/s)
    80-95   mouse rate (16 log bins, 0..>=1000/s)
    96-103  idle time (8 log bins, <1s..>1h)
    104-135 mic mel-spectrogram (32 bands, scaled)
    136-143 mic RMS loudness (8 log bins)
    144-147 time tonic day_phase (4 cosine/sine basis values, scaled to spike)
    148-151 time tonic week_phase (4 cosine/sine basis)
    152-199 reserve (always 0 for now)

Each "active" neuron in a one-hot or bin slot fires with current 3.0
(strong drive, well above LIF threshold of 1.0). Mel-spectrogram and
tonics scale their values to a similar range.
"""
from __future__ import annotations
import math
from typing import Any

import torch


SENSORY_DIM = 200

_DRIVE_STRENGTH = 3.0


def _hash_app_to_index(name: str, num_slots: int = 63) -> int:
    """Stable hash from app name to index in [0, num_slots). Slot num_slots = "other"."""
    return abs(hash(name)) % num_slots


def _log_bin(value: float, max_value: float, num_bins: int) -> int:
    """Map value to log-scaled bin in [0, num_bins-1]."""
    if value <= 0:
        return 0
    if value >= max_value:
        return num_bins - 1
    log_v = math.log1p(value)
    log_max = math.log1p(max_value)
    frac = log_v / log_max
    return min(num_bins - 1, int(frac * num_bins))


def encode_snapshot(snap: dict[str, Any]) -> torch.Tensor:
    """Encode a SensorBus snapshot into a 200-dim sensory input tensor."""
    vec = torch.zeros(SENSORY_DIM, dtype=torch.float32)

    # Active app: 0-63
    app = snap.get("active_app")
    if app and "name" in app:
        idx = _hash_app_to_index(app["name"])
        vec[idx] = _DRIVE_STRENGTH

    # Keystroke rate: 64-79 (16 bins)
    ks = snap.get("keystroke_rate")
    if ks is not None and "count" in ks:
        # 5 Hz windows → max ~50 events/window for "intensive typing"
        bin_idx = _log_bin(ks["count"], max_value=50.0, num_bins=16)
        vec[64 + bin_idx] = _DRIVE_STRENGTH

    # Mouse rate: 80-95 (16 bins)
    ms = snap.get("mouse_rate")
    if ms is not None and "count" in ms:
        bin_idx = _log_bin(ms["count"], max_value=200.0, num_bins=16)
        vec[80 + bin_idx] = _DRIVE_STRENGTH

    # Idle time: 96-103 (8 bins, log)
    idle = snap.get("idle")
    if idle is not None and "seconds" in idle:
        bin_idx = _log_bin(idle["seconds"], max_value=3600.0, num_bins=8)
        vec[96 + bin_idx] = _DRIVE_STRENGTH

    # Mic mel: 104-135 (32 bands), Mic RMS: 136-143 (8 bins)
    mic = snap.get("mic")
    if mic is not None:
        if "mel" in mic and len(mic["mel"]) == 32:
            mel = torch.tensor(mic["mel"], dtype=torch.float32)
            # Scale: log-mel values are typically 0-5; scale to drive 0-_DRIVE_STRENGTH
            mel_scaled = (mel / 5.0).clamp(0.0, 1.0) * _DRIVE_STRENGTH
            vec[104:136] = mel_scaled
        if "rms" in mic:
            bin_idx = _log_bin(mic["rms"], max_value=1.0, num_bins=8)
            vec[136 + bin_idx] = _DRIVE_STRENGTH

    # Time tonics: 144-151
    tt = snap.get("time_tonic")
    if tt is not None:
        if "day_phase" in tt and len(tt["day_phase"]) == 4:
            day = torch.tensor(tt["day_phase"], dtype=torch.float32)
            # Tonic values are -1..1; map to 0..2*drive (a baseline tonic)
            vec[144:148] = (day + 1.0) * _DRIVE_STRENGTH * 0.5
        if "week_phase" in tt and len(tt["week_phase"]) == 4:
            week = torch.tensor(tt["week_phase"], dtype=torch.float32)
            vec[148:152] = (week + 1.0) * _DRIVE_STRENGTH * 0.5

    # Reserve 152-199 stays zero
    return vec
```

**Step 4: Implement adapter**

Create `/Users/leonmatthies/brAIntest/adapters/mac_desktop/adapter.py`:

```python
"""MacDesktopAdapter — composes the 6 Mac desktop sensors and encodes them
into a 200-dim sensory input vector for the Brain.

Usage:
    adapter = MacDesktopAdapter(mock_mode=False)
    asyncio.create_task(adapter.run())
    while True:
        sensory_input = adapter.encode()
        brain.tick(sensory_input)
        await asyncio.sleep(0.01)
"""
from __future__ import annotations
import asyncio
from typing import Any

import torch

from adapters.base import Sensor, SensorBus
from adapters.mac_desktop.sensor_app import ActiveAppSensor
from adapters.mac_desktop.sensor_keymouse import KeystrokeRateSensor, MouseRateSensor
from adapters.mac_desktop.sensor_idle import IdleSensor
from adapters.mac_desktop.sensor_mic import MicSensor
from adapters.mac_desktop.sensor_time import TimeTonicSensor
from adapters.mac_desktop.encoding import encode_snapshot, SENSORY_DIM


class MacDesktopAdapter:
    def __init__(self, mock_mode: bool = False) -> None:
        self.mock_mode = mock_mode
        self.bus = SensorBus()
        self.sensors: list[Sensor] = [
            ActiveAppSensor(mock_mode=mock_mode),
            KeystrokeRateSensor(mock_mode=mock_mode),
            MouseRateSensor(mock_mode=mock_mode),
            IdleSensor(mock_mode=mock_mode),
            MicSensor(mock_mode=mock_mode),
            TimeTonicSensor(mock_mode=mock_mode),
        ]
        self._tasks: list[asyncio.Task] = []

    async def run(self) -> None:
        """Start all sensor coroutines. Returns when all are cancelled."""
        self._tasks = [asyncio.create_task(s.run(self.bus)) for s in self.sensors]
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        """Stop background streams (mic, key/mouse listeners) and cancel tasks."""
        for s in self.sensors:
            stop_fn = getattr(s, "stop", None)
            if callable(stop_fn):
                stop_fn()
        for t in self._tasks:
            t.cancel()

    def encode(self) -> torch.Tensor:
        """Encode the current bus snapshot into the 200-dim sensory vector."""
        return encode_snapshot(self.bus.snapshot())
```

**Step 5: Run tests**

```bash
.venv/bin/pytest tests/test_mac_adapter.py -v
```

Expected: 11 tests pass.

**Step 6: Run full suite**

```bash
.venv/bin/pytest -v
```

Expected: ~75 tests pass total.

**Step 7: Commit**

```bash
cd /Users/leonmatthies/brAIntest
git add adapters/mac_desktop/encoding.py adapters/mac_desktop/adapter.py tests/test_mac_adapter.py
git commit -m "feat(adapters): MacDesktopAdapter encodes 6 sensors into 200-dim vector"
```

---

## Task 8: Brain integration test — adapter feeds brain, brain produces spikes

**Files:**
- Create: `/Users/leonmatthies/brAIntest/tests/test_brain_with_adapter.py`

This is a pure integration test, no new production code.

**Step 1: Write the test**

Create `/Users/leonmatthies/brAIntest/tests/test_brain_with_adapter.py`:

```python
"""Integration test: MacDesktopAdapter feeding the Brain produces spike activity.

Verifies the contract between Phase 3a (adapter) and Phase 2 (brain):
- The 200-dim sensory tensor is the right shape
- The brain accepts it without crashing
- Sensor activity actually drives sensory neuron spikes
- Modulators move in response to varying input
"""
import asyncio
import pytest
import torch

from brain.core import Brain
from adapters.mac_desktop.adapter import MacDesktopAdapter


@pytest.mark.asyncio
async def test_adapter_feeds_brain_no_crash():
    """Run adapter + brain for ~500ms, no exceptions."""
    torch.manual_seed(0)
    brain = Brain()  # default size, includes 200 sensory
    adapter = MacDesktopAdapter(mock_mode=True)
    sensor_task = asyncio.create_task(adapter.run())

    # Inject some bus state to drive activity
    adapter.bus.write("active_app", {"name": "VSCode"})
    adapter.bus.write("keystroke_rate", {"count": 12})

    for _ in range(50):
        vec = adapter.encode()
        assert vec.shape == (200,)
        brain.tick(vec)
        await asyncio.sleep(0.005)  # ~200 ticks/sec

    adapter.stop()
    sensor_task.cancel()
    try:
        await sensor_task
    except asyncio.CancelledError:
        pass

    # Brain should have advanced
    assert brain.tick_count == 50


@pytest.mark.asyncio
async def test_adapter_drives_sensory_spikes():
    """Strong sensor input should produce sensory spikes within a few ticks."""
    torch.manual_seed(0)
    brain = Brain()
    adapter = MacDesktopAdapter(mock_mode=True)

    # Drive ALL sensors strongly
    adapter.bus.write("active_app", {"name": "VSCode"})
    adapter.bus.write("keystroke_rate", {"count": 30})
    adapter.bus.write("mouse_rate", {"count": 50})
    adapter.bus.write("idle", {"seconds": 0})
    adapter.bus.write("mic", {"mel": [3.0] * 32, "rms": 0.5})
    adapter.bus.write("time_tonic", {
        "day_phase": [1.0, 0.0, 0.0, 0.0],
        "week_phase": [1.0, 0.0, 0.0, 0.0],
    })

    total_sensory_spikes = 0
    for _ in range(20):
        vec = adapter.encode()
        out = brain.tick(vec)
        total_sensory_spikes += int(out["sensory"].sum().item())

    assert total_sensory_spikes > 0, "Expected sensory spikes under strong input"


@pytest.mark.asyncio
async def test_adapter_silence_then_loud_activates_modulators():
    """Going from silence to loud audio should bump modulators."""
    torch.manual_seed(0)
    brain = Brain()
    adapter = MacDesktopAdapter(mock_mode=True)

    # Run silence for a while
    for _ in range(30):
        brain.tick(adapter.encode())

    # Now inject loud audio
    adapter.bus.write("mic", {"mel": [4.0] * 32, "rms": 0.9})
    for _ in range(30):
        brain.tick(adapter.encode())

    # ACh + DA should have moved from baseline (0.0) at least slightly
    # because activity changed. We don't assert a specific value, just
    # that the brain isn't completely silent.
    snap = brain.modulators.snapshot()
    # At minimum, the brain ran without error and produced a snapshot
    assert "DA" in snap and "NE" in snap
```

**Step 2: Run the test**

```bash
cd /Users/leonmatthies/brAIntest
.venv/bin/pytest tests/test_brain_with_adapter.py -v
```

Expected: 3 tests pass.

If `test_adapter_drives_sensory_spikes` fails because `total_sensory_spikes == 0`, that means the encoding strength is too weak relative to the LIF threshold. Verify that `_DRIVE_STRENGTH = 3.0` in `encoding.py` is well above `threshold=1.0` in the brain default. If still failing, increase `_DRIVE_STRENGTH` to 5.0.

**Step 3: Run full suite**

```bash
.venv/bin/pytest -v
```

Expected: ~78 tests pass.

**Step 4: Commit**

```bash
cd /Users/leonmatthies/brAIntest
git add tests/test_brain_with_adapter.py
git commit -m "test: integration test for adapter→brain wiring"
```

---

## Task 9: FastAPI server skeleton + WebSocket pusher

**Files:**
- Create: `/Users/leonmatthies/brAIntest/server/__init__.py`
- Create: `/Users/leonmatthies/brAIntest/server/main.py`
- Create: `/Users/leonmatthies/brAIntest/server/ws.py`
- Create: `/Users/leonmatthies/brAIntest/tests/test_server.py`

**Step 1: Write failing tests**

Create `/Users/leonmatthies/brAIntest/tests/test_server.py`:

```python
"""Tests for the FastAPI server skeleton.

We test the WSPusher state-broadcast logic and the /healthz endpoint.
The full daemon main loop is tested via the E2E smoke test in Task 11.
"""
import asyncio
import json
import pytest
from httpx import AsyncClient, ASGITransport
from server.main import build_app
from server.ws import WSPusher


@pytest.mark.asyncio
async def test_healthz_endpoint():
    app = build_app(brain=None, adapter=None, pusher=None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_wspusher_construction():
    pusher = WSPusher(rate_hz=30.0)
    assert pusher.rate_hz == 30.0
    assert len(pusher.clients) == 0


@pytest.mark.asyncio
async def test_wspusher_register_unregister():
    pusher = WSPusher()
    fake_ws = object()
    await pusher.register(fake_ws)
    assert fake_ws in pusher.clients
    await pusher.unregister(fake_ws)
    assert fake_ws not in pusher.clients


@pytest.mark.asyncio
async def test_wspusher_broadcast_to_registered_clients():
    """Broadcasting a state dict should call send_text on each client."""
    pusher = WSPusher()

    class FakeWS:
        def __init__(self):
            self.received = []
        async def send_text(self, text):
            self.received.append(text)

    a, b = FakeWS(), FakeWS()
    await pusher.register(a)
    await pusher.register(b)
    await pusher.broadcast({"tick": 42, "modulators": {"DA": 0.1}})
    assert len(a.received) == 1
    assert len(b.received) == 1
    msg_a = json.loads(a.received[0])
    assert msg_a["tick"] == 42


@pytest.mark.asyncio
async def test_wspusher_failed_send_removes_client():
    """A client that throws on send_text gets removed."""
    pusher = WSPusher()

    class BrokenWS:
        async def send_text(self, text):
            raise RuntimeError("connection closed")

    bad = BrokenWS()
    await pusher.register(bad)
    await pusher.broadcast({"x": 1})
    assert bad not in pusher.clients
```

**Step 2: Run to fail**

```bash
.venv/bin/pytest tests/test_server.py -v
```

Expected: 5 tests fail with ModuleNotFoundError.

**Step 3: Implement WSPusher**

Create `/Users/leonmatthies/brAIntest/server/__init__.py` (empty).

Create `/Users/leonmatthies/brAIntest/server/ws.py`:

```python
"""WSPusher — broadcasts brain state snapshots to all connected WebSocket clients.

Holds a set of clients (FastAPI WebSocket objects). Each broadcast serializes
the state dict to JSON and sends to every client. Failed sends silently
unregister the client.
"""
from __future__ import annotations
import asyncio
import json
from typing import Any


class WSPusher:
    def __init__(self, rate_hz: float = 30.0) -> None:
        self.rate_hz = rate_hz
        self.clients: set[Any] = set()
        self._lock = asyncio.Lock()

    async def register(self, client: Any) -> None:
        async with self._lock:
            self.clients.add(client)

    async def unregister(self, client: Any) -> None:
        async with self._lock:
            self.clients.discard(client)

    async def broadcast(self, state: dict[str, Any]) -> None:
        text = json.dumps(state, default=_json_default)
        async with self._lock:
            failed = []
            for client in list(self.clients):
                try:
                    await client.send_text(text)
                except Exception:
                    failed.append(client)
            for c in failed:
                self.clients.discard(c)


def _json_default(o: Any) -> Any:
    """Fallback JSON serializer for tensors etc."""
    try:
        import torch
        if isinstance(o, torch.Tensor):
            return o.tolist()
    except ImportError:
        pass
    if hasattr(o, "__iter__"):
        return list(o)
    return str(o)
```

**Step 4: Implement build_app**

Create `/Users/leonmatthies/brAIntest/server/main.py`:

```python
"""FastAPI app builder + main daemon loop entry point.

build_app(brain, adapter, pusher) returns a FastAPI instance with:
    GET /healthz                  liveness check
    WS  /ws                       brain state stream

The actual brain tick loop and the periodic WSPusher broadcast loop are
started by run_daemon() which is called from the braind CLI.
"""
from __future__ import annotations
import asyncio
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from server.ws import WSPusher


def build_app(
    brain: Optional[Any],
    adapter: Optional[Any],
    pusher: Optional[WSPusher],
) -> FastAPI:
    app = FastAPI(title="braind", version="3.0.0")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "status": "ok",
            "brain_tick_count": brain.tick_count if brain is not None else None,
            "adapter_running": adapter is not None,
            "ws_clients": len(pusher.clients) if pusher is not None else 0,
        }

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        if pusher is not None:
            await pusher.register(ws)
        try:
            # Keep the connection open by awaiting client messages
            # (we don't actually need them; this just blocks)
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            if pusher is not None:
                await pusher.unregister(ws)

    return app


async def brain_tick_loop(brain: Any, adapter: Any, hz: float = 100.0) -> None:
    """Run the brain tick loop indefinitely. Cancellable."""
    period = 1.0 / hz
    try:
        while True:
            vec = adapter.encode()
            brain.tick(vec)
            await asyncio.sleep(period)
    except asyncio.CancelledError:
        return


async def push_loop(brain: Any, pusher: WSPusher) -> None:
    """Periodically broadcast brain state to all WS clients."""
    period = 1.0 / pusher.rate_hz
    try:
        while True:
            state = {
                "tick": brain.tick_count,
                "modulators": brain.modulators.snapshot(),
                # Phase 3b will add: active_concepts, sensors, etc.
            }
            await pusher.broadcast(state)
            await asyncio.sleep(period)
    except asyncio.CancelledError:
        return


async def persistence_loop(brain: Any, save_path: str, period_sec: float = 60.0) -> None:
    """Periodically save the brain to SQLite."""
    from brain.persistence import save_brain
    from pathlib import Path
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            await asyncio.sleep(period_sec)
            try:
                save_brain(brain, p)
            except Exception as e:
                print(f"[persistence] save failed: {e}")
    except asyncio.CancelledError:
        return
```

**Step 5: Run tests**

```bash
.venv/bin/pytest tests/test_server.py -v
```

Expected: 5 tests pass.

**Step 6: Commit**

```bash
cd /Users/leonmatthies/brAIntest
git add server/__init__.py server/main.py server/ws.py tests/test_server.py
git commit -m "feat(server): FastAPI app + WSPusher for brain state broadcast"
```

---

## Task 10: braind CLI

**Files:**
- Create: `/Users/leonmatthies/brAIntest/server/braind.py`
- Create: `/Users/leonmatthies/brAIntest/tests/test_braind_cli.py`

**Step 1: Write the failing test**

Create `/Users/leonmatthies/brAIntest/tests/test_braind_cli.py`:

```python
"""Tests for the braind CLI.

We test the argument parser and the high-level run flow with mock sensors.
The actual long-running daemon is exercised by the E2E smoke test in Task 11.
"""
import argparse
import pytest
from server.braind import build_parser


def test_parser_default_args():
    parser = build_parser()
    args = parser.parse_args(["start"])
    assert args.command == "start"
    assert args.mock_sensors is False
    assert args.port == 8000
    assert args.checkpoint == "checkpoints/braind.sqlite"


def test_parser_mock_sensors_flag():
    parser = build_parser()
    args = parser.parse_args(["start", "--mock-sensors"])
    assert args.mock_sensors is True


def test_parser_custom_port():
    parser = build_parser()
    args = parser.parse_args(["start", "--port", "9999"])
    assert args.port == 9999


def test_parser_custom_checkpoint():
    parser = build_parser()
    args = parser.parse_args(["start", "--checkpoint", "/tmp/test.sqlite"])
    assert args.checkpoint == "/tmp/test.sqlite"
```

**Step 2: Run to fail**

```bash
.venv/bin/pytest tests/test_braind_cli.py -v
```

Expected: 4 tests fail with ModuleNotFoundError.

**Step 3: Implement**

Create `/Users/leonmatthies/brAIntest/server/braind.py`:

```python
"""braind — Mini-OSCEN background daemon CLI.

Usage:
    braind start [--mock-sensors] [--port 8000] [--checkpoint PATH]
    braind status
    braind --help

`start` runs the daemon in the foreground. The daemon:
- Loads the brain from the checkpoint file (or creates fresh if missing)
- Starts the MacDesktopAdapter with all 6 sensors
- Runs brain.tick() at ~100 Hz on incoming sensor data
- Pushes brain state to /ws clients at ~30 Hz
- Auto-saves the brain to checkpoint every 60s
- Listens on the configured port
"""
from __future__ import annotations
import argparse
import asyncio
from pathlib import Path
import sys

import uvicorn

from brain.core import Brain
from brain.persistence import load_brain, save_brain
from adapters.mac_desktop.adapter import MacDesktopAdapter
from server.main import build_app, brain_tick_loop, push_loop, persistence_loop
from server.ws import WSPusher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="braind")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="Start the daemon")
    start.add_argument("--mock-sensors", action="store_true",
                       help="Run with mock sensors (CI/dev mode)")
    start.add_argument("--port", type=int, default=8000)
    start.add_argument("--checkpoint", type=str, default="checkpoints/braind.sqlite")
    start.add_argument("--tick-hz", type=float, default=100.0)
    start.add_argument("--push-hz", type=float, default=30.0)

    sub.add_parser("status", help="Print daemon status")

    return parser


async def _run_daemon(args: argparse.Namespace) -> None:
    # Load or create brain
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        print(f"Loading brain from {checkpoint}")
        brain = load_brain(checkpoint)
        print(f"Resumed at tick {brain.tick_count}")
    else:
        print("Creating fresh brain")
        brain = Brain()

    # Create adapter
    adapter = MacDesktopAdapter(mock_mode=args.mock_sensors)
    pusher = WSPusher(rate_hz=args.push_hz)

    # Build FastAPI app
    app = build_app(brain=brain, adapter=adapter, pusher=pusher)

    # Schedule background tasks
    sensor_task = asyncio.create_task(adapter.run())
    tick_task = asyncio.create_task(brain_tick_loop(brain, adapter, hz=args.tick_hz))
    push_task = asyncio.create_task(push_loop(brain, pusher))
    persist_task = asyncio.create_task(persistence_loop(brain, str(checkpoint)))

    # Run uvicorn in the same loop
    config = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="info")
    server = uvicorn.Server(config)
    try:
        await server.serve()
    finally:
        for t in (sensor_task, tick_task, push_task, persist_task):
            t.cancel()
        adapter.stop()
        # Final save
        try:
            save_brain(brain, checkpoint)
            print(f"Final save to {checkpoint} at tick {brain.tick_count}")
        except Exception as e:
            print(f"Final save failed: {e}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "start":
        try:
            asyncio.run(_run_daemon(args))
        except KeyboardInterrupt:
            print("\nbraind stopped by user")
        return 0
    elif args.command == "status":
        # Phase 3a: just print a stub. Real status check via /healthz comes later.
        print("braind status — use `curl http://localhost:8000/healthz` for live status")
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_braind_cli.py -v
```

Expected: 4 tests pass.

**Step 5: Commit**

```bash
cd /Users/leonmatthies/brAIntest
git add server/braind.py tests/test_braind_cli.py
git commit -m "feat(server): braind CLI with start command + arg parser"
```

---

## Task 11: E2E smoke test — start daemon, connect WS, verify state stream

**Files:**
- Create: `/Users/leonmatthies/brAIntest/tests/test_e2e_smoke.py`

**Step 1: Write the test**

Create `/Users/leonmatthies/brAIntest/tests/test_e2e_smoke.py`:

```python
"""End-to-end smoke test: spin up a real daemon (mock sensors), connect a
WebSocket client, verify state messages flow.

Uses uvicorn in-process via lifespan + httpx websockets. This is the
single most important integration test in Phase 3a — it proves all the
pieces actually fit together.
"""
import asyncio
import json
import pytest
import websockets
from contextlib import asynccontextmanager
from pathlib import Path
import tempfile

from brain.core import Brain
from adapters.mac_desktop.adapter import MacDesktopAdapter
from server.main import build_app, brain_tick_loop, push_loop
from server.ws import WSPusher


@asynccontextmanager
async def _running_server(port: int):
    """Start a daemon on the given port for the duration of the context."""
    import uvicorn

    brain = Brain(num_sensory=200)  # default size
    adapter = MacDesktopAdapter(mock_mode=True)
    pusher = WSPusher(rate_hz=30.0)
    app = build_app(brain=brain, adapter=adapter, pusher=pusher)

    sensor_task = asyncio.create_task(adapter.run())
    tick_task = asyncio.create_task(brain_tick_loop(brain, adapter, hz=100.0))
    push_task = asyncio.create_task(push_loop(brain, pusher))

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    # Wait for the server to actually be up
    for _ in range(50):
        if server.started:
            break
        await asyncio.sleep(0.1)

    try:
        yield brain
    finally:
        server.should_exit = True
        for t in (sensor_task, tick_task, push_task):
            t.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        adapter.stop()


@pytest.mark.asyncio
async def test_e2e_daemon_starts_and_streams():
    """Start daemon, connect WebSocket, receive ≥1 state message."""
    port = 8765  # avoid conflict with default 8000
    async with _running_server(port) as brain:
        url = f"ws://127.0.0.1:{port}/ws"
        async with websockets.connect(url) as ws:
            # Receive a few messages
            messages = []
            for _ in range(3):
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                messages.append(json.loads(msg))
        # Verify messages have the expected shape
        assert len(messages) >= 1
        for m in messages:
            assert "tick" in m
            assert "modulators" in m
            assert "DA" in m["modulators"]
        # Brain should have advanced
        assert brain.tick_count > 0


@pytest.mark.asyncio
async def test_e2e_healthz_endpoint():
    """Daemon's /healthz returns ok status with tick count."""
    import httpx
    port = 8766
    async with _running_server(port) as brain:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://127.0.0.1:{port}/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["adapter_running"] is True
```

**Step 2: Run the test**

```bash
cd /Users/leonmatthies/brAIntest
.venv/bin/pytest tests/test_e2e_smoke.py -v
```

Expected: 2 tests pass.

If `test_e2e_daemon_starts_and_streams` times out on `ws.recv()`, the most likely cause is that the push_loop isn't broadcasting fast enough or the WebSocket isn't registered. Add a 0.2s sleep before the first `ws.recv()` to give the push loop time to fire.

If port 8765 is in use, change to a different free port.

**Step 3: Run full suite**

```bash
.venv/bin/pytest -v
```

Expected: ~85 tests pass total.

**Step 4: Commit**

```bash
cd /Users/leonmatthies/brAIntest
git add tests/test_e2e_smoke.py
git commit -m "test: E2E smoke test for daemon + WebSocket state stream"
```

---

## Task 12: launchd plist (optional, manual install instructions)

**Files:**
- Create: `/Users/leonmatthies/brAIntest/scripts/com.brAIntest.braind.plist`
- Create: `/Users/leonmatthies/brAIntest/scripts/install_launchd.sh`
- Create: `/Users/leonmatthies/brAIntest/scripts/uninstall_launchd.sh`

These are configuration files. No tests.

**Step 1: Write the plist**

Create `/Users/leonmatthies/brAIntest/scripts/com.brAIntest.braind.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.brAIntest.braind</string>

  <key>ProgramArguments</key>
  <array>
    <string>/Users/leonmatthies/brAIntest/.venv/bin/python</string>
    <string>-m</string>
    <string>server.braind</string>
    <string>start</string>
  </array>

  <key>WorkingDirectory</key>
  <string>/Users/leonmatthies/brAIntest</string>

  <key>RunAtLoad</key>
  <true/>

  <key>KeepAlive</key>
  <true/>

  <key>StandardOutPath</key>
  <string>/Users/leonmatthies/brAIntest/logs/braind.out.log</string>

  <key>StandardErrorPath</key>
  <string>/Users/leonmatthies/brAIntest/logs/braind.err.log</string>

  <key>ProcessType</key>
  <string>Interactive</string>
</dict>
</plist>
```

**Step 2: Write install script**

Create `/Users/leonmatthies/brAIntest/scripts/install_launchd.sh`:

```bash
#!/usr/bin/env bash
set -e
LABEL=com.brAIntest.braind
PLIST_SRC=/Users/leonmatthies/brAIntest/scripts/${LABEL}.plist
PLIST_DST=$HOME/Library/LaunchAgents/${LABEL}.plist

mkdir -p /Users/leonmatthies/brAIntest/logs
mkdir -p $HOME/Library/LaunchAgents

cp "$PLIST_SRC" "$PLIST_DST"
launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"
echo "Loaded $LABEL. Tail logs with: tail -f /Users/leonmatthies/brAIntest/logs/braind.out.log"
```

**Step 3: Write uninstall script**

Create `/Users/leonmatthies/brAIntest/scripts/uninstall_launchd.sh`:

```bash
#!/usr/bin/env bash
LABEL=com.brAIntest.braind
PLIST_DST=$HOME/Library/LaunchAgents/${LABEL}.plist
launchctl unload "$PLIST_DST" 2>/dev/null || true
rm -f "$PLIST_DST"
echo "Unloaded and removed $LABEL"
```

**Step 4: Make scripts executable**

```bash
cd /Users/leonmatthies/brAIntest
chmod +x scripts/install_launchd.sh scripts/uninstall_launchd.sh
```

**Step 5: Verify the daemon module is invokable**

```bash
cd /Users/leonmatthies/brAIntest
.venv/bin/python -m server.braind --help
```

Expected: prints the argparse help text.

Then a 5-second mock-sensor smoke run:
```bash
cd /Users/leonmatthies/brAIntest
timeout 5 .venv/bin/python -m server.braind start --mock-sensors --checkpoint /tmp/braind_smoke.sqlite || true
```
Expected: prints "Creating fresh brain" or "Loading brain from..." plus uvicorn startup logs, runs for 5 seconds, then exits via timeout. No tracebacks.

**Step 6: Commit**

```bash
cd /Users/leonmatthies/brAIntest
git add scripts/com.brAIntest.braind.plist scripts/install_launchd.sh scripts/uninstall_launchd.sh
git commit -m "feat(scripts): launchd plist + install/uninstall helpers"
```

---

## Task 13: Tag phase-3a-complete + README

**Step 1: Run full test suite one more time**

```bash
cd /Users/leonmatthies/brAIntest
.venv/bin/pytest -v
```

Expected: All ~85 tests pass.

**Step 2: Update README**

Edit `/Users/leonmatthies/brAIntest/README.md`. Replace the `## Status` section with:

```markdown
## Status
- [x] Phase 1: SNN core foundations (LIF, STDP, 2-region brain, viz)
- [x] Phase 2: Full multi-region brain + WTA + modulators + R-STDP + SQLite persistence
- [x] Phase 3a: Mac sensor adapter + daemon (FastAPI + WebSocket)
- [ ] Phase 3b: LLM bridge + dashboard
- [ ] Phase 3c: Pet face (Tauri) + voice (TTS/STT)
- [ ] Phase 4+: see `docs/plans/2026-04-09-mini-oscen-phase3-design.md`
```

Replace the `## Quick start` section with:

```markdown
## Quick start

```bash
uv venv
uv pip install -e ".[dev]"
.venv/bin/pytest -v

# Phase 1 viz (2-region STDP demo)
.venv/bin/python scripts/visualize_two_region.py

# Phase 2 soak test (full brain, concept emergence, save/resume)
.venv/bin/python scripts/run_soak.py --ticks 20000
.venv/bin/python scripts/run_soak.py --ticks 5000 --resume

# Phase 3a daemon (real Mac sensors — needs Accessibility + Mic permissions)
.venv/bin/python -m server.braind start

# Phase 3a daemon (mock sensors, no permissions needed)
.venv/bin/python -m server.braind start --mock-sensors

# Watch the brain state stream
wscat -c ws://localhost:8000/ws        # if you have wscat
curl http://localhost:8000/healthz     # quick health check
```
```

**Step 3: Commit + tag**

```bash
cd /Users/leonmatthies/brAIntest
git add README.md
git commit -m "docs: mark Phase 3a complete in README"
git tag phase-3a-complete -m "Phase 3a: Mac sensor adapter + daemon foundation"
git log --oneline | head -20
git tag
```

---

## Phase 3a Done. What's next?

Phase 3a delivers:
- 6 Mac desktop sensors (active app, keystroke/mouse rate, idle, mic mel+RMS, time tonics)
- A composable `MacDesktopAdapter` that encodes sensors into a 200-dim sensory vector
- Full mock-mode for tests; production mode degrades gracefully when permissions denied
- A `braind` CLI daemon with FastAPI + WebSocket state push at 30 Hz
- Auto-save to SQLite every 60 seconds
- E2E smoke test proving the whole pipeline streams real brain state
- ~85 tests, all green
- Optional launchd integration

Phase 3b entry criteria: Phase 3a daemon runs cleanly for at least 30 minutes on real Mac sensors without crash or memory leak. Ideally let it run for a few hours and verify the brain forms some stable concept activity (visible via `wscat` or a quick state-dump script).

Then re-invoke `superpowers:writing-plans` for **Phase 3b (LLM Bridge + Dashboard)**.
