#!/usr/bin/env python3
"""Live System Validator — end-to-end test against the running brAIn daemon.

Connects to the running daemon via WebSocket and HTTP, verifies:
1. Brain state stream delivers sensor data, spike counts, modulators, concepts
2. ConceptTracker assigns clusters (current_cluster != -1)
3. Chat API responds correctly (no emojis, no hallucinated capabilities)

Usage:
    .venv/bin/python scripts/validate_live_system.py [--host HOST] [--port PORT]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Results:
    tests: list[TestResult] = field(default_factory=list)

    def record(self, name: str, passed: bool, detail: str = "") -> None:
        self.tests.append(TestResult(name=name, passed=passed, detail=detail))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))

    @property
    def total(self) -> int:
        return len(self.tests)

    @property
    def passed(self) -> int:
        return sum(1 for t in self.tests if t.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def all_passed(self) -> bool:
        return self.failed == 0


# ---------------------------------------------------------------------------
# Emoji detection
# ---------------------------------------------------------------------------

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed chars
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols ext-A
    "\U00002600-\U000026FF"  # misc symbols
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # zero-width joiner
    "\U00002B50"             # star
    "\U0000203C-\U00003299"  # misc
    "]+",
    flags=re.UNICODE,
)


def _has_emoji(text: str) -> bool:
    return bool(_EMOJI_RE.search(text))


# ---------------------------------------------------------------------------
# Test 1: WebSocket brain state stream
# ---------------------------------------------------------------------------

async def test_brain_state_stream(host: str, port: int, results: Results) -> list[dict]:
    """Connect to WS, read 10 seconds of brain state, validate structure."""
    import websockets

    url = f"ws://{host}:{port}/ws"
    frames: list[dict] = []

    print(f"\n--- Test: Brain State Stream ({url}) ---")

    try:
        async with websockets.connect(url, open_timeout=5) as ws:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    frame = json.loads(raw)
                    frames.append(frame)
                except asyncio.TimeoutError:
                    break
    except Exception as e:
        results.record("ws_connect", False, f"Connection failed: {e}")
        return frames

    results.record("ws_connect", len(frames) > 0, f"received {len(frames)} frames in 10s")

    if not frames:
        # All subsequent tests depend on frames
        for name in ["sensors_present", "spike_counts_present", "sensory_spikes_gt0",
                      "modulators_present", "concepts_present"]:
            results.record(name, False, "no frames received")
        return frames

    # Use last frame for structure checks (most likely to have data)
    last = frames[-1]

    # --- Sensors present ---
    sensors = last.get("sensors", {})
    required_sensor_keys = {"keys", "mouse", "mic_rms", "idle", "app"}
    present_keys = set(sensors.keys()) & required_sensor_keys
    results.record(
        "sensors_present",
        present_keys == required_sensor_keys,
        f"found {sorted(present_keys)}, need {sorted(required_sensor_keys)}",
    )

    # --- Spike counts present and sensory > 0 ---
    spike_counts = last.get("spike_counts", {})
    has_spike_keys = all(k in spike_counts for k in ("sensory", "concept", "wm"))
    results.record(
        "spike_counts_present",
        has_spike_keys,
        f"keys={sorted(spike_counts.keys())}",
    )

    # Check if sensory spikes > 0 in ANY frame (not just the last one)
    any_sensory_spikes = any(
        f.get("spike_counts", {}).get("sensory", 0) > 0 for f in frames
    )
    results.record(
        "sensory_spikes_gt0",
        any_sensory_spikes,
        f"max sensory={max(f.get('spike_counts', {}).get('sensory', 0) for f in frames)}",
    )

    # --- Modulators present ---
    modulators = last.get("modulators", {})
    required_mods = {"DA", "NE", "ACh", "5HT"}
    present_mods = set(modulators.keys()) & required_mods
    results.record(
        "modulators_present",
        present_mods == required_mods,
        f"found {sorted(present_mods)}, need {sorted(required_mods)}",
    )

    # --- Concepts present with current_cluster ---
    concepts = last.get("concepts", {})
    has_current_cluster = "current_cluster" in concepts
    results.record(
        "concepts_present",
        has_current_cluster,
        f"current_cluster={'present' if has_current_cluster else 'MISSING'}, "
        f"keys={sorted(concepts.keys())[:6]}",
    )

    return frames


# ---------------------------------------------------------------------------
# Test 2: Chat API
# ---------------------------------------------------------------------------

async def test_chat_api(host: str, port: int, results: Results) -> None:
    """Send 3 chat messages, verify responses."""
    import httpx

    base_url = f"http://{host}:{port}"
    print(f"\n--- Test: Chat API ({base_url}/api/chat) ---")

    test_cases = [
        {
            "message": "hey was passiert gerade?",
            "check_name": "chat_basic_response",
            "check_fn": lambda text: len(text) > 5,
            "check_detail": lambda text: f"response length={len(text)}",
            "emoji_check": True,
        },
        {
            "message": "kannst du meinen Bildschirm sehen?",
            "check_name": "chat_no_hallucination",
            "check_fn": lambda text: any(
                w in text.lower() for w in ("nein", "nicht", "kann ich nicht", "sehe ich nicht", "kein")
            ),
            "check_detail": lambda text: f"expected denial, got: {text[:80]}...",
            "emoji_check": False,
        },
        {
            "message": "hallo \U0001F3AE",
            "check_name": "chat_no_emoji_response",
            "check_fn": lambda _text: True,  # just check emojis below
            "check_detail": lambda text: f"response: {text[:80]}...",
            "emoji_check": True,
        },
    ]

    async with httpx.AsyncClient(timeout=120.0) as client:
        for tc in test_cases:
            try:
                resp = await client.post(
                    f"{base_url}/api/chat",
                    json={"message": tc["message"]},
                )
                if resp.status_code != 200:
                    results.record(tc["check_name"], False, f"HTTP {resp.status_code}")
                    continue

                body = resp.json()
                text = body.get("text", "")

                # Primary check
                passed = tc["check_fn"](text)
                results.record(tc["check_name"], passed, tc["check_detail"](text))

                # Emoji check
                if tc["emoji_check"]:
                    has_emoji = _has_emoji(text)
                    results.record(
                        f"{tc['check_name']}_no_emoji",
                        not has_emoji,
                        f"emojis found in response" if has_emoji else "clean",
                    )

            except httpx.ConnectError as e:
                results.record(tc["check_name"], False, f"Connection failed: {e}")
            except httpx.ReadTimeout:
                results.record(tc["check_name"], False, "LLM response timed out (120s)")
            except Exception as e:
                results.record(tc["check_name"], False, f"Unexpected error: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Test 3: ConceptTracker cluster assignment
# ---------------------------------------------------------------------------

async def test_concept_cluster(host: str, port: int, results: Results) -> None:
    """Read WS for 10s, check if current_cluster != -1 at least once."""
    import websockets

    url = f"ws://{host}:{port}/ws"
    print(f"\n--- Test: ConceptTracker Cluster Assignment ---")

    cluster_values: list[int] = []

    try:
        async with websockets.connect(url, open_timeout=5) as ws:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    frame = json.loads(raw)
                    cluster = frame.get("concepts", {}).get("current_cluster", -1)
                    cluster_values.append(cluster)
                except asyncio.TimeoutError:
                    break
    except Exception as e:
        results.record("concept_cluster_assigned", False, f"WS error: {e}")
        return

    assigned = [c for c in cluster_values if c != -1]
    total = len(cluster_values)
    results.record(
        "concept_cluster_assigned",
        len(assigned) > 0,
        f"{len(assigned)}/{total} frames had cluster != -1"
        + (f", clusters seen: {sorted(set(assigned))}" if assigned else ""),
    )


# ---------------------------------------------------------------------------
# Healthcheck (pre-flight)
# ---------------------------------------------------------------------------

async def preflight_check(host: str, port: int) -> bool:
    """Quick check that the daemon is reachable."""
    import httpx

    url = f"http://{host}:{port}/healthz"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                body = resp.json()
                print(f"  Daemon reachable: tick={body.get('brain_tick_count')}, "
                      f"ws_clients={body.get('ws_clients')}")
                return True
            else:
                print(f"  Daemon returned HTTP {resp.status_code}")
                return False
    except Exception as e:
        print(f"  Cannot reach daemon at {url}: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(host: str, port: int) -> int:
    results = Results()

    print("=" * 60)
    print("  brAIn Live System Validator")
    print("=" * 60)
    print(f"\n  Target: {host}:{port}")
    print()

    # Pre-flight
    print("--- Pre-flight: healthcheck ---")
    reachable = await preflight_check(host, port)
    if not reachable:
        print("\n  ABORT: daemon not reachable. Start it first:")
        print("    .venv/bin/python -m server.braind start --port", port)
        return 1

    # Run tests
    frames = await test_brain_state_stream(host, port, results)
    await test_chat_api(host, port, results)
    await test_concept_cluster(host, port, results)

    # Summary
    print()
    print("=" * 60)
    if results.all_passed:
        print(f"  PASS  ({results.passed}/{results.total} checks passed)")
    else:
        print(f"  FAIL  ({results.passed}/{results.total} passed, "
              f"{results.failed} failed)")
        print()
        print("  Failed checks:")
        for t in results.tests:
            if not t.passed:
                print(f"    - {t.name}: {t.detail}")
    print("=" * 60)

    return 0 if results.all_passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end test against running brAIn daemon",
    )
    parser.add_argument("--host", default="localhost", help="Daemon host (default: localhost)")
    parser.add_argument("--port", type=int, default=8765, help="Daemon port (default: 8765)")
    args = parser.parse_args()

    return asyncio.run(run(args.host, args.port))


if __name__ == "__main__":
    sys.exit(main())
