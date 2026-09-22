"""Tests for the braind CLI.

We test the argument parser and the high-level run flow with mock sensors.
The actual long-running daemon is exercised by the E2E smoke test in Task 11.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from brain.persistence import load_brain
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


def test_uvicorn_config_ws_backend_is_importable():
    """The daemon's websocket backend must be installable from pyproject alone.

    Regression guard: the daemon used to pin ws="wsproto", but wsproto is not a
    declared dependency. A clean `uv pip install -e ".[dev]"` therefore crashed
    at startup inside uvicorn's config.load() with
    ModuleNotFoundError: No module named 'wsproto'.

    config.load() is exactly where that crash happened, so loading a real config
    is what catches it.
    """
    from fastapi import FastAPI
    from server.braind import build_uvicorn_config

    config = build_uvicorn_config(FastAPI(), 8000)
    config.load()

    assert config.loaded
    assert config.ws_protocol_class is not None


def test_sigterm_ends_with_a_saved_checkpoint(tmp_path):
    """SIGTERM (launchd, console Stop, system shutdown) must end with a saved brain.

    Regression guard: uvicorn re-raises a captured SIGTERM after restoring the
    default handler, which terminated the process inside serve() before the
    daemon's final save could run. Everything learned since the last 60 s
    autosave was lost on every stop. Only a real process receiving a real
    signal exercises that path, hence the subprocess.
    """
    ckpt = tmp_path / "braind.sqlite"
    log = tmp_path / "daemon.log"
    port = 8791
    with log.open("w") as out:
        proc = subprocess.Popen(
            [sys.executable, "-m", "server.braind", "start", "--mock-sensors",
             "--port", str(port), "--checkpoint", str(ckpt)],
            cwd=Path(__file__).resolve().parents[1],
            stdout=out, stderr=subprocess.STDOUT,
        )
    try:
        deadline = time.monotonic() + 90
        while True:
            if proc.poll() is not None:
                pytest.fail(f"daemon exited before becoming healthy:\n{log.read_text()}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1):
                    break
            except (urllib.error.URLError, OSError):
                pass
            if time.monotonic() > deadline:
                pytest.fail(f"daemon never became healthy:\n{log.read_text()}")
            time.sleep(0.5)

        # Nothing is learned until a source is shared (synthetic data in mock mode).
        share = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/consent", method="POST",
            data=json.dumps({"revision": 0, "enabled": {"idle": True}}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(share, timeout=5).close()
        while json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1))["brain_tick_count"] == 0:
            if time.monotonic() > deadline:
                pytest.fail(f"the brain never stepped after sharing a source:\n{log.read_text()}")
            time.sleep(0.2)

        proc.terminate()  # SIGTERM, exactly what the console Stop button sends
        rc = proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()

    assert rc == 0, log.read_text()
    assert ckpt.exists(), log.read_text()
    assert "Final save" in log.read_text()
    assert load_brain(ckpt).tick_count > 0
