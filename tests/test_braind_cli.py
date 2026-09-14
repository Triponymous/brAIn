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
