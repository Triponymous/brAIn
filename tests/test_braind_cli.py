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
