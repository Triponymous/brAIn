"""Minimal local research runner; sensor capture requires --desktop-metadata.

No existing checkpoint is loaded or overwritten. No voice, LLM or OS actions.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import time

import torch
import uvicorn

from brain.core import Brain
from server.capture import CaptureControl
from server.telemetry import TelemetryBuffer, build_telemetry_app


async def run_observation(brain, buffer, collector, capture, tick_hz=100):
    while True:
        start = time.monotonic()
        capture.sample(collector, start)
        if capture.vector is None:
            await asyncio.sleep(0.1)
            continue
        def step():
            with torch.no_grad():
                capture.step(brain, buffer)
        await asyncio.to_thread(step)
        await asyncio.sleep(max(0, 1 / tick_hz - (time.monotonic() - start)))


async def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    brain = Brain()
    buffer = TelemetryBuffer(brain, input_kind="desktop_metadata" if args.desktop_metadata else "disabled", seed=args.seed)
    capture = CaptureControl(buffer.session_id, armed=args.desktop_metadata)
    app = build_telemetry_app(buffer, capture)
    collector = None
    if args.desktop_metadata:
        from adapters.mac_desktop.metadata import DesktopMetadata
        collector = DesktopMetadata()
    print("Observation endpoint: http://127.0.0.1:8001/api/telemetry")
    print("Desktop controls available: " + str(args.desktop_metadata) + "; all inputs start OFF")
    print("Microphone / cloud / disk persistence: OFF")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8001,
                            access_log=False, log_level="warning"))
    async def guarded_capture():
        try:
            await run_observation(brain, buffer, collector, capture)
        except asyncio.CancelledError:
            raise
        except Exception:
            capture.fail()
            buffer.state = "error"
            print("Observation stopped after a capture error; no synthetic fallback.")
    task = asyncio.create_task(guarded_capture()) if collector else None
    async def auto_stop():
        await asyncio.sleep(args.duration)
        server.should_exit = True
    timer = asyncio.create_task(auto_stop()) if args.duration else None
    try:
        await server.serve()
    finally:
        for pending in (task, timer):
            if pending:
                pending.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desktop-metadata", action="store_true", help="Allow dashboard input controls; every input starts OFF")
    parser.add_argument("--seed", type=int, default=4271)
    parser.add_argument("--duration", type=int, default=0, help="Stop automatically after N seconds; 0 waits for Ctrl-C")
    args = parser.parse_args()
    if args.duration < 0:
        parser.error("duration must be non-negative")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
