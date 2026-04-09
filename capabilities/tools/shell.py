"""Sandboxed shell execution with allowlist."""
from __future__ import annotations
import asyncio
import shlex
from typing import Any

_ALLOWED_COMMANDS = {"ls", "cat", "wc", "date", "which", "brew", "pip", "echo", "head", "tail", "find", "grep"}
_BLOCKED_PREFIXES = {"rm", "sudo", "mv", "cp", "chmod", "chown", "kill", "pkill", "dd", "mkfs", "fdisk"}


async def safe_shell(command: str) -> dict[str, Any]:
    try:
        parts = shlex.split(command)
    except ValueError as e:
        return {"error": f"Invalid command: {e}"}
    if not parts:
        return {"error": "Empty command"}
    base_cmd = parts[0].split("/")[-1]
    if base_cmd in _BLOCKED_PREFIXES or any(p.startswith("sudo") for p in parts):
        return {"error": f"Blocked: '{base_cmd}' is not allowed for safety reasons."}
    if base_cmd not in _ALLOWED_COMMANDS:
        return {"error": f"Blocked: '{base_cmd}' is not in the allowlist. Allowed: {', '.join(sorted(_ALLOWED_COMMANDS))}"}
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
    except asyncio.TimeoutError:
        return {"error": "Command timed out (10s limit)"}
    return {
        "stdout": stdout.decode("utf-8", errors="replace")[:5000],
        "stderr": stderr.decode("utf-8", errors="replace")[:2000],
        "returncode": proc.returncode,
    }
