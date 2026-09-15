"""brAIn as a context layer for other models — a read-only MCP server.

Claude Code, Codex, Claude Desktop or any MCP client spawns this over stdio
and gets the brain's read-only tools: brain_state, brain_history,
brain_recall and the rest of BrainTools. Nothing here touches the
organism. Every call is proxied to the running daemon's /api/tools, which
exposes reads only; the daemon stays the one process that holds the brain,
this is a thin pipe to it.

    claude mcp add --scope user brain -- /path/to/brAIn/.venv/bin/brain-mcp
    # Codex (~/.codex/config.toml):  [mcp_servers.brain]  command = ".../.venv/bin/brain-mcp"

BRAIN_URL (default http://127.0.0.1:8000) points at the daemon. With the
daemon down the tools are still listed, so the client knows what brAIn can
do, and a call answers with how to start it.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from typing import Any

import httpx
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from bridge.brain_tools import BrainTools

DEFAULT_URL = os.environ.get("BRAIN_URL", "http://127.0.0.1:8000")

INSTRUCTIONS = (
    "brAIn is a spiking neural network that lives on this person's Mac and learns them from their "
    "desktop life: typing rhythm, which app is in front, idle time, sound level, never content. These "
    "tools read what it has learned. How they seem right now: brain_state, brain_felt. What they did "
    "today: brain_history, brain_recall. Their habits and what is unusual today: brain_habits, "
    "brain_anomalies. Why the brain feels as it does: brain_why. How the two of them have been getting "
    "along: brain_experience. Use them to ground anything about this person's current state, day or "
    "habits in observation instead of guessing. All tools are read-only."
)

_DOWN = ("brAIn daemon is not running at {url}. Start it from the console at http://127.0.0.1:8900 "
         "(or `.venv/bin/python -m server.control --autostart`). The daemon holds the brain that "
         "answers these tools; nothing is learned or served while it is down.")


def brain_tools() -> list[types.Tool]:
    """The daemon's read-only tools, as MCP sees them. Static: listing works without a daemon."""
    return [types.Tool(name=d["name"], description=d["description"], inputSchema=d["input_schema"])
            for d in BrainTools.tool_definitions()]


async def call_brain(name: str, args: dict[str, Any] | None, url: str = DEFAULT_URL,
                     transport: httpx.AsyncBaseTransport | None = None) -> tuple[Any, bool]:
    """Proxy one tool call to the daemon. Returns (result, is_error)."""
    try:
        async with httpx.AsyncClient(base_url=url, timeout=30.0, transport=transport) as client:
            resp = await client.post(f"/api/tools/{name}", json=args or {}, headers={"X-Brain-Via": "mcp"})
    except httpx.HTTPError:
        return _DOWN.format(url=url), True
    if resp.status_code != 200:
        return f"daemon answered {resp.status_code}: {resp.text}", True
    result = resp.json().get("result")
    return result, isinstance(result, dict) and "error" in result


def build_server(url: str = DEFAULT_URL, transport: httpx.AsyncBaseTransport | None = None) -> Server:
    async def on_list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=brain_tools())

    async def on_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        result, is_error = await call_brain(params.name, params.arguments, url, transport)
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=1, default=str)
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)], isError=is_error)

    return Server("brain", version="0.1.0", instructions=INSTRUCTIONS,
                  on_list_tools=on_list_tools, on_call_tool=on_call_tool)


async def serve(url: str = DEFAULT_URL) -> None:
    server = build_server(url)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    parser = argparse.ArgumentParser(prog="brain-mcp", description="brAIn read-only MCP server (stdio)")
    parser.add_argument("--url", default=DEFAULT_URL, help="the brain daemon (default: $BRAIN_URL or http://127.0.0.1:8000)")
    asyncio.run(serve(parser.parse_args().url))


if __name__ == "__main__":
    main()
