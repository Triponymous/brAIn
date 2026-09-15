"""/api/tools — the brain's read-only tools over HTTP.

The same brain_* tools the daemon's own LLM calls, for anything outside
this process: the MCP server (server/mcp.py) that hands them to Claude
Code, Codex or any other frontier model, and any script. Read-only by
construction — only BrainTools are reachable here, never label, teach,
reward, shell or files. Every call is an experience: how often outside
models consult the brain is part of the record.
"""
from __future__ import annotations
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request

from bridge.brain_tools import BrainTools
from bridge.experience import state_of


def build_tools_router(brain: Any, brain_tools: BrainTools) -> APIRouter:
    api = APIRouter()

    @api.get("/api/tools")
    async def list_tools() -> dict[str, Any]:
        return {"tools": brain_tools.tool_definitions()}

    @api.post("/api/tools/{name}")
    async def call_tool(name: str, request: Request,
                        args: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        if name not in BrainTools.NAMES:
            raise HTTPException(status_code=404, detail=f"no read-only brain tool named '{name}'")
        result = brain_tools.execute(name, args)
        log = getattr(brain, "_experience", None)
        if log is not None:
            log.record("llm", "tool_call",
                       {"tool": name, "args": args, "via": request.headers.get("x-brain-via", "api")},
                       state=state_of(brain))
        return {"tool": name, "result": result}

    return api
