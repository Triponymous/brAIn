"""Capability catalog — registry of all available tools."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class CatalogEntry:
    name: str
    description: str
    trigger_keywords: list[str]
    tool_schema: dict[str, Any]
    execute_fn: Callable[..., Any] | None = None
    default_enabled: bool = False


def get_catalog() -> dict[str, CatalogEntry]:
    from capabilities.tools.web_search import web_search
    from capabilities.tools.shell import safe_shell

    return {
        "web_search": CatalogEntry(
            name="web_search",
            description="Search the web via DuckDuckGo and return top results.",
            trigger_keywords=["google", "suchen", "recherche", "search", "nachschlagen", "web"],
            tool_schema={
                "name": "web_search",
                "description": "Search the web and return top-5 results with title, URL, and snippet.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                    "required": ["query"],
                },
            },
            execute_fn=web_search,
        ),
        "shell": CatalogEntry(
            name="shell",
            description="Run a safe shell command (allowlisted commands only).",
            trigger_keywords=["terminal", "command", "befehl", "shell", "cli", "konsole"],
            tool_schema={
                "name": "shell",
                "description": "Execute a safe shell command. Only allowlisted commands are permitted.",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string", "description": "Shell command to run"}},
                    "required": ["command"],
                },
            },
            execute_fn=safe_shell,
        ),
        "local_files": CatalogEntry(
            name="local_files",
            description="Read, write, and list text/markdown files in a local directory.",
            trigger_keywords=["notizen", "notes", "datei", "file", "markdown", "schreiben", "lesen"],
            tool_schema={
                "name": "local_files",
                "description": "Read, write, or list text files in the user's notes directory.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["read", "write", "list"]},
                        "path": {"type": "string", "description": "Relative file path (for read/write)"},
                        "content": {"type": "string", "description": "Content to write (for write action)"},
                    },
                    "required": ["action"],
                },
            },
            execute_fn=None,  # dispatched by action
        ),
    }
