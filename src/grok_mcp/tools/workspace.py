"""Workspace tools the model can call inside the agent() tool loop: read/write/shell/fetch/search.

Schemas are offered to Grok via function calling; run_workspace_tool executes them. Guards live
in the underlying helpers (paths.check_readable / check_writable, the SSRF guard in fetch.py), so they apply the
same way whether a tool is reached from the loop or called directly as an MCP tool.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ..config import settings
from .fetch import _fetch
from .files import _read_internal, _write_internal
from .search import _search_internal
from .shell import SHELL_EXECUTOR, _run_shell

WORKSPACE_TOOL_SCHEMAS: dict[str, dict] = {
    "read": {
        "type": "function",
        "function": {
            "name": "read",
            "description": (
                "Read a local file, list of files, or glob pattern. For large files read a "
                "line range: offset (0-based first line) + limit (number of lines); the "
                "result then starts with '[lines a-b of N]' and numbers each line. Prefer "
                "shell grep -n to locate code before reading big files whole."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path, list of paths (JSON array), or glob pattern",
                    },
                    "offset": {"type": "integer", "description": "First line (0-based)"},
                    "limit": {"type": "integer", "description": "Number of lines"},
                },
                "required": ["path"],
            },
        },
    },
    "write": {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write content to a file. dry_run=true (default) returns diff only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "dry_run": {"type": "boolean", "default": True},
                },
                "required": ["path", "content"],
            },
        },
    },
    "shell": {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command, return stdout+stderr+exit code (timeout "
            "GROK_SHELL_TIMEOUT, default 240 s).",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    },
    "fetch": {
        "type": "function",
        "function": {
            "name": "fetch",
            "description": "HTTP GET a URL, return the response body.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    "search": {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Live web search. Returns synthesized answer with source URLs.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
}

WORKSPACE_TOOLS = tuple(WORKSPACE_TOOL_SCHEMAS)


def validate_workspace_tools(tools: list[str] | None) -> list[str]:
    """Return `tools` as a list, raising ValueError on any name that is not a workspace tool."""
    tools = list(tools or [])
    unknown = [t for t in tools if t not in WORKSPACE_TOOL_SCHEMAS]
    if unknown:
        raise ValueError(f"unknown tools {unknown}; allowed: {list(WORKSPACE_TOOLS)}")
    return tools


async def run_workspace_tool(
    name: str, args: dict, model: str, time_left: float | None = None
) -> Any:
    """Execute one workspace tool and return its raw result (str or JSON-able).

    `model` is the loop's own model — search runs on it rather than on a hidden default.
    """
    if name == "read":
        path = args.get("path", "")
        # Accept JSON array string
        if isinstance(path, str) and path.startswith("["):
            path = json.loads(path)
        return await _read_internal(path, offset=args.get("offset"), limit=args.get("limit"))
    if name == "write":
        # Single-file args only from the loop
        return await _write_internal(
            args["path"], content=args.get("content"), dry_run=args.get("dry_run", True)
        )
    if name == "shell":
        # Off the event loop — a slow command must not stall other requests.
        # Never longer than the agent run has left: the process (not just the awaiting
        # coroutine) must end with the budget, or the 4 shell threads fill up.
        timeout = min(settings.shell_timeout, time_left or settings.shell_timeout)
        return await asyncio.get_running_loop().run_in_executor(
            SHELL_EXECUTOR, _run_shell, args["cmd"], timeout
        )
    if name == "fetch":
        return await _fetch(args["url"])
    if name == "search":
        return await _search_internal(args["query"], model)
    raise ValueError(f"unknown tool '{name}'")
