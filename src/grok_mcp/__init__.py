"""Grok MCP Server — xAI API as MCP tools."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("grok-mcp")
except PackageNotFoundError:
    __version__ = "unknown"
