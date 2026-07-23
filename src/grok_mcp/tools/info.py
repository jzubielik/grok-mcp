"""Server info tools."""

from grok_mcp import __version__

from ..server import mcp


@mcp.tool()
def get_version() -> dict:
    """Return the grok-mcp server version."""
    return {"version": __version__}
