"""FastMCP server for xAI Grok API."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from .client import close_clients, init_clients
from .db import close_db, init_db
from .logjson import configure_json_logging, error_text, loggable

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.applications import Starlette

logger = logging.getLogger("grok_mcp")


_started = False
_process_lifespan = False
"""True when the HTTP app owns startup/shutdown (see _run_http). FastMCP runs its lifespan
per streamable-HTTP *session*; tying shared state to sessions closed the DB pool whenever the
last client disconnected — background jobs then could not record their results — and the next
session re-ran startup, marking still-running jobs as orphaned."""


async def _startup() -> None:
    """Initialise clients, DB, jobs, discovery and the retention purge — once per process.

    On failure everything already started is torn down again before the error propagates.
    """
    global _started  # noqa: PLW0603
    if _started:
        return
    _started = True
    try:
        init_clients()
        await init_db()
        from .tools.jobs import fail_orphaned_jobs  # noqa: PLC0415

        await fail_orphaned_jobs()
        from .tools.context_cache import bootstrap_project_discovery  # noqa: PLC0415

        await bootstrap_project_discovery()
        from .tools.models import refresh_tool_descriptions  # noqa: PLC0415

        await refresh_tool_descriptions()
        from .retention import start_purge_loop  # noqa: PLC0415

        start_purge_loop()
    except BaseException:
        await _shutdown()
        raise
    logger.info("Grok MCP server started")


async def _shutdown() -> None:
    global _started  # noqa: PLW0603
    if not _started:
        return
    _started = False
    from .retention import stop_purge_loop  # noqa: PLC0415
    from .tools.jobs import drain  # noqa: PLC0415

    await drain()
    await stop_purge_loop()
    await close_db()
    await close_clients()
    logger.info("Grok MCP server stopped")


@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[None]:
    """FastMCP lifespan. stdio has one session per process, so it owns startup/shutdown;
    over HTTP the app lifespan does (sessions come and go, the process state must not)."""
    if _process_lifespan:
        yield
        return
    await _startup()
    try:
        yield
    finally:
        await _shutdown()


mcp = FastMCP("Grok", lifespan=server_lifespan)
# FastMCP just installed its Rich console handler — replace it with JSON Lines.
configure_json_logging()

_tool_log = logging.getLogger("grok_mcp.tools")
_call_tool = mcp._tool_manager.call_tool


async def _logged_call_tool(
    name: str,
    arguments: dict,
    context: Context | None = None,
    convert_result: bool = False,
) -> Any:
    """Log every MCP tool call: arguments, result and timing, per GROK_LOG_CONTENT.

    The tool runs unconverted so the log holds its real return value; conversion to MCP
    content happens afterwards, exactly as FastMCP would have done it.
    """
    _tool_log.info("tool_call", extra={"fields": {"tool": name, "args": loggable(arguments)}})
    # FastMCP silently drops unknown arguments; a caller passing a parameter the tool does not
    # have (e.g. a typo) must hear about it instead of getting different behaviour.
    tool = mcp._tool_manager.get_tool(name)
    if tool is not None:
        known = set(tool.parameters.get("properties", {}))
        unknown = sorted(set(arguments) - known)
        if unknown:
            error = f"unknown argument(s) {unknown} for {name}; valid: {sorted(known)}"
            _tool_log.error("tool_error", extra={"fields": {"tool": name, "error": error}})
            raise ToolError(error)
    started = time.monotonic()
    try:
        result = await _call_tool(name, arguments, context=context, convert_result=False)
    except Exception as exc:
        _tool_log.error(
            "tool_error",
            exc_info=True,
            extra={
                "fields": {
                    "tool": name,
                    "duration_s": round(time.monotonic() - started, 3),
                    "error": error_text(exc),
                }
            },
        )
        raise
    _tool_log.info(
        "tool_result",
        extra={
            "fields": {
                "tool": name,
                "duration_s": round(time.monotonic() - started, 3),
                "result": loggable(result),
            }
        },
    )
    if convert_result:
        return mcp._tool_manager.get_tool(name).fn_metadata.convert_result(result)
    return result


mcp._tool_manager.call_tool = _logged_call_tool

# Import tool modules to register them on the mcp instance
import grok_mcp.tools  # noqa: E402, F401


def _resolve_transport() -> str:
    """Map MCP_TRANSPORT env var to a FastMCP transport name. Default: stdio."""
    value = os.environ.get("MCP_TRANSPORT", "stdio").lower()
    if value == "stdio":
        return "stdio"
    if value in ("http", "streamable-http"):
        return "streamable-http"
    raise ValueError(f"Invalid MCP_TRANSPORT '{value}' — use 'stdio' or 'http'.")


def main() -> None:
    """Entry point for the grok-mcp command."""
    from .auth import capture_secrets, drop_privileges  # noqa: PLC0415

    drop_privileges()  # before anything else: the server never runs tools as root
    capture_secrets()  # before any tool can run a command or read /proc/self/environ
    check_config_outside_mounts()
    transport = _resolve_transport()
    if transport == "stdio":
        mcp.run(transport=transport)
        return
    _run_http()


def check_config_outside_mounts() -> None:
    """Refuse to start when the configuration file holding the secrets is under a directory the
    agent can reach (compose passes its host path as GROK_CONFIG_FILE). Mounted directories are
    the boundary for `shell`: anything inside them can be read, whatever the allow-lists say."""
    config = os.environ.get("GROK_CONFIG_FILE")
    if not config:
        return
    if not config.startswith("/"):
        raise RuntimeError(
            f"GROK_CONFIG_FILE must be an absolute path, got {config!r} — start with `make up`, "
            "which resolves it."
        )
    config_path = PurePosixPath(os.path.normpath(config))
    for name in ("WORKSPACE_DIR", "ASSETS_DIR"):
        mount = os.environ.get(name)
        if mount and config_path.is_relative_to(PurePosixPath(os.path.normpath(mount))):
            raise RuntimeError(
                f"The configuration file {config} is inside {name}={mount}, where the agent's "
                "shell can read it (API key, MCP token, DB password). Move it outside the "
                "mounted directories, e.g. ~/.config/grok-mcp/.env, and start with `make up`."
            )


def _transport_security(token: str | None) -> TransportSecuritySettings:
    """Host-header policy for the HTTP endpoint.

    FastMCP's default only accepts Host: 127.0.0.1/localhost/[::1] — DNS-rebinding protection
    for an unauthenticated loopback server. With a token that protection is redundant (a
    rebinding page cannot know the token) and would reject every remote or proxied client, so
    it is turned off.
    """
    from mcp.server.transport_security import TransportSecuritySettings  # noqa: PLC0415

    if token:
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    return mcp.settings.transport_security


def _run_http() -> None:
    """Serve streamable HTTP, behind a bearer token when MCP_AUTH_TOKEN is set."""
    import uvicorn  # noqa: PLC0415

    from .http_auth import BearerAuthMiddleware, check_bind_allowed  # noqa: PLC0415

    host = os.environ.get("MCP_HTTP_HOST", "127.0.0.1")
    port = int(os.environ.get("MCP_HTTP_PORT", "8000"))
    from .auth import get_secret  # noqa: PLC0415

    token = get_secret("MCP_AUTH_TOKEN")
    check_bind_allowed(host, token)
    mcp.settings.host, mcp.settings.port = host, port
    mcp.settings.transport_security = _transport_security(token)
    global _process_lifespan  # noqa: PLW0603
    _process_lifespan = True
    app = mcp.streamable_http_app()
    session_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def process_lifespan(a: Starlette) -> AsyncIterator[None]:
        async with session_lifespan(a):
            await _startup()
            try:
                yield
            finally:
                await _shutdown()

    app.router.lifespan_context = process_lifespan
    if token:
        app = BearerAuthMiddleware(app, token)
    logger.info(
        "MCP streamable HTTP on %s:%d/mcp", host, port, extra={"fields": {"auth": bool(token)}}
    )
    uvicorn.run(app, host=host, port=port, log_level=mcp.settings.log_level.lower())
