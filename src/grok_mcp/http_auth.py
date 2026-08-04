"""Bearer-token guard for the streamable-HTTP MCP endpoint.

The MCP tools include `shell` and `write`, so an unauthenticated endpoint reachable from the
network is remote code execution. The server therefore refuses to bind a non-loopback address
without MCP_AUTH_TOKEN, and with a token set every request must carry it.
"""

from __future__ import annotations

import hmac
import json
from typing import Any

_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


def token_matches(header: str | None, token: str) -> bool:
    """True if an Authorization header value is `Bearer <token>` (constant time on the token).

    The scheme is case-insensitive (RFC 9110 §11.1), so `bearer` / `BEARER` clients work too.
    """
    if not header:
        return False
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer":
        return False
    return hmac.compare_digest(value.encode(), token.encode())


MIN_TOKEN_LENGTH = 32


def check_bind_allowed(host: str, token: str | None) -> None:
    """Raise unless the bind address is loopback or a strong token protects the endpoint."""
    if token is not None and any(c.isspace() for c in token):
        # Usually a stray newline/space from the config file; the client would never match it.
        raise RuntimeError(
            "MCP_AUTH_TOKEN contains whitespace. Generate one with: openssl rand -hex 32"
        )
    if token is not None and len(token) < MIN_TOKEN_LENGTH:
        raise RuntimeError(
            f"MCP_AUTH_TOKEN is too short ({len(token)} < {MIN_TOKEN_LENGTH} characters). "
            "Generate one with: openssl rand -hex 32"
        )
    if token or host in _LOOPBACK:
        return
    raise RuntimeError(
        f"Refusing to serve MCP over HTTP on {host} without authentication: the tools can run "
        "shell commands. Set MCP_AUTH_TOKEN, or bind MCP_HTTP_HOST=127.0.0.1."
    )


class BearerAuthMiddleware:
    """Pure ASGI middleware (does not buffer, so SSE streaming keeps working)."""

    def __init__(self, app: Any, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":  # server startup/shutdown, not a client
            await self.app(scope, receive, send)
            return
        # ASGI servers lowercase header names; every other scope (http, websocket) needs a token.
        headers = dict(scope.get("headers") or [])
        auth = headers.get(b"authorization", b"").decode("latin-1") or None
        if token_matches(auth, self.token):
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":
            # Closing before accept makes the server answer the handshake with 403.
            await send({"type": "websocket.close", "code": 1008})
            return
        body = json.dumps({"error": "unauthorized"}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", b"Bearer"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
