# tests/test_server.py
"""Tests for MCP transport selection."""

from __future__ import annotations

import pytest


def test_resolve_transport_defaults_to_stdio(monkeypatch):
    monkeypatch.delenv("MCP_TRANSPORT", raising=False)
    from grok_mcp.server import _resolve_transport

    assert _resolve_transport() == "stdio"


@pytest.mark.parametrize("value", ["http", "streamable-http", "HTTP"])
def test_resolve_transport_http_aliases(monkeypatch, value):
    monkeypatch.setenv("MCP_TRANSPORT", value)
    from grok_mcp.server import _resolve_transport

    assert _resolve_transport() == "streamable-http"


def test_resolve_transport_rejects_unknown(monkeypatch):
    monkeypatch.setenv("MCP_TRANSPORT", "carrier-pigeon")
    from grok_mcp.server import _resolve_transport

    with pytest.raises(ValueError):
        _resolve_transport()


def test_transport_security_policy():
    import grok_mcp.server as srv

    # With a token the token is the guard: any Host (remote, proxied) is accepted.
    assert srv._transport_security("tok").enable_dns_rebinding_protection is False
    # Without one (loopback only) FastMCP's DNS-rebinding protection stays in place.
    ts = srv._transport_security(None)
    assert ts.enable_dns_rebinding_protection is True
    assert all(h.rsplit(":", 1)[0] in ("127.0.0.1", "localhost", "[::1]") for h in ts.allowed_hosts)


async def test_startup_failure_tears_down_and_can_retry(monkeypatch):
    from unittest.mock import AsyncMock, patch

    import grok_mcp.server as srv

    with (
        patch.object(srv, "init_clients"),
        patch.object(srv, "init_db", AsyncMock(side_effect=RuntimeError("db down"))),
        patch.object(srv, "close_db", AsyncMock()) as close_db,
        patch.object(srv, "close_clients") as close_clients,
        patch("grok_mcp.retention.stop_purge_loop", AsyncMock()),
    ):
        with pytest.raises(RuntimeError, match="db down"):
            await srv._startup()
        close_db.assert_awaited_once()
        close_clients.assert_called_once()
        assert srv._started is False


def test_config_inside_a_mount_refuses_to_start(monkeypatch):
    import grok_mcp.server as srv

    monkeypatch.setenv("WORKSPACE_DIR", "/home/u/Work")
    monkeypatch.delenv("ASSETS_DIR", raising=False)
    monkeypatch.setenv("GROK_CONFIG_FILE", "/home/u/Work/grok-mcp/.env")
    with pytest.raises(RuntimeError, match="inside WORKSPACE_DIR"):
        srv.check_config_outside_mounts()
    monkeypatch.setenv("GROK_CONFIG_FILE", "/home/u/.config/grok-mcp/.env")
    srv.check_config_outside_mounts()
    monkeypatch.setenv("GROK_CONFIG_FILE", "/home/u/Work-other/.env")  # not component-wise inside
    srv.check_config_outside_mounts()


@pytest.mark.parametrize(
    "config", [".env", "./x/.env", "grok/.env", "/home/u/.config/../Work/grok/.env"]
)
def test_config_guard_cannot_be_bypassed(monkeypatch, config):
    import grok_mcp.server as srv

    monkeypatch.setenv("WORKSPACE_DIR", "/home/u/Work/")
    monkeypatch.setenv("ASSETS_DIR", "/home/u/assets")
    monkeypatch.setenv("GROK_CONFIG_FILE", config)
    with pytest.raises(RuntimeError, match="absolute path|inside WORKSPACE_DIR"):
        srv.check_config_outside_mounts()


def test_config_guard_covers_assets_dir(monkeypatch):
    import grok_mcp.server as srv

    monkeypatch.setenv("WORKSPACE_DIR", "/home/u/Work")
    monkeypatch.setenv("ASSETS_DIR", "/home/u/assets")
    monkeypatch.setenv("GROK_CONFIG_FILE", "/home/u/assets/.env")
    with pytest.raises(RuntimeError, match="inside ASSETS_DIR"):
        srv.check_config_outside_mounts()
