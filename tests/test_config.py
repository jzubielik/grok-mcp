"""Tests for grok_mcp.config validation logic."""

from __future__ import annotations

from datetime import timedelta

import pytest


def test_parse_duration_hours():
    from grok_mcp.config import parse_duration

    assert parse_duration("24h") == timedelta(hours=24)


def test_parse_duration_days():
    from grok_mcp.config import parse_duration

    assert parse_duration("7d") == timedelta(days=7)


def test_parse_duration_minutes():
    from grok_mcp.config import parse_duration

    assert parse_duration("30m") == timedelta(minutes=30)


def test_parse_duration_invalid_raises():
    from grok_mcp.config import parse_duration

    with pytest.raises(ValueError, match="Invalid duration"):
        parse_duration("2w")


def test_parse_duration_empty_raises():
    from grok_mcp.config import parse_duration

    with pytest.raises(ValueError):
        parse_duration("")


def test_write_allowed_paths_empty_by_default(monkeypatch):
    monkeypatch.delenv("GROK_WRITE_ALLOWED_PATHS", raising=False)
    from importlib import reload

    import grok_mcp.config as cfg

    reload(cfg)
    s = cfg.Settings()
    assert s.write_allowed_paths == []


def test_write_allowed_paths_parsed(monkeypatch):
    monkeypatch.setenv("GROK_WRITE_ALLOWED_PATHS", "/tmp,/srv/projects")
    from importlib import reload

    import grok_mcp.config as cfg

    reload(cfg)
    s = cfg.Settings()
    assert s.write_allowed_paths == ["/tmp", "/srv/projects"]
