"""Tests for grok_mcp.db — no live PostgreSQL required."""

from __future__ import annotations

import asyncio

import pytest

import grok_mcp.db as db_module


def setup_function():
    """Ensure pool is None before each test."""
    db_module._pool = None


def test_is_db_available_returns_false_when_pool_not_initialized():
    assert db_module.is_db_available() is False


def test_get_pool_raises_runtime_error_when_not_initialized():
    with pytest.raises(RuntimeError, match="Context cache database not available"):
        db_module.get_pool()


def test_close_db_is_noop_when_pool_is_none():
    db_module._pool = None
    # Should not raise
    asyncio.run(db_module.close_db())
    assert db_module._pool is None


def test_init_db_requires_database_url(monkeypatch):
    """No database, no server: history and jobs are not optional."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL is required"):
        asyncio.run(db_module.init_db())
    assert not db_module.is_db_available()


def test_schema_creates_final_tables():
    sql = db_module._SCHEMA_SQL
    for table in ("project_discovery", "chat_history", "jobs"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql
    assert "idx_chat_history_archived" in sql
