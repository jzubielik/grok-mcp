"""Tests for context cache MCP tools."""

from __future__ import annotations

from datetime import UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_conn(fetchrow_return=None, fetch_return=None, execute_return=None):
    """Return a mock asyncpg connection context manager."""
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=fetchrow_return)
    conn.fetch = AsyncMock(return_value=fetch_return or [])
    conn.execute = AsyncMock(return_value=execute_return or "INSERT 0 1")

    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return pool, conn


# ---------------------------------------------------------------------------
# get_discovery_cache
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# store_discovery_cache
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# get_file_analysis
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# store_file_analysis
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# get_active_anomalies
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# store_anomaly
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# build_context_frame
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# get_active_project_context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_active_project_context_no_db():
    with patch("grok_mcp.tools.context_cache.is_db_available", return_value=False):
        from grok_mcp.tools.context_cache import get_active_project_context

        result = await get_active_project_context("/test/project")
    assert result is None


@pytest.mark.asyncio
async def test_get_active_project_context_empty_db():
    pool, conn = _make_mock_conn(fetchrow_return=None)
    with (
        patch("grok_mcp.tools.context_cache.is_db_available", return_value=True),
        patch("grok_mcp.tools.context_cache.get_pool", return_value=pool),
    ):
        from grok_mcp.tools.context_cache import get_active_project_context

        result = await get_active_project_context("/test/project")
    assert result is None


@pytest.mark.asyncio
async def test_get_active_project_context_returns_formatted_string():
    row = {"language": "Python", "framework": "FastAPI"}
    pool, conn = _make_mock_conn(fetchrow_return=row)
    with (
        patch("grok_mcp.tools.context_cache.is_db_available", return_value=True),
        patch("grok_mcp.tools.context_cache.get_pool", return_value=pool),
    ):
        from grok_mcp.tools.context_cache import get_active_project_context

        result = await get_active_project_context("/test/project")
    assert result is not None
    assert "[Project Context]" in result
    assert "Language: Python" in result
    assert "Framework: FastAPI" in result


@pytest.mark.asyncio
async def test_get_active_project_context_returns_none_when_all_fields_null():
    row = {"language": None, "framework": None}
    pool, conn = _make_mock_conn(fetchrow_return=row)
    with (
        patch("grok_mcp.tools.context_cache.is_db_available", return_value=True),
        patch("grok_mcp.tools.context_cache.get_pool", return_value=pool),
    ):
        from grok_mcp.tools.context_cache import get_active_project_context

        result = await get_active_project_context("/test/project")
    assert result is None


@pytest.mark.asyncio
async def test_prune_chat_context_no_db():
    with patch("grok_mcp.tools.context_cache.is_db_available", return_value=False):
        from grok_mcp.tools.context_cache import prune_chat_context

        with pytest.raises(RuntimeError, match="not available"):
            await prune_chat_context(conversation="conv-abc")


@pytest.mark.asyncio
async def test_prune_chat_context_soft():
    """Soft prune (default): runs UPDATE SET archived=TRUE."""
    pool, conn = _make_mock_conn()
    conn.execute = AsyncMock(side_effect=["UPDATE 5"])
    conn.fetchval = AsyncMock(return_value=3)
    with (
        patch("grok_mcp.tools.context_cache.is_db_available", return_value=True),
        patch("grok_mcp.tools.context_cache.get_pool", return_value=pool),
    ):
        from grok_mcp.tools.context_cache import prune_chat_context

        result = await prune_chat_context(conversation="conv-abc")
    sql = conn.execute.call_args[0][0]
    assert "UPDATE" in sql
    assert "archived = TRUE" in sql
    assert result["pruned"] == 5
    assert result["mode"] == "soft"
    assert result["remaining"] == 3


@pytest.mark.asyncio
async def test_prune_chat_context_hard():
    """Hard prune: runs DELETE."""
    pool, conn = _make_mock_conn()
    conn.execute = AsyncMock(side_effect=["DELETE 10"])
    conn.fetchval = AsyncMock(return_value=2)
    with (
        patch("grok_mcp.tools.context_cache.is_db_available", return_value=True),
        patch("grok_mcp.tools.context_cache.get_pool", return_value=pool),
    ):
        from grok_mcp.tools.context_cache import prune_chat_context

        result = await prune_chat_context(conversation="conv-abc", hard=True)
    sql = conn.execute.call_args[0][0]
    assert "DELETE" in sql
    assert result["pruned"] == 10
    assert result["mode"] == "hard"


@pytest.mark.asyncio
async def test_prune_chat_context_before_duration():
    """'before' duration string is converted to absolute datetime before DB call."""
    from datetime import datetime

    pool, conn = _make_mock_conn()
    conn.execute = AsyncMock(side_effect=["UPDATE 2"])
    conn.fetchval = AsyncMock(return_value=0)
    with (
        patch("grok_mcp.tools.context_cache.is_db_available", return_value=True),
        patch("grok_mcp.tools.context_cache.get_pool", return_value=pool),
    ):
        from grok_mcp.tools.context_cache import prune_chat_context

        await prune_chat_context(conversation="conv-abc", before="24h")
    _sql, conv_id, cutoff = conn.execute.call_args[0]
    assert isinstance(cutoff, datetime)
    assert cutoff.tzinfo is not None  # must be timezone-aware
    assert cutoff < datetime.now(UTC)  # cutoff must be in the past


@pytest.mark.asyncio
async def test_prune_targets_exactly_the_named_conversation():
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="DELETE 3")
    conn.fetchval = AsyncMock(return_value=0)
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    with (
        patch("grok_mcp.tools.context_cache.is_db_available", return_value=True),
        patch("grok_mcp.tools.context_cache.get_pool", return_value=pool),
    ):
        from grok_mcp.tools.context_cache import prune_chat_context

        result = await prune_chat_context(conversation="grok-mcp/audit", hard=True)
    assert result["conversation"] == "grok-mcp/audit"
    assert conn.execute.await_args.args[1] == "grok-mcp/audit"


@pytest.mark.asyncio
async def test_prune_rejects_a_blank_conversation():
    from grok_mcp.tools.context_cache import prune_chat_context

    with pytest.raises(ValueError, match="conversation"):
        await prune_chat_context(conversation="  ")


@pytest.mark.asyncio
async def test_list_conversations_returns_name_count_and_last_activity():
    from datetime import datetime

    when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    pool, conn = _make_mock_conn()
    conn.fetch = AsyncMock(
        return_value=[{"conversation_id": "p/t", "messages": 4, "last_activity": when}]
    )
    with (
        patch("grok_mcp.tools.context_cache.is_db_available", return_value=True),
        patch("grok_mcp.tools.context_cache.get_pool", return_value=pool),
    ):
        from grok_mcp.tools.context_cache import list_conversations

        result = await list_conversations(limit=500)
    assert result == {
        "conversations": [
            {"conversation": "p/t", "messages": 4, "last_activity": "2026-01-02T03:04:05+00:00"}
        ]
    }
    assert conn.fetch.await_args.args[1] == 100  # limit capped


@pytest.mark.asyncio
async def test_prune_chat_context_before_invalid_raises():
    with patch("grok_mcp.tools.context_cache.is_db_available", return_value=True):
        from grok_mcp.tools.context_cache import prune_chat_context

        with pytest.raises(ValueError, match="invalid 'before'"):
            await prune_chat_context(conversation="conv-abc", before="bad-value")


# ---------------------------------------------------------------------------
# _detect_project_info
# ---------------------------------------------------------------------------


def test_detect_project_info_python_pyproject(tmp_path):
    from grok_mcp.tools.context_cache import _detect_project_info

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "myapp"\ndependencies = ["fastmcp>=0.1"]\n'
    )
    lang, fw = _detect_project_info(str(tmp_path))
    assert lang == "Python"
    assert fw == "FastMCP"


def test_detect_project_info_python_no_framework(tmp_path):
    from grok_mcp.tools.context_cache import _detect_project_info

    (tmp_path / "pyproject.toml").write_text('[project]\nname = "myapp"\n')
    lang, fw = _detect_project_info(str(tmp_path))
    assert lang == "Python"
    assert fw is None


def test_detect_project_info_typescript(tmp_path):
    import json

    from grok_mcp.tools.context_cache import _detect_project_info

    (tmp_path / "package.json").write_text(
        json.dumps({"devDependencies": {"typescript": "^5"}, "dependencies": {"react": "^18"}})
    )
    lang, fw = _detect_project_info(str(tmp_path))
    assert lang == "TypeScript"
    assert fw == "React"


def test_detect_project_info_javascript_express(tmp_path):
    import json

    from grok_mcp.tools.context_cache import _detect_project_info

    (tmp_path / "package.json").write_text(json.dumps({"dependencies": {"express": "^4"}}))
    lang, fw = _detect_project_info(str(tmp_path))
    assert lang == "JavaScript"
    assert fw == "Express"


def test_detect_project_info_go(tmp_path):
    from grok_mcp.tools.context_cache import _detect_project_info

    (tmp_path / "go.mod").write_text("module example.com/myapp\n\ngo 1.21\n")
    lang, fw = _detect_project_info(str(tmp_path))
    assert lang == "Go"
    assert fw is None


def test_detect_project_info_rust(tmp_path):
    from grok_mcp.tools.context_cache import _detect_project_info

    (tmp_path / "Cargo.toml").write_text('[package]\nname = "myapp"\nversion = "0.1.0"\n')
    lang, fw = _detect_project_info(str(tmp_path))
    assert lang == "Rust"
    assert fw is None


def test_detect_project_info_unknown(tmp_path):
    from grok_mcp.tools.context_cache import _detect_project_info

    lang, fw = _detect_project_info(str(tmp_path))
    assert lang is None
    assert fw is None


# ---------------------------------------------------------------------------
# bootstrap_project_discovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_writes_detected_language(tmp_path):
    """bootstrap_project_discovery stores auto-detected language/framework."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "myapp"\ndependencies = ["fastapi>=0.1"]\n'
    )
    pool, conn = _make_mock_conn()

    async def _fake_git(*_args, **_kwargs):
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"abc1234\n", b""))
        return proc

    with (
        patch("grok_mcp.tools.context_cache.is_db_available", return_value=True),
        patch("grok_mcp.tools.context_cache.get_pool", return_value=pool),
        patch("asyncio.create_subprocess_exec", side_effect=_fake_git),
        patch("os.getcwd", return_value=str(tmp_path)),
    ):
        from grok_mcp.tools.context_cache import bootstrap_project_discovery

        await bootstrap_project_discovery()

    sql, path_arg, hash_arg, lang_arg, fw_arg = conn.execute.call_args[0]
    assert "INSERT INTO project_discovery" in sql
    assert lang_arg == "Python"
    assert fw_arg == "FastAPI"


@pytest.mark.asyncio
async def test_bootstrap_skips_when_db_unavailable():
    """bootstrap_project_discovery is a no-op without a DB."""
    with patch("grok_mcp.tools.context_cache.is_db_available", return_value=False):
        from grok_mcp.tools.context_cache import bootstrap_project_discovery

        await bootstrap_project_discovery()  # must not raise


@pytest.mark.asyncio
async def test_bootstrap_skips_when_not_git_repo(tmp_path):
    """bootstrap_project_discovery skips gracefully outside git repos."""
    pool, conn = _make_mock_conn()

    async def _fake_git(*_args, **_kwargs):
        proc = MagicMock()
        proc.returncode = 128  # not a git repo
        proc.communicate = AsyncMock(return_value=(b"", b"fatal: not a git repo"))
        return proc

    with (
        patch("grok_mcp.tools.context_cache.is_db_available", return_value=True),
        patch("grok_mcp.tools.context_cache.get_pool", return_value=pool),
        patch("asyncio.create_subprocess_exec", side_effect=_fake_git),
        patch("os.getcwd", return_value=str(tmp_path)),
    ):
        from grok_mcp.tools.context_cache import bootstrap_project_discovery

        await bootstrap_project_discovery()

    conn.execute.assert_not_called()


# ---------------------------------------------------------------------------
# refresh_project_git_hash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_project_git_hash_updates_when_stale():
    """Fires an UPDATE when stored hash differs from current HEAD."""
    pool, conn = _make_mock_conn(execute_return="UPDATE 1")
    with (
        patch("grok_mcp.tools.context_cache.is_db_available", return_value=True),
        patch("grok_mcp.tools.context_cache.get_pool", return_value=pool),
    ):
        from grok_mcp.tools.context_cache import refresh_project_git_hash

        await refresh_project_git_hash("/my/project", "newhash123")
    conn.execute.assert_called_once()
    sql, path, h = conn.execute.call_args[0]
    assert "UPDATE project_discovery" in sql
    assert "updated_at = NOW()" in sql  # touched on every call: a live project is never purged
    assert path == "/my/project"
    assert h == "newhash123"


@pytest.mark.asyncio
async def test_refresh_project_git_hash_no_db():
    """No-op when DB is unavailable."""
    with patch("grok_mcp.tools.context_cache.is_db_available", return_value=False):
        from grok_mcp.tools.context_cache import refresh_project_git_hash

        await refresh_project_git_hash("/my/project", "abc123")  # should not raise


@pytest.mark.asyncio
async def test_refresh_project_git_hash_swallows_exceptions():
    """Exceptions are logged and swallowed, never raised."""
    pool, conn = _make_mock_conn()
    conn.execute = AsyncMock(side_effect=Exception("DB gone"))
    with (
        patch("grok_mcp.tools.context_cache.is_db_available", return_value=True),
        patch("grok_mcp.tools.context_cache.get_pool", return_value=pool),
    ):
        from grok_mcp.tools.context_cache import refresh_project_git_hash

        await refresh_project_git_hash("/my/project", "abc123")  # must not raise


@pytest.mark.asyncio
async def test_bootstrap_sets_bootstrap_project_path(tmp_path):
    """bootstrap_project_discovery stores cwd in _BOOTSTRAP_PROJECT_PATH."""
    pool, conn = _make_mock_conn()

    async def _fake_git(*_args, **_kwargs):
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"abc1234\n", b""))
        return proc

    with (
        patch("grok_mcp.tools.context_cache.is_db_available", return_value=True),
        patch("grok_mcp.tools.context_cache.get_pool", return_value=pool),
        patch("asyncio.create_subprocess_exec", side_effect=_fake_git),
        patch("os.getcwd", return_value=str(tmp_path)),
    ):
        import grok_mcp.tools.context_cache as cc

        await cc.bootstrap_project_discovery()

    assert str(tmp_path) == cc._BOOTSTRAP_PROJECT_PATH
    assert str(tmp_path) == cc._BOOTSTRAP_PROJECT_PATH


# ---------------------------------------------------------------------------
# find_project_for_path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_turn_is_one_locked_transaction():
    """Concurrent turns of one conversation must not compute the same sequence."""
    conn = MagicMock()
    conn.execute = AsyncMock()
    conn.fetchval = AsyncMock(return_value=7)
    conn.executemany = AsyncMock()
    tx = MagicMock()
    tx.__aenter__ = AsyncMock()
    tx.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=tx)
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    with patch("grok_mcp.tools.context_cache.get_pool", return_value=pool):
        from grok_mcp.tools.context_cache import save_turn

        await save_turn("c", "q", "a", "m", usage={"reasoning_tokens": 3})
    assert "pg_advisory_xact_lock" in conn.execute.await_args.args[0]
    rows = conn.executemany.await_args.args[1]
    assert [(r[1], r[2]) for r in rows] == [(7, "user"), (8, "assistant")]
    assert rows[1][9] == 3
    tx.__aenter__.assert_awaited_once()


@pytest.mark.asyncio
async def test_prune_naive_before_timestamp_is_utc():
    from datetime import UTC, datetime

    pool, conn = _make_mock_conn()
    conn.execute = AsyncMock(side_effect=["UPDATE 1"])
    conn.fetchval = AsyncMock(return_value=0)
    with (
        patch("grok_mcp.tools.context_cache.is_db_available", return_value=True),
        patch("grok_mcp.tools.context_cache.get_pool", return_value=pool),
    ):
        from grok_mcp.tools.context_cache import prune_chat_context

        result = await prune_chat_context(conversation="conv-abc", before="2026-01-02T03:04:05")
    cutoff = next(a for a in conn.execute.call_args.args if isinstance(a, datetime))
    assert cutoff == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert result["before"] == "2026-01-02T03:04:05+00:00"


def test_like_escape_makes_wildcards_literal():
    # `_` in save_turn would match any character, and a trailing `\` used to eat the final %.
    from grok_mcp.tools.context_cache import _like_escape

    assert _like_escape("save_turn") == r"save\_turn"
    assert _like_escape("100%") == r"100\%"
    assert _like_escape("a\\") == "a\\\\"


@pytest.mark.parametrize("name", ["p/t ", " p/t", "\tp/t\n"])
def test_conversation_name_drops_surrounding_whitespace(name):
    from grok_mcp.tools.context_cache import check_conversation

    assert check_conversation(name) == "p/t"


@pytest.mark.parametrize("name", ["p/ t", "p/​t", "p/‮t", "p/\x85t"])
def test_conversation_name_rejects_invisible_characters(name):
    from grok_mcp.tools.context_cache import check_conversation

    with pytest.raises(ValueError, match="control"):
        check_conversation(name)
