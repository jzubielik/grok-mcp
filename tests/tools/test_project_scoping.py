# tests/tools/test_project_scoping.py
"""Project context must be scoped to the server's own project, never guessed.

Over HTTP one container serves many projects; resolving the project as "most recently
updated row in project_discovery" attributed calls to an unrelated project.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# get_server_project_path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_project_path_returns_bootstrap_when_known(mock_db_pool):
    import grok_mcp.tools.context_cache as cc

    conn = mock_db_pool.acquire.return_value.__aenter__.return_value
    conn.fetchrow.return_value = {"project_path": "/work/proj"}
    with (
        patch.object(cc, "_BOOTSTRAP_PROJECT_PATH", "/work/proj"),
        patch.object(cc, "is_db_available", return_value=True),
    ):
        assert await cc.get_server_project_path() == "/work/proj"


@pytest.mark.asyncio
async def test_server_project_path_none_when_cwd_not_a_project(mock_db_pool):
    """Container case: CWD is /app, which has no project_discovery row."""
    import grok_mcp.tools.context_cache as cc

    conn = mock_db_pool.acquire.return_value.__aenter__.return_value
    conn.fetchrow.return_value = None
    with (
        patch.object(cc, "_BOOTSTRAP_PROJECT_PATH", "/app"),
        patch.object(cc, "is_db_available", return_value=True),
    ):
        assert await cc.get_server_project_path() is None


# ---------------------------------------------------------------------------
# get_active_project_context is scoped by path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_project_context_queries_given_path(mock_db_pool):
    import grok_mcp.tools.context_cache as cc

    conn = mock_db_pool.acquire.return_value.__aenter__.return_value
    conn.fetchrow.return_value = {"language": "Python", "framework": None}
    with patch.object(cc, "is_db_available", return_value=True):
        result = await cc.get_active_project_context("/work/proj")
    assert "Python" in result
    assert "/work/proj" in conn.fetchrow.await_args.args


@pytest.mark.asyncio
async def test_project_context_none_without_path(mock_db_pool):
    """No server project → inject nothing rather than another project's context."""
    import grok_mcp.tools.context_cache as cc

    with patch.object(cc, "is_db_available", return_value=True):
        assert await cc.get_active_project_context(None) is None
