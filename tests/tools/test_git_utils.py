"""Tests for git subprocess utilities."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _mock_proc(returncode: int, stdout: bytes):
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, b""))
    return proc


@pytest.mark.asyncio
async def test_get_git_hash_returns_hash():
    proc = _mock_proc(0, b"abc123def456\n")
    with patch(
        "grok_mcp.tools.git_utils.asyncio.create_subprocess_exec",
        AsyncMock(return_value=proc),
    ):
        from grok_mcp.tools.git_utils import _get_git_hash

        result = await _get_git_hash("/some/project")
    assert result == "abc123def456"


@pytest.mark.asyncio
async def test_get_git_hash_returns_none_on_failure():
    proc = _mock_proc(128, b"")
    with patch(
        "grok_mcp.tools.git_utils.asyncio.create_subprocess_exec",
        AsyncMock(return_value=proc),
    ):
        from grok_mcp.tools.git_utils import _get_git_hash

        result = await _get_git_hash("/not-a-repo")
    assert result is None


@pytest.mark.asyncio
async def test_get_git_hash_returns_none_on_exception():
    with patch(
        "grok_mcp.tools.git_utils.asyncio.create_subprocess_exec",
        AsyncMock(side_effect=FileNotFoundError("git not found")),
    ):
        from grok_mcp.tools.git_utils import _get_git_hash

        result = await _get_git_hash("/some/project")
    assert result is None
