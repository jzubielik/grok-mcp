"""Tests for shell() tool."""

from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_shell_returns_grok_analysis():
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "No errors found"
    mock_client.chat.completions.create.return_value = mock_resp

    fake_proc = MagicMock()
    fake_proc.stdout = "line1\nline2\n"
    fake_proc.stderr = ""
    fake_proc.returncode = 0

    with (
        patch("grok_mcp.tools.shell.get_text_client", return_value=mock_client),
        patch("grok_mcp.tools.shell.run_shell", return_value=fake_proc),
    ):
        from grok_mcp.tools.shell import _shell_internal

        result = await _shell_internal("echo hello", "any errors?", model="m")
    assert result == {"answer": "No errors found", "model": "m", "exit_code": 0}
    prompt_arg = mock_client.chat.completions.create.call_args[1]["messages"][0]["content"]
    assert "[exit code: 0]" in prompt_arg
    assert "line1" in prompt_arg
    assert "any errors?" in prompt_arg


@pytest.mark.asyncio
async def test_shell_includes_stderr():
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "stderr found"
    mock_client.chat.completions.create.return_value = mock_resp

    fake_proc = MagicMock()
    fake_proc.stdout = ""
    fake_proc.stderr = "ERROR: something failed"
    fake_proc.returncode = 1

    with (
        patch("grok_mcp.tools.shell.get_text_client", return_value=mock_client),
        patch("grok_mcp.tools.shell.run_shell", return_value=fake_proc),
    ):
        from grok_mcp.tools.shell import _shell_internal

        await _shell_internal("bad cmd", "what went wrong?", model="m")
    prompt_arg = mock_client.chat.completions.create.call_args[1]["messages"][0]["content"]
    assert "[exit code: 1]" in prompt_arg
    assert "ERROR: something failed" in prompt_arg


@pytest.mark.asyncio
async def test_shell_timeout_raises():
    with patch("grok_mcp.tools.shell.run_shell", side_effect=subprocess.TimeoutExpired("cmd", 30)):
        from grok_mcp.tools.shell import _shell_internal

        with pytest.raises(TimeoutError, match="timed out"):
            await _shell_internal("sleep 999", "anything", model="m", timeout_seconds=30)


@pytest.mark.asyncio
async def test_shell_rejects_nan_timeout():
    # min(nan, ceiling) is nan, and subprocess treats a NaN timeout as none at all.
    with patch("grok_mcp.tools.shell.run_shell") as mock_run:
        from grok_mcp.tools.shell import _shell_internal

        with pytest.raises(ValueError, match="NaN"):
            await _shell_internal("sleep 999", "x", model="m", timeout_seconds=float("nan"))
    mock_run.assert_not_called()


@pytest.mark.asyncio
async def test_shell_caps_timeout_at_the_configured_limit():
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "ok"
    mock_client.chat.completions.create.return_value = mock_resp
    fake_proc = MagicMock(stdout="", stderr="", returncode=0)

    with (
        patch("grok_mcp.tools.shell.get_text_client", return_value=mock_client),
        patch("grok_mcp.tools.shell.run_shell", return_value=fake_proc) as mock_run,
    ):
        from grok_mcp.tools.shell import _shell_internal

        await _shell_internal("echo hi", "ok?", model="m", timeout_seconds=9999)
    assert mock_run.call_args.args[1] == 240


def test_timeout_kills_the_whole_process_group(tmp_path):
    """A timed-out command must not leave its children running."""
    import time

    from grok_mcp.tools.shell import run_shell

    marker = tmp_path / "child.pid"
    with pytest.raises(subprocess.TimeoutExpired):
        run_shell(f"sleep 30 & echo $! > {marker}; wait", timeout=0.5)
    child = int(marker.read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        import os

        os.kill(child, 0)


def test_shell_children_get_no_secrets_no_stdin_and_survive_binary(monkeypatch):
    import grok_mcp.auth as auth
    from grok_mcp.tools.shell import run_shell

    monkeypatch.setenv("XAI_API_KEY", "xai-SHOULD-NOT-LEAK")
    monkeypatch.setenv("MCP_AUTH_TOKEN", "tok-SHOULD-NOT-LEAK")
    monkeypatch.setattr(auth, "_secrets", {})
    out = run_shell('echo "$XAI_API_KEY|$MCP_AUTH_TOKEN"; cat; printf "\\377"', 5).stdout
    assert "SHOULD-NOT-LEAK" not in out
    assert out.startswith("|")  # both variables empty; `cat` got EOF from /dev/null at once
    assert "�" in out  # an undecodable byte is replaced, not fatal


def test_shell_output_is_capped(monkeypatch):
    from grok_mcp.tools import shell

    monkeypatch.setattr(shell, "_MAX_SHELL_OUTPUT", 100)
    out = shell.run_shell("head -c 10000 /dev/zero | tr '\\0' x", 5).stdout
    assert out.startswith("x" * 100) and "truncated" in out and "x" * 101 not in out
