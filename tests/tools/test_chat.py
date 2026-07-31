# tests/tools/test_chat.py
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_response(content="Hello from Grok"):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    mock_response.choices[0].message.tool_calls = None
    return mock_response


@pytest.fixture
def mock_text_client():
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = _make_response()

    with patch("grok_mcp.tools.chat.get_text_client", return_value=mock_client):
        yield mock_client


@pytest.fixture(autouse=True)
def mock_save_turn():
    """Keep history writes away from a real DB in all chat tests."""
    with patch("grok_mcp.tools.chat.save_turn", AsyncMock()):
        yield


def _system_msgs(mock_client):
    messages = mock_client.chat.completions.create.call_args[1]["messages"]
    return [m for m in messages if m["role"] == "system"]


@pytest.mark.asyncio
async def test_chat_returns_text(mock_text_client):
    from grok_mcp.tools.chat import _chat_internal

    result = await _chat_internal("Hello!", model="grok-4.3")
    assert result["answer"] == "Hello from Grok"
    mock_text_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_chat_system_contains_only_history_note_when_no_context(mock_text_client):
    from grok_mcp.tools.chat import _chat_internal

    with patch("grok_mcp.tools.chat.get_active_project_context", AsyncMock(return_value=None)):
        await _chat_internal("Hello!", model="m", conversation="p/t")
    system_msgs = _system_msgs(mock_text_client)
    assert len(system_msgs) == 1
    assert "get_recent_history" in system_msgs[0]["content"]  # history note only


@pytest.mark.asyncio
async def test_chat_injects_context_as_system_when_no_system_given(mock_text_client):
    from grok_mcp.tools.chat import _chat_internal

    ctx = "[Project Context]\nLanguage: Python"
    with patch("grok_mcp.tools.chat.get_active_project_context", AsyncMock(return_value=ctx)):
        await _chat_internal("Hello!", model="m")
    system_msgs = _system_msgs(mock_text_client)
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"].startswith(ctx)


@pytest.mark.asyncio
async def test_chat_prepends_context_before_caller_system(mock_text_client):
    from grok_mcp.tools.chat import _chat_internal

    ctx = "[Project Context]\nLanguage: Python"
    caller_system = "You are a test expert."
    with patch("grok_mcp.tools.chat.get_active_project_context", AsyncMock(return_value=ctx)):
        await _chat_internal("Hello!", system=caller_system, model="m")
    content = _system_msgs(mock_text_client)[0]["content"]
    assert content.startswith(ctx + "\n\n" + caller_system)


@pytest.mark.asyncio
async def test_chat_skips_injection_when_system_already_has_marker(mock_text_client):
    from grok_mcp.tools.chat import _chat_internal

    ctx = "[Project Context]\nLanguage: Python"
    existing = "[Project Context]\nLanguage: Go"
    with patch("grok_mcp.tools.chat.get_active_project_context", AsyncMock(return_value=ctx)):
        await _chat_internal("Hello!", system=existing, model="m")
    content = _system_msgs(mock_text_client)[0]["content"]
    assert content.startswith(existing)
    assert "Language: Python" not in content


# ---------------------------------------------------------------------------
# conversation persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_saves_user_and_assistant_turns(mock_text_client):
    from grok_mcp.tools.chat import _chat_internal

    with (
        patch("grok_mcp.tools.chat.get_active_project_context", AsyncMock(return_value=None)),
        patch("grok_mcp.tools.chat.save_turn", AsyncMock()) as mock_save,
    ):
        result = await _chat_internal("Q2", conversation="conv-abc", model="m")
    assert result["answer"] == "Hello from Grok"
    mock_save.assert_awaited_once()
    assert mock_save.await_args.args == ("conv-abc", "Q2", "Hello from Grok", "m")


@pytest.mark.asyncio
async def test_chat_with_file_paths_injects_content(mock_text_client, tmp_path):
    from grok_mcp.tools.chat import _chat_internal

    f = tmp_path / "notes.txt"
    f.write_text("File content here")
    with (
        patch("grok_mcp.tools.chat.get_active_project_context", AsyncMock(return_value=None)),
        patch("grok_mcp.tools.chat.save_turn", AsyncMock()),
    ):
        await _chat_internal("Summarize it", file_paths=[str(f)], model="m")
    call_kwargs = mock_text_client.chat.completions.create.call_args[1]
    user_msg = call_kwargs["messages"][-1]["content"]
    assert "File content here" in user_msg
    assert "Summarize it" in user_msg


@pytest.mark.asyncio
async def test_chat_missing_file_fails_before_any_api_call(mock_text_client, tmp_path):
    """A mistyped file path must not produce a billed answer about nothing."""
    from grok_mcp.tools.chat import _chat_internal

    with (
        patch("grok_mcp.tools.chat.get_active_project_context", AsyncMock(return_value=None)),
        patch("grok_mcp.tools.chat.save_turn", AsyncMock()),
        pytest.raises(FileNotFoundError, match="file not found"),
    ):
        await _chat_internal("Hello", file_paths=[str(tmp_path / "missing.txt")], model="m")
    mock_text_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_chat_skips_git_check_when_no_project_path(mock_text_client):
    from grok_mcp.tools.chat import _chat_internal

    with (
        patch("grok_mcp.tools.chat.get_active_project_context", AsyncMock(return_value=None)),
        patch("grok_mcp.tools.chat.save_turn", AsyncMock()),
        patch("grok_mcp.tools.chat._get_git_hash") as mock_git,
    ):
        await _chat_internal("Hello!", conversation="conv-noproj", model="m")
    mock_git.assert_not_called()


# ---------------------------------------------------------------------------
# request timeouts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_history_write_failure_keeps_the_answer(mock_text_client, caplog):
    """A failed history write is logged; the paid answer still reaches the caller."""
    import logging

    from grok_mcp.tools.chat import _chat_internal

    with (
        patch("grok_mcp.tools.chat.get_active_project_context", AsyncMock(return_value=None)),
        patch("grok_mcp.tools.chat.save_turn", AsyncMock(side_effect=RuntimeError("db"))),
        caplog.at_level(logging.WARNING),
    ):
        result = await _chat_internal("Hello", model="m", conversation="c")
    assert result["answer"] == "Hello from Grok"
    assert any(r.getMessage() == "history_save_failed" for r in caplog.records)


def test_within_caps_per_call_values():
    from grok_mcp.tools.chat import _within

    assert _within(None, 100) == 100
    assert _within(10, 100) == 10
    assert _within(1000, 100) == 100
    assert _within(0, 100) == 100


def test_within_rejects_nan():
    # JSON-RPC accepts NaN, and `cost >= nan` is never true: the ceiling would not hold.
    from grok_mcp.tools.chat import _within

    with pytest.raises(ValueError, match="NaN"):
        _within(float("nan"), 2.5)
