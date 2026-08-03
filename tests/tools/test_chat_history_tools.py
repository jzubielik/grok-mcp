# tests/tools/test_chat_history_tools.py
"""Model-driven history: chat() offers history tools instead of blind injection."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# History accessors in context_cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_recent_history_returns_messages(mock_db_pool):
    conn = mock_db_pool.acquire.return_value.__aenter__.return_value
    conn.fetch.return_value = [
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "earlier question"},
    ]
    from grok_mcp.tools.context_cache import get_recent_history

    result = await get_recent_history("conv-1", limit=2)
    assert result == [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]


@pytest.mark.asyncio
async def test_search_history_passes_query(mock_db_pool):
    conn = mock_db_pool.acquire.return_value.__aenter__.return_value
    conn.fetch.return_value = [{"role": "assistant", "content": "found it"}]
    from grok_mcp.tools.context_cache import search_history

    result = await search_history("conv-1", "needle", limit=3)
    assert result == [{"role": "assistant", "content": "found it"}]
    args = conn.fetch.await_args.args
    assert "%needle%" in args


@pytest.mark.asyncio
async def test_history_accessors_return_empty_without_db(no_db):
    from grok_mcp.tools.context_cache import get_recent_history, search_history

    assert await get_recent_history("conv-1") == []
    assert await search_history("conv-1", "x") == []


# ---------------------------------------------------------------------------
# chat(): no blind injection, history tools offered and executed on demand
# ---------------------------------------------------------------------------


def _response(content=None, tool_calls=None):
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = tool_calls
    resp.choices = [choice]
    return resp


def _tool_call(call_id: str, name: str, arguments: dict):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


@pytest.fixture
def chat_env():
    """Patch chat module DB/project plumbing for isolated _chat_internal runs."""
    with (
        patch("grok_mcp.tools.chat.get_active_project_context", AsyncMock(return_value=None)),
        patch("grok_mcp.tools.chat.get_server_project_path", AsyncMock(return_value=None)),
        patch("grok_mcp.tools.chat.save_turn", AsyncMock()) as save,
    ):
        yield {"save_turn": save}


@pytest.mark.asyncio
async def test_chat_does_not_inject_history(chat_env):
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = _response(content="hi")
    with patch("grok_mcp.tools.chat.get_text_client", return_value=mock_client):
        from grok_mcp.tools.chat import _chat_internal

        await _chat_internal("Hello", model="m", conversation="conv-1")

    kwargs = mock_client.chat.completions.create.await_args.kwargs
    roles = [m["role"] for m in kwargs["messages"]]
    assert roles.count("user") == 1
    assert "assistant" not in roles  # no history pre-injected


@pytest.mark.asyncio
async def test_chat_offers_history_tools_when_conversation_active(chat_env):
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = _response(content="hi")
    with patch("grok_mcp.tools.chat.get_text_client", return_value=mock_client):
        from grok_mcp.tools.chat import _chat_internal

        await _chat_internal("Hello", model="m", conversation="conv-1")

    kwargs = mock_client.chat.completions.create.await_args.kwargs
    names = [t["function"]["name"] for t in kwargs["tools"]]
    assert {"get_recent_history", "search_history"} <= set(names)


@pytest.mark.asyncio
async def test_chat_executes_history_tool_call_and_loops(chat_env):
    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("t1", "get_recent_history", {"limit": 2})]),
        _response(content="final answer"),
    ]
    history = [{"role": "user", "content": "earlier"}]
    with (
        patch("grok_mcp.tools.chat.get_text_client", return_value=mock_client),
        patch("grok_mcp.tools.chat.get_recent_history", AsyncMock(return_value=history)) as recent,
    ):
        from grok_mcp.tools.chat import _chat_internal

        result = await _chat_internal("What did we discuss?", model="m", conversation="conv-1")

    assert result["answer"] == "final answer"
    recent.assert_awaited_once_with("conv-1", limit=2)
    second_kwargs = mock_client.chat.completions.create.await_args_list[1].kwargs
    tool_msgs = [m for m in second_kwargs["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "earlier" in tool_msgs[0]["content"]


# ---------------------------------------------------------------------------
# conversation: named task history vs. stateless calls
# ---------------------------------------------------------------------------


def _system_text(kwargs) -> str:
    return "".join(m["content"] for m in kwargs["messages"] if m["role"] == "system")


@pytest.mark.asyncio
async def test_call_without_conversation_is_stateless(chat_env):
    """No conversation: nothing saved, no history tools, no history note."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = _response(content="hi")
    with patch("grok_mcp.tools.chat.get_text_client", return_value=mock_client):
        from grok_mcp.tools.chat import _HISTORY_NOTE, _chat_internal

        await _chat_internal("Hello", model="m")

    kwargs = mock_client.chat.completions.create.await_args.kwargs
    names = [t["function"]["name"] for t in kwargs.get("tools") or []]
    assert "get_recent_history" not in names and "search_history" not in names
    assert _HISTORY_NOTE not in _system_text(kwargs)
    chat_env["save_turn"].assert_not_awaited()


@pytest.mark.asyncio
async def test_conversation_is_stored_under_the_exact_name(chat_env):
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = _response(content="hi")
    with patch("grok_mcp.tools.chat.get_text_client", return_value=mock_client):
        from grok_mcp.tools.chat import _HISTORY_NOTE, _chat_internal

        await _chat_internal("Hello", model="m", conversation="grok-mcp/security-audit")

    assert chat_env["save_turn"].await_args.args[0] == "grok-mcp/security-audit"
    assert _HISTORY_NOTE in _system_text(mock_client.chat.completions.create.await_args.kwargs)


@pytest.mark.asyncio
async def test_history_guidance_is_identical_across_conversations(chat_env):
    """The system prompt must not vary with the conversation name (xAI prompt caching)."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = _response(content="hi")
    with patch("grok_mcp.tools.chat.get_text_client", return_value=mock_client):
        from grok_mcp.tools.chat import _chat_internal

        await _chat_internal("Hello", model="m", conversation="proj-a/task-1")
        first = mock_client.chat.completions.create.await_args.kwargs
        await _chat_internal("Hello", model="m", conversation="proj-b/task-2")
        second = mock_client.chat.completions.create.await_args.kwargs

    assert _system_text(first) == _system_text(second)
    assert first["tools"] == second["tools"]
    assert "task-1" not in json.dumps(first["tools"]) + _system_text(first)


@pytest.mark.parametrize("bad", ["", "   ", "x" * 201, "a\nb", "a\x00b"])
@pytest.mark.asyncio
async def test_agent_rejects_invalid_conversation_before_queueing(bad):
    with patch("grok_mcp.tools.chat.start_job", AsyncMock(return_value="job-1")) as start:
        from grok_mcp.tools.agent import agent

        with pytest.raises(ValueError, match="conversation"):
            await agent("x", model="m", conversation=bad, background=True)
    start.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_accepts_a_200_char_conversation():
    with patch("grok_mcp.tools.chat.start_job", AsyncMock(return_value="job-1")) as start:
        from grok_mcp.tools.agent import agent

        await agent("x", model="m", conversation="p/" + "t" * 198, background=True)
    start.assert_awaited_once()
