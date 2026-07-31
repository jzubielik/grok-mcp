# tests/tools/test_effort_and_usage.py
"""Controllable reasoning_effort + real token accounting.

grok-4.x models think by default and bill reasoning tokens as output. Effort is chosen
per call (no per-role defaults); usage is captured so every result can report tokens
and cost.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _response(content="ok", prompt_tokens=100, completion_tokens=50, reasoning_tokens=400):
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = None
    resp.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.completion_tokens_details.reasoning_tokens = reasoning_tokens
    resp.usage = usage
    return resp


@pytest.fixture
def chat_env():
    with (
        patch("grok_mcp.tools.chat.get_active_project_context", AsyncMock(return_value=None)),
        patch("grok_mcp.tools.chat.get_server_project_path", AsyncMock(return_value=None)),
        patch("grok_mcp.tools.chat.save_turn", AsyncMock()) as save,
    ):
        yield save


# ---------------------------------------------------------------------------
# reasoning_effort is per call only, passed through as-is
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_omits_effort_when_unset(chat_env):
    client = AsyncMock()
    client.chat.completions.create.return_value = _response()
    with patch("grok_mcp.tools.chat.get_text_client", return_value=client):
        from grok_mcp.tools.chat import _chat_internal

        await _chat_internal("hi", model="m", conversation="c1")

    assert "reasoning_effort" not in client.chat.completions.create.await_args.kwargs


@pytest.mark.asyncio
async def test_chat_sends_per_call_effort(chat_env):
    client = AsyncMock()
    client.chat.completions.create.return_value = _response()
    with patch("grok_mcp.tools.chat.get_text_client", return_value=client):
        from grok_mcp.tools.chat import _chat_internal

        await _chat_internal("hi", model="m", conversation="c1", reasoning_effort="high")

    assert client.chat.completions.create.await_args.kwargs["reasoning_effort"] == "high"


# ---------------------------------------------------------------------------
# Usage capture
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_usage_is_collected_into_sink(chat_env):
    client = AsyncMock()
    client.chat.completions.create.return_value = _response(
        prompt_tokens=120, completion_tokens=30, reasoning_tokens=900
    )
    sink: dict = {}
    with patch("grok_mcp.tools.chat.get_text_client", return_value=client):
        from grok_mcp.tools.chat import _chat_internal

        await _chat_internal("hi", model="m", conversation="c1", usage_sink=sink)

    assert sink["prompt_tokens"] == 120
    assert sink["completion_tokens"] == 30
    assert sink["reasoning_tokens"] == 900


@pytest.mark.asyncio
async def test_usage_is_persisted_with_assistant_message(chat_env):
    client = AsyncMock()
    client.chat.completions.create.return_value = _response(
        prompt_tokens=10, completion_tokens=5, reasoning_tokens=7
    )
    with patch("grok_mcp.tools.chat.get_text_client", return_value=client):
        from grok_mcp.tools.chat import _chat_internal

        await _chat_internal("hi", model="m", conversation="c1")

    assistant_call = chat_env.await_args_list[-1]
    assert assistant_call.kwargs["usage"]["reasoning_tokens"] == 7


@pytest.mark.asyncio
async def test_job_records_usage(mock_db_pool):
    import grok_mcp.tools.jobs as jobs

    conn = mock_db_pool.acquire.return_value.__aenter__.return_value
    sink = {"prompt_tokens": 11, "completion_tokens": 22, "reasoning_tokens": 33}

    async def work():
        return "done"

    with patch.object(jobs, "is_db_available", return_value=True):
        await jobs.start_job("reason", work, usage_sink=sink)
        import asyncio

        for _ in range(50):
            if conn.execute.await_count >= 2:
                break
            await asyncio.sleep(0.01)

    update = conn.execute.await_args_list[-1].args
    assert 33 in update  # reasoning tokens persisted on the job row
