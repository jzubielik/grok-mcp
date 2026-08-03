"""Guidance for the calling model: flagship defaults to low effort, results carry a hint."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest


def _response(content="ok", prompt_tokens=100):
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = None
    resp.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = 10
    usage.completion_tokens_details.reasoning_tokens = 5
    usage.prompt_tokens_details.cached_tokens = 0
    resp.usage = usage
    return resp


def _rejection(model="grok-4.7"):
    return openai.BadRequestError(
        f"Model {model} does not support parameter reasoningEffort.",
        response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
        body=None,
    )


@pytest.fixture
def client():
    mock = AsyncMock()
    with (
        patch("grok_mcp.tools.chat.get_active_project_context", AsyncMock(return_value=None)),
        patch("grok_mcp.tools.chat.get_server_project_path", AsyncMock(return_value=None)),
        patch("grok_mcp.tools.chat.get_text_client", return_value=mock),
    ):
        yield mock


def _sent(client):
    return [c.kwargs for c in client.chat.completions.create.await_args_list]


@pytest.mark.asyncio
async def test_flagship_without_effort_runs_at_low(client):
    client.chat.completions.create.return_value = _response()
    from grok_mcp.tools.agent import agent

    result = await agent("x", model="grok-4.7")

    assert _sent(client)[0]["reasoning_effort"] == "low"
    assert result["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_non_flagship_without_effort_sends_nothing(client):
    client.chat.completions.create.return_value = _response()
    from grok_mcp.tools.agent import agent

    result = await agent("x", model="grok-4.3")

    assert "reasoning_effort" not in _sent(client)[0]
    assert result["reasoning_effort"] is None


@pytest.mark.asyncio
async def test_explicit_effort_wins_over_the_flagship_default(client):
    client.chat.completions.create.return_value = _response()
    from grok_mcp.tools.agent import agent

    result = await agent("x", model="grok-4.7", reasoning_effort="high")

    assert _sent(client)[0]["reasoning_effort"] == "high"
    assert result["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_rejected_default_is_retried_without_effort_and_remembered(client):
    client.chat.completions.create.side_effect = [_rejection(), _response("a"), _response("b")]
    from grok_mcp.tools.agent import agent

    first = await agent("x", model="grok-4.7")
    second = await agent("y", model="grok-4.7")

    sent = _sent(client)
    assert "reasoning_effort" in sent[0]
    assert "reasoning_effort" not in sent[1]
    assert "reasoning_effort" not in sent[2]
    assert first["answer"] == "a" and first["reasoning_effort"] is None
    assert second["answer"] == "b"


@pytest.mark.asyncio
async def test_rejected_explicit_effort_is_an_error(client):
    client.chat.completions.create.side_effect = _rejection()
    from grok_mcp.tools.agent import agent

    with pytest.raises(openai.BadRequestError):
        await agent("x", model="grok-4.7", reasoning_effort="high")
    assert client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_normal_cheap_run_has_no_hint(client):
    client.chat.completions.create.return_value = _response()
    from grok_mcp.tools.agent import agent

    assert "hint" not in await agent("x", model="grok-4.3")


@pytest.mark.asyncio
async def test_budget_stop_on_flagship_points_at_the_cheapest_tier(client, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("a")
    call = MagicMock()
    call.id = "c1"
    call.function.name = "read"
    call.function.arguments = json.dumps({"path": str(f)})
    first = _response(prompt_tokens=300_000)
    first.choices[0].message.tool_calls = [call]
    client.chat.completions.create.side_effect = [first, _response("over budget")]
    from grok_mcp.tools.agent import agent

    result = await agent("x", model="grok-4.7", tools=["read"], max_cost_usd=0.1)

    assert result["stopped"] == "cost"
    assert "cheapest" in result["hint"] and "flagship" in result["hint"]


@pytest.mark.asyncio
async def test_expensive_flagship_run_gets_a_hint_even_when_it_finished(client):
    # 100K prompt tokens @ $1/M on the test catalog ≈ $0.10; make it pricey with 1M tokens
    client.chat.completions.create.return_value = _response(prompt_tokens=900_000)
    from grok_mcp.tools.agent import agent

    result = await agent("x", model="grok-4.7", max_cost_usd=50)

    assert result["stopped"] == "answer"
    assert "cheapest" in result["hint"]
