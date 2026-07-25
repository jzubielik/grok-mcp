"""Tests for search() tool."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _mock_xai_response(answer: str, urls: list[str]) -> MagicMock:
    """Return Response mock as xai_sdk chat.sample() does."""
    response = MagicMock()
    response.content = answer
    response.citations = list(urls)  # Sequence[str], as xai_sdk Response.citations
    return response


@pytest.mark.asyncio
async def test_search_returns_answer_and_sources():
    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_client.chat.create.return_value = mock_chat
    mock_chat.sample.return_value = _mock_xai_response(
        "etcd issues found", ["https://github.com/k3s-io/k3s/issues/1234"]
    )
    with patch("grok_mcp.tools.search.get_image_client", return_value=mock_client):
        from grok_mcp.tools.search import _search_internal

        result = await _search_internal("k3s etcd error", "m")
    assert result["answer"] == "etcd issues found"
    assert result["sources"] == [{"url": "https://github.com/k3s-io/k3s/issues/1234"}]


@pytest.mark.asyncio
async def test_search_uses_web_search_tool():
    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_client.chat.create.return_value = mock_chat
    mock_chat.sample.return_value = _mock_xai_response("ok", [])
    with (
        patch("grok_mcp.tools.search.get_image_client", return_value=mock_client),
        patch("grok_mcp.tools.search.xai_web_search") as mock_ws,
    ):
        from grok_mcp.tools.search import _search_internal

        await _search_internal("test query", "m")
    mock_ws.assert_called_once()
    mock_client.chat.create.assert_called_once()


@pytest.mark.asyncio
async def test_search_handles_no_citations():
    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_client.chat.create.return_value = mock_chat
    mock_chat.sample.return_value = _mock_xai_response("answer", [])
    with patch("grok_mcp.tools.search.get_image_client", return_value=mock_client):
        from grok_mcp.tools.search import _search_internal

        result = await _search_internal("query", "m")
    assert result["sources"] == []


@pytest.mark.asyncio
async def test_search_runs_in_its_own_executor():
    import threading

    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_client.chat.create.return_value = mock_chat
    seen = {}

    def sample():
        seen["thread"] = threading.current_thread().name
        return _mock_xai_response("ok", [])

    mock_chat.sample.side_effect = sample
    with patch("grok_mcp.tools.search.get_image_client", return_value=mock_client):
        from grok_mcp.tools.search import _search_internal

        await _search_internal("q", "m")
    assert seen["thread"].startswith("search")
