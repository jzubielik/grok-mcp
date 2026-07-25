"""Tool-level fixtures — patches get_text_client at the tool module import path."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_text_client(content: str = "Hello from Grok") -> AsyncMock:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_text_client():
    """Patch get_text_client where the agent loop uses it."""
    mock_client = _make_text_client()
    with patch("grok_mcp.tools.chat.get_text_client", return_value=mock_client):
        yield mock_client
