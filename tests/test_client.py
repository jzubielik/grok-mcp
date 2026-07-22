# tests/test_client.py
import asyncio
from unittest.mock import patch

import pytest

from grok_mcp.client import (
    REQUEST_TIMEOUT,
    close_clients,
    get_image_client,
    get_text_client,
    init_clients,
)


def test_get_text_client_raises_before_init():
    asyncio.run(close_clients())
    with pytest.raises(RuntimeError, match="not initialised"):
        get_text_client()


def test_get_image_client_raises_before_init():
    asyncio.run(close_clients())
    with pytest.raises(RuntimeError, match="not initialised"):
        get_image_client()


def test_init_clients_sets_clients(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    with (
        patch("grok_mcp.client.openai.AsyncOpenAI") as mock_oai,
        patch("grok_mcp.client.Client") as mock_xai,
    ):
        init_clients()
        assert get_text_client() is mock_oai.return_value
        assert get_image_client() is mock_xai.return_value
    asyncio.run(close_clients())


def test_text_client_never_retries(monkeypatch):
    """A retry on a timed-out /chat/completions bills the same generation twice."""
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    with patch("grok_mcp.client.openai.AsyncOpenAI") as mock_oai, patch("grok_mcp.client.Client"):
        init_clients()
    assert mock_oai.call_args[1]["max_retries"] == 0
    asyncio.run(close_clients())


def test_text_client_read_timeout_is_explicit(monkeypatch):
    """The SDK default (600 s) must not be inherited — see REQUEST_TIMEOUT."""
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    with patch("grok_mcp.client.openai.AsyncOpenAI") as mock_oai, patch("grok_mcp.client.Client"):
        init_clients()
    timeout = mock_oai.call_args[1]["timeout"]
    assert timeout is REQUEST_TIMEOUT
    assert timeout.read == 300.0
    assert timeout.connect == 5.0
    asyncio.run(close_clients())


def test_grpc_client_has_a_deadline(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    with patch("grok_mcp.client.openai.AsyncOpenAI"), patch("grok_mcp.client.Client") as mock_xai:
        init_clients()
    assert mock_xai.call_args.kwargs["timeout"] == 300
    asyncio.run(close_clients())
