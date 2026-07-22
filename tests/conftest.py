"""Shared pytest fixtures for the grok-mcp test suite."""

from __future__ import annotations

import copy
import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mutable module state that tests touch directly or through the code under test. Each test
# starts from the import-time values and its changes are undone, so test order cannot matter.
_MODULE_GLOBALS = (
    ("grok_mcp.models_registry", "_effort_unsupported"),
    ("grok_mcp.auth", "_secrets"),
    ("grok_mcp.config", "settings"),  # test_config reload()s the module
)
_pristine: list[tuple[object, str, object, object]] = []


def _reset(value: object, snapshot: object) -> None:
    # Containers are restored in place: code may hold a reference to the original object.
    if isinstance(value, (set, dict)):
        value.clear()
        value.update(snapshot)


@pytest.fixture(autouse=True)
def _isolate_module_globals():
    if not _pristine:
        for modname, attr in _MODULE_GLOBALS:
            mod = importlib.import_module(modname)
            value = getattr(mod, attr)
            _pristine.append((mod, attr, value, copy.copy(value)))
    for mod, attr, value, snapshot in _pristine:
        _reset(value, snapshot)
        setattr(mod, attr, value)
    yield
    for mod, attr, value, snapshot in _pristine:
        _reset(value, snapshot)
        setattr(mod, attr, value)


@pytest.fixture
def mock_db_pool():
    """Patch grok_mcp.db._pool with a mock that supports `async with pool.acquire() as conn`."""
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_pool.close = AsyncMock()
    with patch("grok_mcp.db._pool", mock_pool):
        yield mock_pool


@pytest.fixture
def no_db(monkeypatch):
    """Simulate the database being unavailable (e.g. a transient outage) — no pool."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import grok_mcp.db as db_module  # noqa: PLC0415

    db_module._pool = None
    yield
    db_module._pool = None


@pytest.fixture(autouse=True)
def test_catalog(monkeypatch):
    """Serve a fixed model catalog so require_model() never hits the network.

    Prices are xAI ticks (1e-10 USD): 10_000/token = $1/M.
    """
    import time  # noqa: PLC0415

    import grok_mcp.models_registry as reg  # noqa: PLC0415
    from grok_mcp.models_registry import ModelInfo  # noqa: PLC0415

    def lang(mid: str, images: bool = True, ctx: int = 1_000_000) -> ModelInfo:
        return ModelInfo(
            mid,
            "language",
            [f"{mid}-latest"],
            ["text", "image"] if images else ["text"],
            ["text"],
            context_length=ctx,
            prompt_price=10_000,
            completion_price=20_000,
            cached_prompt_price=2_500,
        )

    catalog = {
        m.id: m
        for m in [
            lang("m"),
            lang("grok-4.3"),
            lang("grok-4.7", ctx=500_000),
            lang("text-only", images=False),
            lang("tiny-ctx", ctx=1_000),
            ModelInfo(
                "grok-imagine-image",
                "image",
                [],
                ["text", "image"],
                ["image"],
                image_price=200_000_000,
            ),
            ModelInfo(
                "grok-imagine-image-quality",
                "image",
                ["grok-imagine-image-pro"],
                ["text", "image"],
                ["image"],
                image_price=500_000_000,
            ),
        ]
    }
    monkeypatch.setattr(reg, "_catalog", catalog)
    monkeypatch.setattr(reg, "_catalog_fetched_at", time.monotonic())
    monkeypatch.setattr(reg, "_catalog_is_fallback", False)
    yield catalog
