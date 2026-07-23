# tests/test_models_registry.py
"""Tests for the dynamic xAI model registry (catalog + role resolution)."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

LANG_PAYLOAD = {
    "models": [
        {
            "id": "grok-4.3",
            "aliases": ["grok-4.3-latest"],
            "version": "1.0",
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
            "prompt_text_token_price": 12500,
            "cached_prompt_text_token_price": 1250,
            "completion_text_token_price": 25000,
            "prompt_text_token_price_long_context": 25000,
            "completion_text_token_price_long_context": 50000,
            "long_context_threshold": 200000,
        },
        {
            "id": "grok-4.6",
            "aliases": ["grok-4.6-latest"],
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "prompt_text_token_price": 20000,
            "completion_text_token_price": 60000,
            "long_context_threshold": 200000,
        },
        {
            "id": "grok-build-0.1",
            "aliases": [],
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "prompt_text_token_price": 10000,
            "completion_text_token_price": 20000,
        },
    ]
}

# /v1/language-models does NOT return context_length — only /v1/models does.
MODELS_PAYLOAD = {
    "data": [
        {"id": "grok-4.3", "context_length": 1000000},
        {"id": "grok-4.6", "context_length": 500000},
        {"id": "grok-build-0.1", "context_length": 256000},
        {"id": "grok-imagine-image-2.0", "context_length": 8000},
    ]
}

IMAGE_PAYLOAD = {
    "models": [
        {
            "id": "grok-imagine-image-2.0",
            "aliases": ["grok-imagine-image-latest"],
            "input_modalities": ["text", "image"],
            "output_modalities": ["image"],
            "image_price": 4,
            "max_prompt_length": 10000,
        }
    ]
}


def _mock_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/language-models":
            return httpx.Response(200, json=LANG_PAYLOAD)
        if path == "/v1/image-generation-models":
            return httpx.Response(200, json=IMAGE_PAYLOAD)
        if path == "/v1/models":
            return httpx.Response(200, json=MODELS_PAYLOAD)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _reset_registry(monkeypatch):
    """Fresh registry state and a fake API key for every test."""
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    import grok_mcp.models_registry as reg

    reg._catalog = None
    reg._catalog_fetched_at = 0.0
    reg._catalog_is_fallback = False
    yield
    reg._catalog = None
    reg._catalog_fetched_at = 0.0
    reg._catalog_is_fallback = False


@pytest.fixture
def catalog_from_mock_api():
    """Patch _fetch_catalog to go through httpx.MockTransport (real parsing, no network)."""
    import grok_mcp.models_registry as reg

    real_fetch = reg._fetch_catalog

    async def fetch_via_mock():
        async with httpx.AsyncClient(
            transport=_mock_transport(), base_url=reg.XAI_API_BASE
        ) as client:
            return await real_fetch(client=client)

    with patch.object(reg, "_fetch_catalog", side_effect=fetch_via_mock) as mock:
        mock.original = real_fetch
        yield mock


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_catalog_parses_language_model_fields(catalog_from_mock_api):
    from grok_mcp.models_registry import get_catalog

    catalog = await get_catalog()
    m = catalog["grok-4.3"]
    assert m.kind == "language"
    assert m.aliases == ["grok-4.3-latest"]
    assert m.context_length == 1000000
    assert m.long_context_threshold == 200000
    assert m.prompt_price == 12500
    assert m.completion_price == 25000
    assert "image" in m.input_modalities


@pytest.mark.asyncio
async def test_fetch_catalog_parses_image_models(catalog_from_mock_api):
    from grok_mcp.models_registry import get_catalog

    catalog = await get_catalog()
    img = catalog["grok-imagine-image-2.0"]
    assert img.kind == "image"
    assert img.image_price == 4


# ---------------------------------------------------------------------------
# Caching / TTL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_catalog_uses_cache_within_ttl(catalog_from_mock_api):
    from grok_mcp.models_registry import get_catalog

    await get_catalog()
    await get_catalog()
    assert catalog_from_mock_api.call_count == 1


@pytest.mark.asyncio
async def test_get_catalog_refetches_after_ttl_expiry(catalog_from_mock_api):
    import grok_mcp.models_registry as reg

    await reg.get_catalog()
    reg._catalog_fetched_at -= reg._catalog_ttl() + 1
    await reg.get_catalog()
    assert catalog_from_mock_api.call_count == 2


@pytest.mark.asyncio
async def test_get_catalog_force_refetches(catalog_from_mock_api):
    from grok_mcp.models_registry import get_catalog

    await get_catalog()
    await get_catalog(force=True)
    assert catalog_from_mock_api.call_count == 2


@pytest.mark.asyncio
async def test_fetch_failure_falls_back_to_builtin_snapshot():
    import grok_mcp.models_registry as reg

    with patch.object(reg, "_fetch_catalog", side_effect=httpx.ConnectError("down")):
        catalog = await reg.get_catalog()
    assert catalog  # builtin snapshot is non-empty
    assert all(m.kind in ("language", "image") for m in catalog.values())


@pytest.mark.asyncio
async def test_fetch_failure_keeps_last_good_catalog(catalog_from_mock_api):
    import grok_mcp.models_registry as reg

    await reg.get_catalog()
    catalog_from_mock_api.side_effect = httpx.ConnectError("down")
    reg._catalog_fetched_at -= reg._catalog_ttl() + 1
    catalog = await reg.get_catalog()
    assert "grok-4.3" in catalog


@pytest.mark.asyncio
async def test_failed_first_fetch_is_cached_briefly_then_retried(catalog_from_mock_api):
    import grok_mcp.models_registry as reg

    catalog_from_mock_api.side_effect = httpx.ConnectError("down")
    first = await reg.get_catalog()
    second = await reg.get_catalog()
    assert first == second == reg._BUILTIN_SNAPSHOT
    assert catalog_from_mock_api.call_count == 1  # the failing fetch is not repeated
    reg._catalog_fetched_at -= reg._RETRY_TTL_S + 1
    catalog_from_mock_api.side_effect = None
    catalog_from_mock_api.return_value = {"live-model": reg._BUILTIN_SNAPSHOT["grok-4.3"]}
    assert "live-model" in await reg.get_catalog()
    assert catalog_from_mock_api.call_count == 2


# ---------------------------------------------------------------------------
# Refresh diff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_catalog_reports_added_and_removed(catalog_from_mock_api):
    import copy

    import grok_mcp.models_registry as reg

    await reg.get_catalog()

    new_lang = copy.deepcopy(LANG_PAYLOAD)
    new_lang["models"] = [m for m in new_lang["models"] if m["id"] != "grok-build-0.1"]
    new_lang["models"].append({"id": "grok-5", "aliases": [], "context_length": 2000000})
    new_lang["models"][0]["prompt_text_token_price"] = 99999

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/language-models":
            return httpx.Response(200, json=new_lang)
        if path == "/v1/image-generation-models":
            return httpx.Response(200, json=IMAGE_PAYLOAD)
        return httpx.Response(200, json=MODELS_PAYLOAD)

    async def fetch_v2():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url=reg.XAI_API_BASE
        ) as client:
            return await catalog_from_mock_api.original(client=client)

    catalog_from_mock_api.side_effect = fetch_v2
    diff = await reg.refresh_catalog()
    assert "grok-5" in diff["added"]
    assert "grok-build-0.1" in diff["removed"]
    assert any(c["id"] == "grok-4.3" for c in diff["changed"])


# ---------------------------------------------------------------------------
# Explicit model choice: validation, description listing, cost
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_require_model_returns_canonical_id_for_alias(catalog_from_mock_api):
    from grok_mcp.models_registry import require_model

    assert await require_model("grok-4.3-latest") == "grok-4.3"


@pytest.mark.asyncio
async def test_require_model_rejects_missing_and_unknown(catalog_from_mock_api):
    from grok_mcp.models_registry import require_model

    with pytest.raises(ValueError, match="model is required"):
        await require_model(None)
    with pytest.raises(ValueError, match="not a language model"):
        await require_model("nope")


@pytest.mark.asyncio
async def test_require_model_checks_kind(catalog_from_mock_api):
    from grok_mcp.models_registry import require_model

    with pytest.raises(ValueError, match="not a language model"):
        await require_model("grok-imagine-image-2.0")
    assert await require_model("grok-imagine-image-2.0", kind="image") == "grok-imagine-image-2.0"


def test_describe_models_lists_context_prices_and_image_support():
    from grok_mcp.models_registry import ModelInfo, describe_models

    catalog = {
        "a": ModelInfo(
            "a",
            "language",
            [],
            ["text", "image"],
            ["text"],
            context_length=500_000,
            prompt_price=20_000,
            completion_price=60_000,
            cached_prompt_price=5_000,
        ),
        "img": ModelInfo("img", "image", [], ["text"], ["image"], image_price=200_000_000),
    }
    text = describe_models(catalog)
    assert "a [images]: 500K ctx" in text
    assert "in $2.00 (cached $0.50) / out $6.00" in text
    assert "img" not in text
    assert describe_models(catalog, "image") == "  img: $0.020/image"


def test_cost_usd_bills_cached_and_reasoning_tokens():
    from grok_mcp.models_registry import ModelInfo, cost_usd

    info = ModelInfo(
        "a",
        "language",
        [],
        [],
        [],
        prompt_price=20_000,
        completion_price=60_000,
        cached_prompt_price=5_000,
    )
    usage = {
        "prompt_tokens": 1_000_000,
        "cached_tokens": 600_000,
        "completion_tokens": 1_000,
        "reasoning_tokens": 9_000,
    }
    # 400K @ $2/M + 600K @ $0.50/M + 10K out @ $6/M
    assert cost_usd(info, usage) == pytest.approx(0.8 + 0.3 + 0.06)
    # Unknown model: the conservative fallback ($5 in incl. cached, $15 out), never None.
    assert cost_usd(None, usage) == pytest.approx(5.0 + 0.15)


def _lang(mid, prompt, completion, aliases=()):
    from grok_mcp.models_registry import ModelInfo

    return ModelInfo(
        mid,
        "language",
        list(aliases),
        ["text"],
        ["text"],
        context_length=500_000,
        prompt_price=prompt,
        completion_price=completion,
    )


def test_model_tags_derive_cheapest_flagship_and_code_from_the_catalog():
    from grok_mcp.models_registry import model_tags

    catalog = {
        m.id: m
        for m in [
            _lang("grok-4.7", 20_000, 60_000),
            _lang("grok-4.3", 12_500, 25_000),
            _lang("grok-build-0.1", 10_000, 20_000, aliases=["grok-code-fast-1"]),
        ]
    }
    tags = model_tags(catalog)
    assert tags["grok-4.7"] == ["flagship"]
    assert tags["grok-4.3"] == []
    assert tags["grok-build-0.1"] == ["cheapest", "code"]


def test_model_tags_follow_the_catalog_not_model_names():
    from grok_mcp.models_registry import model_tags

    catalog = {
        m.id: m for m in [_lang("grok-5.0", 30_000, 90_000), _lang("grok-4.7", 20_000, 60_000)]
    }
    tags = model_tags(catalog)
    assert tags["grok-5.0"] == ["flagship"]
    assert tags["grok-4.7"] == ["cheapest"]


def test_model_tags_single_model_is_both_and_images_get_none():
    from grok_mcp.models_registry import ModelInfo, model_tags

    catalog = {"only": _lang("only", 1, 2)}
    assert model_tags(catalog) == {"only": ["cheapest", "flagship"]}
    img = {"i": ModelInfo("i", "image", [], ["text"], ["image"], image_price=1)}
    assert model_tags(img, "image") == {}


def test_describe_models_appends_tags_and_a_legend():
    from grok_mcp.models_registry import describe_models

    catalog = {
        m.id: m for m in [_lang("grok-4.7", 20_000, 60_000), _lang("grok-4.3", 12_500, 25_000)]
    }
    text = describe_models(catalog)
    lines = {ln.split(":")[0].strip(): ln for ln in text.splitlines() if "ctx" in ln}
    assert lines["grok-4.7"].endswith("[flagship]")
    assert lines["grok-4.3"].endswith("[cheapest]")
    assert "cheapest" in text.splitlines()[0] and "flagship" in text.splitlines()[0]


def test_flagship_defaults_to_low_effort_until_the_model_rejects_it():
    from grok_mcp import models_registry as reg

    catalog = {
        m.id: m for m in [_lang("grok-4.7", 20_000, 60_000), _lang("grok-4.3", 12_500, 25_000)]
    }
    assert reg.default_reasoning_effort(catalog, "grok-4.7") == "low"
    assert reg.default_reasoning_effort(catalog, "grok-4.3") is None
    reg.mark_effort_unsupported("grok-4.7")
    assert reg.default_reasoning_effort(catalog, "grok-4.7") is None


def test_cost_without_live_price_uses_conservative_fallback():
    from grok_mcp.models_registry import ModelInfo, cost_usd, has_price

    unpriced = ModelInfo("x", "language", [], ["text"], ["text"])
    assert not has_price(unpriced)
    # 1M prompt + 1M completion tokens at the $5 / $15 fallback.
    assert cost_usd(unpriced, {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}) == 20.0
    assert cost_usd(None, {"prompt_tokens": 0}) == 0.0


@pytest.mark.asyncio
async def test_list_models_refresh_reports_changes(catalog_from_mock_api):
    from grok_mcp.tools import models

    plain = await models.list_models()
    assert "changes" not in plain

    with patch.object(models, "refresh_tool_descriptions") as refresh_descriptions:
        result = await models.list_models(kind="image", refresh=True)

    refresh_descriptions.assert_awaited_once()
    assert set(result["changes"]) == {"added", "removed", "changed", "count"}
    assert result["models"] and all(m["kind"] == "image" for m in result["models"])
