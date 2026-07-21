"""Dynamic xAI model registry — live catalog from the API + role → model resolution.

Catalog source of truth: GET /v1/language-models and /v1/image-generation-models. Cached in
memory with a TTL (GROK_MODELS_TTL seconds).

There are no default models: every tool that calls a Grok model takes an explicit `model`,
validated here against the catalog (require_model). Tool descriptions list the current
catalog (describe_models) so the caller can choose deliberately.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import httpx

from .auth import load_api_key
from .config import settings

logger = logging.getLogger("grok_mcp.models_registry")

XAI_API_BASE = "https://api.x.ai"


@dataclass
class ModelInfo:
    id: str
    kind: str  # "language" | "image"
    aliases: list[str]
    input_modalities: list[str]
    output_modalities: list[str]
    context_length: int | None = None
    prompt_price: int | None = None
    completion_price: int | None = None
    cached_prompt_price: int | None = None
    long_context_threshold: int | None = None
    long_context_prompt_price: int | None = None
    long_context_completion_price: int | None = None
    image_price: int | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, [])}


# Last-resort snapshot used only when the API is unreachable and no cache exists.
# Kept intentionally minimal — the live catalog is the source of truth.
_BUILTIN_SNAPSHOT: dict[str, ModelInfo] = {
    m.id: m
    for m in [
        ModelInfo(
            "grok-4.6",
            "language",
            [],
            ["text", "image"],
            ["text"],
            context_length=500_000,
            long_context_threshold=200_000,
        ),
        ModelInfo(
            "grok-4.5",
            "language",
            [],
            ["text", "image"],
            ["text"],
            context_length=500_000,
            long_context_threshold=200_000,
        ),
        ModelInfo(
            "grok-4.3", "language", [], ["text", "image"], ["text"], context_length=1_000_000
        ),
        ModelInfo(
            "grok-4.20-0309-reasoning", "language", [], ["text"], ["text"], context_length=1_000_000
        ),
        ModelInfo(
            "grok-4.20-0309-non-reasoning",
            "language",
            [],
            ["text"],
            ["text"],
            context_length=1_000_000,
        ),
        ModelInfo("grok-build-0.1", "language", [], ["text"], ["text"], context_length=256_000),
        ModelInfo("grok-imagine-image", "image", [], ["text", "image"], ["image"]),
        ModelInfo("grok-imagine-image-2.0", "image", [], ["text", "image"], ["image"]),
        ModelInfo("grok-imagine-image-quality", "image", [], ["text", "image"], ["image"]),
    ]
}

_catalog: dict[str, ModelInfo] | None = None
_catalog_fetched_at: float = 0.0
# After a failed fetch the fallback is cached for this long instead of the full TTL: every
# call must not repeat the slow failing request, yet a recovered API is picked up soon.
_RETRY_TTL_S = 60.0
_catalog_is_fallback = False

# xAI prices are integer "ticks" of 1e-10 USD (per token for text, per image for images).
_TICK_USD = 1e-10


def _catalog_ttl() -> float:
    return settings.models_ttl


def _parse_model(item: dict, kind: str) -> ModelInfo:
    return ModelInfo(
        id=item["id"],
        kind=kind,
        aliases=list(item.get("aliases") or []),
        input_modalities=list(item.get("input_modalities") or []),
        output_modalities=list(item.get("output_modalities") or []),
        context_length=item.get("context_length") or item.get("max_prompt_length"),
        prompt_price=item.get("prompt_text_token_price"),
        completion_price=item.get("completion_text_token_price"),
        cached_prompt_price=item.get("cached_prompt_text_token_price"),
        long_context_threshold=item.get("long_context_threshold"),
        long_context_prompt_price=item.get("prompt_text_token_price_long_context"),
        long_context_completion_price=item.get("completion_text_token_price_long_context"),
        image_price=item.get("image_price"),
    )


_ENDPOINTS = (
    ("/v1/language-models", "language"),
    ("/v1/image-generation-models", "image"),
)


async def _fetch_catalog(client: httpx.AsyncClient | None = None) -> dict[str, ModelInfo]:
    """Fetch and parse the full model catalog from the xAI API."""
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(
            base_url=XAI_API_BASE,
            headers={"Authorization": f"Bearer {load_api_key()}"},
            timeout=30.0,
        )
    try:
        catalog: dict[str, ModelInfo] = {}
        for path, kind in _ENDPOINTS:
            resp = await client.get(path)
            resp.raise_for_status()
            payload = resp.json()
            for item in payload.get("models", payload.get("data", [])):
                info = _parse_model(item, kind)
                catalog[info.id] = info
        # /v1/language-models omits context_length — merge it from /v1/models.
        resp = await client.get("/v1/models")
        resp.raise_for_status()
        payload = resp.json()
        for item in payload.get("data", payload.get("models", [])):
            info = catalog.get(item.get("id"))
            if info is not None and info.context_length is None:
                info.context_length = item.get("context_length")
        return catalog
    finally:
        if own_client:
            await client.aclose()


async def get_catalog(force: bool = False) -> dict[str, ModelInfo]:
    """Return the model catalog, refetching when the TTL expired or force=True.

    Falls back to the last good catalog (or the builtin snapshot) when the API
    is unreachable — never raises on fetch failure.
    """
    global _catalog, _catalog_fetched_at, _catalog_is_fallback  # noqa: PLW0603
    ttl = _RETRY_TTL_S if _catalog_is_fallback else _catalog_ttl()
    fresh = _catalog is not None and (time.monotonic() - _catalog_fetched_at) < ttl
    if fresh and not force:
        return _catalog
    try:
        _catalog = await _fetch_catalog()
        _catalog_is_fallback = False
    except Exception as exc:  # noqa: BLE001
        if _catalog is not None:
            logger.warning("Model catalog refresh failed (%s) — using last good catalog", exc)
        else:
            logger.warning("Model catalog fetch failed (%s) — using builtin snapshot", exc)
            _catalog = dict(_BUILTIN_SNAPSHOT)
        _catalog_is_fallback = True
    _catalog_fetched_at = time.monotonic()
    return _catalog


async def refresh_catalog() -> dict:
    """Force a catalog refetch and return a diff against the previous catalog."""
    old = dict(_catalog) if _catalog is not None else {}
    new = await get_catalog(force=True)
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = []
    watched = (
        "prompt_price",
        "completion_price",
        "context_length",
        "long_context_threshold",
        "image_price",
    )
    for mid in sorted(set(new) & set(old)):
        deltas = {
            f: {"old": getattr(old[mid], f), "new": getattr(new[mid], f)}
            for f in watched
            if getattr(old[mid], f) != getattr(new[mid], f)
        }
        if deltas:
            changed.append({"id": mid, "changes": deltas})
    return {"added": added, "removed": removed, "changed": changed, "count": len(new)}


def find_model(catalog: dict[str, ModelInfo], name: str) -> ModelInfo | None:
    """Look up a model by id or alias."""
    if name in catalog:
        return catalog[name]
    for info in catalog.values():
        if name in info.aliases:
            return info
    return None


def accepts_images(info: ModelInfo) -> bool:
    return "image" in info.input_modalities


def _model_choices(catalog: dict[str, ModelInfo], kind: str, images: bool = False) -> list[str]:
    return sorted(
        m.id for m in catalog.values() if m.kind == kind and (not images or accepts_images(m))
    )


async def require_model(model: str | None, kind: str = "language", images: bool = False) -> str:
    """Validate an explicitly chosen model against the catalog; return its canonical id.

    Raises ValueError naming the valid choices when the model is missing, unknown, of the
    wrong kind, or (images=True) cannot take image input.
    """
    catalog = await get_catalog()
    choices = _model_choices(catalog, kind, images)
    if not model:
        raise ValueError(f"model is required — choose one of {choices} (see list_models)")
    info = find_model(catalog, model)
    if info is None or info.kind != kind or (images and not accepts_images(info)):
        what = ("an " if kind[0] in "aeiou" else "a ") + f"{kind} model"
        what += " accepting image input" if images else ""
        raise ValueError(f"'{model}' is not {what} — choose one of {choices}")
    return info.id


def _usd_per_million(ticks: int | None) -> str:
    return "?" if ticks is None else f"${ticks * _TICK_USD * 1e6:.2f}"


def _newest_first(m: ModelInfo) -> tuple[float, str]:
    """Sort key: highest grok-<version> first (grok-4.7 before grok-4.3), others after."""
    match = re.match(r"grok-(\d+\.\d+)", m.id)
    return (-float(match.group(1)) if match else 0.0, m.id)


_TAG_LEGEND = {
    "cheapest": "lowest price",
    "flagship": "newest and deepest, priciest",
    "code": "tuned for code",
}

# Models that rejected `reasoning_effort` at run time. The catalog does not say which models
# support it, so a server-applied default is dropped for good the first time one refuses.
_effort_unsupported: set[str] = set()


def model_tags(catalog: dict[str, ModelInfo], kind: str = "language") -> dict[str, list[str]]:
    """Role tags derived from the live catalog — never from model names.

    cheapest = lowest in+out price, flagship = newest generation, code = id/alias names it.
    Only language models are tagged; everything else maps to no entry.
    """
    if kind != "language":
        return {}
    models = [m for m in catalog.values() if m.kind == kind]
    priced = [m for m in models if m.prompt_price is not None and m.completion_price is not None]
    tags: dict[str, list[str]] = {m.id: [] for m in models}
    if priced:
        cheapest = min(priced, key=lambda m: (m.prompt_price + m.completion_price, m.id))
        tags[cheapest.id].append("cheapest")
    if models:
        tags[min(models, key=_newest_first).id].append("flagship")
    for m in models:
        if any("build" in n or "code" in n for n in [m.id, *m.aliases]):
            tags[m.id].append("code")
    return tags


def default_reasoning_effort(catalog: dict[str, ModelInfo], model: str) -> str | None:
    """Effort applied when the caller sets none: flagship runs at low, others untouched."""
    if model in _effort_unsupported or "flagship" not in model_tags(catalog).get(model, []):
        return None
    return "low"


def mark_effort_unsupported(model: str) -> None:
    _effort_unsupported.add(model)


def describe_models(catalog: dict[str, ModelInfo], kind: str = "language") -> str:
    """One line per model for tool descriptions: id, context, price and role tags."""
    lines = []
    tags = model_tags(catalog, kind)
    models = sorted((m for m in catalog.values() if m.kind == kind), key=_newest_first)
    for m in models:
        if kind == "image":
            price = "?" if m.image_price is None else f"${m.image_price * _TICK_USD:.3f}/image"
            lines.append(f"  {m.id}: {price}")
        else:
            ctx = f"{m.context_length // 1000}K ctx" if m.context_length else "? ctx"
            images = " [images]" if accepts_images(m) else ""
            role = f" [{', '.join(tags[m.id])}]" if tags.get(m.id) else ""
            lines.append(
                f"  {m.id}{images}: {ctx}, per 1M tokens in {_usd_per_million(m.prompt_price)} "
                f"(cached {_usd_per_million(m.cached_prompt_price)}) / "
                f"out {_usd_per_million(m.completion_price)}{role}"
            )
    used = [t for t in _TAG_LEGEND if any(t in v for v in tags.values())]
    if used:
        lines.insert(0, "  tags: " + "; ".join(f"{t} = {_TAG_LEGEND[t]}" for t in used))
    return "\n".join(lines)


# Used when a model has no live price (the catalog could not be fetched and the builtin
# snapshot has none): deliberately above every current model ($5 in / $15 out per 1M tokens),
# so the cost limit still holds — it just stops a little early.
_FALLBACK_PROMPT_PRICE = 50_000
_FALLBACK_COMPLETION_PRICE = 150_000


def has_price(info: ModelInfo | None) -> bool:
    return info is not None and info.prompt_price is not None and info.completion_price is not None


def cost_usd(info: ModelInfo | None, usage: dict) -> float:
    """Estimated USD cost of `usage` (prompt/cached/completion/reasoning tokens) on `info`.

    Reasoning tokens are billed as completion tokens. Long-context surcharges are ignored —
    this is an estimate for the caller, not an invoice. Without a live price a conservative
    fallback is used (see has_price), never None — the cost limit must always apply.
    """
    if has_price(info):
        prompt_price, completion_price = info.prompt_price, info.completion_price
        cached_price = (
            info.cached_prompt_price if info.cached_prompt_price is not None else prompt_price
        )
    else:
        prompt_price = cached_price = _FALLBACK_PROMPT_PRICE
        completion_price = _FALLBACK_COMPLETION_PRICE
    prompt = usage.get("prompt_tokens", 0)
    cached = min(usage.get("cached_tokens", 0), prompt)
    out = usage.get("completion_tokens", 0) + usage.get("reasoning_tokens", 0)
    ticks = (prompt - cached) * prompt_price + cached * cached_price + out * completion_price
    return round(ticks * _TICK_USD, 6)
