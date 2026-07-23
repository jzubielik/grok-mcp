"""MCP tools exposing the live model catalog, and catalog-driven tool descriptions.

There are no default models — callers pass `model` explicitly. To make that choice
informed, every model-taking tool's description carries a `{models}` placeholder that is
filled from the live catalog at startup and on list_models(refresh=True).
"""

from __future__ import annotations

from ..models_registry import describe_models, get_catalog, refresh_catalog
from ..server import mcp

# Tools whose description lists catalog models → which kind of model they take.
_MODEL_LIST_TOOLS = {
    "agent": "language",
    "search": "language",
    "shell": "language",
    "read": "language",
    "generate_image": "image",
}
_templates: dict[str, str] = {}


async def refresh_tool_descriptions() -> None:
    """Fill each tool's `{models}` placeholder with the current catalog listing."""
    catalog = await get_catalog()
    for name, kind in _MODEL_LIST_TOOLS.items():
        tool = mcp._tool_manager.get_tool(name)
        if tool is None:
            continue
        template = _templates.setdefault(name, tool.description)
        tool.description = template.replace("{models}", describe_models(catalog, kind))


@mcp.tool(
    description=(
        "List the xAI models currently available on the API, with pricing, context window, "
        "modalities and aliases. Data comes from the live xAI catalog (cached, TTL "
        "GROK_MODELS_TTL, default 24h). kind: 'language' | 'image' to filter. "
        "refresh=true refetches the catalog, adds 'changes' (added/removed model ids and "
        "price/context changes since the last fetch) and refreshes the model lists in tool "
        "descriptions (clients see them after re-listing tools). "
        "Prices are xAI ticks: 1 tick = 1e-10 USD per token (text) or per image."
    )
)
async def list_models(kind: str | None = None, refresh: bool = False) -> dict:
    changes = None
    if refresh:
        changes = await refresh_catalog()
        await refresh_tool_descriptions()
    catalog = await get_catalog()
    models = [m.to_dict() for m in catalog.values() if kind is None or m.kind == kind]
    result = {"models": sorted(models, key=lambda m: (m["kind"], m["id"])), "count": len(models)}
    if changes is not None:
        result["changes"] = changes
    return result
