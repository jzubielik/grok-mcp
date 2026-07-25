"""search() tool — xAI web search via native xai_sdk Agent Tools API."""

from __future__ import annotations

import asyncio
import concurrent.futures

from xai_sdk.chat import user as xai_user
from xai_sdk.tools import web_search as xai_web_search

from ..client import get_image_client
from ..models_registry import require_model
from ..server import mcp

# Searches block a thread for up to the gRPC deadline; a dedicated pool keeps them from
# starving the default executor that file reads and DNS lookups share.
SEARCH_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="search")


def _search_sync(query: str, model: str) -> dict:
    client = get_image_client()
    chat = client.chat.create(model=model, tools=[xai_web_search()])
    chat.append(xai_user(query))
    response = chat.sample()
    answer = response.content or ""
    # Response.citations is a list of URL strings; the API carries no titles.
    sources = [{"url": c} for c in response.citations or []]
    return {"answer": answer, "sources": sources}


async def _search_internal(query: str, model: str) -> dict:
    model = await require_model(model)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(SEARCH_EXECUTOR, _search_sync, query, model)
    return {**result, "model": model}


@mcp.tool(
    description=(
        "Search the web via xAI's live search and get a synthesized answer with citations. "
        "Results never enter the caller's context — Grok retrieves and summarizes directly. "
        "Returns: {answer, sources: [{url}], model}. "
        "Use for: current error diagnosis, latest versions, CVE checks, best practices. "
        "model (required): language model that runs the search — choose deliberately:\n"
        "{models}"
    )
)
async def search(query: str, model: str) -> dict:
    return await _search_internal(query, model)
