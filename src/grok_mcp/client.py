"""Client factory for xAI API — text (OpenAI-compat) + image (gRPC xai_sdk)."""

from __future__ import annotations

import contextlib

import httpx
import openai
from xai_sdk import Client

from .auth import load_api_key

_text_client: openai.AsyncOpenAI | None = None
_image_client: Client | None = None

XAI_BASE_URL = "https://api.x.ai/v1"

# Both of the openai SDK defaults are wrong for this server and must stay explicit.
#
# Its default read timeout is 600 s and its default max_retries is 2. A reasoning run
# legitimately thinks for longer than ten minutes, so the default turned a healthy call
# into a timeout, and the retry then re-sent the same /chat/completions three times in
# total. Dropping our socket does not stop the generation upstream, so each attempt is
# billed while none of them returns an answer — a reasoning job was observed to hang at
# exactly 600 s and be silently retried.
#
# Retries are off: /chat/completions is expensive and not idempotent, so a retry can
# only ever duplicate cost. Anything worth re-running is re-run deliberately as a job,
# where list_jobs shows it.
REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=60.0, pool=60.0)
GRPC_TIMEOUT_S = 300.0  # xai_sdk Client deadline per call (search, image generation)


class UpstreamError(RuntimeError):
    """xAI rejected a request; the message is xAI's own text, surfaced verbatim."""


def get_text_client() -> openai.AsyncOpenAI:
    """Return the shared async OpenAI-compat client."""
    if _text_client is None:
        raise RuntimeError("Text client not initialised — server lifespan not started")
    return _text_client


def get_image_client() -> Client:
    """Return the shared xai_sdk gRPC client."""
    if _image_client is None:
        raise RuntimeError("Image client not initialised — server lifespan not started")
    return _image_client


def init_clients() -> None:
    """Create and store global client instances. Called in server lifespan."""
    global _text_client, _image_client
    api_key = load_api_key()
    _text_client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url=XAI_BASE_URL,
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )
    # gRPC calls otherwise have no deadline: a stalled search or image call would pin its
    # worker thread forever.
    _image_client = Client(api_key=api_key, timeout=GRPC_TIMEOUT_S)


async def close_clients() -> None:
    """Tear down client instances (closing the HTTP pool and the gRPC channel)."""
    global _text_client, _image_client
    if _text_client is not None:
        with contextlib.suppress(Exception):
            await _text_client.close()
    if _image_client is not None:
        with contextlib.suppress(Exception):
            _image_client.close()
    _text_client = None
    _image_client = None
