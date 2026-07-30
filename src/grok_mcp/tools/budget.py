"""Input budget: fail fast on prompts and attachments that cannot fit or cannot be seen."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from ..config import settings
from ..models_registry import find_model, get_catalog
from ..paths import check_readable

# Rough cost of one image in the prompt, for the pre-flight input estimate.
_IMAGE_TOKEN_ESTIMATE = 1_500


def _est_tokens(text: str) -> int:
    """Token estimate for text the API has not counted yet (conservative)."""
    return int(len(text) / settings.chars_per_token) + 1


async def precheck_input(
    prompt: str,
    model: str,
    system: str | None = None,
    file_paths: list[str] | None = None,
    image_paths: list[str] | None = None,
) -> None:
    """Fail fast — before any API call or job — on inputs that cannot work.

    Raises FileNotFoundError for image paths the server cannot see, and ValueError when the
    estimated input exceeds GROK_INPUT_MAX_RATIO of the model's context window.
    """
    # Allow-list first, so "not found" never reveals whether an outside path exists.
    for path in [*(image_paths or []), *(file_paths or [])]:
        check_readable(path)
    for path in [*(image_paths or []), *(file_paths or [])]:
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"file not found on the server: {path}. The server only sees its mounted "
                f"directories ({_visible_roots()}); host paths elsewhere, e.g. /tmp, are "
                "invisible to it — copy the file into one of them."
            )
    # One byte budget for the whole call, like files._collect: a per-file cap alone would let
    # many attachments add up to any size.
    total = 0
    for path in [*(image_paths or []), *(file_paths or [])]:
        with contextlib.suppress(OSError):
            total += Path(path).stat().st_size
    if total > settings.max_read_bytes:
        raise ValueError(
            f"attachments total {total} bytes, over the {settings.max_read_bytes}-byte "
            "GROK_MAX_READ_BYTES budget shared by file_paths and image_paths — pass fewer "
            "files (let Grok read what it needs with tools=['read'])"
        )
    catalog = await get_catalog()
    info = find_model(catalog, model)
    if info is None or not info.context_length:
        return
    chars = len(prompt) + len(system or "")
    for fpath in file_paths or []:
        with contextlib.suppress(OSError):
            chars += Path(fpath).stat().st_size
    estimate = int(chars / settings.chars_per_token) + _IMAGE_TOKEN_ESTIMATE * len(
        image_paths or []
    )
    limit = int(settings.input_max_ratio * info.context_length)
    if estimate > limit:
        fits = sorted(
            m.id
            for m in catalog.values()
            if m.kind == "language"
            and m.context_length
            and estimate <= settings.input_max_ratio * m.context_length
        )
        raise ValueError(
            f"input is ~{estimate} tokens but {info.id} accepts ~{limit} "
            f"({settings.input_max_ratio:.0%} of its {info.context_length}-token window). "
            f"Pass fewer files (let Grok read what it needs with tools=['read']) or use a "
            f"model with a larger window: {fits or 'none fits'}."
        )


def _visible_roots() -> str:
    roots = [os.environ.get(v) for v in ("WORKSPACE_DIR", "ASSETS_DIR")]
    return ", ".join(r for r in roots if r) or "the server's working tree"
