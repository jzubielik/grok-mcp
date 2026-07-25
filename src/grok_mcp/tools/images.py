"""Image generation tools — wraps xAI Aurora via xai_sdk gRPC."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

from ..client import UpstreamError, get_image_client
from ..models_registry import require_model
from ..paths import (
    check_readable,
    checked_output_path,
    makedirs_checked,
    read_bytes_capped,
    write_bytes_nofollow,
)
from ..server import mcp

# NOTE: REST /images/edits returns 403 — use xai_sdk gRPC for all image generation.

# The model hands back JPEG bytes that we save under a .png name, so a reference fed
# back from an earlier round is typically a .png file holding JPEG. Sniff the real
# format instead of trusting the extension — a wrong MIME gets the ref rejected.
_MAGIC_MIME: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def _sniff_mime(raw: bytes) -> str:
    """Return the MIME type of image bytes, falling back to PNG when unrecognised."""
    for magic, candidate in _MAGIC_MIME:
        if raw.startswith(magic):
            return candidate
    # WEBP is "RIFF" + 4 size bytes + "WEBP", so it needs a windowed check.
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"  # unknown formats keep the historical assumption


def _data_uri(path: str) -> str:
    """Read an image file and return it as a base64 data URI with a sniffed MIME type."""
    raw = read_bytes_capped(path)
    return f"data:{_sniff_mime(raw)};base64,{base64.b64encode(raw).decode()}"


def _check_ref_paths(ref_image_paths: list[str]) -> list[str]:
    """Validate the reference images; return them as an ordered list of existing paths."""
    if not ref_image_paths:
        raise ValueError("ref_image_paths is empty — pass at least one reference image.")
    for path in ref_image_paths:
        check_readable(path)  # before is_file, so the error never reveals outside paths
        if not Path(path).is_file():
            raise FileNotFoundError(f"Reference image not found: {path}")
    return list(ref_image_paths)


def _generate_image_internal(
    prompt: str,
    output_path: str,
    model: str,
    aspect_ratio: str = "1:1",
    resolution: str = "1k",
    ref_image_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Core image generation logic, callable from tests."""
    client = get_image_client()
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "model": model,
        "image_format": "base64",
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
    }
    if ref_image_paths:
        # NOTE: xai_sdk gRPC takes the plural "image_urls" — REST /images/edits returns 403.
        # Order is preserved because prompts address the references positionally
        # ("geometry from the first, edge character from the second").
        kwargs["image_urls"] = [_data_uri(path) for path in ref_image_paths]

    try:
        response = client.image.sample(**kwargs)
    except Exception as exc:
        # grpc.RpcError keeps the readable text in .details(); the per-model cap on
        # reference images is enforced there, not here, so surface it verbatim
        # instead of letting a raw gRPC traceback reach the caller.
        details = getattr(exc, "details", None)
        if callable(details):
            raise UpstreamError(str(details())) from exc
        raise

    # The public `base64` property is a data URI; xai_sdk's own decoder is private API.
    image_bytes = base64.b64decode(response.base64.partition("base64,")[2])

    out = Path(output_path)  # resolved and checked by the caller; never re-resolved
    makedirs_checked(out.parent)
    write_bytes_nofollow(out, image_bytes)

    return {
        "path": str(out),
        "model": model,
        "prompt": prompt,
        # xAI picks the codec — today it returns JPEG whatever the output path is
        # named, and the API exposes no way to ask for another (ImageFormat is
        # BASE64/URL, i.e. transport, not codec). Report what was actually written
        # rather than letting the extension speak for the bytes.
        "format": _sniff_mime(image_bytes).removeprefix("image/"),
    }


async def _generate(
    prompt: str,
    output_path: str,
    model: str,
    aspect_ratio: str = "1:1",
    resolution: str = "1k",
    ref_image_paths: list[str] | None = None,
) -> dict:
    """Validate the model and generate off the event loop. Path checks are the callers' job."""
    model = await require_model(model, kind="image")
    return await asyncio.to_thread(
        _generate_image_internal,
        prompt=prompt,
        output_path=output_path,
        model=model,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        ref_image_paths=ref_image_paths,
    )


@mcp.tool(
    description=(
        "Generate an image using the xAI image model, optionally guided by reference images. "
        "Writes the bytes the model returns to output_path verbatim. xAI currently returns "
        "JPEG whatever extension you choose and exposes no codec control, so the result's "
        "'format' field names what was actually written — check it before trusting the suffix. "
        "ref_image_paths (optional): local paths to reference images, in order — the prompt may "
        "address them positionally, e.g. 'geometry from the first, edge character from the second'. "
        "The server forwards every path you pass; the per-model cap (3 for the current image "
        "models) is enforced by the backend, whose message is surfaced verbatim if you exceed it. "
        "The output does NOT inherit a reference's proportions, so pass aspect_ratio and "
        "resolution explicitly (e.g. '16:9' + '2k') for anything but a 1k square. References are "
        "MIME-sniffed, so a previous result fed back in works regardless of its extension. "
        "model (required): image model — choose deliberately by quality vs price; quality-tier "
        "models transfer reference style better:\n"
        "{models}"
    )
)
async def generate_image(
    prompt: str,
    output_path: str,
    model: str,
    aspect_ratio: str = "1:1",
    resolution: str = "1k",
    ref_image_paths: list[str] | None = None,
) -> dict:
    out = checked_output_path(output_path)
    paths = _check_ref_paths(ref_image_paths) if ref_image_paths else None
    return await _generate(prompt, str(out), model, aspect_ratio, resolution, paths)
