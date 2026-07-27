"""TTS and STT MCP tools — xAI's native /v1/tts and /v1/stt endpoints.

These are not the OpenAI-compatible /audio/speech and /audio/transcriptions routes: xAI answers
those with 403 "Team is not authorized" even for teams that can use TTS/STT.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx

from ..auth import load_api_key
from ..client import REQUEST_TIMEOUT, XAI_BASE_URL, UpstreamError
from ..paths import (
    check_readable,
    checked_output_path,
    makedirs_checked,
    read_bytes_capped,
    write_bytes_nofollow,
)
from ..server import mcp

logger = logging.getLogger("grok_mcp.audio")

CODECS = ("mp3", "wav", "pcm", "mulaw", "alaw")


def _http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=XAI_BASE_URL,
        headers={"Authorization": f"Bearer {load_api_key()}"},
        timeout=REQUEST_TIMEOUT,
    )


def _save(out: Path, data: bytes) -> None:
    makedirs_checked(out.parent)
    write_bytes_nofollow(out, data)


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.is_error:
        raise UpstreamError(f"xAI {resp.request.url.path} returned {resp.status_code}: {resp.text}")


async def _tts_internal(
    text: str,
    output_path: str,
    voice: str = "eve",
    language: str = "auto",
    codec: str = "mp3",
    speed: float = 1.0,
) -> dict[str, Any]:
    if codec not in CODECS:
        raise ValueError(f"codec must be one of {CODECS}, got {codec!r}")
    body = {
        "text": text,
        "voice_id": voice,
        "language": language,
        "output_format": {"codec": codec},
        "speed": speed,
    }
    async with _http_client() as client:
        resp = await client.post("/tts", json=body)
    _raise_for_status(resp)
    out = Path(output_path)  # resolved and checked by the caller; never re-resolved
    await asyncio.to_thread(_save, out, resp.content)
    return {
        "path": str(out),
        "voice": voice,
        "language": language,
        "format": codec,
        "characters": len(text),
        "bytes": len(resp.content),
    }


@mcp.tool(
    description=(
        "Convert text to speech using xAI TTS (max 60,000 characters; speech tags like [pause] "
        "or [laugh] are supported). Saves audio to output_path. "
        "voice: built-in (eve, ara, rex, sal, leo) or a custom voice id. "
        "language: 'auto' or a BCP-47 code (en, de, fr, es-ES, pt-BR, ja, ...). "
        "codec: mp3 (default), wav, pcm, mulaw, alaw. speed: 0.7-1.5."
    )
)
async def text_to_speech(
    text: str,
    output_path: str,
    voice: str = "eve",
    language: str = "auto",
    codec: str = "mp3",
    speed: float = 1.0,
) -> dict:
    out = checked_output_path(output_path)
    return await _tts_internal(text, str(out), voice, language, codec, speed)


async def _transcribe_internal(audio_path: str, language: str | None = None) -> dict[str, Any]:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(audio_path)
    # Regular file, capped, descriptor-checked, off the event loop (a FIFO must not hang it).
    audio = await asyncio.to_thread(read_bytes_capped, path)
    # Option fields must precede the file part — xAI may ignore fields sent after it.
    data = {"language": language} if language else {}
    async with _http_client() as client:
        resp = await client.post("/stt", data=data, files={"file": (path.name, audio)})
    _raise_for_status(resp)
    payload = resp.json()
    return {
        "text": payload.get("text", ""),
        "language": payload.get("language") or language,
        "duration": payload.get("duration"),
    }


@mcp.tool(
    description=(
        "Transcribe an audio file with xAI STT. Container formats are auto-detected (wav, mp3, "
        "ogg, opus, flac, aac, mp4, m4a, mkv; up to GROK_MAX_READ_BYTES, 20 MB by default). language: optional code (en, de, "
        "fr, ...) to bias recognition; detected when omitted. Returns {text, language, duration}."
    )
)
async def transcribe(audio_path: str, language: str | None = None) -> dict:
    check_readable(audio_path)
    return await _transcribe_internal(audio_path, language)
