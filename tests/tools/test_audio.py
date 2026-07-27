"""Tests for _tts_internal and _transcribe_internal — xAI /v1/tts and /v1/stt."""

from __future__ import annotations

import json

import httpx
import pytest

from grok_mcp.tools import audio


@pytest.fixture
def xai(monkeypatch):
    """Route the audio module's HTTP client through a MockTransport; collect requests."""
    seen: list[httpx.Request] = []
    responses: dict[str, httpx.Response] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses[request.url.path]

    monkeypatch.setattr(
        audio,
        "_http_client",
        lambda: httpx.AsyncClient(
            base_url="https://api.x.ai/v1", transport=httpx.MockTransport(handler)
        ),
    )
    return seen, responses


async def test_tts_posts_native_endpoint_and_saves_audio(xai, tmp_path, monkeypatch):
    monkeypatch.setattr("grok_mcp.paths.settings.write_allowed_paths", [str(tmp_path)])
    seen, responses = xai
    responses["/v1/tts"] = httpx.Response(
        200, content=b"MP3", headers={"content-type": "audio/mpeg"}
    )
    out = tmp_path / "sub" / "speech.mp3"

    result = await audio._tts_internal("Grüß Gott", str(out), voice="ara", language="de")

    assert out.read_bytes() == b"MP3"
    assert result == {
        "path": str(out),
        "voice": "ara",
        "language": "de",
        "format": "mp3",
        "characters": 9,
        "bytes": 3,
    }
    body = json.loads(seen[0].content)
    assert body == {
        "text": "Grüß Gott",
        "voice_id": "ara",
        "language": "de",
        "output_format": {"codec": "mp3"},
        "speed": 1.0,
    }


async def test_tts_rejects_unknown_codec(tmp_path):
    with pytest.raises(ValueError, match="codec"):
        await audio._tts_internal("x", str(tmp_path / "a"), codec="ogg")


async def test_tts_surfaces_api_errors(xai, tmp_path):
    _, responses = xai
    responses["/v1/tts"] = httpx.Response(403, text='{"error":"denied"}')
    with pytest.raises(RuntimeError, match="403"):
        await audio._tts_internal("x", str(tmp_path / "a.mp3"))


async def test_stt_sends_options_before_file_and_returns_transcript(xai, tmp_path):
    seen, responses = xai
    responses["/v1/stt"] = httpx.Response(
        200, json={"text": "hello world", "language": "en", "duration": 1.25, "words": []}
    )
    f = tmp_path / "a.mp3"
    f.write_bytes(b"AUDIO")

    result = await audio._transcribe_internal(str(f), language="en")

    assert result == {"text": "hello world", "language": "en", "duration": 1.25}
    raw = seen[0].content
    assert raw.index(b'name="language"') < raw.index(b'name="file"')


async def test_stt_without_language_omits_field(xai, tmp_path):
    seen, responses = xai
    responses["/v1/stt"] = httpx.Response(200, json={"text": "hej", "language": "pl"})
    f = tmp_path / "a.wav"
    f.write_bytes(b"AUDIO")
    result = await audio._transcribe_internal(str(f))
    assert result["language"] == "pl"
    assert b'name="language"' not in seen[0].content


async def test_stt_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        await audio._transcribe_internal(str(tmp_path / "nope.mp3"))


async def test_tts_tool_refuses_output_outside_allowed_paths(monkeypatch, tmp_path):
    monkeypatch.setattr("grok_mcp.paths.settings.write_allowed_paths", [str(tmp_path / "ok")])
    with pytest.raises(PermissionError, match="GROK_WRITE_ALLOWED_PATHS"):
        await audio.text_to_speech("x", str(tmp_path / "evil.mp3"))


async def test_transcribe_tool_checks_read_allow_list(monkeypatch, tmp_path):
    monkeypatch.setattr("grok_mcp.paths.settings.read_allowed_paths", [str(tmp_path / "ok")])
    with pytest.raises(PermissionError, match="GROK_READ_ALLOWED_PATHS"):
        await audio.transcribe(str(tmp_path / "secret.mp3"))


def test_save_does_not_create_dirs_outside_allowed_paths(monkeypatch, tmp_path):
    allowed, outside = tmp_path / "ok", tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (allowed / "link").symlink_to(outside)
    monkeypatch.setattr("grok_mcp.paths.settings.write_allowed_paths", [str(allowed)])
    with pytest.raises(PermissionError):
        audio._save(allowed / "link" / "new" / "a.mp3", b"x")
    assert list(outside.iterdir()) == []
