# tests/tools/test_images.py
from unittest.mock import MagicMock, patch

import pytest

from grok_mcp.client import UpstreamError
from grok_mcp.tools.images import (
    _check_ref_paths,
    _data_uri,
    _generate_image_internal,
    generate_image,
)


@pytest.fixture(autouse=True)
def _writable_tmp(monkeypatch, tmp_path):
    """The image tools only write under GROK_WRITE_ALLOWED_PATHS."""
    monkeypatch.setattr("grok_mcp.paths.settings.write_allowed_paths", [str(tmp_path)])


async def test_generate_image_refuses_output_outside_allowed_paths(monkeypatch, tmp_path):
    from grok_mcp.tools.images import generate_image

    monkeypatch.setattr("grok_mcp.paths.settings.write_allowed_paths", [str(tmp_path / "ok")])
    with pytest.raises(PermissionError, match="GROK_WRITE_ALLOWED_PATHS"):
        await generate_image("p", str(tmp_path / "elsewhere.png"), model="m")


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff\xe0"
GIF_MAGIC = b"GIF89a"
WEBP_MAGIC = b"RIFF\x00\x00\x00\x00WEBP"


def _write_ref(path, magic=PNG_MAGIC):
    path.write_bytes(magic + b"\x00" * 50)
    return str(path)


def _as_data_uri(raw: bytes) -> str:
    """What ImageResponse.base64 returns."""
    import base64

    return "data:image/jpeg;base64," + base64.b64encode(raw).decode()


@pytest.fixture
def mock_image_client(tmp_path):
    """Provide a mocked xai_sdk Client with a fake image response."""
    fake_png = PNG_MAGIC + b"\x00" * 100  # minimal fake PNG bytes
    mock_response = MagicMock()
    mock_response.base64 = _as_data_uri(fake_png)

    mock_client = MagicMock()
    mock_client.image.sample.return_value = mock_response

    with patch("grok_mcp.tools.images.get_image_client", return_value=mock_client):
        yield mock_client, tmp_path


def test_generate_image_saves_png(mock_image_client):
    client, tmp_path = mock_image_client
    output = tmp_path / "out.png"

    result = _generate_image_internal(
        prompt="mossy cave stalactite",
        output_path=str(output),
        model="grok-imagine-image",
    )

    assert output.exists()
    assert result["path"] == str(output)
    assert "model" in result
    client.image.sample.assert_called_once()


# --- reference images (issue #60) ------------------------------------------


def test_single_ref_is_sent_as_a_one_element_list(mock_image_client, tmp_path):
    """One ref goes through image_urls too — verified against the live API to
    behave identically to the legacy singular image_url field."""
    client, _ = mock_image_client
    ref = _write_ref(tmp_path / "ref.png")

    _generate_image_internal(
        prompt="darker version",
        output_path=str(tmp_path / "out.png"),
        model="grok-imagine-image-pro",
        ref_image_paths=[ref],
    )

    call_kwargs = client.image.sample.call_args[1]
    assert "image_url" not in call_kwargs  # the singular field is no longer used
    assert len(call_kwargs["image_urls"]) == 1


@pytest.mark.asyncio
async def test_three_refs_are_all_forwarded_in_order(mock_image_client, tmp_path):
    """Issue #60: the tool accepted one ref while the API takes three."""
    client, _ = mock_image_client
    refs = [_write_ref(tmp_path / f"ref{i}.png") for i in range(3)]

    await generate_image(
        prompt="geometry from the first, edge character from the second",
        model="grok-imagine-image-quality",
        output_path=str(tmp_path / "out.png"),
        ref_image_paths=refs,
    )

    sent = client.image.sample.call_args[1]["image_urls"]
    assert len(sent) == 3
    # order must survive — the prompt refers to refs positionally
    assert [uri[:40] for uri in sent] == [_data_uri(r)[:40] for r in refs]


def test_empty_ref_list_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        _check_ref_paths([])


def test_missing_ref_file_error_names_the_path(tmp_path):
    missing = str(tmp_path / "nope.png")

    with pytest.raises(FileNotFoundError, match="nope.png"):
        _check_ref_paths([missing])


# --- MIME sniffing ---------------------------------------------------------


@pytest.mark.parametrize(
    ("magic", "expected"),
    [
        (PNG_MAGIC, "image/png"),
        (JPEG_MAGIC, "image/jpeg"),
        (GIF_MAGIC, "image/gif"),
        (WEBP_MAGIC, "image/webp"),
        (b"\x00\x01\x02\x03", "image/png"),  # unknown falls back to PNG
    ],
)
def test_data_uri_sniffs_the_real_format(tmp_path, magic, expected):
    """The model returns JPEG bytes that we save under .png, so a ref fed back
    from a previous round is typically a .png file containing JPEG."""
    ref = tmp_path / "ref.png"
    ref.write_bytes(magic + b"\x00" * 50)

    assert _data_uri(str(ref)).startswith(f"data:{expected};base64,")


# --- output format honesty -------------------------------------------------


def test_result_reports_the_real_output_format(mock_image_client, tmp_path):
    """xAI returns JPEG bytes regardless of the .png output path, and the API
    exposes no codec control — so the result must name the real format."""
    client, _ = mock_image_client
    jpeg_bytes = JPEG_MAGIC + b"\x00" * 100
    client.image.sample.return_value.base64 = _as_data_uri(jpeg_bytes)
    output = tmp_path / "out.png"

    result = _generate_image_internal(
        prompt="mossy wall",
        output_path=str(output),
        model="grok-imagine-image",
    )

    assert result["format"] == "jpeg"
    assert result["path"] == str(output)  # the caller's path is left alone
    assert output.read_bytes() == jpeg_bytes  # bytes written verbatim


# --- backend errors --------------------------------------------------------


def test_backend_rejection_surfaces_a_readable_message(mock_image_client, tmp_path):
    """The per-model ref cap is enforced by xAI; its message must reach the
    caller instead of a raw gRPC traceback."""
    client, _ = mock_image_client
    ref = _write_ref(tmp_path / "ref.png")

    class FakeRpcError(Exception):
        def details(self):
            return "This model supports at most 3 input image(s), but 4 were provided."

    client.image.sample.side_effect = FakeRpcError("rpc blew up")

    with pytest.raises(UpstreamError, match="at most 3 input image"):
        _generate_image_internal(
            prompt="four refs",
            output_path=str(tmp_path / "out.png"),
            model="grok-imagine-image-quality",
            ref_image_paths=[ref],
        )


# --- shared knobs ----------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_image_with_refs_forwards_aspect_ratio_and_resolution(
    mock_image_client, tmp_path
):
    """aspect_ratio/resolution must reach the backend — the output never inherits the ref's."""
    client, _ = mock_image_client
    ref = _write_ref(tmp_path / "ref.png")

    await generate_image(
        prompt="parallax band",
        ref_image_paths=[ref],
        model="grok-imagine-image-quality",
        output_path=str(tmp_path / "out.png"),
        aspect_ratio="16:9",
        resolution="2k",
    )

    call_kwargs = client.image.sample.call_args[1]
    assert call_kwargs["aspect_ratio"] == "16:9"
    assert call_kwargs["resolution"] == "2k"


@pytest.mark.asyncio
async def test_refs_only_add_image_urls_to_the_same_backend_keys(mock_image_client, tmp_path):
    """With or without refs the backend gets the same knobs — only refs add image_urls."""
    client, _ = mock_image_client
    ref = _write_ref(tmp_path / "ref.png")

    if True:
        await generate_image(
            prompt="plain",
            model="grok-imagine-image",
            output_path=str(tmp_path / "a.png"),
            aspect_ratio="16:9",
            resolution="2k",
        )
        plain_keys = set(client.image.sample.call_args[1])

        await generate_image(
            prompt="with ref",
            model="grok-imagine-image",
            ref_image_paths=[ref],
            output_path=str(tmp_path / "b.png"),
            aspect_ratio="16:9",
            resolution="2k",
        )
        ref_keys = set(client.image.sample.call_args[1])

    assert ref_keys - plain_keys == {"image_urls"}
    assert plain_keys - ref_keys == set()


@pytest.mark.asyncio
async def test_generate_image_requires_an_image_model(mock_image_client, tmp_path):
    with pytest.raises(ValueError, match="image model"):
        await generate_image(prompt="x", output_path=str(tmp_path / "o.png"), model="grok-4.3")


def test_output_dirs_are_not_created_outside_allowed_paths(mock_image_client, tmp_path):
    # A parent that leads out of the allow-list (a symlink swapped in after the tool's check)
    # must not get directories created there.
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "link").symlink_to(outside)
    with pytest.raises(PermissionError):
        _generate_image_internal(
            prompt="p", output_path=str(tmp_path / "link" / "new" / "o.png"), model="m"
        )
    assert list(outside.iterdir()) == []


async def test_generate_image_passes_the_checked_resolved_path(mock_image_client, tmp_path):
    seen = {}

    async def fake_generate(prompt, output_path, *a, **k):
        seen["path"] = output_path

    (tmp_path / "real").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "real")
    with patch("grok_mcp.tools.images._generate", fake_generate):
        await generate_image("p", str(tmp_path / "alias" / "o.png"), model="m")
    assert seen["path"] == str(tmp_path / "real" / "o.png")


@pytest.mark.asyncio
@pytest.mark.parametrize("refs", [None, []])
async def test_no_refs_sends_no_image_urls(mock_image_client, tmp_path, refs):
    client, _ = mock_image_client

    await generate_image(
        prompt="plain",
        model="grok-imagine-image",
        output_path=str(tmp_path / "out.png"),
        ref_image_paths=refs,
    )

    assert "image_urls" not in client.image.sample.call_args[1]


@pytest.mark.asyncio
async def test_missing_ref_is_rejected_before_calling_the_backend(mock_image_client, tmp_path):
    client, _ = mock_image_client

    with pytest.raises(FileNotFoundError):
        await generate_image(
            prompt="x",
            model="grok-imagine-image",
            output_path=str(tmp_path / "out.png"),
            ref_image_paths=[str(tmp_path / "nope.png")],
        )

    client.image.sample.assert_not_called()
