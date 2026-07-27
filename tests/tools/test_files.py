"""Tests for read() and write() tools."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# --- read() ---


@pytest.mark.asyncio
async def test_read_single_file(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    from grok_mcp.tools.files import _read_internal

    result = await _read_internal(str(f))
    assert result == "hello world"


@pytest.mark.asyncio
async def test_read_list_of_files(tmp_path):
    (tmp_path / "a.txt").write_text("aaa")
    (tmp_path / "b.txt").write_text("bbb")
    from grok_mcp.tools.files import _read_internal

    result = await _read_internal([str(tmp_path / "a.txt"), str(tmp_path / "b.txt")])
    assert isinstance(result, dict)
    assert result[str(tmp_path / "a.txt")] == "aaa"
    assert result[str(tmp_path / "b.txt")] == "bbb"


@pytest.mark.asyncio
async def test_read_glob_pattern(tmp_path):
    (tmp_path / "x.yaml").write_text("x: 1")
    (tmp_path / "y.yaml").write_text("y: 2")
    from grok_mcp.tools.files import _read_internal

    result = await _read_internal(str(tmp_path / "*.yaml"))
    assert isinstance(result, dict)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_read_missing_file_raises(tmp_path):
    from grok_mcp.tools.files import _read_internal

    with pytest.raises(FileNotFoundError, match="no-exist.txt"):
        await _read_internal(str(tmp_path / "no-exist.txt"))


@pytest.mark.asyncio
async def test_read_with_question_calls_grok(tmp_path):
    f = tmp_path / "cfg.yaml"
    f.write_text("memory: 512Mi")
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Looks fine"
    mock_client.chat.completions.create.return_value = mock_response
    with patch("grok_mcp.tools.files.get_text_client", return_value=mock_client):
        from grok_mcp.tools.files import _read_internal

        result = await _read_internal(str(f), question="Any OOM risk?", model="m")
    assert result == "Looks fine"
    mock_client.chat.completions.create.assert_called_once()


# --- write() ---


@pytest.mark.asyncio
async def test_write_dry_run_returns_diff(tmp_path, monkeypatch):
    monkeypatch.setattr("grok_mcp.paths.settings.write_allowed_paths", [str(tmp_path)])
    f = tmp_path / "cfg.yaml"
    f.write_text("memory: 512Mi\n")
    from grok_mcp.tools.files import _write_internal

    result = await _write_internal(str(f), "memory: 1Gi\n", dry_run=True)
    assert result["dry_run"] is True
    assert "-memory: 512Mi" in result["diff"]
    assert "+memory: 1Gi" in result["diff"]


@pytest.mark.asyncio
async def test_write_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr("grok_mcp.paths.settings.write_allowed_paths", [str(tmp_path)])
    f = tmp_path / "new.txt"
    f.write_text("old")
    from grok_mcp.tools.files import _write_internal

    result = await _write_internal(str(f), "new content", dry_run=False, backup=False)
    assert result["written"] is True
    assert f.read_text() == "new content"


@pytest.mark.asyncio
async def test_write_creates_backup(tmp_path, monkeypatch):
    monkeypatch.setattr("grok_mcp.paths.settings.write_allowed_paths", [str(tmp_path)])
    f = tmp_path / "data.txt"
    f.write_text("original")
    from grok_mcp.tools.files import _write_internal

    await _write_internal(str(f), "updated", dry_run=False, backup=True)
    assert (tmp_path / "data.txt.bak").read_text() == "original"


@pytest.mark.asyncio
async def test_write_rejects_path_outside_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr("grok_mcp.paths.settings.write_allowed_paths", [str(tmp_path)])
    from grok_mcp.tools.files import _write_internal

    with pytest.raises(PermissionError):
        await _write_internal("/etc/passwd", "evil", dry_run=False)


@pytest.mark.asyncio
async def test_write_multi_file_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr("grok_mcp.paths.settings.write_allowed_paths", [str(tmp_path)])
    (tmp_path / "a.txt").write_text("aaa")
    (tmp_path / "b.txt").write_text("bbb")
    from grok_mcp.tools.files import _write_internal

    result = await _write_internal(
        [
            {"path": str(tmp_path / "a.txt"), "content": "AAA"},
            {"path": str(tmp_path / "b.txt"), "content": "BBB"},
        ],
        dry_run=True,
    )
    assert isinstance(result, list)
    assert all(r["dry_run"] for r in result)


@pytest.mark.asyncio
async def test_write_raises_if_parent_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("grok_mcp.paths.settings.write_allowed_paths", [str(tmp_path)])
    from grok_mcp.tools.files import _write_internal

    with pytest.raises(FileNotFoundError):
        await _write_internal(str(tmp_path / "no_dir" / "file.txt"), "x", dry_run=False)


@pytest.mark.asyncio
async def test_write_denied_when_allowed_paths_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("grok_mcp.paths.settings.write_allowed_paths", [])
    from grok_mcp.tools.files import _write_internal

    with pytest.raises(PermissionError):
        await _write_internal(str(tmp_path / "file.txt"), "x", dry_run=False)


@pytest.mark.asyncio
async def test_read_line_range_is_numbered_with_a_header(tmp_path):
    from grok_mcp.tools.files import _read_internal

    f = tmp_path / "big.cs"
    f.write_text("\n".join(f"line{i}" for i in range(1, 101)))
    out = await _read_internal(str(f), offset=10, limit=3)
    assert out == "[lines 11-13 of 100]\n11: line11\n12: line12\n13: line13"


@pytest.mark.asyncio
async def test_read_line_range_of_a_file_larger_than_the_cap(tmp_path, monkeypatch):
    """The cap applies to the bytes returned, so a big file can still be paged through."""
    from grok_mcp.config import settings
    from grok_mcp.tools.files import _read_internal

    f = tmp_path / "big.log"
    f.write_text("".join(f"line{i}\n" for i in range(1, 201)))  # ~1.7 KB
    monkeypatch.setattr(settings, "max_read_bytes", 100)
    out = await _read_internal(str(f), offset=150, limit=2)
    assert out == "[lines 151-152 of 200]\n151: line151\n152: line152"
    with pytest.raises(ValueError, match="Line range too large"):
        await _read_internal(str(f), offset=0, limit=50)


@pytest.mark.asyncio
async def test_read_line_range_scan_is_bounded(tmp_path, monkeypatch):
    from grok_mcp.config import settings
    from grok_mcp.tools.files import _read_internal

    f = tmp_path / "huge.log"
    f.write_text("x" * 99 + "\n" + ("y" * 9 + "\n") * 1000)  # 10 100 bytes
    monkeypatch.setattr(settings, "max_read_bytes", 100)  # scan cap 5000
    assert (await _read_internal(str(f), offset=0, limit=1)).startswith("[lines 1-1 of 491+]")
    with pytest.raises(ValueError, match="grep it instead"):
        await _read_internal(str(f), offset=900, limit=1)


@pytest.mark.asyncio
async def test_read_line_range_needs_a_single_file(tmp_path):
    from grok_mcp.tools.files import _read_internal

    with pytest.raises(ValueError, match="single file"):
        await _read_internal(str(tmp_path / "*.cs"), offset=0, limit=5)


@pytest.mark.parametrize("pattern", ["*.py", "./*.py"])
def test_glob_relative_pattern_matches_cwd(tmp_path, monkeypatch, pattern):
    from grok_mcp.tools.files import _glob_files

    (tmp_path / "a.py").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("x")
    monkeypatch.chdir(tmp_path)
    assert [os.path.normpath(m) for m in _glob_files(pattern)] == ["a.py"]


def test_glob_without_doublestar_does_not_descend(tmp_path, monkeypatch):
    from grok_mcp.tools import files

    (tmp_path / "dir" / "deep" / "deeper").mkdir(parents=True)
    (tmp_path / "dir" / "a.md").write_text("x")
    (tmp_path / "dir" / "deep" / "b.md").write_text("x")
    walked = []
    real_walk = os.walk

    def counting_walk(*a, **k):
        for entry in real_walk(*a, **k):
            walked.append(os.path.normpath(entry[0]))
            yield entry

    monkeypatch.setattr(files.os, "walk", counting_walk)
    assert list(files._glob_files(f"{tmp_path}/dir/*.md")) == [f"{tmp_path}/dir/a.md"]
    assert walked == [f"{tmp_path}/dir"]
    walked.clear()
    assert len(list(files._glob_files(f"{tmp_path}/dir/**/*.md"))) == 2
    assert len(walked) == 3


def test_glob_star_segment_descends_exactly_one_level(tmp_path):
    from grok_mcp.tools.files import _glob_files

    (tmp_path / "x" / "y" / "z").mkdir(parents=True)
    (tmp_path / "x" / "y" / "f.txt").write_text("x")
    (tmp_path / "x" / "y" / "z" / "f.txt").write_text("x")
    assert list(_glob_files(f"{tmp_path}/x/*/f.txt")) == [f"{tmp_path}/x/y/f.txt"]


def test_glob_from_ancestor_walks_only_allowed_roots(tmp_path, monkeypatch):
    from grok_mcp.config import settings
    from grok_mcp.tools import files

    for d in ("ws/sub", "ws/sub/nested", "other/deep"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "ws" / "sub" / "a.txt").write_text("x")
    (tmp_path / "other" / "deep" / "b.txt").write_text("x")
    # The nested allowed path must not be walked a second time.
    allowed = [str(tmp_path / "ws"), str(tmp_path / "ws" / "sub" / "nested")]
    monkeypatch.setattr(settings, "read_allowed_paths", allowed)
    walked = []
    real_walk = os.walk

    def counting_walk(*a, **k):
        for entry in real_walk(*a, **k):
            walked.append(os.path.normpath(entry[0]))
            yield entry

    monkeypatch.setattr(files.os, "walk", counting_walk)
    assert list(files._glob_files(f"{tmp_path}/**/*.txt")) == [f"{tmp_path}/ws/sub/a.txt"]
    # Not even tmp_path itself is listed — that alone would reveal "other".
    assert walked == [f"{tmp_path}/ws", f"{tmp_path}/ws/sub", f"{tmp_path}/ws/sub/nested"]
