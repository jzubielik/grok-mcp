import pytest

from grok_mcp.config import settings
from grok_mcp.paths import check_readable, check_writable, is_under


def test_is_under_is_component_wise(tmp_path):
    root = tmp_path / "work"
    sibling = tmp_path / "work-evil"
    root.mkdir()
    sibling.mkdir()
    assert is_under(root / "a.txt", [str(root)])
    assert not is_under(sibling / "a.txt", [str(root)])


def test_is_under_rejects_dotdot_escape(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    assert not is_under(root / ".." / "secret", [str(root)])


def test_is_under_resolves_symlinks(tmp_path):
    root = tmp_path / "work"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "link").symlink_to(outside)
    assert not is_under(root / "link" / "x", [str(root)])


def test_check_writable_empty_denies(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "write_allowed_paths", [])
    with pytest.raises(PermissionError, match="GROK_WRITE_ALLOWED_PATHS"):
        check_writable(tmp_path / "x")


def test_check_readable_empty_allows(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "read_allowed_paths", [])
    check_readable(tmp_path / "x")


def test_check_readable_outside_denied(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "read_allowed_paths", [str(tmp_path / "work")])
    with pytest.raises(PermissionError, match="GROK_READ_ALLOWED_PATHS"):
        check_readable(tmp_path / "work-evil" / "x")
    check_readable(tmp_path / "work" / "x")


async def test_glob_skips_outside_matches_silently(monkeypatch, tmp_path):
    from grok_mcp.tools.files import _read_internal

    inside, outside = tmp_path / "in", tmp_path / "out"
    inside.mkdir()
    outside.mkdir()
    (inside / "a.txt").write_text("A")
    (outside / "secret.txt").write_text("S")
    monkeypatch.setattr(settings, "read_allowed_paths", [str(inside)])
    result = await _read_internal(str(tmp_path / "*" / "*.txt"))
    assert result == {str(inside / "a.txt"): "A"}


async def test_proc_environ_is_never_readable(monkeypatch):
    from grok_mcp.tools.files import _read_internal

    monkeypatch.setattr(settings, "read_allowed_paths", [])  # even with no allow-list
    for path in ("/proc/self/environ", "/sys/kernel", "/dev/zero"):
        with pytest.raises(PermissionError, match="system pseudo-files"):
            await _read_internal(path)


def test_non_regular_files_and_oversized_files_refused(monkeypatch, tmp_path):
    import os

    from grok_mcp.paths import read_bytes_capped

    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="Not a regular file"):
        read_bytes_capped(fifo)
    big = tmp_path / "big.txt"
    big.write_text("x" * 100)
    monkeypatch.setattr(settings, "max_read_bytes", 10)
    with pytest.raises(ValueError, match="too large"):
        read_bytes_capped(big)


async def test_backup_symlink_cannot_redirect_a_write(monkeypatch, tmp_path):
    """A planted `file.bak -> outside` must not let a write escape the allow-list."""
    from grok_mcp.tools.files import _write_single

    allowed, outside = tmp_path / "allowed", tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (outside / "target").write_text("orig")
    (allowed / "a.txt").write_text("old")
    (allowed / "a.txt.bak").symlink_to(outside / "target")
    monkeypatch.setattr(settings, "write_allowed_paths", [str(allowed)])
    with pytest.raises(PermissionError):
        await _write_single(str(allowed / "a.txt"), "new", dry_run=False, backup=True)
    assert (outside / "target").read_text() == "orig"


async def test_write_does_not_follow_a_symlinked_target(monkeypatch, tmp_path):
    from grok_mcp.paths import write_text_nofollow

    (tmp_path / "real").write_text("orig")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    with pytest.raises(OSError):
        write_text_nofollow(tmp_path / "link", "new")
    assert (tmp_path / "real").read_text() == "orig"


def test_read_checks_the_descriptor_not_the_name(monkeypatch, tmp_path):
    """A directory swapped for a symlink after the name check must not redirect the read."""
    from grok_mcp.paths import read_bytes_capped

    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(settings, "read_allowed_paths", [str(ws)])
    (ws / "d").symlink_to("/proc")  # what a racing `shell` would leave behind
    # /proc/version opens fine, so only the check on the opened descriptor can refuse it.
    with pytest.raises(PermissionError, match="system pseudo-files"):
        read_bytes_capped(ws / "d" / "version")


def test_growing_file_cannot_exceed_the_cap(tmp_path):
    from grok_mcp.paths import read_bytes_capped

    f = tmp_path / "f"
    f.write_bytes(b"x" * 50)
    with pytest.raises(ValueError, match="too large"):
        read_bytes_capped(f, max_bytes=10)


def test_fifo_does_not_block_and_is_refused(tmp_path):
    import os

    from grok_mcp.paths import read_bytes_capped

    fifo = tmp_path / "call.wav"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="Not a regular file"):
        read_bytes_capped(fifo)  # returns at once: O_NONBLOCK


def test_env_files_are_not_readable(tmp_path):
    for name in (".env", ".env.prod", ".env.local"):
        with pytest.raises(PermissionError, match="credential files"):
            check_readable(tmp_path / name)
    check_readable(tmp_path / ".env.example")


async def test_read_call_shares_one_byte_budget(monkeypatch, tmp_path):
    from grok_mcp.tools.files import _read_internal

    for i in range(3):
        (tmp_path / f"f{i}.txt").write_text("x" * 40)
    monkeypatch.setattr(settings, "max_read_bytes", 100)  # 3 × 40 > 100
    with pytest.raises(ValueError, match="too large"):
        await _read_internal(str(tmp_path / "*.txt"))


async def test_dry_run_does_not_disclose_unreadable_files(monkeypatch, tmp_path):
    from grok_mcp.tools.files import _write_single

    (tmp_path / "secret.txt").write_text("TOPSECRET")
    monkeypatch.setattr(settings, "write_allowed_paths", [str(tmp_path)])
    monkeypatch.setattr(settings, "read_allowed_paths", [str(tmp_path / "other")])
    with pytest.raises(PermissionError):
        await _write_single(str(tmp_path / "secret.txt"), "x", dry_run=True, backup=True)


async def test_glob_ignores_directory_symlink_loops(tmp_path):
    """Two `a -> .` links (any cloned repo can carry them) must not make a glob spin."""
    import asyncio

    from grok_mcp.tools.files import _read_internal

    (tmp_path / "x.txt").write_text("X")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "y.txt").write_text("Y")
    (tmp_path / "a").symlink_to(".")
    (tmp_path / "b").symlink_to(".")
    result = await asyncio.wait_for(_read_internal(str(tmp_path / "**" / "*.txt")), 5)
    assert sorted(result) == [str(tmp_path / "sub" / "y.txt"), str(tmp_path / "x.txt")]
    assert await asyncio.wait_for(_read_internal(str(tmp_path / "**" / "*.zzz")), 5) == {}


def test_glob_regex_semantics():
    from grok_mcp.tools.files import _glob_regex

    r = _glob_regex("/w/**/*.py")
    assert r.match("/w/a.py") and r.match("/w/x/y/a.py")
    assert not r.match("/w/a.pyc") and not r.match("/v/a.py")
    assert _glob_regex("/w/*.py").match("/w/a.py")
    assert not _glob_regex("/w/*.py").match("/w/x/a.py")


async def test_plain_overwrite_needs_no_read_access(monkeypatch, tmp_path):
    """Only a diff or a backup disclose the old content; a plain overwrite needs write access."""
    from grok_mcp.tools.files import _write_single

    f = tmp_path / "log.txt"
    f.write_text("old")
    monkeypatch.setattr(settings, "write_allowed_paths", [str(tmp_path)])
    monkeypatch.setattr(settings, "read_allowed_paths", [str(tmp_path / "other")])
    result = await _write_single(str(f), "new", dry_run=False, backup=False)
    assert result["written"] and f.read_text() == "new"


async def test_budget_counts_raw_bytes_not_replacement_chars(monkeypatch, tmp_path):
    from grok_mcp.tools.files import _read_internal

    (tmp_path / "a.bin").write_bytes(b"\xff" * 40)  # 40 bytes → 120 bytes once re-encoded
    (tmp_path / "b.bin").write_bytes(b"\xff" * 40)
    monkeypatch.setattr(settings, "max_read_bytes", 100)
    result = await _read_internal([str(tmp_path / "a.bin"), str(tmp_path / "b.bin")])
    assert len(result) == 2  # 80 raw bytes fit the 100-byte budget


def test_write_refuses_a_fifo_without_blocking(monkeypatch, tmp_path):
    import os

    from grok_mcp.paths import write_bytes_nofollow

    monkeypatch.setattr(settings, "write_allowed_paths", [str(tmp_path)])
    fifo = tmp_path / "out.mp3"
    os.mkfifo(fifo)
    with pytest.raises(OSError):  # ENXIO at once: no reader, O_NONBLOCK
        write_bytes_nofollow(fifo, b"x")


def test_write_never_unlinks_a_file_it_did_not_create(monkeypatch, tmp_path):
    from grok_mcp.paths import write_bytes_nofollow

    allowed, outside = tmp_path / "allowed", tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (outside / "keep.txt").write_text("precious")
    (allowed / "d").symlink_to(outside)  # a swapped directory component
    monkeypatch.setattr(settings, "write_allowed_paths", [str(allowed)])
    with pytest.raises(PermissionError):
        write_bytes_nofollow(allowed / "d" / "keep.txt", b"x")
    assert (outside / "keep.txt").read_text() == "precious"
    with pytest.raises(PermissionError):
        write_bytes_nofollow(allowed / "d" / "new.txt", b"x")
    assert not (outside / "new.txt").exists()  # created by this call → removed again


def test_write_creates_files_for_the_owner_only(monkeypatch, tmp_path):
    import os
    import stat

    from grok_mcp.paths import write_bytes_nofollow

    monkeypatch.setattr(settings, "write_allowed_paths", [str(tmp_path)])
    old_umask = os.umask(0)  # the mode itself must leave others out, not the umask
    try:
        write_bytes_nofollow(tmp_path / "new.png", b"x")
        existing = tmp_path / "existing.txt"
        existing.write_text("old")
        existing.chmod(0o604)
        write_bytes_nofollow(existing, b"new")
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE((tmp_path / "new.png").stat().st_mode) == 0o600
    assert stat.S_IMODE(existing.stat().st_mode) == 0o604  # an overwrite keeps the mode
    assert existing.read_bytes() == b"new"


async def test_question_mark_is_a_wildcard(tmp_path):
    from grok_mcp.tools.files import _read_internal

    (tmp_path / "a1.py").write_text("1")
    (tmp_path / "a2.py").write_text("2")
    result = await _read_internal(str(tmp_path / "a?.py"))
    assert sorted(result) == [str(tmp_path / "a1.py"), str(tmp_path / "a2.py")]


async def test_multi_file_write_validates_all_before_writing_any(monkeypatch, tmp_path):
    from grok_mcp.tools.files import _write_internal

    (tmp_path / "ok.txt").write_text("old")
    (tmp_path / ".env").write_text("SECRET=1")
    monkeypatch.setattr(settings, "write_allowed_paths", [str(tmp_path)])
    items = [
        {"path": str(tmp_path / "ok.txt"), "content": "new"},
        {"path": str(tmp_path / ".env"), "content": "x"},  # backup would read a .env
    ]
    with pytest.raises(PermissionError):
        await _write_internal(items, dry_run=False, backup=True)
    assert (tmp_path / "ok.txt").read_text() == "old"  # nothing written


def test_makedirs_checked_creates_nested_dirs(monkeypatch, tmp_path):
    from grok_mcp.paths import makedirs_checked

    monkeypatch.setattr(settings, "write_allowed_paths", [str(tmp_path)])
    makedirs_checked(tmp_path / "a" / "b" / "c")
    assert (tmp_path / "a" / "b" / "c").is_dir()
    makedirs_checked(tmp_path / "a" / "b" / "c")  # existing → no-op


def test_makedirs_checked_removes_dir_created_through_symlink(monkeypatch, tmp_path):
    # A parent swapped for a symlink (e.g. by a racing shell command) must not leave a
    # directory outside the allow-list behind.
    from grok_mcp.paths import makedirs_checked

    allowed, outside = tmp_path / "allowed", tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (allowed / "link").symlink_to(outside)
    monkeypatch.setattr(settings, "write_allowed_paths", [str(allowed)])
    with pytest.raises(PermissionError):
        makedirs_checked(allowed / "link" / "sub" / "deeper")
    assert list(outside.iterdir()) == []


def test_makedirs_checked_catches_swap_after_check(monkeypatch, tmp_path):
    import os

    from grok_mcp import paths

    allowed, outside = tmp_path / "allowed", tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    monkeypatch.setattr(settings, "write_allowed_paths", [str(allowed)])
    real_mkdir = os.mkdir

    def racing_mkdir(p, *a):
        if p == allowed / "x":  # the race: x appears as a symlink just before mkdir
            os.symlink(outside, p)
            raise FileExistsError(p)
        real_mkdir(p, *a)

    monkeypatch.setattr(paths.os, "mkdir", racing_mkdir)
    with pytest.raises(PermissionError):
        paths.makedirs_checked(allowed / "x" / "y")
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("rel", [".git/hooks/post-checkout", ".git/config", "sub/.git/HEAD"])
def test_check_writable_refuses_git_dir(monkeypatch, tmp_path, rel):
    monkeypatch.setattr(settings, "write_allowed_paths", [str(tmp_path)])
    with pytest.raises(PermissionError, match=r"\.git"):
        check_writable(tmp_path / rel)
    check_writable(tmp_path / ".gitignore")  # only the directory component counts


async def test_precheck_shares_one_budget_across_attachments(monkeypatch, tmp_path):
    from grok_mcp.tools.budget import precheck_input

    (tmp_path / "a.txt").write_text("x" * 60)
    (tmp_path / "b.png").write_bytes(b"x" * 60)
    monkeypatch.setattr(settings, "max_read_bytes", 100)  # each fits, together 120 > 100
    with pytest.raises(ValueError, match="GROK_MAX_READ_BYTES budget"):
        await precheck_input("p", "m", None, [str(tmp_path / "a.txt")], [str(tmp_path / "b.png")])


async def test_attachment_reads_share_the_remaining_budget(monkeypatch, tmp_path):
    """Files that grew after the precheck still cannot exceed the shared budget."""
    from grok_mcp.tools.chat import _ToolLoop

    (tmp_path / "a.txt").write_text("x" * 60)
    (tmp_path / "b.png").write_bytes(b"x" * 60)
    monkeypatch.setattr(settings, "max_read_bytes", 100)
    loop = _ToolLoop.__new__(_ToolLoop)
    loop.conversation, loop.messages = None, []
    with pytest.raises(ValueError, match="too large"):
        await loop.build_messages("p", None, [str(tmp_path / "a.txt")], [str(tmp_path / "b.png")])


@pytest.mark.parametrize("name", [".git", ".GIT", ".Git"])
def test_git_write_block_is_case_insensitive(monkeypatch, tmp_path, name):
    monkeypatch.setattr(settings, "write_allowed_paths", [str(tmp_path)])
    with pytest.raises(PermissionError, match=".git directory"):
        check_writable(tmp_path / name / "hooks" / "pre-commit")


async def test_glob_does_not_walk_outside_the_read_allow_list(monkeypatch, tmp_path):
    import os

    from grok_mcp.tools import files

    (tmp_path / "ok" / "sub").mkdir(parents=True)
    (tmp_path / "ok" / "sub" / "a.txt").write_text("A")
    for i in range(20):
        (tmp_path / "outside" / f"d{i}").mkdir(parents=True)
    monkeypatch.setattr(settings, "read_allowed_paths", [str(tmp_path / "ok" / "sub")])
    walked = []
    real_walk = os.walk

    def counting_walk(*args, **kwargs):
        for entry in real_walk(*args, **kwargs):
            walked.append(entry[0])
            yield entry

    monkeypatch.setattr(files.os, "walk", counting_walk)
    result = await files._read_internal(str(tmp_path / "**" / "*.txt"))
    assert list(result) == [str(tmp_path / "ok" / "sub" / "a.txt")]
    assert not [d for d in walked if "outside" in d]
    walked.clear()
    assert await files._read_internal(str(tmp_path / "outside" / "**" / "*.txt")) == {}
    assert walked == []
