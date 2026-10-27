"""Path guards shared by every tool that reads or writes local files."""

from __future__ import annotations

import contextlib
import os
import stat
from collections.abc import Iterator
from pathlib import Path

from .config import settings

# Never readable, whatever the allow-list says: /proc/self/environ and friends expose the
# process's secrets and memory, /dev holds endless streams (/dev/zero) and devices.
_FORBIDDEN_ROOTS = (Path("/proc"), Path("/sys"), Path("/dev"))


def is_under(path: str | Path, prefixes: list[str] | tuple[Path, ...]) -> bool:
    """True if `path` resolves inside one of `prefixes` (component-wise, symlinks resolved)."""
    p = Path(path).resolve()
    return any(p.is_relative_to(Path(prefix).resolve()) for prefix in prefixes)


def _is_env_file(path: str | Path) -> bool:
    name = Path(path).name
    return name == ".env" or (name.startswith(".env.") and name != ".env.example")


def check_readable(path: str | Path) -> None:
    """Raise PermissionError for system pseudo-files, credential files (.env, .env.*), or a
    path outside GROK_READ_ALLOWED_PATHS."""
    if is_under(path, _FORBIDDEN_ROOTS):
        raise PermissionError(f"Reading system pseudo-files is not allowed: {path}")
    if _is_env_file(path) or _is_env_file(Path(path).resolve()):
        raise PermissionError(f"Reading credential files (.env) is not allowed: {path}")
    allowed = settings.read_allowed_paths
    if allowed and not is_under(path, allowed):
        raise PermissionError(f"Path not in GROK_READ_ALLOWED_PATHS: {path}")


def check_writable(path: str | Path) -> None:
    """Raise PermissionError unless `path` is inside GROK_WRITE_ALLOWED_PATHS (empty → deny)
    and outside any .git directory."""
    # A file written into a mount can run on the host: .git/hooks/* fire on the user's next
    # git command, .git/config can set core.fsmonitor / core.hooksPath. The file tools refuse
    # .git/ (the shell can still write there — it is not a sandbox).
    # casefold: on a case-insensitive filesystem (macOS/Windows mounts) `.GIT` is `.git`.
    parts = (*Path(path).parts, *Path(path).resolve().parts)
    if any(part.casefold() == ".git" for part in parts):
        raise PermissionError(f"Writing inside a .git directory is not allowed: {path}")
    if not is_under(path, settings.write_allowed_paths):
        raise PermissionError(f"Path not in GROK_WRITE_ALLOWED_PATHS: {path}")


def checked_output_path(path: str | Path) -> Path:
    """Resolve an output path once and check it; callers write to exactly this path, so a
    symlink swapped in after the check cannot change where a later re-resolve would lead."""
    out = Path(path).resolve()  # a symlinked path inside the allowed tree is fine
    check_writable(out)
    return out


def makedirs_checked(directory: str | Path) -> None:
    """mkdir -p that never leaves a directory outside GROK_WRITE_ALLOWED_PATHS.

    Path.mkdir(parents=True) follows a parent swapped for a symlink between the caller's
    check and the mkdir (a race a `shell` command could run) and creates directories
    wherever it leads. Here each missing component is created alone and its real path
    checked right after; one that landed outside is removed again.
    """
    missing: list[Path] = []
    p = Path(directory)
    while not os.path.lexists(p) and p != p.parent:
        missing.append(p)
        p = p.parent
    for component in reversed(missing):
        try:
            os.mkdir(component)
            created = True
        except FileExistsError:
            created = False
        real = os.path.realpath(component)
        try:
            check_writable(real)
        except PermissionError:
            if created:
                with contextlib.suppress(OSError):
                    os.rmdir(real)
            raise


def _fd_path(fd: int, fallback: str | Path) -> str:
    """The real path of an open descriptor — what was actually opened, after every symlink
    and any directory swapped in meanwhile (Linux /proc/self/fd; elsewhere the resolved name)."""
    try:
        return os.readlink(f"/proc/self/fd/{fd}")
    except OSError:
        return str(Path(fallback).resolve())


@contextlib.contextmanager
def _open_regular(path: str | Path) -> Iterator[int]:
    """Open `path` read-only and yield the descriptor once it passed the checks.

    Open first, check the open descriptor second: the allow-list and type checks apply to the
    object actually read, so swapping a directory for a symlink between check and open (a race
    a `shell` command could run) cannot redirect the read to /proc/self/environ. O_NONBLOCK
    keeps a FIFO from blocking the open; only regular files are read.
    """
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOCTTY)
    try:
        check_readable(_fd_path(fd, path))
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"Not a regular file: {path}")
        yield fd
    finally:
        os.close(fd)


def read_bytes_capped(path: str | Path, max_bytes: int | None = None) -> bytes:
    """Read a regular file of at most `max_bytes` (default GROK_MAX_READ_BYTES), through
    _open_regular's descriptor checks — never more than the cap even if the file grows."""
    limit = settings.max_read_bytes if max_bytes is None else max_bytes
    with _open_regular(path) as fd:
        chunks, size = [], 0
        while chunk := os.read(fd, min(1 << 20, limit + 1 - size)):
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                raise ValueError(
                    f"File too large (> {limit} bytes, GROK_MAX_READ_BYTES): {path} — read a "
                    "line range (offset/limit) or grep it instead"
                )
        return b"".join(chunks)


# How far past GROK_MAX_READ_BYTES a ranged read may scan: the returned bytes are capped, but
# reaching line 10M of a huge log (or counting its lines) must not run unbounded.
_SCAN_FACTOR = 50
_LINE_CHUNK = 1 << 16


def read_line_range(
    path: str | Path, offset: int | None, limit: int | None, max_bytes: int | None = None
) -> tuple[list[str], int, bool]:
    """Lines [offset, offset+limit) of a file, streamed — the file is never loaded whole.

    `max_bytes` (default GROK_MAX_READ_BYTES) caps the raw bytes of the returned lines, not
    the file size. Returns (lines, total, complete): `total` counts the lines scanned — all of
    them unless the scan cap (_SCAN_FACTOR × the cap) stopped the count early (complete=False).
    Lines end at "\n" (a trailing "\r" is dropped too).
    """
    cap = settings.max_read_bytes if max_bytes is None else max_bytes
    scan_cap = _SCAN_FACTOR * max(cap, 1)
    start = max(0, offset or 0)
    end = None if limit is None else start + max(0, limit)
    out: list[str] = []
    pending: list[bytes] = []  # pieces of the current selected line (it may span chunks)
    taken = scanned = total = 0
    piece = b""
    with _open_regular(path) as fd, os.fdopen(fd, "rb", closefd=False) as f:
        while chunk := f.readline(_LINE_CHUNK):
            piece = chunk
            scanned += len(piece)
            if scanned > scan_cap:
                if end is None or total < end:
                    raise ValueError(
                        f"Line range of {path} reaches past the first {scan_cap} bytes "
                        f"({_SCAN_FACTOR}× GROK_MAX_READ_BYTES) — grep it instead"
                    )
                return out, total, False
            if total >= start and (end is None or total < end):
                taken += len(piece)
                if taken > cap:
                    raise ValueError(
                        f"Line range too large (> {cap} bytes, GROK_MAX_READ_BYTES): {path} — "
                        "narrow offset/limit"
                    )
                pending.append(piece)
            if piece.endswith(b"\n"):
                total += 1
                if pending:
                    # "\n" never occurs inside a UTF-8 sequence: a whole line decodes alone.
                    out.append(_decode_line(pending))
                    pending = []
    if piece and not piece.endswith(b"\n"):  # a last line without a newline still counts
        total += 1
        if pending:
            out.append(_decode_line(pending))
    return out, total, True


def _decode_line(pieces: list[bytes]) -> str:
    return b"".join(pieces).rstrip(b"\n").rstrip(b"\r").decode("utf-8", "replace")


def read_text_capped(path: str | Path, max_bytes: int | None = None) -> str:
    return read_bytes_capped(path, max_bytes).decode("utf-8", errors="replace")


def write_bytes_nofollow(path: str | Path, data: bytes) -> None:
    """Write `data`, re-checking the allow-list against the descriptor actually opened.

    - O_NOFOLLOW refuses a symlink at the final component (a hostile repo's
      `file.bak -> ~/.ssh/authorized_keys`); the descriptor check catches a directory component
      swapped after the caller's check.
    - O_NONBLOCK + S_ISREG: a FIFO or device planted at the target is refused instead of
      blocking the server forever.
    - Only a file this call created (O_EXCL) is removed again when the check fails — never one
      that already existed where a swapped path led.
    - A new file is 0o640: only the server's uid and group read it. An existing file keeps its mode.
    """
    flags = os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOCTTY
    try:
        fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o640)
        created = True
    except FileExistsError:
        fd = os.open(path, flags)
        created = False
    try:
        real = _fd_path(fd, path)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError(f"Not a regular file: {path}")
            check_writable(real)
        except (PermissionError, ValueError):
            if created:
                with contextlib.suppress(OSError):
                    os.unlink(real)
            raise
        os.ftruncate(fd, 0)
        _write_all(fd, data)
    finally:
        os.close(fd)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def write_text_nofollow(path: str | Path, text: str) -> None:
    write_bytes_nofollow(path, text.encode("utf-8"))
