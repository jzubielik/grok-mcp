"""read() and write() tools — unified file I/O with optional Grok analysis."""

from __future__ import annotations

import asyncio
import difflib
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path

# typing.TypedDict is rejected by pydantic on Python < 3.12; FastMCP builds the schema with it.
from typing_extensions import TypedDict

from ..client import get_text_client
from ..config import settings
from ..models_registry import require_model
from ..paths import (
    check_readable,
    check_writable,
    read_bytes_capped,
    read_line_range,
    read_text_capped,
    write_text_nofollow,
)
from ..server import mcp

_MAX_GLOB_MATCHES = 2000


class WriteItem(TypedDict):
    """One file of a multi-file write()."""

    path: str
    content: str


_MAX_GLOB_VISITED = 200_000
_GLOB_DEADLINE_S = 20


def _is_glob(path: str) -> bool:
    """`*` and `?` are the supported wildcards (`**` = any depth); `[` is a literal character."""
    return "*" in path or "?" in path


def _glob_regex(pattern: str) -> re.Pattern[str]:
    """Translate a glob (`**` = any depth, `*` / `?` within one segment) to a regex."""
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def _parts(path: str) -> list[str]:
    return os.path.normpath(path).split(os.sep)


def _glob_files(pattern: str) -> Iterator[str]:
    """Files matching `pattern`, walking the tree without following directory symlinks.

    glob.iglob(recursive=True) follows them: two `a -> .` links in a cloned repo make it run
    forever, pinning a worker thread. This walk never leaves the real tree, and stops after
    _MAX_GLOB_VISITED entries or _GLOB_DEADLINE_S seconds.
    """
    head = pattern.split("*", 1)[0].split("?", 1)[0]
    root = head if head.endswith("/") else (os.path.dirname(head) or ".")
    # Match normalised paths: the walk of root "." yields "./a.py", which "*.py" (or the
    # equivalent "./*.py") must still match.
    norm_pattern = os.path.normpath(pattern)
    regex = _glob_regex(norm_pattern)
    # Without "**" a match lies exactly this many directory levels below root — never walk
    # deeper than that ("docs/*.md" must not crawl all of docs/).
    max_depth = None
    if "**" not in pattern:
        root_parts = 0 if os.path.normpath(root) == "." else len(_parts(root))
        max_depth = len(_parts(norm_pattern)) - root_parts - 1
    # With an allow-list, only walk directories that are inside an allowed path or lead to
    # one: crawling the rest only burns the visit budget (and probes outside trees) for
    # matches that would be skipped anyway.
    allowed = [Path(a).resolve() for a in settings.read_allowed_paths]

    def walkable(d: str) -> bool:
        real = Path(d).resolve()
        return any(real.is_relative_to(a) or a.is_relative_to(real) for a in allowed)

    if allowed and not walkable(root):
        return
    starts = [root]
    real_root = Path(root).resolve()
    if allowed and not any(real_root.is_relative_to(a) for a in allowed):
        # root is an ancestor of allowed paths (e.g. "/"): start at those paths themselves —
        # walking down from the ancestor would list every directory on the way to them.
        # Nested allowed paths are walked once, via the outermost one.
        under = sorted({a for a in allowed if a.is_relative_to(real_root)}, key=lambda a: a.parts)
        outer = [a for a in under if not any(a != b and a.is_relative_to(b) for b in under)]
        # Joined onto root, so results keep the pattern's own (possibly relative) spelling.
        starts = [os.path.join(root, a.relative_to(real_root)) for a in outer]
    deadline = time.monotonic() + _GLOB_DEADLINE_S
    visited = 0
    walk = (entry for start in starts for entry in os.walk(start, followlinks=False))
    for dirpath, dirnames, filenames in walk:
        if allowed:
            dirnames[:] = [d for d in dirnames if walkable(os.path.join(dirpath, d))]
        if max_depth is not None:
            rel = os.path.relpath(dirpath, root)
            if (0 if rel == "." else len(_parts(rel))) >= max_depth:
                dirnames[:] = []
        dirnames.sort()
        visited += len(dirnames) + len(filenames)
        if visited > _MAX_GLOB_VISITED or time.monotonic() > deadline:
            raise ValueError("glob walked too much of the tree — narrow it")
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            if regex.match(os.path.normpath(full)) and os.path.isfile(full):
                yield full


def _readable(path: Path) -> bool:
    try:
        check_readable(path)
    except PermissionError:
        return False
    return True


def _line_range(path: str, offset: int | None, limit: int | None, budget: list[int]) -> str:
    """Lines [offset, offset+limit), numbered, under a '[lines a-b of N]' header.

    Streamed, so a file larger than GROK_MAX_READ_BYTES can still be paged through; only the
    returned lines are charged to the budget. "of N+" means the count stopped at the scan cap.
    """
    lines, total, complete = read_line_range(path, offset, limit, max_bytes=budget[0])
    start = max(0, offset or 0)
    body = "\n".join(f"{start + i + 1}: {line}" for i, line in enumerate(lines))
    return f"[lines {start + 1}-{start + len(lines)} of {total}{'' if complete else '+'}]\n{body}"


def _read_one(path: str, budget: list[int]) -> str:
    """Read one file, charging it to the call's shared byte budget (budget[0] bytes left)."""
    if not Path(path).exists():
        raise FileNotFoundError(path)
    if budget[0] <= 0:
        raise ValueError("read exceeded GROK_MAX_READ_BYTES across its files — narrow it")
    raw = read_bytes_capped(path, max_bytes=budget[0])
    budget[0] -= len(raw)
    return raw.decode("utf-8", errors="replace")


def _collect(path: str | list[str], offset: int | None, limit: int | None) -> str | dict[str, str]:
    """Blocking part of read(): run in a worker thread so a big tree never stalls the loop.

    All files of one call share GROK_MAX_READ_BYTES, so a list or glob cannot add up to more.
    Every read goes through read_text_capped, which checks the descriptor actually opened.
    """
    budget = [settings.max_read_bytes]
    if offset is not None or limit is not None:
        if not isinstance(path, str) or _is_glob(path):
            raise ValueError("offset/limit need a single file path")
        check_readable(path)
        return _line_range(path, offset, limit, budget)
    if isinstance(path, list):
        for p in path:
            check_readable(p)  # allow-list before existence: never reveal outside paths
        return {p: _read_one(p, budget) for p in path}
    if not _is_glob(path):
        check_readable(path)
        return _read_one(path, budget)
    result: dict[str, str] = {}
    for n, m in enumerate(_glob_files(path)):
        if n >= _MAX_GLOB_MATCHES:
            raise ValueError(f"glob matched more than {_MAX_GLOB_MATCHES} files — narrow it")
        # Skip matches outside the allow-list instead of raising: an error naming them
        # would tell the caller which outside files exist.
        if _readable(Path(m)):
            result[m] = _read_one(m, budget)
    return dict(sorted(result.items()))


async def _read_internal(
    path: str | list[str],
    question: str | None = None,
    model: str | None = None,
    offset: int | None = None,
    limit: int | None = None,
) -> str | dict[str, str]:
    """Core read logic — usable from the agent() tool loop and the MCP tool."""
    collected = await asyncio.to_thread(_collect, path, offset, limit)
    if question is None:
        return collected
    answer, _ = await _ask(path, collected, question, model)
    return answer


async def _ask(
    path: str | list[str], collected: str | dict[str, str], question: str, model: str | None
) -> tuple[str, str]:
    """Send the collected file contents to Grok with `question`; return (answer, model)."""
    result = collected if isinstance(collected, dict) else {str(path): collected}

    model = await require_model(model)

    # Assemble prompt and call Grok
    parts = []
    if isinstance(result, dict):
        for fpath, content in result.items():
            parts.append(
                f"```{Path(fpath).suffix.lstrip('.') or 'text'}\n# {fpath}\n{content}\n```"
            )
    prompt = "\n\n".join(parts) + f"\n\n{question}"
    client = get_text_client()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or "", model


@mcp.tool(
    description=(
        "Read one or more local files and optionally send to Grok for analysis. "
        "path: single file path, list of paths, or glob pattern (e.g. 'clusters/**/*.yaml'). "
        "question: if provided, file contents are sent to Grok with this question; "
        "Grok's answer is returned instead of raw content. "
        "offset/limit (single file): read lines [offset, offset+limit), numbered, with a "
        "'[lines a-b of N]' header. "
        "Raises FileNotFoundError if any path is missing. "
        "Returns {path, content} for one file, {files: {path: content}} for several, or "
        "{answer, model} with a question. "
        "model: required together with question — choose deliberately:\n"
        "{models}"
    )
)
async def read(
    path: str | list[str],
    question: str | None = None,
    model: str | None = None,
    offset: int | None = None,
    limit: int | None = None,
) -> dict:
    if question is not None:
        # Resolve the model once: _ask returns the id it validated.
        collected = await asyncio.to_thread(_collect, path, offset, limit)
        answer, model = await _ask(path, collected, question, model)
        return {"answer": answer, "model": model}
    result = await _read_internal(path, offset=offset, limit=limit)
    if isinstance(result, str):
        return {"path": path, "content": result}
    return {"files": result}


def _make_diff(original: str, updated: str, path: str) -> str:
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def _write_sync(path: str, content: str, dry_run: bool, backup: bool) -> dict:
    # Resolve once and use that path throughout, so what was checked is what gets written.
    fp = Path(path).resolve()
    check_writable(fp)
    if not fp.parent.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {fp.parent}")
    # The diff and the backup both disclose the current content: only then must it be
    # readable (and small enough to load); a plain overwrite needs write access alone.
    original = ""
    if fp.exists() and (dry_run or backup):
        check_readable(fp)
        original = read_text_capped(fp)
    if dry_run:
        return {"dry_run": True, "path": path, "diff": _make_diff(original, content, path)}
    if backup and fp.exists():
        bak = fp.with_name(fp.name + ".bak")
        check_writable(bak)
        write_text_nofollow(bak, original)  # a planted `file.bak` symlink is refused
    write_text_nofollow(fp, content)
    return {"written": True, "path": path, "bytes": len(content.encode())}


async def _write_single(path: str, content: str, dry_run: bool, backup: bool) -> dict:
    return await asyncio.to_thread(_write_sync, path, content, dry_run, backup)


async def _write_internal(
    path: str | list[WriteItem],
    content: str | None = None,
    dry_run: bool = True,
    backup: bool = True,
) -> dict | list[dict]:
    if isinstance(path, list):
        # Validate all before writing any — including what the diff / backup must read.
        for item in path:
            check_writable(item["path"])
            target = Path(item["path"])
            if not target.parent.exists():
                raise FileNotFoundError(f"Parent directory does not exist: {target.parent}")
            if target.exists() and (dry_run or backup):
                check_readable(target)
        return [
            await _write_single(item["path"], item["content"], dry_run, backup) for item in path
        ]
    if content is None:
        raise ValueError("content is required for a single-file write")
    return await _write_single(path, content, dry_run, backup)


@mcp.tool(
    description=(
        "Write one or more files. dry_run=True (default) returns a unified diff without writing. "
        "dry_run=False writes the file; backup=True (default) saves original as <path>.bak. "
        "Single file: write(path, content, dry_run=True). "
        "Multi-file: write([{path, content}, ...], dry_run=True). "
        "Raises PermissionError if path is not under GROK_WRITE_ALLOWED_PATHS. "
        "Raises FileNotFoundError if parent directory does not exist."
    )
)
async def write(
    path: str | list[WriteItem],
    content: str | None = None,
    dry_run: bool = True,
    backup: bool = True,
) -> dict:
    result = await _write_internal(path, content=content, dry_run=dry_run, backup=backup)
    return {"files": result} if isinstance(result, list) else result
