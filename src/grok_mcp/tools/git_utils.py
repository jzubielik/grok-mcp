"""Git helper: the current HEAD of a project, for project-discovery freshness."""

from __future__ import annotations

import asyncio
import logging
from asyncio.subprocess import DEVNULL, PIPE

from ..auth import scrubbed_env

logger = logging.getLogger("grok_mcp.tools.git_utils")


async def _get_git_hash(project_path: str) -> str | None:
    """Return current HEAD hash for project_path, or None if unavailable."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "rev-parse",
            "HEAD",
            cwd=project_path,
            stdin=DEVNULL,
            env=scrubbed_env(),
            stdout=PIPE,
            stderr=DEVNULL,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode("utf-8", errors="replace").strip() if proc.returncode == 0 else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("_get_git_hash failed for %s: %s", project_path, exc)
        return None
