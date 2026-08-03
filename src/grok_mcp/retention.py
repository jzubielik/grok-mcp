"""Retention purge: delete or clear stored content once its GROK_RETENTION_* period ends.

Runs once at startup and then every 24 h while the server is up. A period of 0 keeps that
data forever.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from .config import settings
from .db import get_pool, is_db_available

logger = logging.getLogger("grok_mcp.retention")

PURGE_INTERVAL_S = 24 * 3600

# (setting name, SQL) — $1 is the period in days. Each statement touches only rows past it.
_PURGES: tuple[tuple[str, str], ...] = (
    (
        "retention_chat_days",
        "DELETE FROM chat_history WHERE created_at < NOW() - make_interval(days => $1)",
    ),
    (
        "retention_job_content_days",
        "UPDATE jobs SET result = NULL, error = NULL "
        "WHERE finished_at < NOW() - make_interval(days => $1) "
        "AND (result IS NOT NULL OR error IS NOT NULL)",
    ),
    (
        "retention_jobs_days",
        "DELETE FROM jobs WHERE created_at < NOW() - make_interval(days => $1) "
        "AND status <> 'running'",
    ),
    (
        "retention_projects_days",
        "DELETE FROM project_discovery WHERE updated_at < NOW() - make_interval(days => $1)",
    ),
)


async def purge_expired() -> dict[str, int]:
    """Apply every retention period; return rows affected per setting. Never raises."""
    if not is_db_available():
        return {}
    # Only finished jobs age out (by finished_at / status), so a job left 'running' by a dead
    # process would be kept forever — close orphans first. Imported here: tools.jobs pulls in
    # the server module.
    from .tools.jobs import fail_orphaned_jobs

    await fail_orphaned_jobs()
    counts: dict[str, int] = {}
    try:
        async with get_pool().acquire() as conn:
            for setting, sql in _PURGES:
                days = int(getattr(settings, setting))
                if days <= 0:
                    continue
                tag = await conn.execute(sql, days)  # e.g. "DELETE 12" / "UPDATE 3"
                counts[setting] = int(tag.rsplit(" ", 1)[-1])
    except Exception as exc:  # noqa: BLE001
        logger.warning("retention_purge_failed", extra={"fields": {"error": type(exc).__name__}})
        return counts
    logger.info("retention_purge", extra={"fields": counts})
    return counts


async def _purge_loop() -> None:
    while True:
        await purge_expired()
        await asyncio.sleep(PURGE_INTERVAL_S)


_task: asyncio.Task | None = None


def start_purge_loop() -> None:
    """Start the periodic purge (idempotent)."""
    global _task  # noqa: PLW0603
    if _task is None or _task.done():
        _task = asyncio.create_task(_purge_loop())


async def stop_purge_loop() -> None:
    global _task  # noqa: PLW0603
    if _task is not None:
        _task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _task
        _task = None
