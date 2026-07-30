"""Background jobs — long calls return a job_id immediately and outlive the client.

MCP clients abort a tool call at their own timeouts. Claude Code, for one, applies a 5 minute
idle timeout to HTTP servers (CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT) whenever the server sends
neither a response nor a progress notification. A job decouples the work
from that connection: the tool returns a job_id at once, the server keeps running, and
the result is persisted in Postgres so it can be collected later — after a client
timeout, a reconnect, or an entirely new session.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from uuid import uuid4

from ..db import get_pool, is_db_available
from ..logjson import bind_job_ref, current_job_id, error_text, redact
from ..server import mcp

logger = logging.getLogger("grok_mcp.tools.jobs")

# A runaway result (a model echoing a huge file) must not bloat the jobs table or every
# get_job response; far above any normal answer.
_MAX_STORED_TEXT = 1_000_000
_TRUNCATION_MARKER = "\n[... truncated by grok-mcp: stored text capped at {n} chars]"

# Strong refs to running tasks — asyncio only keeps weak ones, so without this a
# job could be garbage-collected mid-flight.
_TASKS: set[asyncio.Task] = set()

# Liveness: every process heartbeats the jobs it runs. Several processes may share one
# database (stdio sessions next to the compose server), so "running" alone does not mean
# "dead after my restart" — only a job whose heartbeat stopped is orphaned.
HEARTBEAT_S = 30
ORPHAN_AFTER_S = 120
_RUNNING: set[str] = set()
_heartbeat_task: asyncio.Task | None = None


async def _heartbeat_loop() -> None:
    while _RUNNING:
        await asyncio.sleep(HEARTBEAT_S)
        if not _RUNNING or not is_db_available():
            continue
        try:
            async with get_pool().acquire() as conn:
                await conn.execute(
                    "UPDATE jobs SET heartbeat_at = NOW() WHERE job_id = ANY($1::text[])",
                    list(_RUNNING),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("job_heartbeat_failed", extra={"fields": {"error": error_text(exc)}})


def _track(job_id: str) -> None:
    global _heartbeat_task  # noqa: PLW0603
    _RUNNING.add(job_id)
    if _heartbeat_task is None or _heartbeat_task.done():
        _heartbeat_task = asyncio.create_task(_heartbeat_loop())


async def fail_orphaned_jobs() -> int:
    """Mark jobs whose process died as failed; return how many.

    Jobs run as tasks inside a server process and only the task itself records its end, so
    a killed process leaves them 'running' forever. A job counts as dead once its heartbeat
    is older than ORPHAN_AFTER_S — jobs of other live processes keep theirs fresh.
    """
    if not is_db_available():
        return 0
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            tag = await conn.execute(
                "UPDATE jobs SET status = 'error', finished_at = NOW(), "
                "error = 'interrupted at ' || "
                "to_char(NOW() AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"') || "
                "' — the process running the job stopped; re-run it' "
                "WHERE status = 'running' AND COALESCE(heartbeat_at, created_at) "
                "< NOW() - make_interval(secs => $1)",
                float(ORPHAN_AFTER_S),
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("failed to mark orphaned jobs: %s", exc)
        return 0
    count = int(tag.split()[-1])
    if count:
        logger.warning("orphaned_jobs_failed", extra={"fields": {"count": count}})
    return count


def _spawn(coro: Awaitable[Any]) -> asyncio.Task:
    """Start a background task held by a strong reference until it ends."""
    task = asyncio.ensure_future(coro)
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return task


async def _insert_job(tool: str, model: str | None) -> str:
    if not is_db_available():
        raise RuntimeError("Background jobs require DATABASE_URL (jobs table).")
    job_id = uuid4().hex
    async with get_pool().acquire() as conn:
        await conn.execute(
            "INSERT INTO jobs (job_id, tool, status, model, heartbeat_at) "
            "VALUES ($1, $2, $3, $4, NOW())",
            job_id,
            tool,
            "running",
            model,
        )
    return job_id


async def start_job(
    tool: str,
    work: Callable[[], Awaitable[Any]],
    model: str | None = None,
    usage_sink: dict | None = None,
) -> str:
    """Persist a running job, start `work` in the background, return its job_id."""
    job_id = await _insert_job(tool, model)
    _spawn(_run_job(job_id, work, usage_sink))
    logger.info("job %s started (tool=%s)", job_id, tool)
    return job_id


async def adopt_task(
    tool: str,
    task: asyncio.Task,
    model: str | None = None,
    usage_sink: dict | None = None,
    job_ref: dict | None = None,
) -> str:
    """Turn an already-running task into a job (used when a foreground call runs long).

    The task keeps running untouched; its result is recorded like any job's. job_ref (from
    bind_job_ref before the task was created) makes the task's own log lines carry the id.
    """
    job_id = await _insert_job(tool, model)
    if job_ref is not None:
        job_ref["id"] = job_id
    _spawn(_run_job(job_id, lambda: task, usage_sink))
    logger.info("job %s adopted from a long foreground call (tool=%s)", job_id, tool)
    return job_id


async def drain(timeout: float = 10) -> None:
    """At shutdown: give running jobs a moment to record their result, then cancel the rest
    (and the heartbeat) before the database pool is closed underneath them."""
    pending = {t for t in _TASKS if not t.done()}
    if pending:
        _, pending = await asyncio.wait(pending, timeout=timeout)
    for task in [*pending, *([_heartbeat_task] if _heartbeat_task else [])]:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


async def run_with_handoff(
    tool: str,
    work: Callable[[], Awaitable[Any]],
    model: str | None,
    usage: dict,
    timeout: float | None,
) -> tuple[str, Any]:
    """Run `work` in the foreground; ("done", result) or ("job", job_id) after a handoff.

    Past `timeout` the SAME running task becomes a job, so the caller can return before its
    client gives up and nothing is lost. If the caller is cancelled first (client disconnect,
    MCP cancel), the task keeps running and is adopted too — paid work is never left
    unobserved. timeout=None waits for the result.

    Adoption always runs as a task of its own: under the MCP SDK (anyio) cancellation is
    level-triggered, so a cancel arriving during the adoption's DB insert must not abort it.
    """
    ref = bind_job_ref()
    task = _spawn(work())  # strong reference from the start: a cancelled caller can't drop it
    try:
        return "done", await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except TimeoutError:
        adoption = _spawn(adopt_task(tool, task, model=model, usage_sink=usage, job_ref=ref))
        try:
            job_id = await asyncio.shield(adoption)
        except asyncio.CancelledError:
            adoption.add_done_callback(_log_adoption)  # finishes on its own; just record it
            raise
        except Exception as exc:  # noqa: BLE001 — no job row possible: keep waiting instead
            logger.warning("handoff_failed", extra={"fields": {"error": error_text(exc)}})
            return "done", await task
        return "job", job_id
    except asyncio.CancelledError:
        _spawn(
            adopt_task(tool, task, model=model, usage_sink=usage, job_ref=ref)
        ).add_done_callback(_log_adoption)
        raise


def _log_adoption(adoption: asyncio.Task) -> None:
    """Record how the adoption of a cancelled caller's run ended."""
    if adoption.cancelled():
        logger.error("cancelled_run_not_adopted", extra={"fields": {"error": "CancelledError"}})
    elif (exc := adoption.exception()) is not None:
        logger.error("cancelled_run_not_adopted", extra={"fields": {"error": error_text(exc)}})
    else:
        logger.info("job %s adopted from a cancelled call", adoption.result())


async def _run_job(
    job_id: str, work: Callable[[], Awaitable[Any]], usage_sink: dict | None = None
) -> None:
    current_job_id.set(job_id)  # every log line of this run carries the job id
    _track(job_id)
    try:
        result = await work()
        text = result if isinstance(result, str) else json.dumps(result)
        await _finish(job_id, "done", result=text, usage=usage_sink)
        logger.info("job %s done (usage=%s)", job_id, usage_sink or {})
    except Exception as exc:  # noqa: BLE001
        logger.warning("job %s failed: %s", job_id, error_text(exc))
        await _finish(job_id, "error", error=str(exc), usage=usage_sink)
    finally:
        _RUNNING.discard(job_id)


async def _finish(
    job_id: str,
    status: str,
    result: str | None = None,
    error: str | None = None,
    usage: dict | None = None,
) -> None:
    # The run is finished and paid for: a transient DB error must not leave the row
    # 'running' until the orphan sweep reports it as interrupted. Retry before giving up.
    for attempt, delay in enumerate((0, 1, 3, 10)):
        await asyncio.sleep(delay)
        try:
            await _persist(job_id, status, result, error, usage)
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "job_persist_failed",
                extra={"fields": {"attempt": attempt + 1, "error": error_text(exc)}},
            )
    logger.error("job %s: result could not be persisted", job_id)


async def _persist(
    job_id: str,
    status: str,
    result: str | None,
    error: str | None,
    usage: dict | None,
) -> None:
    async with get_pool().acquire() as conn:
        await conn.execute(
            "UPDATE jobs SET status = $2, result = $3, error = $4, "
            "prompt_tokens = $5, completion_tokens = $6, reasoning_tokens = $7, "
            "finished_at = NOW() WHERE job_id = $1",
            job_id,
            status,
            None if result is None else _cap_text(_redact_result(result), as_json=True),
            None if error is None else _cap_text(redact(error, key_value=False)),
            (usage or {}).get("prompt_tokens"),
            (usage or {}).get("completion_tokens"),
            (usage or {}).get("reasoning_tokens"),
        )


def _redact_result(result: str) -> str:
    """Mask credentials in a job result. A JSON result is masked string by string: masking the
    serialised text could cut it (a private key without its END line is masked to the end of
    the text) and leave a document that no longer parses."""
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        return redact(result, key_value=False)
    return json.dumps(_redact_strings(data))


def _redact_strings(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value, key_value=False)
    if isinstance(value, list):
        return [_redact_strings(v) for v in value]
    if isinstance(value, dict):
        return {k: _redact_strings(v) for k, v in value.items()}
    return value


def _cap_text(text: str, *, as_json: bool = False) -> str:
    """Cap stored text at _MAX_STORED_TEXT. A JSON result cut mid-way would no longer parse,
    so an oversized one is stored as a JSON envelope around its first chars instead."""
    if len(text) <= _MAX_STORED_TEXT:
        return text
    if as_json:
        try:
            json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            return json.dumps({"truncated": True, "text": text[:_MAX_STORED_TEXT]})
    return text[:_MAX_STORED_TEXT] + _TRUNCATION_MARKER.format(n=_MAX_STORED_TEXT)


def _row_to_dict(row: Mapping[str, Any]) -> dict:
    out = {}
    for key in (
        "job_id",
        "tool",
        "status",
        "result",
        "error",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
    ):
        if key in row and row[key] is not None:
            out[key] = row[key]
    for key in ("created_at", "finished_at"):
        if key in row and row[key] is not None:
            out[key] = str(row[key])
    # Structured results (agent() dicts) are stored as JSON — hand them back as objects.
    if isinstance(out.get("result"), str) and out["result"].startswith("{"):
        with contextlib.suppress(json.JSONDecodeError):
            out["result"] = json.loads(out["result"])
    return out


async def _get_job_internal(job_id: str) -> dict:
    if not is_db_available():
        raise RuntimeError("Background jobs require the database (unreachable).")
    await fail_orphaned_jobs()  # a job whose process died must not read 'running' forever
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT job_id, tool, status, result, error, model, prompt_tokens, "
                "completion_tokens, reasoning_tokens, created_at, finished_at "
                "FROM jobs WHERE job_id = $1",
                job_id,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("job_lookup_failed", extra={"fields": {"error": error_text(exc)}})
        raise RuntimeError("job lookup failed (database error)") from exc
    if row is None:
        raise ValueError(f"no such job: {job_id}")
    return _row_to_dict(row)


_MAX_LIST_JOBS = 100


async def _list_jobs_internal(limit: int = 10) -> dict:
    limit = max(1, min(int(limit), _MAX_LIST_JOBS))
    if not is_db_available():
        raise RuntimeError("Background jobs require the database (unreachable).")
    await fail_orphaned_jobs()
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT job_id, tool, status, model, prompt_tokens, completion_tokens, "
                "reasoning_tokens, created_at, finished_at FROM jobs "
                "ORDER BY created_at DESC LIMIT $1",
                limit,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("job_listing_failed", extra={"fields": {"error": error_text(exc)}})
        raise RuntimeError("job listing failed (database error)") from exc
    return {"jobs": [_row_to_dict(r) for r in rows]}


@mcp.tool(
    description=(
        "Collect the result of a background job started with agent(background=true). "
        "Returns {status: running|done|error, result|error}. Safe to poll — the job keeps "
        "running on the server regardless of client timeouts, reconnects or new sessions."
    )
)
async def get_job(job_id: str) -> dict:
    return await _get_job_internal(job_id)


@mcp.tool(
    description=(
        "List recent background jobs (newest first) with their status; limit 1-100. "
        "Use to recover a job_id you lost — e.g. after a client timeout or session restart."
    )
)
async def list_jobs(limit: int = 10) -> dict:
    return await _list_jobs_internal(limit)
