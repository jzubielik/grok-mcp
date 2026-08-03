# tests/tools/test_jobs.py
"""Background jobs: long calls return a job_id immediately and survive client timeouts."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def db(mock_db_pool):
    """mock_db_pool plus is_db_available patched True in the jobs module."""
    import grok_mcp.tools.jobs as jobs

    with patch.object(jobs, "is_db_available", return_value=True):
        yield mock_db_pool.acquire.return_value.__aenter__.return_value


async def _wait_for(predicate, timeout: float = 1.0):
    """Give the background task a chance to run."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# start_job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_job_returns_id_without_waiting(db):
    from grok_mcp.tools.jobs import start_job

    started = asyncio.Event()
    release = asyncio.Event()

    async def work():
        started.set()
        await release.wait()
        return "late result"

    job_id = await start_job("reason", work)
    assert job_id  # returned before work finished
    assert not release.is_set()
    await asyncio.wait_for(started.wait(), timeout=1)
    release.set()


@pytest.mark.asyncio
async def test_start_job_inserts_running_row(db):
    from grok_mcp.tools.jobs import start_job

    async def work():
        return "ok"

    job_id = await start_job("chat", work)
    insert = db.execute.await_args_list[0].args
    assert job_id in insert
    assert "running" in insert
    assert "chat" in insert


@pytest.mark.asyncio
async def test_completed_job_is_marked_done_with_result(db):
    from grok_mcp.tools.jobs import start_job

    async def work():
        return "the answer"

    await start_job("reason", work)
    assert await _wait_for(lambda: db.execute.await_count >= 2)
    update = db.execute.await_args_list[-1].args
    assert "done" in update
    assert "the answer" in update


@pytest.mark.asyncio
async def test_failed_job_is_marked_error(db):
    from grok_mcp.tools.jobs import start_job

    async def work():
        raise RuntimeError("upstream exploded")

    await start_job("reason", work)
    assert await _wait_for(lambda: db.execute.await_count >= 2)
    update = db.execute.await_args_list[-1].args
    assert "error" in update
    assert any("upstream exploded" in str(a) for a in update)


@pytest.mark.asyncio
async def test_start_job_requires_db():
    import grok_mcp.tools.jobs as jobs

    async def work():
        return "x"

    with patch.object(jobs, "is_db_available", return_value=False), pytest.raises(RuntimeError):
        await jobs.start_job("chat", work)


# ---------------------------------------------------------------------------
# get_job / list_jobs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_job_returns_result(db):
    from grok_mcp.tools.jobs import _get_job_internal

    db.fetchrow.return_value = {
        "job_id": "j1",
        "tool": "reason",
        "status": "done",
        "result": "answer text",
        "error": None,
        "model": "grok-4.6",
        "created_at": None,
        "finished_at": None,
    }
    result = await _get_job_internal("j1")
    assert result["status"] == "done"
    assert result["result"] == "answer text"


@pytest.mark.asyncio
async def test_get_job_unknown_id(db):
    from grok_mcp.tools.jobs import _get_job_internal

    db.fetchrow.return_value = None
    with pytest.raises(ValueError, match="no such job"):
        await _get_job_internal("nope")


@pytest.mark.asyncio
async def test_list_jobs_returns_recent(db):
    from grok_mcp.tools.jobs import _list_jobs_internal

    db.fetch.return_value = [
        {
            "job_id": "j2",
            "tool": "reason",
            "status": "running",
            "created_at": None,
            "finished_at": None,
        },
    ]
    result = await _list_jobs_internal(limit=5)
    assert result["jobs"][0]["job_id"] == "j2"


# ---------------------------------------------------------------------------
# agent() background wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_background_returns_job_id_immediately():
    from grok_mcp.tools.agent import agent

    with patch("grok_mcp.tools.chat.start_job", AsyncMock(return_value="job-123")) as start:
        payload = await agent("long prompt", model="m", background=True)

    assert payload["job_id"] == "job-123"
    assert payload["status"] == "running"
    start.assert_awaited_once()
    assert start.await_args.args[0] == "agent"


@pytest.mark.asyncio
async def test_agent_background_with_another_model_returns_job_id():
    from grok_mcp.tools.agent import agent

    with patch("grok_mcp.tools.chat.start_job", AsyncMock(return_value="job-9")) as start:
        payload = await agent("hard problem", model="grok-4.7", background=True)

    assert payload["job_id"] == "job-9"
    assert start.await_args.args[0] == "agent"


def test_structured_job_result_is_returned_as_an_object():
    """agent() results are stored as JSON and must come back nested, not as escaped text."""
    from grok_mcp.tools.jobs import _row_to_dict

    row = {"job_id": "j", "status": "done", "result": '{"answer": "hi", "cost_usd": 0.1}'}
    assert _row_to_dict(row)["result"] == {"answer": "hi", "cost_usd": 0.1}


def test_plain_text_job_result_stays_text():
    from grok_mcp.tools.jobs import _row_to_dict

    assert _row_to_dict({"job_id": "j", "result": "plain"})["result"] == "plain"


@pytest.mark.asyncio
async def test_fails_only_jobs_whose_heartbeat_stopped(mock_db_pool):
    from grok_mcp.tools.jobs import fail_orphaned_jobs

    conn = mock_db_pool.acquire.return_value.__aenter__.return_value
    conn.execute.return_value = "UPDATE 2"

    assert await fail_orphaned_jobs() == 2
    sql = conn.execute.await_args.args[0]
    assert "SET status = 'error'" in sql
    assert "WHERE status = 'running'" in sql
    assert "heartbeat_at" in sql  # only jobs whose process stopped heartbeating


@pytest.mark.asyncio
async def test_fail_orphaned_jobs_is_a_noop_without_db(no_db):
    from grok_mcp.tools.jobs import fail_orphaned_jobs

    assert await fail_orphaned_jobs() == 0


@pytest.fixture(autouse=True)
def _no_orphan_sweep(request, monkeypatch):
    """Job reads sweep orphans first; tests of the reads themselves don't model that query."""
    if "heartbeat" not in request.node.name and "orphan" not in request.node.name:
        monkeypatch.setattr("grok_mcp.tools.jobs.fail_orphaned_jobs", AsyncMock(return_value=0))


@pytest.mark.asyncio
async def test_job_reads_sweep_orphans_first(mock_db_pool, monkeypatch):
    from grok_mcp.tools import jobs

    sweep = AsyncMock(return_value=0)
    monkeypatch.setattr(jobs, "fail_orphaned_jobs", sweep)
    conn = mock_db_pool.acquire.return_value.__aenter__.return_value
    conn.fetchrow = AsyncMock(return_value=None)
    with pytest.raises(ValueError):
        await jobs._get_job_internal("x")  # unknown id
    sweep.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_caller_leaves_a_job_behind(monkeypatch):
    """A client that cancels before the handoff must not leave paid work unobserved —
    under anyio's level-triggered cancel scopes, as the MCP SDK runs requests."""
    import asyncio

    import anyio

    from grok_mcp.tools import jobs

    adopted = {}

    async def fake_adopt(tool, task, model=None, usage_sink=None, job_ref=None):
        await asyncio.sleep(0)  # a real DB insert awaits — must survive the cancel scope
        adopted["task"] = task
        return "job-c"

    monkeypatch.setattr(jobs, "adopt_task", fake_adopt)

    async def work():
        await asyncio.sleep(0.2)
        return {"answer": "done anyway"}

    with anyio.CancelScope() as scope:

        async def cancel_soon():
            await asyncio.sleep(0.05)
            scope.cancel()

        asyncio.get_running_loop().create_task(cancel_soon())
        await jobs.run_with_handoff("agent", work, "m", {}, timeout=5)
    await asyncio.sleep(0.05)
    assert (await adopted["task"])["answer"] == "done anyway"


@pytest.mark.asyncio
async def test_adopted_run_logs_carry_job_id(monkeypatch):
    """Log lines of a run handed off after it started still get the job id."""
    import asyncio
    import json
    import logging

    from grok_mcp import logjson
    from grok_mcp.tools import jobs

    lines = []

    async def fake_adopt(tool, task, model=None, usage_sink=None, job_ref=None):
        job_ref["id"] = "job-log"
        return "job-log"

    monkeypatch.setattr(jobs, "adopt_task", fake_adopt)

    async def work():
        await asyncio.sleep(0.1)
        record = logging.LogRecord("t", logging.INFO, __file__, 1, "inside", None, None)
        lines.append(json.loads(logjson.JsonFormatter().format(record)))
        return {}

    kind, job_id = await jobs.run_with_handoff("agent", work, "m", {}, timeout=0.01)
    assert (kind, job_id) == ("job", "job-log")
    await asyncio.sleep(0.2)
    assert lines and lines[0]["job_id"] == "job-log"


@pytest.mark.asyncio
async def test_finish_retries_transient_db_errors(monkeypatch):
    from grok_mcp.tools import jobs

    calls = []

    async def flaky(*args):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("db blip")

    async def no_sleep(_):
        return None

    monkeypatch.setattr(jobs, "_persist", flaky)
    monkeypatch.setattr(jobs.asyncio, "sleep", no_sleep)
    await jobs._finish("j", "done", result="r")
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_drain_cancels_jobs_still_running_after_timeout():
    import asyncio

    from grok_mcp.tools import jobs

    task = asyncio.create_task(asyncio.sleep(10))
    jobs._TASKS.add(task)
    await jobs.drain(timeout=0.05)
    assert task.cancelled()


@pytest.mark.asyncio
async def test_cancel_during_the_adoption_insert_still_records_the_job(monkeypatch):
    """A client giving up exactly at the handoff (mid-INSERT) must not lose the run."""
    import asyncio

    from grok_mcp.tools import jobs

    adopted = {}

    async def slow_adopt(tool, task, model=None, usage_sink=None, job_ref=None):
        await asyncio.sleep(0.2)  # the DB round-trip
        adopted["task"] = task
        return "job-x"

    monkeypatch.setattr(jobs, "adopt_task", slow_adopt)

    async def work():
        await asyncio.sleep(0.5)
        return {"answer": "ok"}

    caller = asyncio.create_task(jobs.run_with_handoff("agent", work, "m", {}, timeout=0.05))
    await asyncio.sleep(0.1)  # timeout passed, adoption in flight
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    await asyncio.sleep(0.25)
    assert (await adopted["task"])["answer"] == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize(("given", "sent"), [(10_000, 100), (0, 1), (-5, 1), (7, 7)])
async def test_list_jobs_clamps_limit(db, given, sent):
    from grok_mcp.tools.jobs import _list_jobs_internal

    db.fetch.return_value = []
    await _list_jobs_internal(limit=given)
    assert db.fetch.call_args.args[-1] == sent


def test_cap_text_keeps_json_valid(monkeypatch):
    import json

    from grok_mcp.tools import jobs

    monkeypatch.setattr(jobs, "_MAX_STORED_TEXT", 50)
    big = json.dumps({"text": "x" * 200})
    stored = jobs._cap_text(big, as_json=True)
    assert json.loads(stored) == {"truncated": True, "text": big[:50]}
    assert jobs._row_to_dict({"result": stored})["result"]["truncated"] is True
    plain = jobs._cap_text("y" * 200)
    assert plain.startswith("y" * 50) and "truncated" in plain and len(plain) < 200
    assert jobs._cap_text("short", as_json=True) == "short"


@pytest.mark.asyncio
async def test_persist_caps_result_and_error(db, monkeypatch):
    from grok_mcp.tools import jobs

    monkeypatch.setattr(jobs, "_MAX_STORED_TEXT", 10)
    await jobs._persist("j", "error", "r" * 100, "e" * 100, None)
    args = db.execute.call_args.args
    assert len(args[3]) < 100 and len(args[4]) < 100


@pytest.mark.asyncio
async def test_persist_redacts_a_json_result_field_by_field(db):
    # A PEM BEGIN marker without END used to mask the serialised JSON to its end, so the stored
    # result no longer parsed and get_job lost model, tokens and cost.
    import json

    from grok_mcp.tools import jobs

    marker = "-----BEGIN " + "PRIVATE KEY-----"  # in pieces: secret scanners flag the whole
    result = json.dumps({"answer": "files start with " + marker, "model": "m"})
    await jobs._persist("j", "done", result, None, None)
    stored = json.loads(db.execute.call_args.args[3])
    assert stored["model"] == "m"
    assert "REDACTED" in stored["answer"]
