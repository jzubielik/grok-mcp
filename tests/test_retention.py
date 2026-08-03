from unittest.mock import AsyncMock, MagicMock

import pytest

from grok_mcp import retention
from grok_mcp.config import settings


@pytest.fixture()
def conn(monkeypatch):
    c = MagicMock()
    c.execute = AsyncMock(return_value="DELETE 3")
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=c)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(retention, "is_db_available", lambda: True)
    monkeypatch.setattr(retention, "get_pool", lambda: pool)
    return c


async def test_purge_runs_every_period_with_its_days(conn, monkeypatch):
    for name, days in [
        ("retention_chat_days", 30),
        ("retention_job_content_days", 7),
        ("retention_jobs_days", 365),
        ("retention_projects_days", 90),
    ]:
        monkeypatch.setattr(settings, name, days)
    counts = await retention.purge_expired()
    assert counts == dict.fromkeys(
        [
            "retention_chat_days",
            "retention_job_content_days",
            "retention_jobs_days",
            "retention_projects_days",
        ],
        3,
    )
    sqls = [(call.args[0], call.args[1]) for call in conn.execute.call_args_list]
    assert any("chat_history" in s and d == 30 for s, d in sqls)
    assert any("SET result = NULL" in s and d == 7 for s, d in sqls)


async def test_zero_keeps_forever(conn, monkeypatch):
    for name in (
        "retention_chat_days",
        "retention_job_content_days",
        "retention_jobs_days",
        "retention_projects_days",
    ):
        monkeypatch.setattr(settings, name, 0)
    assert await retention.purge_expired() == {}
    conn.execute.assert_not_called()


async def test_no_db_is_noop(monkeypatch):
    monkeypatch.setattr(retention, "is_db_available", lambda: False)
    assert await retention.purge_expired() == {}


async def test_db_error_never_raises(conn, monkeypatch):
    monkeypatch.setattr(settings, "retention_chat_days", 30)
    conn.execute.side_effect = RuntimeError("boom")
    await retention.purge_expired()


async def test_purge_fails_orphaned_jobs_first(conn, monkeypatch):
    # A job stuck in 'running' has no finished_at: without the sweep no period ever applies.
    order = []

    async def sweep():
        order.append("sweep")
        return 1

    async def execute(sql, days):
        order.append("purge")
        return "UPDATE 0"

    monkeypatch.setattr("grok_mcp.tools.jobs.fail_orphaned_jobs", sweep)
    conn.execute = execute
    monkeypatch.setattr(settings, "retention_job_content_days", 7)
    await retention.purge_expired()
    assert order[0] == "sweep" and "purge" in order
