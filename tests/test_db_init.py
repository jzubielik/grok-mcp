# tests/test_db_init.py
"""Process-wide DB init: one pool per process, never nulled by a later failure.

FastMCP runs the server lifespan for EVERY streamable-HTTP session, so init_db must
create the schema once and a single failure must never null the shared global pool
under other live sessions.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def db_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x/y")
    import grok_mcp.db as db

    db._pool = None
    yield db
    db._pool = None


def _pool_mock():
    pool = MagicMock()
    pool.close = AsyncMock()
    return pool


@pytest.mark.asyncio
async def test_init_db_is_idempotent(db_env):
    """A second session must reuse the existing pool, not build another one."""
    db_env._pool = _pool_mock()
    with patch("asyncpg.create_pool", AsyncMock()) as create:
        await db_env.init_db()
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_init_creates_single_pool(db_env):
    """Many sessions starting at once must not each build a pool and race schema creation."""
    created = []

    async def slow_create_pool(*args, **kwargs):
        await asyncio.sleep(0.02)
        pool = _pool_mock()
        created.append(pool)
        return pool

    with (
        patch("asyncpg.create_pool", side_effect=slow_create_pool),
        patch.object(db_env, "_create_schema", AsyncMock()) as create_schema,
    ):
        await asyncio.gather(*[db_env.init_db() for _ in range(5)])

    assert len(created) == 1
    assert create_schema.await_count == 1


@pytest.mark.asyncio
async def test_schema_failure_does_not_publish_pool(db_env):
    """A failed schema creation must leave no half-initialised global pool behind."""
    pool = _pool_mock()
    with (
        patch("asyncpg.create_pool", AsyncMock(return_value=pool)),
        patch.object(db_env, "_create_schema", AsyncMock(side_effect=RuntimeError("boom"))),
        pytest.raises(RuntimeError),
    ):
        await db_env.init_db()

    assert db_env._pool is None
    pool.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_schema_failure_never_clobbers_a_working_pool(db_env):
    """The failure path must not disable the DB for sessions that already work."""
    working = _pool_mock()
    db_env._pool = working
    with patch("asyncpg.create_pool", AsyncMock(side_effect=RuntimeError("boom"))):
        await db_env.init_db()  # idempotent: returns without touching the pool
    assert db_env._pool is working
    assert db_env.is_db_available()


# ---------------------------------------------------------------------------
# Lifespan must not tear down shared state while other sessions are live
# ---------------------------------------------------------------------------


@pytest.fixture()
def lifespan_env():
    import grok_mcp.server as server

    with (
        patch.object(server, "init_clients") as init_clients,
        patch.object(server, "close_clients") as close_clients,
        patch.object(server, "init_db", AsyncMock()),
        patch.object(server, "close_db", AsyncMock()) as close_db,
        patch("grok_mcp.tools.jobs.fail_orphaned_jobs", AsyncMock()) as orphans,
        patch("grok_mcp.tools.context_cache.bootstrap_project_discovery", AsyncMock()),
        patch("grok_mcp.tools.models.refresh_tool_descriptions", AsyncMock()),
        patch("grok_mcp.retention.start_purge_loop"),
        patch("grok_mcp.retention.stop_purge_loop", AsyncMock()),
    ):
        yield server, init_clients, close_clients, close_db, orphans


@pytest.mark.asyncio
async def test_stdio_lifespan_starts_and_stops_the_process_state(lifespan_env):
    server, init_clients, close_clients, close_db, _ = lifespan_env
    server._process_lifespan = False
    async with server.server_lifespan(server.mcp):
        assert init_clients.call_count == 1
    close_db.assert_awaited_once()
    close_clients.assert_called_once()


@pytest.mark.asyncio
async def test_http_sessions_never_touch_process_state(lifespan_env):
    """Sessions ending must not close the DB (background jobs still need it), and a new
    session must not re-run startup (it would mark running jobs as orphaned)."""
    server, init_clients, close_clients, close_db, orphans = lifespan_env
    server._process_lifespan = True
    await server._startup()
    for _ in range(2):  # one client disconnects, another connects
        async with server.server_lifespan(server.mcp):
            pass
    assert init_clients.call_count == 1
    assert orphans.await_count == 1
    close_db.assert_not_awaited()
    close_clients.assert_not_called()
    await server._shutdown()
    close_db.assert_awaited_once()


@pytest.mark.asyncio
async def test_init_db_small_lazy_pool_with_separate_password(db_env, monkeypatch):
    monkeypatch.setenv("PGPASSWORD", "any:/@chars")
    with (
        patch("asyncpg.create_pool", AsyncMock(return_value=_pool_mock())) as create,
        patch.object(db_env, "_create_schema", AsyncMock()),
    ):
        await db_env.init_db()
    kwargs = create.call_args.kwargs
    assert (kwargs["min_size"], kwargs["max_size"]) == (0, 3)
    assert kwargs["password"] == "any:/@chars"


@pytest.mark.asyncio
async def test_init_db_without_pgpassword_passes_none(db_env, monkeypatch):
    monkeypatch.delenv("PGPASSWORD", raising=False)
    with (
        patch("asyncpg.create_pool", AsyncMock(return_value=_pool_mock())) as create,
        patch.object(db_env, "_create_schema", AsyncMock()),
    ):
        await db_env.init_db()
    assert "password" not in create.call_args.kwargs


@pytest.mark.asyncio
async def test_schema_creation_takes_advisory_lock_first(db_env):
    conn = MagicMock()
    conn.execute = AsyncMock()
    conn.transaction.return_value.__aenter__ = AsyncMock()
    conn.transaction.return_value.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    await db_env._create_schema(pool)
    first = conn.execute.call_args_list[0].args
    assert first[0] == "SELECT pg_advisory_xact_lock($1)"
    assert isinstance(first[1], int) and first[1] < 2**63


@pytest.mark.asyncio
async def test_init_db_retries_the_first_real_connection(db_env, monkeypatch):
    """A failure on the schema creation's first acquire is retried too."""
    import asyncpg

    pools = [_pool_mock(), _pool_mock()]
    for p in pools:
        p.acquire = MagicMock()
    conn = MagicMock()
    conn.execute = AsyncMock()
    conn.transaction.return_value.__aenter__ = AsyncMock()
    conn.transaction.return_value.__aexit__ = AsyncMock(return_value=False)
    pools[0].acquire.return_value.__aenter__ = AsyncMock(
        side_effect=asyncpg.exceptions.TooManyConnectionsError("full")
    )
    pools[1].acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pools[1].acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(db_env.asyncio, "sleep", AsyncMock())
    with patch("asyncpg.create_pool", AsyncMock(side_effect=pools)) as create:
        await db_env.init_db()
    assert create.await_count == 2
    pools[0].close.assert_awaited_once()
    assert db_env._pool is pools[1]


@pytest.mark.asyncio
async def test_init_db_retries_when_create_pool_cannot_connect(db_env, monkeypatch):
    """init_size=1 connects at creation: a database still starting fails there and is retried."""

    good = _pool_mock()
    conn = MagicMock()
    conn.execute = AsyncMock()
    conn.transaction.return_value.__aenter__ = AsyncMock()
    conn.transaction.return_value.__aexit__ = AsyncMock(return_value=False)
    good.acquire = MagicMock()
    good.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    good.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(db_env.asyncio, "sleep", AsyncMock())
    with patch(
        "asyncpg.create_pool",
        AsyncMock(side_effect=[ConnectionRefusedError("starting"), good]),
    ) as create:
        await db_env.init_db()
    assert create.await_count == 2
    assert create.await_args.kwargs["init_size"] == 1
    assert create.await_args.kwargs["min_size"] == 0
    assert db_env._pool is good


@pytest.mark.asyncio
async def test_init_db_retries_while_the_database_is_starting(db_env, monkeypatch):
    """57P03 (the database system is starting up) is transient and must be retried."""
    import asyncpg

    good = _pool_mock()
    conn = MagicMock()
    conn.execute = AsyncMock()
    conn.transaction.return_value.__aenter__ = AsyncMock()
    conn.transaction.return_value.__aexit__ = AsyncMock(return_value=False)
    good.acquire = MagicMock()
    good.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    good.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(db_env.asyncio, "sleep", AsyncMock())
    starting = asyncpg.exceptions.CannotConnectNowError("the database system is starting up")
    with patch("asyncpg.create_pool", AsyncMock(side_effect=[starting, good])) as create:
        await db_env.init_db()
    assert create.await_count == 2
    assert db_env._pool is good
