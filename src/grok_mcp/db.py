"""PostgreSQL connection pool and schema for the context cache."""

from __future__ import annotations

import asyncio
import logging

import asyncpg

from .auth import get_secret

logger = logging.getLogger("grok_mcp.db")

_pool: asyncpg.Pool | None = None

# Serialises concurrent init_db() calls — every HTTP session runs the lifespan.
_init_lock = asyncio.Lock()

# Idempotent schema — CREATE TABLE IF NOT EXISTS + CREATE INDEX IF NOT EXISTS, safe on every start
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS project_discovery (
    project_path TEXT PRIMARY KEY,
    git_hash     TEXT NOT NULL,
    language     TEXT,
    framework    TEXT,
    updated_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_history (
    conversation_id   TEXT    NOT NULL,
    sequence          INTEGER NOT NULL,
    role              TEXT    NOT NULL,
    content           TEXT    NOT NULL,
    model             TEXT    NOT NULL,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    file_paths        TEXT[],
    git_hash          TEXT,
    archived          BOOLEAN DEFAULT FALSE,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    reasoning_tokens  INTEGER,
    PRIMARY KEY (conversation_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_chat_history_archived
    ON chat_history (conversation_id, archived, created_at);
-- The retention purge filters on created_at.
CREATE INDEX IF NOT EXISTS idx_chat_history_created ON chat_history (created_at);

CREATE TABLE IF NOT EXISTS jobs (
    job_id            TEXT PRIMARY KEY,
    tool              TEXT NOT NULL,
    status            TEXT NOT NULL,
    result            TEXT,
    error             TEXT,
    model             TEXT,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    finished_at       TIMESTAMPTZ,
    heartbeat_at      TIMESTAMPTZ,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    reasoning_tokens  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_finished ON jobs (finished_at);
"""


async def init_db() -> None:
    """Create the connection pool and the schema — once per process.

    The database is required: conversation history and background jobs live there, and
    without jobs a long run cannot outlive the MCP client's timeout. A missing DATABASE_URL
    stops the server at startup instead of letting it run degraded. No-op if a pool already
    exists. Retries up to 3 times on transient connection errors.
    """
    global _pool  # noqa: PLW0603
    db_url = get_secret("DATABASE_URL")
    if not db_url:
        raise RuntimeError(
            "DATABASE_URL is required (PostgreSQL for conversation history and background "
            "jobs), e.g. postgresql://grok:<password>@127.0.0.1:5432/context_cache"
        )
    if _pool is not None:
        return
    async with _init_lock:
        if _pool is not None:  # another session won the race while we waited
            return
        logger.info("Connecting to PostgreSQL context cache…")
        backoff_delays = [1, 2, 4]
        for attempt, delay in enumerate(backoff_delays, 1):
            # Small and lazy: every stdio session is its own process with its own pool, all
            # sharing one Postgres (max_connections) — min_size=0 keeps no idle connections.
            # init_size=1 (asyncpg >= 0.32) connects once at creation, so a database that is
            # still starting fails here, inside the retry. The pool is only published once
            # the schema exists, so a failure never leaves a half-initialised pool visible to others.
            pool: asyncpg.Pool | None = None
            try:
                pool = await asyncpg.create_pool(
                    db_url, init_size=1, min_size=0, max_size=3, **_password_kwargs()
                )
                await _create_schema(pool)
                break
            except (
                asyncpg.exceptions.TooManyConnectionsError,
                # 57P03: the server is starting up or recovering — transient by definition.
                asyncpg.exceptions.CannotConnectNowError,
                # SQLSTATE class 08 (connection exceptions), raised before a session exists.
                asyncpg.exceptions.PostgresConnectionError,
                OSError,
                ConnectionError,
            ) as exc:
                if pool is not None:
                    await pool.close()
                if attempt < len(backoff_delays):
                    logger.warning(
                        "DB connection attempt %d/3 failed: %s — retrying in %ds",
                        attempt,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("DB connection failed after 3 attempts: %s", exc)
                    raise
            except Exception:
                if pool is not None:
                    await pool.close()
                raise
        _pool = pool
        logger.info("Context cache database ready")


async def close_db() -> None:
    """Close the connection pool if it exists."""
    global _pool  # noqa: PLW0603
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Context cache database connection closed")


def get_pool() -> asyncpg.Pool:
    """Return the active pool. Raises RuntimeError if DB not initialised."""
    if _pool is None:
        raise RuntimeError(
            "Context cache database not available. Set DATABASE_URL and restart grok-mcp."
        )
    return _pool


def is_db_available() -> bool:
    """Return True if the DB pool is initialised."""
    return _pool is not None


def _password_kwargs() -> dict[str, str]:
    """PGPASSWORD travels apart from DATABASE_URL so any character works in it (no URL
    escaping); asyncpg merges it with the DSN."""
    password = get_secret("PGPASSWORD")
    return {"password": password} if password else {}


# Arbitrary constant naming the schema-creation lock across every process.
_SCHEMA_LOCK_ID = 7_264_011_350_118_420_001


async def _create_schema(pool: asyncpg.Pool | None = None) -> None:
    """Execute the idempotent schema SQL against `pool` (defaults to the global one)."""
    if pool is None:
        pool = get_pool()
    async with pool.acquire() as conn, conn.transaction():
        # Several processes starting at once against a fresh database would run the DDL
        # concurrently and trip over each other; the lock serialises them until commit.
        await conn.execute("SELECT pg_advisory_xact_lock($1)", _SCHEMA_LOCK_ID)
        await conn.execute(_SCHEMA_SQL)
