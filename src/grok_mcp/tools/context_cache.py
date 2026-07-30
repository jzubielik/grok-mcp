"""Conversation history and project discovery — PostgreSQL-backed (DATABASE_URL)."""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import UTC, datetime

from ..auth import scrubbed_env
from ..config import parse_duration
from ..db import get_pool, is_db_available
from ..logjson import redact
from ..server import mcp

logger = logging.getLogger("grok_mcp.tools.context_cache")

_MAX_CONVERSATION_CHARS = 200
_MAX_LIST_CONVERSATIONS = 100


def check_conversation(conversation: str | None) -> str | None:
    """Validate a caller-chosen conversation name; None means a stateless call.

    Surrounding whitespace is dropped, so "p/t" and "p/t " are one conversation; otherwise
    the name is stored as given.
    """
    if conversation is None:
        return None
    conversation = conversation.strip()
    if not conversation:
        raise ValueError("conversation must not be empty — omit it for a one-off call")
    if len(conversation) > _MAX_CONVERSATION_CHARS:
        raise ValueError(f"conversation is longer than {_MAX_CONVERSATION_CHARS} characters")
    # Cc: C0/C1 controls; Cf: invisible format characters (zero-width, bidi overrides); Zl/Zp:
    # line and paragraph separators — all would make different names look alike when listed.
    if any(unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp") for c in conversation):
        raise ValueError("conversation must not contain control characters")
    return conversation


# Project path resolved at bootstrap (os.getcwd() when the server started).
# Used by get_server_project_path: the project this server runs in, never a guess.
_BOOTSTRAP_PROJECT_PATH: str | None = None


async def get_recent_history(conversation_id: str, limit: int = 10) -> list[dict]:
    """Return the last `limit` messages of a conversation, oldest first.

    Model-facing history accessor (agent() history tools). Never raises; [] without DB.
    """
    if not is_db_available():
        return []
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT role, content FROM chat_history "
                "WHERE conversation_id = $1 AND archived = FALSE "
                "ORDER BY sequence DESC LIMIT $2",
                conversation_id,
                limit,
            )
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_recent_history failed for %s: %s", conversation_id, exc)
        return []


def _like_escape(text: str) -> str:
    """Make `%`, `_` and `\\` match themselves in an ILIKE pattern (`\\` is its default escape):
    `_` in an identifier such as save_turn would otherwise match any character."""
    return re.sub(r"([\\%_])", r"\\\1", text)


async def search_history(conversation_id: str, query: str, limit: int = 5) -> list[dict]:
    """Case-insensitive substring search over a conversation's history, newest first.

    Model-facing history accessor (agent() history tools). Never raises; [] without DB.
    """
    if not is_db_available():
        return []
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT role, content FROM chat_history "
                "WHERE conversation_id = $1 AND archived = FALSE AND content ILIKE $2 "
                "ORDER BY sequence DESC LIMIT $3",
                conversation_id,
                f"%{_like_escape(query)}%",
                limit,
            )
        return [{"role": r["role"], "content": r["content"]} for r in rows]
    except Exception as exc:  # noqa: BLE001
        logger.warning("search_history failed for %s: %s", conversation_id, exc)
        return []


_INSERT_MESSAGE = """
    INSERT INTO chat_history
        (conversation_id, sequence, role, content, model,
         file_paths, git_hash, prompt_tokens, completion_tokens, reasoning_tokens, created_at)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
"""


async def save_turn(
    conversation_id: str,
    user_prompt: str,
    answer: str,
    model: str,
    file_paths: list[str] | None = None,
    git_hash: str | None = None,
    usage: dict | None = None,
) -> None:
    """Store one user/assistant exchange as the next two messages of the conversation.

    One transaction under an advisory lock on the conversation: several clients may work on
    the same named conversation, and two concurrent turns must not compute the same sequence
    (one of them would be lost). Credential formats are masked before storage. Raises on errors.
    """
    usage = usage or {}
    async with get_pool().acquire() as conn, conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", conversation_id)
        seq = await conn.fetchval(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM chat_history WHERE conversation_id = $1",
            conversation_id,
        )
        await conn.executemany(
            _INSERT_MESSAGE,
            [
                (
                    conversation_id,
                    seq,
                    "user",
                    redact(user_prompt, key_value=False),
                    model,
                    file_paths,
                    git_hash,
                    None,
                    None,
                    None,
                ),
                (
                    conversation_id,
                    seq + 1,
                    "assistant",
                    redact(answer, key_value=False),
                    model,
                    None,
                    git_hash,
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                    usage.get("reasoning_tokens"),
                ),
            ],
        )


async def get_server_project_path() -> str | None:
    """Return the server's OWN project path (CWD at startup), if it is a known project.

    Returns None when the server does not run inside a project — e.g. in a container
    where CWD is /app. Callers must then go without project context instead of guessing a
    project: over HTTP one server instance serves many projects, and picking the
    "most recently updated" row attributes calls to an unrelated project.
    """
    path = _BOOTSTRAP_PROJECT_PATH
    if not path or not is_db_available():
        return None
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT project_path FROM project_discovery WHERE project_path = $1", path
            )
        return path if row else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to resolve server project path: %s", exc)
        return None


async def get_active_project_context(project_path: str | None) -> str | None:
    """Return a compact project context string for the given project.

    Intended to be injected into Grok system prompts automatically.
    Returns None when no project_path is given (the server is not bound to a project),
    the DB is unavailable, or the row has no usable fields.
    Never raises — exceptions are logged and swallowed.
    """
    if project_path is None or not is_db_available():
        return None
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT language, framework FROM project_discovery WHERE project_path = $1",
                project_path,
            )
        if row is None:
            return None

        language = row["language"]
        framework = row["framework"]

        if not language and not framework:
            return None

        lines = ["[Project Context]"]
        if language:
            lines.append(f"Language: {language}")
        if framework:
            lines.append(f"Framework: {framework}")

        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load active project context: %s", exc)
        return None


async def refresh_project_git_hash(project_path: str, git_hash: str) -> None:
    """Lazily keep project_discovery.git_hash current without a server restart.

    Called on every agent() call after the current git HEAD is resolved. Also refreshes
    updated_at, so a project still in use is never removed by the retention purge
    (GROK_RETENTION_PROJECTS_DAYS) — which would silently change its conversation id.
    No-op when DB is unavailable or the project_path is not in project_discovery.
    Never raises — exceptions are logged and swallowed.
    """
    if not is_db_available():
        return
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE project_discovery
                SET git_hash = $2, updated_at = NOW()
                WHERE project_path = $1
                """,
                project_path,
                git_hash,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("refresh_project_git_hash failed: %s", exc)


def _detect_python_framework(data: dict) -> str | None:
    """Infer framework from parsed pyproject.toml data."""
    tools = data.get("tool", {})
    if "fastmcp" in tools or "mcp" in tools:
        return "FastMCP"
    if "fastapi" in tools:
        return "FastAPI"
    deps: set[str] = set()
    for dep in data.get("project", {}).get("dependencies", []):
        name = dep.split("[")[0].split(">=")[0].split("==")[0].strip().lower()
        deps.add(name)
    for name, label in [
        ("fastmcp", "FastMCP"),
        ("mcp", "FastMCP"),
        ("fastapi", "FastAPI"),
        ("django", "Django"),
        ("flask", "Flask"),
        ("aiohttp", "aiohttp"),
    ]:
        if name in deps:
            return label
    return None


def _detect_project_info(cwd: str) -> tuple[str | None, str | None]:
    """Scan well-known config files and return (language, framework).

    Only reads the first matching config file found.  Returns (None, None)
    when the project type is unrecognised.  Never raises.
    """
    import json as _json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    root = Path(cwd)

    # Python — pyproject.toml
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib  # noqa: PLC0415 — stdlib since Python 3.11

            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            return "Python", _detect_python_framework(data)
        except Exception:  # noqa: BLE001
            return "Python", None

    # JavaScript / TypeScript — package.json
    pkg = root / "package.json"
    if pkg.exists():
        try:
            data = _json.loads(pkg.read_text(encoding="utf-8"))
            all_deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
            language = "TypeScript" if "typescript" in all_deps else "JavaScript"
            framework = None
            for dep, label in [
                ("next", "Next.js"),
                ("react", "React"),
                ("vue", "Vue"),
                ("@angular/core", "Angular"),
                ("express", "Express"),
                ("fastify", "Fastify"),
            ]:
                if dep in all_deps:
                    framework = label
                    break
            return language, framework
        except Exception:  # noqa: BLE001
            return "JavaScript", None

    # Go — go.mod
    if (root / "go.mod").exists():
        return "Go", None

    # Rust — Cargo.toml
    if (root / "Cargo.toml").exists():
        return "Rust", None

    return None, None


async def bootstrap_project_discovery() -> None:
    """Upsert a project_discovery row for the current working directory at startup.

    Auto-detects language and framework from project config files (pyproject.toml,
    package.json, go.mod, Cargo.toml).  Detected values are only written when the
    existing row has language = NULL — a later discovery run with
    richer data always take precedence.

    Skips gracefully when DB is unavailable or the directory is not a git repo.
    """
    if not is_db_available():
        return
    import asyncio  # noqa: PLC0415
    import os  # noqa: PLC0415
    from asyncio.subprocess import DEVNULL, PIPE  # noqa: PLC0415

    global _BOOTSTRAP_PROJECT_PATH
    cwd = os.getcwd()
    _BOOTSTRAP_PROJECT_PATH = cwd
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "rev-parse",
            "HEAD",
            cwd=cwd,
            stdin=DEVNULL,
            env=scrubbed_env(),
            stdout=PIPE,
            stderr=DEVNULL,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return
        git_hash = stdout.decode("utf-8", errors="replace").strip()
    except Exception as exc:  # noqa: BLE001
        logger.debug("bootstrap_project_discovery: not a git repo (%s)", exc)
        return

    language, framework = _detect_project_info(cwd)

    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO project_discovery
                    (project_path, git_hash, language, framework, updated_at)
                VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (project_path) DO UPDATE SET
                    git_hash   = EXCLUDED.git_hash,
                    language   = COALESCE(project_discovery.language, EXCLUDED.language),
                    framework  = COALESCE(project_discovery.framework, EXCLUDED.framework),
                    updated_at = NOW()
                """,
                cwd,
                git_hash,
                language,
                framework,
            )
        logger.info(
            "bootstrap_project_discovery: registered %s @ %s (lang=%s fw=%s)",
            cwd,
            git_hash[:7],
            language,
            framework,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("bootstrap_project_discovery failed: %s", exc)


@mcp.tool(
    description=(
        "Prune messages of a conversation (the name passed to agent(conversation=...)). "
        "soft (default, hard=False): marks messages as archived — they stay in DB for audit but "
        "are excluded from future context loads. "
        "hard=True: permanently deletes messages. "
        "before: duration ('24h', '7d', '30m') or ISO 8601 UTC timestamp — prune messages older "
        "than this. Omit to prune ALL messages in the conversation. "
        "Returns a summary of pruned and remaining message counts."
    )
)
async def prune_chat_context(
    conversation: str,
    hard: bool = False,
    before: str | None = None,
) -> dict:
    conv_id = check_conversation(conversation)
    if not is_db_available():
        raise RuntimeError("conversation history not available (database unreachable)")

    # Resolve cutoff datetime
    cutoff: datetime | None = None
    if before is not None:
        try:
            cutoff = datetime.now(UTC) - parse_duration(before)
        except ValueError:
            try:
                cutoff = datetime.fromisoformat(before.replace("Z", "+00:00"))
                if cutoff.tzinfo is None:  # documented as UTC; a naive value means UTC too
                    cutoff = cutoff.replace(tzinfo=UTC)
            except (ValueError, AttributeError):
                raise ValueError(
                    f"invalid 'before' value '{before}' — expected format like "
                    "'24h', '7d', or ISO 8601 timestamp"
                ) from None

    pool = get_pool()

    try:
        async with pool.acquire() as conn:
            if hard:
                tag = await conn.execute(
                    "DELETE FROM chat_history "
                    "WHERE conversation_id = $1 "
                    "AND ($2::timestamptz IS NULL OR created_at < $2)",
                    conv_id,
                    cutoff,
                )
                verb = "hard-deleted"
            else:
                tag = await conn.execute(
                    "UPDATE chat_history SET archived = TRUE "
                    "WHERE conversation_id = $1 "
                    "AND ($2::timestamptz IS NULL OR created_at < $2) "
                    "AND archived = FALSE",
                    conv_id,
                    cutoff,
                )
                verb = "soft-pruned"

            # asyncpg returns DML command tags like "DELETE 10" or "UPDATE 5"
            count = int(tag.split()[-1])

            remaining = await conn.fetchval(
                "SELECT COUNT(*) FROM chat_history WHERE conversation_id = $1 AND archived = FALSE",
                conv_id,
            )

    except Exception as exc:
        logger.warning("prune_chat_context failed: %s", exc)
        raise RuntimeError("prune_chat_context failed (database error)") from exc

    return {
        "conversation": conv_id,
        "pruned": count,
        "mode": "hard" if verb == "hard-deleted" else "soft",
        "before": cutoff.isoformat() if cutoff else "all",
        "remaining": remaining,
    }


@mcp.tool(
    description=(
        "List stored conversations (most recently active first) with their message count and "
        "last activity; limit 1-100. Use to find a conversation name to continue or prune."
    )
)
async def list_conversations(limit: int = 20) -> dict:
    limit = max(1, min(int(limit), _MAX_LIST_CONVERSATIONS))
    if not is_db_available():
        raise RuntimeError("conversation history not available (database unreachable)")
    async with get_pool().acquire() as conn:
        rows = await conn.fetch(
            "SELECT conversation_id, COUNT(*) AS messages, MAX(created_at) AS last_activity "
            "FROM chat_history WHERE archived = FALSE GROUP BY conversation_id "
            "ORDER BY last_activity DESC LIMIT $1",
            limit,
        )
    return {
        "conversations": [
            {
                "conversation": r["conversation_id"],
                "messages": r["messages"],
                "last_activity": r["last_activity"].isoformat() if r["last_activity"] else None,
            }
            for r in rows
        ]
    }
