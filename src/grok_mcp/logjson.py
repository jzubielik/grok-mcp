"""JSON Lines logging: one JSON object per line on stderr, for every logger.

Structured fields go in `extra={"fields": {...}}`; the message becomes `event`. A job id set
via `current_job_id` is added to every line logged while that job runs, so a background
run's lines can be grouped with `jq 'select(.job_id=="...")'`.

Content policy (GROK_LOG_CONTENT): "metadata" (default) logs prompts, answers and other free
text as {chars, hmac} only, and tracebacks without the exception message; "full" logs
everything. Secrets (API keys, bearer tokens, passwords, private keys) are masked on every
line in both modes.
"""

from __future__ import annotations

import contextvars
import hashlib
import hmac
import json
import logging
import re
import secrets
import sys
import traceback
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from .config import settings

current_job_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_job_id", default=None
)

# A foreground run becomes a job only after it started, and a task's context is fixed when it
# is created — so it carries a mutable holder that receives the job id at handoff.
_job_ref: contextvars.ContextVar[dict | None] = contextvars.ContextVar("_job_ref", default=None)


def bind_job_ref() -> dict:
    """Give tasks created from here on a holder that a later handoff fills with the job id."""
    ref: dict = {}
    _job_ref.set(ref)
    return ref


# Tool arguments / result keys that carry bulk data (file bodies), logged as sizes only.
_DATA_KEYS = frozenset({"content", "files"})

# String fields logged verbatim in metadata mode: identifiers, settings and paths, never free
# text. Any other string becomes {chars, hmac}; numbers, booleans and None are always kept.
_META_KEYS = frozenset(
    {
        "tool",
        "tools",
        "model",
        "reasoning_effort",
        "status",
        "stopped",
        "job_id",
        "path",
        "paths",
        "file_paths",
        "audio_path",
        "image_paths",
        "output_path",
        "aspect_ratio",
        "resolution",
        "voice",
        "language",
        "format",
        "background",
        "dry_run",
        "backup",
        "before",
        "mode",
        "conversation",
    }
)

# The key=value rule must stay last: redact(key_value=False) drops it.
_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bxai-[A-Za-z0-9]{16,}"), "xai-***"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), "sk-***"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}|\bgithub_pat_\w{20,}"), "gh***"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AKIA***"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer ***"),
    # A password may contain "/" and "@": mask up to the LAST "@" that is followed by a host.
    # The repeats are bounded so a long run of "x://u:" without spaces stays linear.
    (
        re.compile(r"(\b\w{1,32}://[^:/\s@]{1,256}:)\S{0,512}@(?=[^@\s/]+(?:[/:?#\s]|$))"),
        r"\1***@",
    ),
    (
        re.compile(
            r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key)"
            r"(\"?\s*[=:]\s*\"?)[^\s\"',;&]{4,}"
        ),
        r"\1\2***",
    ),
]


_KEY_BEGIN = re.compile(r"-----BEGIN [A-Z ]{0,40}PRIVATE KEY-----")
# Whole END marker: its tail alone ("PRIVATE KEY-----") also ends every BEGIN marker.
_KEY_END = re.compile(r"-----END [A-Z ]{0,40}PRIVATE KEY-----")
# A key cut before its END is masked line by line while each line still looks like part of
# it: blank, a PEM header (encrypted keys), or a base64 run to the end of the line behind a
# short prefix (`12: ` of a numbered read, `key.pem:3:` of grep -n, `> ` of a quote). Lines
# break at real newlines or escaped ones, however deeply (`\n`, `\\n` of JSON in JSON). The
# first other line ends the key, so prose after a mere mention of the marker survives.
_LINE_BREAK = re.compile(r"\r\n|[\r\n]|\\+[rn]")
_KEY_LINE = re.compile(
    r"[ \t]*(?:(?:Proc-Type|DEK-Info|Content-Domain|Comment|Key-Info|MIC-Info"
    r"|Originator-ID-[A-Za-z]+|Recipient-ID-[A-Za-z]+):[^\r\n]*"
    r"|[^\r\n]{0,16}?[A-Za-z0-9+/=]{16,}[ \t]*)?"
)


def _key_body_end(text: str, start: int, limit: int) -> int:
    """Where the body of a key cut before its END line stops: past the last line, from `start`
    up to `limit`, that still looks like part of a key. Each line is looked at once."""
    pos = last = start
    while pos <= limit:
        brk = _LINE_BREAK.search(text, pos, limit)
        line_end = brk.start() if brk else limit
        if not _KEY_LINE.fullmatch(text, pos, line_end):
            break
        if text[pos:line_end].strip():
            last = line_end
        if brk is None:
            break
        pos = brk.end()
    return last


def _redact_private_keys(text: str) -> str:
    """Mask PEM private keys in one left-to-right pass (a lazy `BEGIN.*?END` regex is
    quadratic on input full of BEGIN markers without an END). A key without its own END line
    (cut output, or the next BEGIN comes first) is masked through the lines that still look like
    key material only — see _KEY_LINE."""
    out, pos = [], 0
    # The next END marker, found once and reused while still ahead: searching again from every
    # BEGIN would rescan the same text (quadratic on BEGIN after BEGIN). None is final — no END
    # after one position means none after any later one either.
    end: re.Match[str] | None = _KEY_END.search(text)
    m = _KEY_BEGIN.search(text)
    while m is not None:
        out.append(text[pos : m.start()])
        out.append("-----PRIVATE KEY REDACTED-----")
        if end is not None and end.start() < m.end():
            end = _KEY_END.search(text, m.end())
        nxt = _KEY_BEGIN.search(text, m.end())
        if end is not None and (nxt is None or end.start() < nxt.start()):
            pos = end.end()
        else:
            pos = _key_body_end(text, m.end(), nxt.start() if nxt else len(text))
        m = nxt
    out.append(text[pos:])
    return "".join(out)


def redact(text: str, key_value: bool = True) -> str:
    """Mask credentials in free text (keys, tokens, passwords, private keys, URL userinfo).

    key_value=False skips the `password=...` rule, which can also hit ordinary code; stored
    conversation text uses that, so only unambiguous credential formats are masked there.
    """
    if "PRIVATE KEY" in text:
        text = _redact_private_keys(text)
    patterns = _SECRET_PATTERNS if key_value else _SECRET_PATTERNS[:-1]
    for pattern, repl in patterns:
        text = pattern.sub(repl, text)
    return text


def redact_value(value: Any) -> Any:
    """Apply `redact` to every string inside dicts / lists."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_value(v) for v in value]
    return value


# Per-process key: digests group identical text within one server run but cannot be reversed
# by hashing guesses (a short prompt, a shell command, an e-mail address) or linked across
# restarts — a plain sha256 of low-entropy text would be both.
_DIGEST_KEY = secrets.token_bytes(32)


def _digest(text: str) -> dict[str, Any]:
    mac = hmac.new(_DIGEST_KEY, text.encode(), hashlib.sha256).hexdigest()[:16]
    return {"chars": len(text), "hmac": mac}


def summarize(value: Any, key: str | None = None) -> Any:
    """Metadata view of a value: free-text strings become {chars, hmac}, URLs their host."""
    if isinstance(value, dict):
        return {k: summarize(v, k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [summarize(v, key) for v in value]
    if isinstance(value, str):
        if key == "url":
            return urlsplit(value).hostname or _digest(value)
        return value if key in _META_KEYS else _digest(value)
    return value


def full_content() -> bool:
    return settings.log_content == "full"


def loggable(value: Any) -> Any:
    """What a log line may hold of a tool's arguments or result, per GROK_LOG_CONTENT."""
    value = strip_data(value)
    return value if full_content() else summarize(value)


def error_text(exc: BaseException) -> str:
    """Exception for a log line: the type always, the message only in full mode."""
    return f"{type(exc).__name__}: {exc}" if full_content() else type(exc).__name__


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": self._message(record),
        }
        job_id = current_job_id.get() or (_job_ref.get() or {}).get("id")
        if job_id:
            line["job_id"] = job_id
        line.update(getattr(record, "fields", None) or {})
        if record.exc_info:
            line["traceback"] = self._traceback(record.exc_info)
        return json.dumps(redact_value(line), default=str, ensure_ascii=False)

    @staticmethod
    def _message(record: logging.LogRecord) -> str:
        # `logger.warning("... %s", exc)` would put the exception text — which may quote a
        # prompt or a stored value — into the event; metadata mode keeps only its type.
        # Third-party loggers (the MCP SDK, uvicorn) pre-format exception text into the message
        # — e.g. a validation error quoting the request body. Keep only the event.
        if (
            not full_content()
            and not record.name.startswith("grok_mcp")
            and record.levelno >= logging.WARNING
        ):
            return f"{record.name}: {record.levelname.lower()} (message hidden in metadata mode)"
        if not full_content() and isinstance(record.args, tuple) and record.args:
            args = tuple(
                type(a).__name__ if isinstance(a, BaseException) else a for a in record.args
            )
            return record.msg % args if isinstance(record.msg, str) else str(record.msg)
        return record.getMessage()

    def _traceback(self, exc_info: Any) -> str:
        if full_content():
            return self.formatException(exc_info)
        # Frames only: the message line may quote a prompt or a model answer.
        etype, _, tb = exc_info
        return "".join(traceback.format_tb(tb)) + (etype.__name__ if etype else "")


def strip_data(value: Any) -> Any:
    """Replace bulk-data entries (file contents) with their size; keep everything else whole."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in _DATA_KEYS and isinstance(item, str):
                out[f"{key}_chars"] = len(item)
            elif key in _DATA_KEYS and isinstance(item, dict):
                out[f"{key}_chars"] = {
                    k: len(v) if isinstance(v, str) else v for k, v in item.items()
                }
            else:
                out[key] = strip_data(item)
        return out
    if isinstance(value, list):
        return [strip_data(v) for v in value]
    return value


def _route_to_root(*names: str) -> None:
    for name in names:
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True


# Per-request chatter that says nothing our tool_call/tool_result lines don't: FastMCP's
# "Processing request of type …", session-transport churn, every POST /mcp, every HTTP call
# to xAI. Warnings and errors from them still get through.
_NOISY_LOGGERS = (
    "mcp.server.lowlevel.server",
    "mcp.server.streamable_http_manager",
    "mcp.server.streamable_http",
    "uvicorn.access",
    "httpx",
)


def _quiet_noise() -> None:
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def configure_json_logging(level: int = logging.INFO) -> None:
    """Send every logger (ours, FastMCP, uvicorn, httpx) through one JSON handler."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    _route_to_root("uvicorn", "uvicorn.error", "uvicorn.access")
    _quiet_noise()

    # uvicorn re-applies its own handlers when a server starts — reroute them right after.
    import uvicorn.config  # noqa: PLC0415

    original = uvicorn.config.Config.configure_logging
    if not getattr(original, "_json_wrapped", False):

        def configure_logging(self: Any) -> None:
            original(self)
            _route_to_root("uvicorn", "uvicorn.error", "uvicorn.access")
            _quiet_noise()

        configure_logging._json_wrapped = True  # type: ignore[attr-defined]
        uvicorn.config.Config.configure_logging = configure_logging
