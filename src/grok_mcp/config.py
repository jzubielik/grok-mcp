"""Server settings — GROK_* environment variables (see .env.example)."""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DURATION_RE = re.compile(r"^(\d+)([hdm])$")


def parse_duration(s: str) -> timedelta:
    """Parse a duration string like '24h', '7d', '30m' into a timedelta.

    Raises ValueError for unrecognized formats.
    """
    m = _DURATION_RE.match(s.strip())
    if not m:
        raise ValueError(f"Invalid duration '{s}' — use format like '24h', '7d', '30m'")
    value, unit = int(m.group(1)), m.group(2)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    return timedelta(minutes=value)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GROK_",
        # Environment only — no .env from the working directory: in stdio mode that is the
        # client's project, whose unrelated .env must not reconfigure the server.
        extra="ignore",
    )

    # --- agent() limits (all GROK_-prefixed env vars) ---------------------------------
    foreground_timeout: float = 60
    # Seconds a foreground agent() call may run before it is moved to a background job
    # (the caller gets {job_id} and collects with get_job — nothing is lost).
    job_max_seconds: float = 1800
    # Wall-clock budget of one agent() run; when reached the loop ends with a final
    # tool-less answer (stopped="time"). Ceiling: max_seconds may only lower it.
    max_cost_usd: float = 2.5
    # Estimated USD budget of one agent() run (stopped="cost"). Ceiling: max_cost_usd may only lower it.
    max_iterations: int = 100
    # Model rounds in the tool loop (stopped="max_iterations"). Ceiling: max_iterations may only lower it.
    input_max_ratio: float = 0.6
    # Prompt + file_paths + images may use at most this share of the model's context
    # window; larger inputs are rejected before any API call.
    tool_result_max_ratio: float = 0.15
    # One tool result may use at most this share of the window; bigger results are
    # replaced by a message telling the model to read ranges / grep instead.
    context_max_ratio: float = 0.8
    # The conversation may grow to this share of the window; beyond it old tool results
    # are trimmed once, then new results are refused, then the loop ends.
    chars_per_token: float = 3.0
    # Token estimate for text not yet seen by the API (conservative: code ~3.5, non-English text ~3).
    shell_timeout: float = 240
    # Seconds one shell command (agent tool or shell()) may run.

    write_allowed_paths: str | list[str] = []
    # Comma-separated list of path prefixes write() is allowed to write to.
    # e.g. GROK_WRITE_ALLOWED_PATHS=/tmp,/home/you/projects
    # Default: empty → all writes denied.

    log_content: Literal["metadata", "full"] = "metadata"
    # GROK_LOG_CONTENT — "metadata": prompts, answers and commands are logged as size + hash;
    # "full": logged whole (debugging). Secrets are masked in both modes.

    retention_chat_days: int = 30
    # Conversation messages older than this are deleted. 0 = keep forever.
    retention_job_content_days: int = 7
    # Background job result / error text is cleared this long after the job ends;
    # the job row (status, tokens, timing) stays. 0 = keep forever.
    retention_jobs_days: int = 365
    # Job rows (metadata) older than this are deleted. 0 = keep forever.
    retention_projects_days: int = 90
    # Project discovery rows not refreshed for this long are deleted. 0 = keep forever.

    models_ttl: float = 86400
    # GROK_MODELS_TTL — seconds between live model-catalog refreshes.

    max_read_bytes: int = 20_000_000
    # GROK_MAX_READ_BYTES — one byte budget per call: summed over every file of a `read` and
    # over all file_paths + image_paths of an `agent` call. read(offset/limit) counts only the
    # returned range.

    read_allowed_paths: str | list[str] = []
    # Comma-separated path prefixes read() / file_paths / image and audio inputs may read.
    # Default: empty → no restriction (the process can read what its user can).

    @field_validator("write_allowed_paths", "read_allowed_paths", mode="before")
    @classmethod
    def parse_allowed_paths(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        if isinstance(v, list):
            return [str(item) for item in v]
        return []


settings = Settings()
