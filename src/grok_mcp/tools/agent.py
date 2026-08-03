"""agent() — the single MCP entry point for prompting Grok (text, images, tool loop)."""

from __future__ import annotations

from ..config import settings
from ..models_registry import require_model
from ..server import mcp
from .budget import precheck_input
from .chat import _chat_internal, _start_background, _within
from .context_cache import check_conversation
from .jobs import run_with_handoff
from .workspace import WORKSPACE_TOOLS, validate_workspace_tools


@mcp.tool(
    description=(
        "Send a prompt to Grok; optionally let it call tools until it has an answer.\n"
        "model (required): choose deliberately — start with the `cheapest` tag, escalate to "
        "`flagship` only when a cheap run was not enough. "
        "Current xAI catalog ([images] = accepts image_paths; list_models for live data):\n"
        "{models}\n"
        f"tools: workspace tools Grok may call on its own — subset of {list(WORKSPACE_TOOLS)}; "
        "default none. With tools=['read'] (+'shell' for git/grep) Grok opens the files it "
        "needs itself — name the files in the prompt, not whole trees. read supports line "
        "ranges (offset/limit). write is limited to GROK_WRITE_ALLOWED_PATHS and defaults "
        "to dry_run.\n"
        "image_paths / file_paths must be visible to the server (its mounted directories) — "
        "host /tmp paths are not.\n"
        "reasoning_effort: low|medium|high (some models also none); an explicit value is sent "
        "as-is. Support varies by model (some reject it, others reject 'none'). Unset: a "
        "`flagship` model runs at low, any other uses its own default.\n"
        f"Limits (server env; max_seconds / max_cost_usd / max_iterations may only LOWER them): "
        f"a foreground call still running after {settings.foreground_timeout:g} s moves to the "
        f"background and returns {{job_id}} — collect with get_job; max_seconds (at most "
        f"{settings.job_max_seconds:g}), max_cost_usd (at most {settings.max_cost_usd:g}), "
        f"max_iterations (model rounds, at most {settings.max_iterations}). Input over {settings.input_max_ratio:.0%} of the "
        f"model's context window is rejected up front; a single tool result over "
        f"{settings.tool_result_max_ratio:.0%} is refused (read ranges / grep instead); the "
        f"conversation is kept under {settings.context_max_ratio:.0%} (old results trimmed "
        "once, then new ones refused). A repeated identical tool call is refused. Whatever "
        "stops the loop, a final tool-less round answers from what was gathered — 'stopped' "
        "says why: answer | max_iterations | context | loop | time | cost.\n"
        "conversation: a name for the task this call belongs to — pass "
        "conversation='<project>/<task>' (e.g. 'grok-mcp/security-audit') to keep context "
        "across calls of one task; Grok pulls earlier prompts and answers on demand. Use a new "
        "name for a new task; omit it for a one-off call (no history kept or offered). The "
        "server cannot tell projects apart over HTTP, hence the project prefix. "
        "file_paths injects file contents server-side.\n"
        "background=true returns {job_id} at once.\n"
        "Returns {answer, model, reasoning_effort, tokens{prompt,cached,completion,reasoning}, "
        "cost_usd, tool_calls, iterations, stopped, hint?} — hint appears when the run ended "
        "early or was expensive on a flagship; follow it."
    )
)
async def agent(
    prompt: str,
    model: str,
    tools: list[str] | None = None,
    image_paths: list[str] | None = None,
    system: str | None = None,
    conversation: str | None = None,
    file_paths: list[str] | None = None,
    reasoning_effort: str | None = None,
    max_iterations: int | None = None,
    max_seconds: float | None = None,
    max_cost_usd: float | None = None,
    background: bool = False,
) -> dict:
    # Fail fast — before any job is queued or any API call is made.
    conversation = check_conversation(conversation)
    tools = validate_workspace_tools(tools)
    model = await require_model(model, images=bool(image_paths))
    await precheck_input(prompt, model, system, file_paths, image_paths)
    for limit, ceiling in (
        (max_iterations, settings.max_iterations),
        (max_seconds, settings.job_max_seconds),
        (max_cost_usd, settings.max_cost_usd),
    ):
        _within(limit, ceiling)  # raises on NaN here, not inside a job already queued
    usage: dict = {}

    async def work() -> dict:
        return await _chat_internal(
            prompt,
            model=model,
            system=system,
            conversation=conversation,
            file_paths=file_paths,
            image_paths=image_paths,
            reasoning_effort=reasoning_effort,
            tools=tools,
            max_iterations=max_iterations,
            max_seconds=max_seconds,
            max_cost_usd=max_cost_usd,
            usage_sink=usage,
            validated=True,
        )

    if background:
        return await _start_background("agent", work, model, usage)

    # Foreground: answer directly if quick; otherwise hand the SAME running task over to the
    # job runner so the client never hits its own timeout and no work is lost.
    kind, value = await run_with_handoff(
        "agent", work, model, usage, timeout=settings.foreground_timeout
    )
    if kind == "done":
        return value
    job_id = value
    return {
        "job_id": job_id,
        "status": "running",
        "collect_with": "get_job",
        "note": f"still running after {settings.foreground_timeout:g} s — moved to the "
        "background; collect the result with get_job",
    }
