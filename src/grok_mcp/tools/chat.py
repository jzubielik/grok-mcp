"""Prompt engine behind agent(): explicit model, one tool loop, structured result.

The model is always chosen by the caller (validated against the live catalog). The loop
keeps the message prefix append-only so xAI's automatic prompt cache keeps hitting; it
only rewrites history once, when the context nears the model's window.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import mimetypes
import time
from typing import Any

import httpx
import openai

from ..client import get_text_client
from ..config import settings
from ..logjson import error_text, loggable, strip_data
from ..models_registry import (
    cost_usd,
    default_reasoning_effort,
    find_model,
    get_catalog,
    has_price,
    mark_effort_unsupported,
    model_tags,
    require_model,
)
from ..paths import check_readable, read_bytes_capped
from .budget import _est_tokens, precheck_input
from .context_cache import (
    check_conversation,
    get_active_project_context,
    get_recent_history,
    get_server_project_path,
    refresh_project_git_hash,
    save_turn,
    search_history,
)
from .git_utils import _get_git_hash
from .jobs import start_job
from .workspace import WORKSPACE_TOOL_SCHEMAS, run_workspace_tool, validate_workspace_tools

logger = logging.getLogger("grok_mcp.chat")

# Tool results from the last N messages survive a trim — the model is usually mid-thought.
_KEEP_RECENT_TOOL_RESULTS = 4
# The Nth identical tool call (same name + args) gets a warning instead of running;
# the loop ends at _REPEAT_STOP.
_REPEAT_WARN = 3
_REPEAT_STOP = 5

_HISTORY_TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_recent_history",
            "description": (
                "Fetch the latest messages of this task's history (oldest first): earlier "
                "prompts and your own final answers. Use when the request continues the "
                "latest work, e.g. 'continue' or 'fix the previous result'."
            ),
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 10, "maximum": 50}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_history",
            "description": (
                "Search this task's history (earlier prompts and your own final answers) "
                "for messages containing a phrase. Use when the request names a specific "
                "earlier topic, file or decision."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 5, "maximum": 20},
                },
                "required": ["query"],
            },
        },
    },
]

# Static on purpose: a byte-identical system prompt keeps xAI's prompt cache warm
# across rounds AND across calls. Never put timestamps, hashes or ids in here.
_HISTORY_NOTE = (
    "This request belongs to an ongoing task. Its history holds the earlier prompts and "
    "your own final answers (tool results are not stored) and is NOT shown to you "
    "automatically. Call get_recent_history when the request continues the latest work "
    "(e.g. 'continue', 'fix the previous result'); call search_history when it names a "
    "specific earlier topic, file or decision. Skip history when the request is "
    "self-contained."
)

_FINAL_NOTES = {
    "max_iterations": "Tool budget (max_iterations) is exhausted.",
    "context": "The conversation is close to your context window.",
    "loop": "You kept repeating the same tool call with the same arguments.",
    "time": "The time budget (max_seconds) is exhausted.",
    "cost": "The cost budget (max_cost_usd) is exhausted.",
}


# A finished run on a flagship model above this estimated cost earns a "was it worth it" hint.
_EXPENSIVE_USD = 0.5

_STOP_ADVICE = {
    "cost": "split the task or use ranged reads/grep instead of whole files",
    "time": "split the task or run it in the background with a narrower scope",
    "max_iterations": "give a narrower task or name the files to look at",
    "loop": "rephrase the task — the same tool call kept repeating",
    "context": "use ranged reads/grep instead of whole files",
}


def _hint(tags: list[str], stopped: str, cost: float | None) -> str | None:
    """Advice for the calling model, built from role tags — never from model names."""
    flagship = "flagship" in tags
    pricey = "the cheapest-tagged model at reasoning_effort=low"
    if stopped in _STOP_ADVICE:
        text = f"Run ended early ({stopped}): {_STOP_ADVICE[stopped]}."
        if flagship:
            text += f" This was a flagship run — if the task is routine, retry on {pricey}."
        return text
    if flagship and cost is not None and cost >= _EXPENSIVE_USD:
        return (
            f"Expensive flagship run (~${cost:.2f}). If the task was routine, "
            f"{pricey} usually suffices."
        )
    return None


def _is_context_overflow(exc: Exception) -> bool:
    text = str(exc)
    return "input_too_large" in text or "context window" in text


def _args_summary(args: dict) -> str:
    return ", ".join(f"{k}={str(v)[:80]}" for k, v in args.items())


async def _run_tool(
    name: str,
    args: dict,
    conversation: str | None,
    model: str,
    workspace_tools: list[str],
    time_left: float | None = None,
) -> tuple[str, str | None]:
    """Execute one model tool call; return (text for the model, error message or None).

    The error is reported out of band — never inferred from the text, which may be file
    content that merely looks like JSON (a file starting with `{"error"` is just a file).
    """
    try:
        if name in workspace_tools:
            result: Any = await run_workspace_tool(name, args, model=model, time_left=time_left)
        elif name == "get_recent_history" and conversation:
            limit = min(max(int(args.get("limit", 10)), 1), 50)
            result = await get_recent_history(conversation, limit=limit)
        elif name == "search_history" and conversation:
            result = await search_history(
                conversation,
                str(args.get("query", "")),
                limit=min(max(int(args.get("limit", 5)), 1), 20),
            )
        else:
            return _tool_error(f"tool '{name}' is not available in this call")
        return (result if isinstance(result, str) else json.dumps(result)), None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "agent_tool_failed",
            extra={"fields": {"tool": name, "args": loggable(args), "error": error_text(exc)}},
        )
        return _tool_error(str(exc) or type(exc).__name__)


def _tool_error(message: str) -> tuple[str, str]:
    return json.dumps({"error": message}), message


def _accumulate_usage(sink: dict, response: Any) -> None:
    """Add one response's token usage into `sink` (totals across tool-loop rounds)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    for key, value in (
        ("prompt_tokens", getattr(usage, "prompt_tokens", None)),
        ("cached_tokens", getattr(prompt_details, "cached_tokens", None)),
        ("completion_tokens", getattr(usage, "completion_tokens", None)),
        ("reasoning_tokens", getattr(details, "reasoning_tokens", None)),
    ):
        if isinstance(value, int):
            sink[key] = sink.get(key, 0) + value


def _image_part(raw: bytes, path: str) -> dict:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    b64 = base64.b64encode(raw).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _trim_tool_results(
    messages: list[dict], calls: dict[str, str], keep: int = _KEEP_RECENT_TOOL_RESULTS
) -> int:
    """Replace old tool results with one-line stubs; return how many chars were freed.

    One deliberate cut: it invalidates the prompt cache once, instead of on every round.
    """
    tool_idx = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    freed = 0
    for i in tool_idx[: len(tool_idx) - keep]:
        content = messages[i]["content"]
        if content.startswith("[trimmed"):
            continue
        what = calls.get(messages[i]["tool_call_id"], "tool call")
        stub = f"[trimmed to save context: {what} returned {len(content)} chars — re-run if needed]"
        messages[i]["content"] = stub
        freed += len(content) - len(stub)
    return freed


def _fit_result(
    result: str,
    est_context: int,
    result_limit: int | None,
    context_limit: int | None,
    messages: list[dict],
    calls: dict[str, str],
    trimmed: bool,
) -> tuple[str, str | None, int]:
    """Keep a tool result inside the context budget, BEFORE it reaches the API.

    Returns (result to send, what happened: None | "trimmed" | "full", tokens freed by
    trimming). An oversized result is replaced by instructions; when the conversation would
    overflow, old results are trimmed once, and after that new results are refused ("full"
    ends the loop). The caller must subtract the freed tokens from its estimate — otherwise
    the next result of the same round is refused although the trim made room.
    """
    tokens = _est_tokens(result)
    if result_limit and tokens > result_limit:
        return (
            json.dumps(
                {
                    "error": f"result too large: ~{tokens} tokens (one result may use ~{result_limit}). "
                    "Read a line range (read with offset/limit) or locate code with "
                    "shell grep -n first."
                }
            ),
            None,
            0,
        )
    if not context_limit or est_context + tokens <= context_limit:
        return result, None, 0
    freed_tokens = 0
    if not trimmed:
        freed = _trim_tool_results(messages, calls)
        freed_tokens = int(freed / settings.chars_per_token)
        logger.info("agent_context_trimmed", extra={"fields": {"freed_chars": freed}})
        if est_context - freed_tokens + tokens <= context_limit:
            return result, "trimmed", freed_tokens
    return (
        json.dumps(
            {
                "error": f"no room left: the conversation is at ~{est_context} of ~{context_limit} "
                "usable tokens. Answer from what you have."
            }
        ),
        "full",
        freed_tokens,
    )


def _within(requested: float | None, ceiling: float) -> float:
    """The per-call value, capped at the server's ceiling; the ceiling when not given."""
    if requested is not None and math.isnan(requested):
        # NaN slips through every comparison: `cost >= nan` is never true, so no limit would hold.
        raise ValueError("limits must be numbers, got NaN")
    return ceiling if not requested or requested <= 0 else min(requested, ceiling)


async def _project_context(system: str | None) -> tuple[str | None, str | None]:
    """Prepend THIS server's project context to `system`; return (system, git hash or None)."""
    server_project = await get_server_project_path()
    # Only ever inject THIS server's project context — never another project's.
    project_ctx = await get_active_project_context(server_project)
    if project_ctx and not (system and system.startswith("[Project Context]")):
        system = project_ctx + ("\n\n" + system if system else "")

    # Keep this server's project row fresh. Only when it runs inside a known project — in a
    # container (cwd /app) there is none, and forking git on every call would be pure waste.
    current_hash: str | None = None
    if server_project:
        current_hash = await _get_git_hash(server_project)
        if current_hash:
            await refresh_project_git_hash(server_project, current_hash)
    return system, current_hash


class _ToolLoop:
    """State of one agent() tool loop; each method is one step of the run."""

    def __init__(
        self,
        *,
        model: str,
        info: Any,
        conversation: str | None,
        workspace_tools: list[str],
        reasoning_effort: str | None,
        effort_defaulted: bool,
        rounds: int,
        deadline: float,
        cost_limit: float,
        usage: dict,
    ) -> None:
        self.client = get_text_client()
        self.model = model
        self.info = info
        self.conversation = conversation
        self.workspace_tools = workspace_tools
        self.reasoning_effort = reasoning_effort
        self.effort_defaulted = effort_defaulted
        self.rounds = rounds
        self.deadline = deadline
        self.cost_limit = cost_limit
        self.usage = usage

        # History is NOT injected — the model pulls it on demand via tools, and only when the
        # caller named a conversation; without one the call is stateless.
        self.tool_schemas: list[dict] = list(_HISTORY_TOOL_SCHEMAS) if conversation else []
        self.tool_schemas += [WORKSPACE_TOOL_SCHEMAS[t] for t in workspace_tools]
        # The per-call HTTP timeout is set in _create from the run's remaining time.
        self.base_kwargs: dict[str, Any] = {"model": model}
        if reasoning_effort:
            self.base_kwargs["reasoning_effort"] = reasoning_effort

        window = info.context_length if info else None
        self.context_window = window
        self.result_limit = int(settings.tool_result_max_ratio * window) if window else None
        self.context_limit = int(settings.context_max_ratio * window) if window else None

        self.messages: list[dict[str, Any]] = []
        self.tool_log: list[dict] = []
        self.call_names: dict[str, str] = {}  # tool_call_id → "name(args)" for trim stubs
        self.seen: dict[str, int] = {}  # "name|args" → count, for loop detection
        self.trimmed = False
        self.est_context = 0
        self.iterations = 0
        self.answer: str | None = None
        self.stopped = "max_iterations"

    async def build_messages(
        self,
        prompt: str,
        system: str | None,
        file_paths: list[str] | None,
        image_paths: list[str] | None,
    ) -> None:
        if self.conversation:
            system = (system + "\n\n" + _HISTORY_NOTE) if system else _HISTORY_NOTE
        if system:
            self.messages.append({"role": "system", "content": system})

        # File bodies are sent to the model but never copied into chat_history (the
        # file_paths column records which files were attached).
        # One GROK_MAX_READ_BYTES budget across every attachment (precheck_input sums the
        # sizes); each read gets only what is left, so files grown since cannot exceed it.
        remaining = settings.max_read_bytes
        if file_paths:
            parts = []
            for fpath in file_paths:
                check_readable(fpath)
                # Unreadable or too large → error, not skip; read off the event loop.
                raw = await asyncio.to_thread(read_bytes_capped, fpath, remaining)
                remaining -= len(raw)
                parts.append(raw.decode("utf-8", errors="replace"))
            if parts:
                prompt = "\n\n".join(parts) + "\n\n" + prompt

        if image_paths:
            images = []
            for ipath in image_paths:
                check_readable(ipath)
                raw = await asyncio.to_thread(read_bytes_capped, ipath, remaining)
                remaining -= len(raw)
                images.append(_image_part(raw, ipath))
            content: Any = [{"type": "text", "text": prompt}, *images]
        else:
            content = prompt
        self.messages.append({"role": "user", "content": content})

    def _request(self, tool_choice: str) -> dict[str, Any]:
        kwargs = dict(self.base_kwargs, messages=self.messages)
        if self.tool_schemas:
            kwargs["tools"] = self.tool_schemas
            kwargs["tool_choice"] = tool_choice
        return kwargs

    async def _create(self, kwargs: dict[str, Any]) -> Any:
        # One model call never outlives the run's budget (the final answer keeps 60 s).
        kwargs["timeout"] = httpx.Timeout(
            connect=5.0, read=max(60.0, self.deadline - time.monotonic()), write=60.0, pool=60.0
        )
        try:
            return await self.client.chat.completions.create(**kwargs)
        except openai.BadRequestError as exc:
            if not (self.effort_defaulted and "reasoningEffort" in str(exc)):
                raise
            logger.info("reasoning_effort_unsupported", extra={"fields": {"model": self.model}})
            mark_effort_unsupported(self.model)
            self.effort_defaulted, self.reasoning_effort = False, None
            self.base_kwargs.pop("reasoning_effort", None)
            kwargs.pop("reasoning_effort", None)
            return await self.client.chat.completions.create(**kwargs)

    async def run(self) -> None:
        for _ in range(self.rounds):
            if self.iterations and time.monotonic() > self.deadline:
                self.stopped = "time"
                break
            stop = await self._round()
            if stop:
                self.stopped = stop
                break
        if self.answer is None:
            self.answer = await self._final_answer()

    async def _round(self) -> str | None:
        """One model round plus its tool calls; return why the loop stops, or None."""
        self.iterations += 1
        try:
            response = await self._create(self._request("auto"))
        except openai.APITimeoutError:
            # The read timeout follows the run budget: past the deadline this is the time limit
            # doing its job — end with a final answer, don't fail the paid rounds so far.
            if time.monotonic() < self.deadline:
                raise
            return "time"
        except openai.BadRequestError as exc:
            if not _is_context_overflow(exc):
                raise
            logger.warning("agent_context_overflow", extra={"fields": {"error": error_text(exc)}})
            _trim_tool_results(self.messages, self.call_names, keep=0)
            return "context"
        _accumulate_usage(self.usage, response)
        msg = response.choices[0].message
        last_prompt = getattr(getattr(response, "usage", None), "prompt_tokens", None)
        self.est_context = (last_prompt if isinstance(last_prompt, int) else 0) + _est_tokens(
            (msg.content or "")
            + "".join(tc.function.arguments or "" for tc in msg.tool_calls or [])
        )
        if not msg.tool_calls:
            self.answer = msg.content or ""
            return "answer"

        self.messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            }
        )
        looping = context_full = False
        for tc in msg.tool_calls:
            repeated, full = await self._call_tool(tc)
            looping = looping or repeated
            context_full = context_full or full
        if looping:
            return "loop"
        if context_full:
            return "context"
        if cost_usd(self.info, self.usage) >= self.cost_limit:
            return "cost"
        return self._trim_near_limit(last_prompt)

    async def _call_tool(self, tc: Any) -> tuple[bool, bool]:
        """Run one tool call and append its result; return (loop detected, context full)."""
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        key = f"{name}|{json.dumps(args, sort_keys=True)}"
        self.seen[key] = self.seen.get(key, 0) + 1
        looping = False
        if self.seen[key] >= _REPEAT_WARN:
            result, error = _tool_error(
                "identical call already made — use its result or change approach"
            )
            looping = self.seen[key] >= _REPEAT_STOP
        else:
            # A tool call is bounded by the run's remaining time too: max_seconds must hold
            # even while one slow call (a dripping fetch, a long search) is in flight.
            remaining = max(1.0, self.deadline - time.monotonic())
            try:
                result, error = await asyncio.wait_for(
                    _run_tool(
                        name,
                        args,
                        self.conversation,
                        self.model,
                        self.workspace_tools,
                        remaining,
                    ),
                    remaining + 1,  # shell's own timeout fires first; this is a backstop
                )
            except TimeoutError:
                result, error = _tool_error("tool call exceeded the run's time budget")
        fitted, refused, freed = _fit_result(
            result,
            self.est_context,
            self.result_limit,
            self.context_limit,
            self.messages,
            self.call_names,
            self.trimmed,
        )
        if fitted is not result:  # replaced by our own refusal JSON
            error = json.loads(fitted)["error"]
        result = fitted
        if freed or refused == "trimmed":
            self.trimmed = True  # the one trim is spent, whether or not it made enough room
        self.est_context += _est_tokens(result) - freed
        self.call_names[tc.id] = f"{name}({_args_summary(args)})"
        entry = {"tool": name, "args": strip_data(args), "result_chars": len(result)}
        if error:
            entry["error"] = error
        self.tool_log.append(entry)
        logger.info(
            "agent_tool",
            extra={"fields": {**loggable(entry), "model": self.model, "round": self.iterations}},
        )
        self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        return looping, refused == "full"

    def _trim_near_limit(self, last_prompt: Any) -> str | None:
        """Trim old tool results once when the API-reported prompt nears the limit."""
        near_limit = (
            self.context_limit and isinstance(last_prompt, int) and last_prompt > self.context_limit
        )
        if not near_limit:
            return None
        if self.trimmed or not _trim_tool_results(self.messages, self.call_names):
            return "context"
        self.trimmed = True
        logger.info(
            "agent_context_trimmed",
            extra={"fields": {"prompt_tokens": last_prompt, "context_window": self.context_window}},
        )
        return None

    async def _final_answer(self) -> str:
        # Never throw away the work: one last tool-less round answers from what was gathered.
        self.messages.append(
            {
                "role": "user",
                "content": f"{_FINAL_NOTES[self.stopped]} Do not call tools. Answer now from what "
                "you gathered, and say what you could not check.",
            }
        )
        kwargs = self._request("none")
        try:
            response = await self._create(kwargs)
        except openai.APITimeoutError:
            logger.warning(
                "agent_final_answer_timeout", extra={"fields": {"round": self.iterations}}
            )
            return ""  # tool_calls, usage and cost still reach the caller
        except openai.BadRequestError as exc:
            if not _is_context_overflow(exc):
                raise
            # Last resort: drop every tool result body, keep only what was asked and called.
            _trim_tool_results(self.messages, self.call_names, keep=0)
            response = await self._create(kwargs)
        _accumulate_usage(self.usage, response)
        return response.choices[0].message.content or ""


async def _chat_internal(
    prompt: str,
    model: str | None = None,
    system: str | None = None,
    conversation: str | None = None,
    file_paths: list[str] | None = None,
    image_paths: list[str] | None = None,
    reasoning_effort: str | None = None,
    tools: list[str] | None = None,
    max_iterations: int | None = None,
    max_seconds: float | None = None,
    max_cost_usd: float | None = None,
    usage_sink: dict | None = None,
    validated: bool = False,
) -> dict:
    """Run one agent() turn and return {answer, model, tokens, cost_usd, tool_calls, ...}.

    validated=True: the caller (agent()) already ran check_conversation,
    validate_workspace_tools, require_model and precheck_input on these exact inputs, so
    they are used as given.
    """
    started = time.monotonic()
    if validated:
        workspace_tools = list(tools or [])
    else:
        conversation = check_conversation(conversation)
        workspace_tools = validate_workspace_tools(tools)
        model = await require_model(model, images=bool(image_paths))
        await precheck_input(prompt, model, system, file_paths, image_paths)
    # Server settings are ceilings: a call may only lower them (like GROK_SHELL_TIMEOUT).
    rounds = max(1, _within(max_iterations, settings.max_iterations))
    deadline = started + _within(max_seconds, settings.job_max_seconds)
    cost_limit = _within(max_cost_usd, settings.max_cost_usd)
    usage: dict = usage_sink if usage_sink is not None else {}
    catalog = await get_catalog()
    info = find_model(catalog, model)
    # Flagship models think hard by default and bill that as output; unless the caller says
    # otherwise they run at low. A model that refuses the parameter drops the default for good.
    effort_defaulted = not reasoning_effort
    if effort_defaulted:
        reasoning_effort = default_reasoning_effort(catalog, model)
        effort_defaulted = reasoning_effort is not None

    system, current_hash = await _project_context(system)
    loop = _ToolLoop(
        model=model,
        info=info,
        conversation=conversation,
        workspace_tools=workspace_tools,
        reasoning_effort=reasoning_effort,
        effort_defaulted=effort_defaulted,
        rounds=rounds,
        deadline=deadline,
        cost_limit=cost_limit,
        usage=usage,
    )
    await loop.build_messages(prompt, system, file_paths, image_paths)
    await loop.run()
    answer = loop.answer or ""

    if conversation:
        # The answer is already paid for and goes back to the caller regardless; a history
        # write that fails is logged, not silently dropped and not raised over the answer.
        try:
            await save_turn(
                conversation,
                prompt,
                answer,
                model,
                file_paths=file_paths,
                git_hash=current_hash,
                usage=usage or None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("history_save_failed", extra={"fields": {"error": error_text(exc)}})

    cost = cost_usd(info, usage)
    result = {
        "answer": answer,
        "model": model,
        "reasoning_effort": loop.reasoning_effort,
        "tokens": {
            "prompt": usage.get("prompt_tokens", 0),
            "cached": usage.get("cached_tokens", 0),
            "completion": usage.get("completion_tokens", 0),
            "reasoning": usage.get("reasoning_tokens", 0),
        },
        "cost_usd": cost,
        **(
            {}
            if has_price(info)
            else {"cost_note": "no live price for this model — conservative estimate"}
        ),
        "tool_calls": loop.tool_log,
        "iterations": loop.iterations,
        "stopped": loop.stopped,
    }
    hint = _hint(model_tags(catalog).get(model, []), loop.stopped, cost)
    if hint:
        result["hint"] = hint
    return result


async def _start_background(
    tool: str, work: Any, model: str | None, usage: dict | None = None
) -> dict:
    """Hand `work` to the job runner and return {job_id, status, collect_with}."""
    job_id = await start_job(tool, work, model=model, usage_sink=usage)
    return {"job_id": job_id, "status": "running", "collect_with": "get_job"}
