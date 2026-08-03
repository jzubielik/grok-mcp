"""Tests for agent() — the single prompt/tool-loop entry point with an explicit model."""

from __future__ import annotations

import asyncio
import inspect
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _response(content=None, tool_calls=None, prompt_tokens=None, cached=None):
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = tool_calls
    resp.choices = [choice]
    if prompt_tokens is None:
        resp.usage = None
    else:
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = prompt_tokens
        resp.usage.completion_tokens = 10
        resp.usage.completion_tokens_details.reasoning_tokens = 5
        resp.usage.prompt_tokens_details.cached_tokens = cached or 0
    return resp


def _tool_call(call_id: str, name: str, arguments: dict):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


@pytest.fixture
def env():
    """Isolate the engine from DB/project plumbing; yields the mocked text client."""
    client = AsyncMock()
    with (
        patch("grok_mcp.tools.chat.get_active_project_context", AsyncMock(return_value=None)),
        patch("grok_mcp.tools.chat.get_server_project_path", AsyncMock(return_value=None)),
        patch("grok_mcp.tools.chat.save_turn", AsyncMock()),
        patch("grok_mcp.tools.chat.get_text_client", return_value=client),
    ):
        yield client


def _calls(client) -> list[dict]:
    return [c.kwargs for c in client.chat.completions.create.await_args_list]


def _tool_result(client, call_index: int) -> str:
    msgs = _calls(client)[call_index]["messages"]
    return [m for m in msgs if m["role"] == "tool"][-1]["content"]


def _offered(client, call_index: int = 0) -> set[str]:
    return {t["function"]["name"] for t in _calls(client)[call_index].get("tools", [])}


# ---------------------------------------------------------------------------
# model choice
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_returns_structured_result(env):
    env.chat.completions.create.return_value = _response(content="Done!")
    from grok_mcp.tools.agent import agent

    result = await agent("Simple task", model="m")

    assert result["answer"] == "Done!"
    assert result["model"] == "m"
    assert result["stopped"] == "answer"
    assert result["iterations"] == 1
    assert result["tool_calls"] == []
    assert set(result["tokens"]) == {"prompt", "cached", "completion", "reasoning"}


@pytest.mark.asyncio
async def test_agent_requires_model(env):
    from grok_mcp.tools.agent import agent

    with pytest.raises(ValueError, match="model is required"):
        await agent("x", model="")
    env.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_rejects_unknown_model_listing_choices(env):
    from grok_mcp.tools.agent import agent

    with pytest.raises(ValueError, match="grok-4.3"):
        await agent("x", model="grok-9000")
    env.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_accepts_alias_and_reports_canonical_id(env):
    env.chat.completions.create.return_value = _response(content="ok")
    from grok_mcp.tools.agent import agent

    result = await agent("x", model="grok-4.3-latest")

    assert _calls(env)[0]["model"] == "grok-4.3"
    assert result["model"] == "grok-4.3"


@pytest.mark.asyncio
async def test_agent_passes_reasoning_effort_as_is(env):
    env.chat.completions.create.return_value = _response(content="ok")
    from grok_mcp.tools.agent import agent

    await agent("x", model="m", reasoning_effort="high")
    assert _calls(env)[0]["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_agent_omits_reasoning_effort_when_unset(env):
    env.chat.completions.create.return_value = _response(content="ok")
    from grok_mcp.tools.agent import agent

    await agent("x", model="m")
    assert "reasoning_effort" not in _calls(env)[0]


# ---------------------------------------------------------------------------
# usage and cost
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_reports_cached_tokens_and_cost(env):
    env.chat.completions.create.return_value = _response(
        content="ok", prompt_tokens=1_000_000, cached=800_000
    )
    from grok_mcp.tools.agent import agent

    result = await agent("x", model="m")

    assert result["tokens"] == {
        "prompt": 1_000_000,
        "cached": 800_000,
        "completion": 10,
        "reasoning": 5,
    }
    # 200K uncached @ $1/M + 800K cached @ $0.25/M + 15 out @ $2/M
    assert result["cost_usd"] == pytest.approx(0.2 + 0.2 + 15 * 2e-6)


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_sends_image_paths_as_image_parts(env, tmp_path):
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG fake")
    env.chat.completions.create.return_value = _response(content="a cat")
    from grok_mcp.tools.agent import agent

    await agent("What is this?", model="m", image_paths=[str(img)])

    user = [m for m in _calls(env)[0]["messages"] if m["role"] == "user"][0]
    assert user["content"][0] == {"type": "text", "text": "What is this?"}
    assert user["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_agent_rejects_text_only_model_for_images(env, tmp_path):
    img = tmp_path / "shot.png"
    img.write_bytes(b"x")
    from grok_mcp.tools.agent import agent

    with pytest.raises(ValueError, match="image input"):
        await agent("x", model="text-only", image_paths=[str(img)])


# ---------------------------------------------------------------------------
# workspace tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_rejects_unknown_tool_before_calling_model(env):
    from grok_mcp.tools.agent import agent

    with pytest.raises(ValueError, match="nope"):
        await agent("x", model="m", tools=["read", "nope"])
    env.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_offers_only_requested_workspace_tools(env):
    env.chat.completions.create.return_value = _response(content="ok")
    from grok_mcp.tools.agent import agent

    await agent("x", model="m", tools=["read"])

    offered = _offered(env)
    assert "read" in offered
    assert not offered & {"write", "shell", "fetch", "search", "escalate_model"}


@pytest.mark.asyncio
async def test_agent_reads_files_and_logs_tool_calls(env, tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("file content")
    env.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("c1", "read", {"path": str(f)})]),
        _response(content="Content is: file content"),
    ]
    from grok_mcp.tools.agent import agent

    result = await agent("Read the file", model="m", tools=["read"])

    assert result["answer"] == "Content is: file content"
    assert _tool_result(env, 1) == "file content"
    assert result["tool_calls"] == [
        {"tool": "read", "args": {"path": str(f)}, "result_chars": len("file content")}
    ]
    assert result["iterations"] == 2


@pytest.mark.asyncio
async def test_agent_search_tool_runs_on_the_agents_model(env):
    env.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("c1", "search", {"query": "q"})]),
        _response(content="found"),
    ]
    with patch(
        "grok_mcp.tools.workspace._search_internal", AsyncMock(return_value={"answer": "a"})
    ) as search:
        from grok_mcp.tools.agent import agent

        await agent("x", model="grok-4.7", tools=["search"])

    search.assert_awaited_once_with("q", "grok-4.7")


@pytest.mark.asyncio
async def test_agent_shell_runs_without_any_env_flag(env):
    env.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("c1", "shell", {"cmd": "echo hello-shell"})]),
        _response(content="ran"),
    ]
    from grok_mcp.tools.agent import agent

    await agent("run it", model="m", tools=["shell"])

    out = _tool_result(env, 1)
    assert "[exit code: 0]" in out
    assert "hello-shell" in out


@pytest.mark.asyncio
async def test_agent_write_still_enforces_allowed_paths(env, tmp_path, monkeypatch):
    monkeypatch.setattr("grok_mcp.paths.settings.write_allowed_paths", [])
    target = tmp_path / "out.txt"
    env.chat.completions.create.side_effect = [
        _response(
            tool_calls=[
                _tool_call("c1", "write", {"path": str(target), "content": "x", "dry_run": False})
            ]
        ),
        _response(content="tried"),
    ]
    from grok_mcp.tools.agent import agent

    result = await agent("write it", model="m", tools=["write"])

    assert "WRITE_ALLOWED_PATHS" in _tool_result(env, 1)
    assert "WRITE_ALLOWED_PATHS" in result["tool_calls"][0]["error"]
    assert not target.exists()


@pytest.mark.asyncio
async def test_agent_refuses_tool_that_was_not_requested(env):
    env.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("c1", "shell", {"cmd": "echo pwned"})]),
        _response(content="Handled"),
    ]
    from grok_mcp.tools.agent import agent

    result = await agent("x", model="m", tools=["read"])

    assert result["answer"] == "Handled"
    out = _tool_result(env, 1)
    assert "not available" in out
    assert "pwned" not in out


# ---------------------------------------------------------------------------
# loop termination — every stop still yields an answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_iterations_ends_with_a_forced_tool_less_answer(env, tmp_path):
    calls = [
        _response(tool_calls=[_tool_call(f"c{i}", "read", {"path": str(tmp_path / f"f{i}")})])
        for i in range(3)
    ]
    env.chat.completions.create.side_effect = [*calls, _response(content="partial answer")]
    from grok_mcp.tools.agent import agent

    result = await agent("Loop", model="m", tools=["read"], max_iterations=3)

    assert result["answer"] == "partial answer"
    assert result["stopped"] == "max_iterations"
    assert result["iterations"] == 3
    final = _calls(env)[3]
    assert final["tool_choice"] == "none"
    assert "max_iterations" in final["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_max_iterations_cannot_exceed_the_server_ceiling(env, tmp_path, monkeypatch):
    """A caller may lower the limits, never raise them past the server's settings."""
    import grok_mcp.tools.chat as chat_module

    monkeypatch.setattr(chat_module.settings, "max_iterations", 5)
    calls = [
        _response(tool_calls=[_tool_call(f"c{i}", "read", {"path": str(tmp_path / f"f{i}")})])
        for i in range(10)
    ]
    env.chat.completions.create.side_effect = [*calls, _response(content="done")]
    from grok_mcp.tools.agent import agent

    result = await agent("Loop", model="m", tools=["read"], max_iterations=150)

    assert result["iterations"] == 5
    assert result["stopped"] == "max_iterations"


@pytest.mark.asyncio
async def test_repeated_identical_call_is_refused_then_loop_stops(env, tmp_path):
    same = _tool_call("c", "read", {"path": str(tmp_path / "same")})
    env.chat.completions.create.side_effect = [
        *[_response(tool_calls=[same]) for _ in range(5)],
        _response(content="gave up"),
    ]
    from grok_mcp.tools.agent import agent

    result = await agent("x", model="m", tools=["read"])

    assert result["stopped"] == "loop"
    assert result["answer"] == "gave up"
    assert "identical call" in result["tool_calls"][2]["error"]
    assert env.chat.completions.create.await_count == 6


@pytest.mark.asyncio
async def test_context_near_limit_trims_old_tool_results_once(env, tmp_path):
    files = []
    for i in range(6):
        f = tmp_path / f"f{i}.txt"
        f.write_text(f"content-{i}" * 10)
        files.append(f)
    # tiny-ctx has a 1000-token window: report 900 prompt tokens (90%) on round 6
    rounds = [
        _response(
            tool_calls=[_tool_call(f"c{i}", "read", {"path": str(files[i])})],
            prompt_tokens=900 if i == 5 else 100,
        )
        for i in range(6)
    ]
    env.chat.completions.create.side_effect = [*rounds, _response(content="done")]
    from grok_mcp.tools.agent import agent

    result = await agent("x", model="tiny-ctx", tools=["read"])

    assert result["answer"] == "done"
    tool_msgs = [m for m in _calls(env)[6]["messages"] if m["role"] == "tool"]
    assert tool_msgs[0]["content"].startswith("[trimmed")
    assert "read(path=" in tool_msgs[0]["content"]
    assert not tool_msgs[-1]["content"].startswith("[trimmed")


@pytest.mark.asyncio
async def test_context_still_full_after_trim_stops_with_answer(env, tmp_path):
    # 95% full on the first round and nothing old enough to trim → stop right away.
    f = tmp_path / "f.txt"
    f.write_text("x")
    env.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("c0", "read", {"path": str(f)})], prompt_tokens=950),
        _response(content="best effort"),
    ]
    from grok_mcp.tools.agent import agent

    result = await agent("x", model="tiny-ctx", tools=["read"])

    assert result["stopped"] == "context"
    assert result["answer"] == "best effort"
    assert result["iterations"] == 1


# ---------------------------------------------------------------------------
# prompt cache friendliness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_prompt_is_byte_identical_across_calls(env):
    env.chat.completions.create.return_value = _response(content="ok")
    with patch(
        "grok_mcp.tools.chat.get_active_project_context",
        AsyncMock(return_value="[Project Context]\nPython"),
    ):
        from grok_mcp.tools.agent import agent

        await agent("first", model="m", conversation="c1")
        await agent("second", model="m", conversation="c1")

    first, second = _calls(env)
    assert first["messages"][0] == second["messages"][0]
    assert first["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_tool_rounds_only_append_to_messages(env, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("a")
    env.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("c1", "read", {"path": str(f)})]),
        _response(content="ok"),
    ]
    from grok_mcp.tools.agent import agent

    await agent("x", model="m", tools=["read"])

    first, second = (c["messages"] for c in _calls(env))
    assert second[: len(first)] == first  # prefix unchanged → cache hit


# ---------------------------------------------------------------------------
# background jobs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_background_returns_job_dict(env):
    with patch("grok_mcp.tools.chat.start_job", AsyncMock(return_value="job-1")) as start:
        from grok_mcp.tools.agent import agent

        result = await agent("long task", model="m", tools=["read"], background=True)

    assert result == {"job_id": "job-1", "status": "running", "collect_with": "get_job"}
    assert start.await_args.args[0] == "agent"


@pytest.mark.asyncio
async def test_agent_background_validates_tools_and_model_up_front(env):
    with patch("grok_mcp.tools.chat.start_job", AsyncMock(return_value="job-1")) as start:
        from grok_mcp.tools.agent import agent

        with pytest.raises(ValueError):
            await agent("x", model="m", tools=["bogus"], background=True)
        with pytest.raises(ValueError):
            await agent("x", model="nope", background=True)
        for limit in ("max_iterations", "max_seconds", "max_cost_usd"):
            with pytest.raises(ValueError, match="NaN"):
                await agent("x", model="m", background=True, **{limit: float("nan")})

    start.assert_not_awaited()


# ---------------------------------------------------------------------------
# limits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_defaults_come_from_settings():
    from grok_mcp.config import settings

    assert settings.max_iterations == 100
    assert settings.foreground_timeout == 60
    assert settings.job_max_seconds == 1800
    assert settings.max_cost_usd == 2.5
    assert settings.shell_timeout == 240


@pytest.mark.asyncio
async def test_model_call_timeout_follows_the_run_budget(env):
    """The HTTP read budget of a model call never exceeds what the run has left."""
    env.chat.completions.create.return_value = _response(content="ok")
    from grok_mcp.tools.agent import agent

    await agent("x", model="m", max_seconds=300)
    timeout = _calls(env)[0]["timeout"]
    assert 60 <= timeout.read <= 300
    assert timeout.connect == 5.0


@pytest.mark.asyncio
async def test_oversized_input_is_rejected_before_any_api_call(env, tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("x" * 3000)  # ~1000 tokens > 60% of tiny-ctx's 1000
    from grok_mcp.tools.agent import agent

    with pytest.raises(ValueError, match="larger window") as err:
        await agent("x", model="tiny-ctx", file_paths=[str(big)])
    assert "grok-4.3" in str(err.value)  # names the models that would fit
    env.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_invisible_image_path_is_rejected_with_a_hint(env, monkeypatch):
    monkeypatch.setenv("WORKSPACE_DIR", "/home/you/projects")
    from grok_mcp.tools.agent import agent

    with pytest.raises(FileNotFoundError, match="/home/you/projects"):
        await agent("x", model="m", image_paths=["/tmp/nowhere/shot.png"])
    env.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_oversized_tool_result_is_replaced_by_instructions(env, tmp_path):
    f = tmp_path / "huge.cs"
    f.write_text("line\n" * 200)  # ~334 tokens > 15% of tiny-ctx (150)
    env.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("c1", "read", {"path": str(f)})], prompt_tokens=10),
        _response(content="ok"),
    ]
    from grok_mcp.tools.agent import agent

    result = await agent("x", model="tiny-ctx", tools=["read"])

    sent = _tool_result(env, 1)
    assert "result too large" in sent and "offset/limit" in sent
    assert "result too large" in result["tool_calls"][0]["error"]


@pytest.mark.asyncio
async def test_results_that_would_overflow_the_context_stop_the_loop(env, tmp_path):
    files = []
    for i in range(3):
        f = tmp_path / f"f{i}.txt"
        f.write_text("y" * 300)  # ~100 tokens each, under the per-result cap
        files.append(f)
    env.chat.completions.create.side_effect = [
        _response(
            tool_calls=[_tool_call(f"c{i}", "read", {"path": str(files[i])}) for i in range(3)],
            prompt_tokens=650,  # + 3 x ~100 > 80% of 1000
        ),
        _response(content="best effort"),
    ]
    from grok_mcp.tools.agent import agent

    result = await agent("x", model="tiny-ctx", tools=["read"])

    assert result["stopped"] == "context"
    assert result["answer"] == "best effort"
    assert "no room left" in result["tool_calls"][-1]["error"]


@pytest.mark.asyncio
async def test_api_context_overflow_ends_with_an_answer_not_an_error(env, tmp_path):
    import httpx
    import openai

    f = tmp_path / "a.txt"
    f.write_text("a")
    overflow = openai.BadRequestError(
        "Failed to start sampling: [input_too_large] The prompt is too long",
        response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
        body=None,
    )
    env.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("c1", "read", {"path": str(f)})]),
        overflow,
        _response(content="answer from what was gathered"),
    ]
    from grok_mcp.tools.agent import agent

    result = await agent("x", model="m", tools=["read"])

    assert result["stopped"] == "context"
    assert result["answer"] == "answer from what was gathered"


@pytest.mark.asyncio
async def test_time_budget_ends_the_loop(env, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("a")
    env.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("c1", "read", {"path": str(f)})]),
        _response(content="out of time"),
    ]
    from grok_mcp.tools.agent import agent

    result = await agent("x", model="m", tools=["read"], max_seconds=1e-9)

    assert result["stopped"] == "time"
    assert result["iterations"] == 1


@pytest.mark.asyncio
async def test_cost_budget_ends_the_loop(env, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("a")
    env.chat.completions.create.side_effect = [
        # 300K prompt tokens @ $1/M = $0.30 > max_cost_usd, well inside the window
        _response(tool_calls=[_tool_call("c1", "read", {"path": str(f)})], prompt_tokens=300_000),
        _response(content="over budget"),
    ]
    from grok_mcp.tools.agent import agent

    result = await agent("x", model="m", tools=["read"], max_cost_usd=0.1)

    assert result["stopped"] == "cost"
    assert result["answer"] == "over budget"


@pytest.mark.asyncio
async def test_slow_foreground_call_moves_to_the_background(env, monkeypatch):
    import asyncio

    import grok_mcp.tools.agent as agent_module

    # Patch the object agent() reads — test_config reloads grok_mcp.config.
    monkeypatch.setattr(agent_module.settings, "foreground_timeout", 0.05)

    async def slow(**_kwargs):
        await asyncio.sleep(0.3)
        return _response(content="late answer")

    env.chat.completions.create.side_effect = slow
    adopted = {}

    async def fake_adopt(tool, task, model=None, usage_sink=None, job_ref=None):
        adopted["task"] = task
        return "job-7"

    with patch("grok_mcp.tools.jobs.adopt_task", fake_adopt):
        from grok_mcp.tools.agent import agent

        result = await agent("x", model="m")

    assert result["job_id"] == "job-7"
    assert "moved to the background" in result["note"]
    finished = await adopted["task"]  # the same work keeps running and completes
    assert finished["answer"] == "late answer"


@pytest.mark.asyncio
async def test_slow_tool_call_is_bounded_by_the_run_budget(env, monkeypatch):
    """max_seconds holds while a single tool call is in flight, not only between rounds."""
    import grok_mcp.tools.chat as chat_module

    async def stuck(*args, **kwargs):
        await asyncio.sleep(30)

    monkeypatch.setattr(chat_module, "_run_tool", stuck)
    env.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("c1", "read", {"path": "/x"})]),
        _response(content="gave up"),
    ]
    from grok_mcp.tools.agent import agent

    started = asyncio.get_running_loop().time()
    result = await agent("x", model="m", tools=["read"], max_seconds=1)
    assert asyncio.get_running_loop().time() - started < 5
    assert "time budget" in result["tool_calls"][0]["error"]


@pytest.mark.asyncio
async def test_file_content_that_looks_like_an_error_does_not_crash_the_run(env, tmp_path):
    """A file starting with `{"error"` (a log line, a hostile repo) is content, not an error."""
    f = tmp_path / "trap.log"
    f.write_text('{"error": broken json, not ours')
    env.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("c1", "read", {"path": str(f)})]),
        _response(content="read it"),
    ]
    from grok_mcp.tools.agent import agent

    result = await agent("x", model="m", tools=["read"])
    assert result["answer"] == "read it"
    assert "error" not in result["tool_calls"][0]


@pytest.mark.asyncio
async def test_model_timeout_past_the_deadline_ends_with_a_final_answer(env):
    """The read timeout follows the run budget; hitting it is "time", not a failed run."""
    import httpx
    import openai

    async def slow_then_timeout(**kwargs):
        await asyncio.sleep(0.15)
        raise openai.APITimeoutError(request=httpx.Request("POST", "https://api.x.ai/v1"))

    steps = [slow_then_timeout, _response(content="final")]

    async def create(**kwargs):
        step = steps.pop(0)
        return await step(**kwargs) if inspect.iscoroutinefunction(step) else step

    env.chat.completions.create.side_effect = create
    from grok_mcp.tools.agent import agent

    result = await agent("x", model="m", max_seconds=0.1)
    assert result["stopped"] == "time"
    assert result["answer"] == "final"


def test_fit_result_reports_the_tokens_it_freed():
    """The caller must learn how much the trim freed, or the round's next result is refused."""
    from grok_mcp.tools.chat import _fit_result

    messages = [{"role": "tool", "tool_call_id": f"t{i}", "content": "x" * 1000} for i in range(6)]
    calls = {f"t{i}": f"read(f{i})" for i in range(6)}
    result, refused, freed = _fit_result("y" * 300, 700, None, 800, messages, calls, False)
    assert refused == "trimmed" and freed > 0
    # With the freed tokens subtracted, a small second result of the same round fits.
    est = 700 + len(result) // 3 - freed
    _, refused2, _ = _fit_result("z" * 30, est, None, 800, messages, calls, True)
    assert refused2 is None


def test_exception_without_message_still_reports_an_error():
    from grok_mcp.tools.chat import _tool_error

    text, error = _tool_error(str(TimeoutError()) or "TimeoutError")
    assert error == "TimeoutError" and "TimeoutError" in text
