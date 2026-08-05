---
name: grok-mcp
description: Use when working with Grok MCP tools (agent prompts and tool loops, image analysis, image generation), OR when estimating token cost of a task running on your current model, OR when a task involves >50K tokens, large files, or full codebase analysis — to decide whether to route through Grok for cost savings and select the right model.
---

# Grok MCP — Agent Skills Reference

Use this file to decide **which tool to call, which model to use, and when not to call Grok at all**.
**There are no default models — you pass `model` on every call and you choose it deliberately:**
the cheapest model that can do the job; go up only when quality is insufficient.

---

## When to use Grok MCP vs. doing it yourself

| Task | Use Grok MCP | Do it yourself |
|------|-------------|-----------------|
| User explicitly asks for Grok | ✅ always honour the request | — |
| Generate or edit an image | ✅ always | ✗ most models cannot generate images |
| Analyze a local file image | ✅ `agent(model=..., image_paths=[...])` | ✗ requires file on disk |
| Large codebase analysis (>200K tokens) | ✅ `agent(model=<large-context>, tools=["read"])` | ✗ context limit |
| Compare your answer with Grok's | ✅ call both | ✅ call both |
| General text / reasoning (unprompted) | ✗ unnecessary cost | ✅ prefer |
| Code generation (unprompted) | ✗ unless your current model is unavailable | ✅ prefer |

**Default rule: do it yourself first. Reach for Grok when:**
- The user explicitly asks for Grok (e.g. "ask Grok", "use Grok", "generate with Grok")
- Your current model cannot do the task (image generation, local file vision, context exhausted)
- The user asks to compare your answer with Grok's

---

## Choosing a model — your decision, every call

Every tool that runs a Grok model (`agent`, `read` with a question, `shell`, `search`,
`generate_image*`) **requires `model`**. There are no roles, no defaults, no `auto` router.
A missing or unknown model fails up front with the list of valid choices.

**Where the choices come from:** each tool's description lists the current xAI catalog —
id, context window, price per 1M tokens (normal and cached) and `[images]` for models that
accept image input. The list is filled from the live xAI API when the server starts;
`list_models(refresh=true)` refetches the catalog, reports what changed and refreshes the
tool descriptions. **Never hardcode a model name or price from memory — read the list.**

**Role tags.** The server marks models from the live catalog, never by name — so the policy
below survives model swaps: `cheapest` (lowest price), `flagship` (newest generation,
deepest, priciest), `code` (tuned for code). Pick by tag and by the prices shown.

**Default policy — start cheap, escalate only on evidence:**

1. Routine work (questions, summaries, extraction, classification, code edits, file review)
   → the `cheapest` model, `reasoning_effort="low"` where the model accepts it.
2. Escalate to `flagship` only when a cheap run's answer was clearly insufficient, or the
   task is deep multi-step analysis across many files. Not "to be safe".
3. Long or tool-heavy runs: `tools=["read"]` + named files, `background=True`, and an
   explicit `max_cost_usd` you chose from the estimate below — never rely on the server default.
4. Read the result: `cost_usd`, `stopped` and `hint`. A `hint` means the run ended early or
   was expensive on a flagship — follow it (retry on `cheapest`, narrow the task) instead of
   re-running the same call.

| Need | Pick |
|------|------|
| Input larger than ~200K tokens | a model whose context window fits — check `ctx` |
| Image input | a model marked `[images]` |
| Image generation | iterate on the cheapest image model, final render on a quality one |

### Cost gate — mandatory for every `agent()` / `read` / `shell` / `search` call

**1. Estimate first, and say it.** Before the call, write one line to the user:
`Estimate: ~<N>K input tokens, ~<R> rounds, <model> → ~$<X>`. For a tiny
single-shot call (a short question) one clause is enough ("~2K tokens, <$0.01"). Work it out
from the prices in the tool description (never from memory):

- Single shot (no `tools`): `input_tokens × in-price + expected_output × out-price`
  (1 token ≈ 4 chars; reasoning tokens bill as output).
- Tool loop: every round re-sends the whole conversation, so cost grows with rounds. Use
  `rounds × average_context × cached-price` plus the new tokens and output of each round.
  Past `cost_usd` values from this session are the best calibration — prefer them to theory.
- Unsure? Take the upper bound: it is what you will pass as `max_cost_usd` anyway.

**2. Above $2 you need the user's acceptance.** If the **estimate** exceeds **$2**, do NOT call yet: show the estimate, the model, and what is gained
over a cheaper run, and wait for an explicit yes. Acceptance covers that task only, not later
ones. At or below $2, proceed without asking.

**3. Always pass `max_cost_usd`**: estimate + ~30% headroom, capped at $2 unless the user
accepted more (then at most what they accepted). The server's `GROK_MAX_COST_USD` ($2.5 by
default) is a hard ceiling, not a budget: a higher `max_cost_usd` is capped to it, so a run that
needs more must be split or the operator must raise the ceiling.

**4. Always report what the job ate.** After every finished call — including a background job
collected with `get_job` — tell the user: model, `cost_usd`, and tokens
(`prompt` / `cached` / `completion` / `reasoning`), plus `iterations` and `stopped`. Do it even
when the cost is tiny, and also when a run failed or stopped early. A job still `running` has no
cost yet — say so, then report it when collected. If actual cost exceeded the estimate by more
than ~50%, say that too.

`reasoning_effort` (`low`/`medium`/`high`, some models also `none`) is per call and passed to
the API. **Support varies by model** (the catalog does not say which): some reject it, others
reject `none`. When you leave it unset, a `flagship` model runs at `low` instead of its own
(expensive) default, and any other model uses its own default; if a model refuses the
parameter the server drops the default for it. Set it explicitly (`high`) when depth
matters — an explicit value is always sent as-is, and a rejection is returned as the API error.

| Task | Model |
|------|-------|
| Quick question, summary, extraction, classification | cheapest model in the list |
| Code generation / refactoring | a code model if listed, else a cheap general model |
| Deep analysis, multi-step reasoning, review of many files | the newest flagship (highest version) |
| Input larger than ~200K tokens | a model whose context window fits — check `ctx` |
| Image input | a model marked `[images]` |
| Image generation | iterate on the cheapest image model, final render on a quality one |

---

---

## Tool → Model quick reference

### `agent(prompt, model, tools?, image_paths?, system?, conversation?, file_paths?, reasoning_effort?, max_iterations?, max_seconds?, max_cost_usd?, background=False)`

**The one tool for sending text (and images) to Grok.**

| param | default | notes |
|-------|---------|-------|
| `model` | **required** | see *Choosing a model* |
| `tools` | `[]` | workspace tools Grok may call itself: `read`, `write`, `shell`, `fetch`, `search`. Unknown name → error up front |
| `image_paths` | — | local images sent with the prompt (model must be `[images]`); must be visible to the server — see *Paths* |
| `max_iterations` | `100` | model rounds in the tool loop (server ceiling — you can only lower it) |
| `max_seconds` | `1800` | wall-clock budget of the run (server ceiling) |
| `max_cost_usd` | `2.5` | estimated USD budget of the run (server ceiling — a higher value is capped) |
| `conversation` | — (stateless) | task name `"<project>/<task>"`, e.g. `"grok-mcp/security-audit"` — keeps context across the calls of one task; Grok pulls history on demand (`get_recent_history`/`search_history`). New task → new name; one-off call → omit (no history kept; a background job still keeps its result for a few days). The project prefix matters: over HTTP the server cannot tell projects apart. ≤ 200 chars, no control characters |
| `file_paths` | — | file contents injected server-side (you pick the files) |
| `reasoning_effort` | `low` on `flagship`, else model default | per call; an explicit value is sent as-is |
| `background` | `False` | returns `{job_id}` at once; collect with `get_job(job_id)`. Not required for long runs — see *Limits* |

Unknown parameters are **rejected** with the list of valid ones.

Returns (pretty JSON):

```json
{
  "answer": "...",
  "model": "grok-4.3",
  "reasoning_effort": "low",
  "tokens": {"prompt": 6909, "cached": 576, "completion": 49, "reasoning": 141},
  "cost_usd": 0.0085,
  "tool_calls": [{"tool": "read", "args": {"path": "README.md"}, "result_chars": 22463}],
  "iterations": 2,
  "stopped": "answer",
  "hint": "only present when the run ended early or was expensive on a flagship"
}
```

`stopped` says how the loop ended: `answer` (normal), `max_iterations`, `context` (window nearly
full), `loop` (the same tool call repeated), `time` (`max_seconds`) or `cost` (`max_cost_usd`).
In every case the last round runs without tools, so `answer` always holds what Grok could
conclude from what it gathered — work is never thrown away.

#### Limits (server env vars, defaults)

| Limit | Default | Env var | What happens |
|-------|---------|---------|--------------|
| Foreground call duration | 60 s | `GROK_FOREGROUND_TIMEOUT` | the call **moves to the background**: you get `{job_id, status: "running", note}` — collect with `get_job`. Nothing is lost; you never hit a client timeout |
| Run duration | 1800 s | `GROK_JOB_MAX_SECONDS` / `max_seconds` | final answer, `stopped: "time"` |
| Run cost | $2.5 | `GROK_MAX_COST_USD` / `max_cost_usd` | final answer, `stopped: "cost"` |
| Model rounds | 100 | `GROK_MAX_ITERATIONS` / `max_iterations` | final answer, `stopped: "max_iterations"` |
| Input (prompt + file_paths + images) | 60% of the model's window | `GROK_INPUT_MAX_RATIO` | rejected **before** any API call, naming models whose window fits |
| One tool result | 15% of the window | `GROK_TOOL_RESULT_MAX_RATIO` | Grok gets "result too large — read a line range or grep" instead |
| Whole conversation | 80% of the window | `GROK_CONTEXT_MAX_RATIO` | old results trimmed once, then new ones refused, then `stopped: "context"` |
| Shell command | 240 s | `GROK_SHELL_TIMEOUT` | command killed, error returned to Grok |

#### Paths

With the Docker setup the server sees only its mounted directories (the workspace and
`ASSETS_DIR`): `image_paths`, `file_paths` and paths Grok reads must live there — **host `/tmp`
or scratchpad paths are invisible**; copy the file into the workspace first. A stdio server sees
what its user can, limited by `GROK_READ_ALLOWED_PATHS` when set. A path the server
cannot see is rejected up front with the list of visible roots.

> **Automatic project context:** when the server runs inside a git project (stdio), `agent()` prepends the detected language and framework to the system prompt. Nothing is injected in the Docker setup. If `system` already starts with `[Project Context]`, injection is skipped.

**Let Grok gather its own context.** Don't read files yourself and paste them into the
prompt — pass `tools=["read"]` (add `shell` for `git log`/`grep`/test runs) and describe *what* to
look at. File contents then never enter your context:

```
agent(prompt="Review src/grok_mcp/tools/agent.py and chat.py for duplicated logic.",
      model="<flagship from the list>", tools=["read", "shell"])
```

**Scope the reading.** Name the files (or a narrow directory) in the prompt — "read whatever you
need" over a whole repo is how a run reaches 1.7M tokens. For big files tell Grok to locate code
with `shell` `grep -n` and read only line ranges (`read` with `offset`/`limit`).

Use `file_paths` instead when you already know exactly which files matter (one round-trip, no loop).

**Long runs and cost.** Each round re-sends the whole conversation; xAI caches the unchanged
prefix automatically (`tokens.cached`, billed ~75% cheaper), so rounds mostly pay for what is new.
Trimming old results costs one cache miss. A third identical tool call is refused; a fifth ends
the loop. A long call simply comes back as `{job_id}` after 60 s — then poll `get_job`;
`background=True` only skips that first minute of waiting.

**Images.** `agent(prompt="What does this screenshot show?", model=..., image_paths=["<workspace>/shot.png"])` — the file must be in a directory the server sees (see *Paths*).
Ask one specific question per call; combine with `tools` when the image must be compared with code.

---

### `generate_image(prompt, output_path, model, aspect_ratio?, resolution?, ref_image_paths?)`

`model` is required — the tool description lists image models with their per-image price.

Cost discipline:
- Iterate with the cheapest image model until the prompt is right, then switch to a
  higher-quality model only for the final render.
- Returns `{path, model, prompt, format}`.

`output_path` must be an absolute path. Parent directories are created automatically.

With reference images (quality-tier models transfer style better):

- Pass `ref_image_paths` when style consistency with an existing image matters. It takes up to 3 absolute paths, in order — the prompt may address them positionally, e.g. *"geometry from the first, edge character from the second"*. Every path you pass is forwarded; nothing is silently dropped.
- The per-model cap (3 for the current image models) is enforced by xAI, whose message is surfaced verbatim if you exceed it.
- Reference files are MIME-sniffed from their magic bytes, so a previous result can be fed straight back in as a reference regardless of the extension it was saved under.
- `aspect_ratio` (default `"1:1"`) and `resolution` (default `"1k"`, also `"2k"`) apply as without references. The output does **not** inherit the reference image's proportions — a 16:10 ref still yields a square unless you say otherwise, so pass both explicitly whenever you need more than a 1k square.

---

## Cost comparison: your current model vs. Grok

For exact prices of your current model, check your provider's current price list rather than
hardcoding numbers here —
pricing changes between model generations. For exact current Grok prices, read the model list
in the tool description or call `list_models` — **never assume a remembered price is
still correct**; this file does not track live pricing. Every `agent()` result reports its own
`cost_usd`.

### When to suggest Grok as a cost-saving alternative

If a task will consume a large number of tokens on your current model, it's usually
cheaper to offload to Grok — but confirm with a live price lookup first,
since both sides' pricing can change. Rough decision rule:

| Estimated tokens | Suggested action |
|-----------------|-----------------|
| < 10K | Stay on your current model — not worth switching |
| 10K–50K | Stay on your current model unless user prefers Grok |
| 50K–200K | Suggest `agent()` with a cheap model, after checking live price |
| > 200K | Suggest `agent()` with a model whose context window fits; check price and `ctx` first |

### How to estimate tokens before a call

Rule of thumb: **1 token ≈ 4 characters ≈ 0.75 words**.

| Content | Approximate tokens |
|---------|-------------------|
| 1 source file (200 lines) | ~1 500 |
| 10 source files | ~15 000 |
| Full small project (50 files) | ~75 000 |
| Full medium project (200 files) | ~300 000 |
| Single conversation turn answer | 200–2 000 |
| Long research answer | 2 000–8 000 |

### How to surface this to the user

When a task looks expensive on your current model, look up live prices for both sides,
then proactively say:

> "This task involves ~X tokens. Running it here costs roughly $Y.
> Using Grok `agent()` with <model> would cost approximately $Z.
> Want me to route this through Grok instead?"

Do not switch silently — always ask first.

---

## Decision flowchart

```
User explicitly asks for Grok?
  └─ yes → honour it, pick model by task type below

Need to generate an image?
  └─ yes → generate_image(model=<cheapest image model>) while iterating,
           a quality image model for the final render

Need to analyze a local image file?
  └─ yes → agent(model=<[images] model>, image_paths=[...])

Need text / reasoning?
  ├─ Estimate token count first (see table above)
  ├─ Cost on your current model looks high? → check live prices, suggest Grok, wait for user confirmation
  ├─ Follow-up to an earlier Grok call of the same task? → reuse its conversation="<project>/<task>"
  ├─ Needs files/commands on disk? → agent(..., tools=["read", "shell"]) — don't paste files yourself
  ├─ Simple task → agent(model=<cheapest>)
  ├─ Deep analysis / many files → agent(model=<newest flagship>, background=True)
  └─ Input beyond a model's window → pick a model whose ctx fits
```

---

## Context management

### `prune_chat_context(conversation, hard?, before?)` / `list_conversations(limit?)`

Prune a conversation's history when it has grown stale or too large; `list_conversations`
shows the stored names (message count, last activity). Often simpler: start a new task name.

| Situation | Action |
|-----------|--------|
| Context feels noisy, want fresh start but keep logs | `prune_chat_context(conversation="p/t")` (soft, prunes all) |
| Prune only old messages, keep recent | `prune_chat_context(conversation="p/t", before="7d")` |
| Permanently delete messages (no audit trail needed) | `prune_chat_context(conversation="p/t", hard=True)` |
| Forgot the name | `list_conversations()` |

Soft prune (default): messages are archived — excluded from model context but retained in DB.
Hard prune: messages are permanently deleted.

History is not loaded automatically: when `conversation` is passed, `agent` gives Grok
`get_recent_history` and `search_history` tools and it pulls what it needs. Messages older than
`GROK_RETENTION_CHAT_DAYS` (default 30) are deleted automatically.

---

## Known limitations

| Limitation | Detail |
|------------|--------|
| Output is JPEG, not PNG | xAI returns JPEG whatever extension you pick and exposes no codec control; the result's `format` key names the real format |
| Use absolute image paths | Relative paths resolve against the server's working directory, not yours |
| Not every model takes images | Use a model marked `[images]` for `image_paths` |
| `reasoning_effort` support varies | some models reject it and some reject `none` — an explicit value is returned as the API error; a server-applied default is dropped automatically |
| Context windows vary by model | Check live context window via `list_models` before sending very large inputs — input larger than a model's window is rejected up front, and the error names models whose window fits |
| Cached token discount | xAI caches identical prompt prefixes automatically (~75% cheaper); `tokens.cached` in each result shows the hit |

---

## Agentic tools

Standalone tools that let Grok read/write files, run commands, and search the web without exposing intermediate results to your context. `read`/`write`/`shell`/`search` are also MCP tools of their own; inside `agent(tools=[...])` Grok can use them plus `fetch` (HTTP GET, public addresses only), with identical guards.

### `read(path, question?, model?, offset?, limit?)`

| path form | returns |
|-----------|---------|
| single path | `{path, content}` |
| list of paths | `{files: {path: content}}` |
| glob `"src/**/*.py"` | `{files: {path: content}}` — all matching files |
| single path + `offset`/`limit` | numbered lines `[lines a-b of N]` — for big files |
| any + `question` (+ required `model`) | `{answer, model}` — Grok's answer about the content |

**Path guard:** `GROK_READ_ALLOWED_PATHS` when set (the Docker setup limits it to the mounted
directories); glob matches outside it are skipped.

---

### `write(path, content, dry_run=True, backup=True)`

**Path guard:** `GROK_WRITE_ALLOWED_PATHS=/tmp,/home/you/projects` — empty list → deny all writes.
Never inside a `.git` directory (hooks there would run on the host).

| param | default | notes |
|-------|---------|-------|
| `dry_run` | `True` | Returns unified diff, does not touch disk |
| `backup` | `True` | Saves `<path>.bak` before overwriting |

Multi-file: `write([{"path": ..., "content": ...}, ...])` — validates all paths before writing any.

Raises `PermissionError` if path is outside allowed prefixes.
Raises `FileNotFoundError` if parent directory is missing.

---

### `shell(cmd, question, model, timeout_seconds?)`

No env guard — always available.

Runs `cmd` in a shell subprocess, pipes `stdout + stderr + exit code` to Grok with `question`, returns `{answer, model, exit_code}`. Output never enters your context. Timeout: `timeout_seconds`, default and maximum `GROK_SHELL_TIMEOUT` (240 s).

In the Docker server the shell has Python 3.12 (`pip install --user` works until restart), `git`, `rg`, `jq`, `curl`, `make`, `patch`, `unzip`, `file`, `ps`, `less` — no compilers or Node.

---

### `search(query, model)`

No guard. Calls xAI web search via native xai_sdk Agent Tools API (`web_search()`) on the given model, returns:

```json
{"answer": "...", "sources": [{"url": "..."}], "model": "..."}
```

Inside `agent(tools=["search"])` the search runs on the agent's own model.

Use for: current docs, CVE checks, latest versions, error diagnosis.

---

### Workspace tools inside `agent()`

`agent(prompt, tools=[...])` runs a function-calling loop: Grok decides which files to read, what
to run or search, until it has a final answer. Intermediate tool calls never surface to you.
`shell` has a 240 s timeout per command; `fetch` is an HTTP GET returning the body (subject to the
tool-result limit);
`write` defaults to `dry_run` and is limited to `GROK_WRITE_ALLOWED_PATHS`. Every call is listed
in the result's `tool_calls` (tool, args, result size, error).
