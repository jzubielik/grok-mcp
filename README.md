# Grok MCP

An MCP server for the xAI Grok API, providing text chat, deep reasoning, vision analysis, and image generation as MCP tools usable from any MCP client.

> Independent project — not affiliated with or endorsed by xAI. "Grok" is a trademark of xAI.

## Prerequisites

- **Python 3.11+**
- **PDM** (Python package manager)
- **xAI API key** — obtain from [console.x.ai](https://console.x.ai)
- **PostgreSQL** — required (conversation history, background jobs); Docker Compose starts one.
  The server refuses to start without `DATABASE_URL`

## Setup

### 1. Install dependencies

```bash
pdm install
```

### 2. Authentication

```bash
export XAI_API_KEY=your_api_key_here
```

> Never hardcode API keys in source code or configuration files.

### 3. Run the server

**Option A: Docker Compose (recommended)** — server + PostgreSQL (conversation history, jobs),
MCP over HTTP on port 8000 — the server `.mcp.json.example` points at.

```bash
mkdir -p -m 700 ~/.config/grok-mcp
cp .env.example ~/.config/grok-mcp/.env && chmod 600 ~/.config/grok-mcp/.env
# set XAI_API_KEY, MCP_AUTH_TOKEN, POSTGRES_PASSWORD, WORKSPACE_DIR
# POSTGRES_PASSWORD may contain any characters; wrap it in single quotes if it has a `$`
# (compose expands `$VAR` in this file)
make up            # also: make down / restart / logs / ps
```

**Keep the configuration outside the workspace.** `WORKSPACE_DIR` (and `ASSETS_DIR`) are the
directories the agent works in — and everything under them is readable through `shell`, whatever
the allow-lists say. Mount only the projects the agent should touch, and keep the config file
(API key, MCP token, DB password) elsewhere: the server refuses to start when the file is inside a
mounted directory. `make up` points Compose at `~/.config/grok-mcp/.env`; use
`make up GROK_CONFIG=/path/other.env` for another one.

`WORKSPACE_DIR` is mounted into the container at the **same path**, so file paths you pass from
the host resolve 1:1. The file tools (`read`, `write`, image and audio paths) are limited by
`GROK_READ_ALLOWED_PATHS` / `GROK_WRITE_ALLOWED_PATHS`, which compose defaults to the mounts
(`WORKSPACE_DIR` plus the optional `ASSETS_DIR`). With `shell` enabled the read allow-list is not
a confidentiality boundary: a command can read anything in the container, mounts included.
`WORKSPACE_DIR` and `ASSETS_DIR` must be real paths — `make up` refuses symlinks. Ports are
published on `127.0.0.1` only. See `.env.example` for every variable.

**Option B: stdio** — each MCP client session starts its own server process. Point them all
at the same PostgreSQL (e.g. the compose `postgres` service, published on `127.0.0.1:5432`):
every session then shares conversation history and background jobs, and `list_jobs`/`get_job` see
jobs started by any of them. Processes keep their running jobs alive with a heartbeat, so one
session starting or stopping never marks another session's jobs as failed.

```bash
# The password travels apart from the URL, so it may contain any characters
export PGPASSWORD='<POSTGRES_PASSWORD>'
export DATABASE_URL=postgresql://grok@127.0.0.1:5432/context_cache
export GROK_WRITE_ALLOWED_PATHS=$HOME/projects   # write is denied everywhere until set
pdm run grok-mcp
```

## Connecting an MCP client

### HTTP (Docker Compose server)

The server listens on `http://127.0.0.1:8000/mcp` and requires the header
`Authorization: Bearer <MCP_AUTH_TOKEN>`. Most MCP clients accept a standard `mcpServers`
entry (see `.mcp.json.example`):

```json
{
  "mcpServers": {
    "grok": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp",
      "headers": {
        "Authorization": "Bearer <MCP_AUTH_TOKEN>"
      }
    }
  }
}
```

Keep a config file holding the token **outside `WORKSPACE_DIR`** — inside it, the agent's
shell could read the token. Prefer your MCP client's user-level configuration over a
per-project file.

### stdio

The client starts the server itself with the command `grok-mcp` (the absolute path
`/path/to/grok-mcp/.venv/bin/grok-mcp` after `pdm install`). A stdio server gets only the
environment you give it, so set the variables in the entry:

```json
{
  "mcpServers": {
    "grok": {
      "command": "/path/to/grok-mcp/.venv/bin/grok-mcp",
      "env": {
        "XAI_API_KEY": "xai-...",
        "PGPASSWORD": "<POSTGRES_PASSWORD>",
        "DATABASE_URL": "postgresql://grok@127.0.0.1:5432/context_cache",
        "GROK_WRITE_ALLOWED_PATHS": "/home/you/projects",
        "GROK_READ_ALLOWED_PATHS": "/home/you/projects"
      }
    }
  }
}
```

`XAI_API_KEY` and `DATABASE_URL` are required. Every session started this way shares the same
database (history, jobs) — see Setup, option B.

### Claude Code

#### Option 0: Docker Compose server (HTTP)

```bash
claude mcp add --scope user --transport http grok http://127.0.0.1:8000/mcp \
  --header "Authorization: Bearer $MCP_AUTH_TOKEN"
```

#### Option 1: Global install (stdio)

Registers the server at user scope — available in every project:

Claude Code passes a stdio server only the environment you give it, so set the variables in
the registration:

```bash
claude mcp add --scope user grok \
  -e XAI_API_KEY=xai-... \
  -e PGPASSWORD=<POSTGRES_PASSWORD> \
  -e DATABASE_URL=postgresql://grok@127.0.0.1:5432/context_cache \
  -e GROK_WRITE_ALLOWED_PATHS=$HOME/projects \
  -e GROK_READ_ALLOWED_PATHS=$HOME/projects \
  -- /path/to/grok-mcp/.venv/bin/grok-mcp
```

Replace `/path/to/grok-mcp` with the absolute path to this repository. Every session started
this way shares the same database (history, jobs) — see Setup, option B.

#### Option 2: Project-only (`.mcp.json`)

Copy `.mcp.json.example` to `.mcp.json` (gitignored) and put your `MCP_AUTH_TOKEN` in it. It
registers the Docker Compose server only for this project directory. **Only for a project
outside `WORKSPACE_DIR`** — inside it, the agent's shell could read the token; prefer the
user-scope registration (Option 0), which keeps it in `~/.claude.json`.

### Install the skill (optional)

The `grok-mcp` skill is a decision guide for agents that support skills: when and how to use
the Grok MCP tools efficiently — which model to pick, how to estimate costs, and when to route
work to Grok instead of doing it on your current model. Copy `skills/grok-mcp` into your
agent's skills directory, or:

```bash
make deploy-skills SKILLS_DIR=/path/to/your/agent/skills   # re-run after every update
```

For Claude Code, the user-level skills directory is `~/.claude/skills`
(`make deploy-skills SKILLS_DIR=$HOME/.claude/skills`).

---

## Available Tools

There are **no default models**: every tool that runs a Grok model takes a required `model`.
Each such tool's description lists the current xAI catalog (id, context window, price per 1M
tokens incl. cached price, image support), filled from the live API at server start — see
[Choosing a model](#choosing-a-model).

| Tool | Model | Description |
|------|-------|-------------|
| `agent` | required | Prompt (+ optional images) → answer, optionally with a tool loop (`read`/`write`/`shell`/`fetch`/`search`). Returns JSON with answer, tokens, cost and tool calls. Keeps history across calls of one task when you name it with `conversation`. |
| `generate_image` | required (image model) | Generate an image from a text prompt, optionally guided by up to 3 reference images. |
| `read` / `shell` / `search` | required when Grok answers | File read (+ question), command output analysis, web search. |
| `write` | — | Write one or more files (dry-run diff by default), only under `GROK_WRITE_ALLOWED_PATHS`. |
| `list_models` | — | Live xAI catalog with prices; `refresh=true` refetches, reports what changed and updates tool descriptions. |
| `get_job` / `list_jobs` | — | Collect results of `background=true` runs and of calls moved to the background after `GROK_FOREGROUND_TIMEOUT`. Jobs run inside a server process and heartbeat every 30 s; if that process dies, the job is marked `error` ("interrupted at … — the process running the job stopped; re-run it") once its heartbeat is 2 min old |
| `text_to_speech` | — | Convert text to speech (xAI `/v1/tts`), saves to output_path |
| `transcribe` | — | Transcribe an audio file (xAI `/v1/stt`) |
| `list_conversations` | — | Stored conversation names with message count and last activity. |
| `prune_chat_context` | — | Archive or delete old messages of a named conversation. |
| `get_version` | — | The server version. |

### Tool parameters

**`agent`**
- `prompt` (str) — the user message / task
- `model` (str, **required**) — xAI model id or alias; the tool description lists the choices
- `tools` (list[str], optional) — workspace tools Grok may call itself: `read`, `write`, `shell`, `fetch`, `search`. Default none. `write` is limited to `GROK_WRITE_ALLOWED_PATHS`
- `image_paths` (list[str], optional) — local images sent with the prompt (model must accept image input)
- `max_iterations` (int, optional) — model rounds in the tool loop; default `GROK_MAX_ITERATIONS` (100)
- `max_seconds` (float, optional) — wall-clock budget; default `GROK_JOB_MAX_SECONDS` (1800)
- `max_cost_usd` (float, optional) — estimated cost budget; default `GROK_MAX_COST_USD` (2.5)
- `system` (str, optional) — system prompt
- `conversation` (str, optional) — task name, e.g. `"grok-mcp/security-audit"`, to keep context across calls; omit for a stateless one-off call (see [Conversation history](#conversation-history))
- `file_paths` (list[str], optional) — absolute paths to files whose contents are prepended to the prompt server-side (avoids double token cost)
- `reasoning_effort` (str, optional) — `low`/`medium`/`high` (some models also `none`), passed to the API as-is; support varies by model, unset = `low` on the model tagged flagship, otherwise the model's own default
- `background` (bool, default false) — return `{job_id}` immediately; collect with `get_job`

Returns `{answer, model, reasoning_effort, tokens{prompt,cached,completion,reasoning}, cost_usd,
tool_calls, iterations, stopped}`. `stopped` is `answer`, `max_iterations`, `context`, `loop`,
`time` or `cost`; in every case a final tool-less round answers from what was gathered, so work
is never lost. The message prefix is append-only, so xAI's automatic prompt cache covers most of
each round (`tokens.cached`). Unknown arguments are rejected, not ignored.

**Limits** (env vars, `GROK_` prefix):

| Variable | Default | Effect |
|----------|---------|--------|
| `GROK_FOREGROUND_TIMEOUT` | `60` | a foreground `agent` call still running after this many seconds is moved to a background job and returns `{job_id}` (collect with `get_job`) |
| `GROK_JOB_MAX_SECONDS` | `1800` | run time budget → `stopped: "time"` (ceiling; a call may lower it with `max_seconds`) |
| `GROK_MAX_COST_USD` | `2.5` | run cost budget → `stopped: "cost"` (ceiling; a call may lower it with `max_cost_usd`; models without a live price are costed at a conservative $5/$15 per 1M tokens) |
| `GROK_MAX_ITERATIONS` | `100` | model rounds → `stopped: "max_iterations"` (ceiling; a call may lower it with `max_iterations`) |
| `GROK_INPUT_MAX_RATIO` | `0.6` | prompt + `file_paths` + images above this share of the model's window are rejected up front |
| `GROK_TOOL_RESULT_MAX_RATIO` | `0.15` | a single tool result above this share is replaced by "read a range / grep" instructions |
| `GROK_CONTEXT_MAX_RATIO` | `0.8` | conversation ceiling: old results trimmed once, then new ones refused, then `stopped: "context"` |
| `GROK_CHARS_PER_TOKEN` | `3` | estimate for text the API has not counted yet |
| `GROK_SHELL_TIMEOUT` | `240` | seconds per shell command (agent tool and `shell()`) |

In the Docker image, `shell` commands find Python 3.12 (the image's own, without the server's
packages; `pip install --user` works and lasts until the container restarts), `git`, `rg`,
`jq`, `curl`, `make`, `patch`, `unzip`, `file`, `ps` and `less`. For compilers, Node or
anything else, build your own image `FROM` this one.

Paths in `image_paths` / `file_paths` must be visible to the server — in Docker only the
mounted `WORKSPACE_DIR` and `ASSETS_DIR`; anything else is rejected up front.

**`generate_image`**
- `prompt` (str) — description of the image to generate
- `output_path` (str) — where to save the result
- `model` (str, **required**) — image model id; the tool description lists them with per-image price
- `aspect_ratio` (str, optional) — e.g. `"1:1"`, `"16:9"` (default `"1:1"`)
- `resolution` (str, optional) — `"1k"` (default) or `"2k"`
- `ref_image_paths` (list[str], optional) — absolute paths to reference images, in order; the prompt may address them positionally (e.g. *"geometry from the first, edge character from the second"*). Quality-tier models transfer style better.

The tool writes the model's bytes to `output_path` verbatim. xAI currently returns **JPEG** no matter which extension you pick, and the API offers no codec control — its `format` field selects the transport (base64 or URL), not the image format. The JSON result therefore carries a `format` key naming what was actually written; trust it over the file suffix.

With references, the output does **not** inherit the reference image's proportions — pass `aspect_ratio` and `resolution` explicitly for anything other than a 1k square.

The server forwards every reference path you pass. The cap on how many a model accepts (3 for the current image models) is enforced by xAI, and its message is surfaced verbatim rather than silently truncating the list. Reference files are MIME-sniffed from their magic bytes, so a `.png` that actually holds JPEG — which is what the model returns — is sent with the right content type.

**`text_to_speech`**
- `text` (str) — the text to convert to speech
- `output_path` (str) — where to save the audio file
- `voice` (str, optional) — built-in `eve`, `ara`, `rex`, `sal`, `leo` or a custom voice id (default: `eve`)
- `language` (str, optional) — `auto` or a BCP-47 code such as `en`, `de`, `pt-BR` (default: `auto`)
- `codec` (str, optional) — `mp3`, `wav`, `pcm`, `mulaw`, `alaw` (default: `mp3`)
- `speed` (float, optional) — 0.7–1.5 (default: 1.0)
- up to 60,000 characters; speech tags such as `[pause]` and `[laugh]` work

**`transcribe`**
- `audio_path` (str) — absolute path to the audio file (wav, mp3, ogg, opus, flac, aac, mp4, m4a, mkv; up to `GROK_MAX_READ_BYTES`, 20 MB by default)
- `language` (str, optional) — language code (e.g. `"en"`) to bias recognition; detected when omitted
- returns `{text, language, duration}`

**`list_conversations`**
- `limit` (int, optional) — 1-100, default 20; most recently active first
- returns `{conversations: [{conversation, messages, last_activity}]}`

**`prune_chat_context`**
- `conversation` (str, **required**) — the conversation to prune (the name passed to `agent`)
- `hard` (bool, optional) — `False` (default) archives messages (excluded from future loads but kept for audit); `True` permanently deletes them
- `before` (str, optional) — prune only messages older than this cutoff. Accepts a duration (`"24h"`, `"7d"`, `"30m"`) or an ISO 8601 UTC timestamp. Omit to prune all messages in the conversation.

## Choosing a model

The server keeps no model defaults — the caller picks the model deliberately for each
call. The catalog is fetched live from xAI
(`/v1/language-models`, `/v1/image-generation-models`, `/v1/models`), cached for
`GROK_MODELS_TTL` seconds (default 24h), and written into the description of every tool that
takes a model, e.g.:

```
grok-4.7 [images]: 500K ctx, per 1M tokens in $2.00 (cached $0.50) / out $6.00
grok-4.3 [images]: 1000K ctx, per 1M tokens in $1.25 (cached $0.20) / out $2.50
```

Rule of thumb: the cheapest model that can do the job; the newest flagship for deep analysis;
a model whose context window fits for very large inputs. `list_models(refresh=true)` refetches the
catalog, reports what changed and refreshes the tool descriptions. An unknown model
fails before any API call, listing the valid ids.

### Audio models

TTS and STT use xAI's native `/v1/tts` and `/v1/stt` endpoints, each with a single model, so
they take no `model` argument. See [xAI pricing](https://docs.x.ai/docs/models) for current
rates.

## Conversation history

History is scoped to a **task** the caller names. Pass `conversation="<project>/<task>"`
(e.g. `"grok-mcp/security-audit"`) to keep context across the calls of one task; use a new name
for a new task; omit `conversation` for a one-off call. The server cannot tell projects apart
over HTTP (one server serves every client), hence the project prefix in the name.

- **With `conversation`:** each exchange (the prompt and Grok's final answer — tool results are
  not stored) is saved under exactly that name. History is **not** injected into the call: the
  model gets two tools — `get_recent_history` (the latest messages) and `search_history` (a text
  search) — and pulls what it needs. Archived messages are excluded.
- **Without it:** no history is kept or offered. (A run that becomes a background job still
  keeps its result for `GROK_RETENTION_JOB_CONTENT_DAYS` — see [Data & privacy](#data--privacy).)

A name is at most 200 characters, not blank, without control or invisible format characters;
surrounding whitespace is dropped. `list_conversations`
shows the stored names with their message count and last activity.

Use `prune_chat_context(conversation=...)` to archive (`hard=False`) or delete (`hard=True`)
messages, e.g. after a large refactor or to honour an erasure request. Independently, messages
older than `GROK_RETENTION_CHAT_DAYS` (default 30) are deleted automatically — see
[Data & privacy](#data--privacy).

History is stored in the `chat_history` table.

## Logs

The server logs **JSON Lines** to stderr — one JSON object per line, from every component
(grok-mcp, FastMCP, uvicorn, httpx). Each line has `ts` (UTC), `level`, `logger`, `event` and
event fields; lines logged during a background job also carry `job_id`.

**Content policy — `GROK_LOG_CONTENT`:**

| Mode | What tool arguments and results contain |
|------|------------------------------------------|
| `metadata` (default) | identifiers, settings, paths and numbers verbatim (`model`, `tools`, `file_paths`, token usage, timing); every other string — prompts, answers, shell commands — as `{"chars": N, "hmac": "<16 hex>"}`, keyed with a random per-process secret (groupable within one run, not reversible by hashing guesses, not linkable across restarts); URLs as their host; exceptions as their type; tracebacks without the message line |
| `full` | everything verbatim — for debugging only |

In both modes credentials are masked on every line (xAI/OpenAI/GitHub/AWS keys, bearer tokens,
`password=`/`token=`/`api_key=` values, private keys, passwords in URLs). File bodies (`write`
content, `read` results) are always logged as sizes.

| `event` | Fields |
|---------|--------|
| `tool_call` | `tool`, `args` |
| `tool_result` | `tool`, `duration_s`, `result` |
| `tool_error` | `tool`, `duration_s`, `error`, `traceback` |
| `agent_tool` | a tool Grok called inside `agent`: `tool`, `args`, `result_chars`, `error`, `model`, `round` |
| `agent_context_trimmed` | old tool results replaced by stubs (`freed_chars` or `prompt_tokens`/`context_window`) |
| `agent_context_overflow` | the API rejected the input as too large; the run ends with a final answer |
| `orphaned_jobs_failed` | jobs whose process stopped heartbeating, marked `error` (`count`) — checked at startup, on `get_job`/`list_jobs` and in the 24 h retention purge |
| `retention_purge` | rows deleted / cleared per retention setting |

```bash
docker compose logs -f --no-log-prefix grok-mcp | jq 'select(.event | startswith("tool") or startswith("agent"))'
docker compose logs --no-log-prefix grok-mcp | jq 'select(.job_id == "<job_id>")'
```

Docker rotates the container log at 50 MB × 5 files — that bounds disk use, not time. Set a
time-based retention in your log pipeline (journald `MaxRetentionSec`, Loki
`retention_period`, ...); 30–90 days is a common choice for operational logs.

## Data & privacy

**What is stored** (PostgreSQL):

| Table | Content | Default retention |
|-------|---------|-------------------|
| `chat_history` | conversation messages — the prompt as written and the answer; files passed via `file_paths` are sent to xAI but only their paths are stored | 30 days — `GROK_RETENTION_CHAT_DAYS` |
| `jobs` — `result`, `error` | background job result and error text | cleared 7 days after the job ends — `GROK_RETENTION_JOB_CONTENT_DAYS` |
| `jobs` — other columns | status, model, token usage, timing | 365 days — `GROK_RETENTION_JOBS_DAYS` |
| `project_discovery` | project paths, git hash, detected language | 90 days since last refresh — `GROK_RETENTION_PROJECTS_DAYS` |

The purge runs at startup and every 24 hours; `0` keeps that data forever. Unambiguous
credential formats (API keys, bearer tokens, private keys, URL passwords) are masked before
text is stored. Delete a conversation at any time with `prune_chat_context` (e.g. to honour an
erasure request).

**Not covered by the purge:** generated images and audio written to your directories, and the
container logs (see [Logs](#logs)).

**Third party.** Prompts, files you attach and tool results are sent to the xAI API and
processed under [xAI's terms and data policy](https://x.ai/legal). If you run this server for
other people, you are the data controller: document the processing, set retention periods that
fit your purpose, and have a data processing agreement with xAI where the law requires one
(e.g. GDPR Art. 28).

## Security

- **Command execution:** `shell` and `agent(tools=["shell"])` run arbitrary commands with the
  server's privileges — anyone who can reach the MCP endpoint can run code wherever the server
  runs. Keep the endpoint on loopback or behind `MCP_AUTH_TOKEN`, run it in the provided
  container, and mount only the directories it needs. An agent that combines `fetch`
  (untrusted web content) with `shell` is exposed to prompt injection.
- **HTTP transport:** the server refuses to bind a non-loopback address without
  `MCP_AUTH_TOKEN`; with a token set, every request needs `Authorization: Bearer <token>`.
  Without a token only `Host: localhost/127.0.0.1` is accepted (DNS-rebinding protection).
  With a token, any host name (remote or behind a proxy) is accepted — the token is the guard.
- **Files:** `write`, `generate_image*` and `text_to_speech` outputs only under
  `GROK_WRITE_ALLOWED_PATHS` (empty → denied); `read`, `file_paths`, image and audio inputs
  only under `GROK_READ_ALLOWED_PATHS` when set.
- **`fetch`** (inside `agent`) only reaches unicast global addresses — loopback, private,
  link-local (cloud metadata), multicast, reserved and IPv6 forms embedding such an IPv4 address
  are refused on every hop. The connection is
  pinned to the address that was checked, so DNS rebinding cannot swap it. Bodies are capped
  at 5 MB.
- **Container:** the server runs as your uid with no capabilities, `no-new-privileges`, private
  `HOME` on a `0700` tmpfs, and `init: true` so tini reaps the orphans background commands
  leave behind. Docker gives PID 1 the whole environment, secrets included, so tini must not
  run under the server's uid (`/proc/1/environ` would hand them to `shell`): the container
  starts as root with only `SETUID`/`SETGID`, tini stays root, and the server drops to
  `GROK_RUN_UID`/`GROK_RUN_GID` before any tool runs. Running the image elsewhere, keep both
  halves: `docker run --init --user 0 -e GROK_RUN_UID=$(id -u) -e GROK_RUN_GID=$(id -g)
  --cap-drop ALL --cap-add SETUID --cap-add SETGID ...`. `make verify-secrets` checks a
  running stack. Debug with a plain `docker compose exec` (root): `docker exec` hands every
  session the container environment, and one opened as the server's uid (`-u 1000`) shows it
  to `shell` for as long as it runs.
- **Secrets stay with the server:** `XAI_API_KEY`, `MCP_AUTH_TOKEN`, `DATABASE_URL` and `PGPASSWORD` are
  removed from the process environment at startup — commands run by `shell` get an environment
  without them, the server process is non-dumpable (its `/proc` entries are closed to other
  processes), and the file tools never read `/proc`, `/sys` or `/dev`. `MCP_AUTH_TOKEN` must be at
  least 32 characters with no whitespace (`openssl rand -hex 32`).
- **Bounded resources:** files the server loads are regular files within `GROK_MAX_READ_BYTES`
  (20 MB), one budget per call — summed over all files of a `read` and over all `file_paths` +
  `image_paths` of an `agent` call; `read(offset/limit)` caps the returned range, not the file
  size; `shell` keeps 1 MB of output per stream, gets no stdin and is killed with its whole
  process group on timeout; compose caps memory and processes (`GROK_MEM_LIMIT`,
  `GROK_PIDS_LIMIT`).
- **Writes never follow a symlink** at the target (including the `.bak` backup); reads and
  writes check the file descriptor actually opened, so swapping a directory for a symlink
  between check and use gets nowhere. `read` and `file_paths` refuse `.env` / `.env.*` files
  — a name guard for the file tools only: `shell` can read any file under the mounts.
- **Keep secrets out of reach:** with Docker, keep the config file (`~/.config/grok-mcp/.env`) **outside**
  `WORKSPACE_DIR` / `ASSETS_DIR` — the server refuses to start otherwise — and likewise any
  `.mcp.json` holding the MCP token. With stdio,
  the keys you pass to a stdio server live in your MCP client's config (for Claude Code:
  `~/.claude.json`), which the server running as you can read; prefer the Docker setup
  when that matters.
- **Anything written into a mount can run on the host:** git hooks, `.envrc` (direnv), IDE
  tasks and similar run with your user's rights the next time you work in that tree. The write
  tools refuse paths inside a `.git/` directory; `shell` is not limited — review what the agent
  leaves in a mounted repo before running tools there.
- `shell` is unrestricted by design; the boundary is the container and its mounts.

### Network isolation (recommended on cloud hosts)

The `fetch` checks protect one tool. `shell` can reach anything the container can — on a cloud
VM that includes the instance metadata service at `169.254.169.254`, which hands out the
credentials of the role attached to the machine. A prompt-injected agent could read them and
act on your cloud account. Block it at the network level, where every process is covered:

- **AWS:** require IMDSv2 with a hop limit of 1 — containers behind Docker's bridge then
  cannot reach the metadata service:
  `aws ec2 modify-instance-metadata-options --instance-id <id> --http-tokens required --http-put-response-hop-limit 1`
- **Any host — iptables:** drop forwarded container traffic to the metadata service and your
  LAN in the `DOCKER-USER` chain (container-to-container traffic on the compose network is
  not affected):
  ```bash
  iptables -I DOCKER-USER -d 169.254.0.0/16 -j DROP
  iptables -I DOCKER-USER -d 192.168.1.0/24 -j DROP   # your LAN range
  ```
- **Strictest — egress proxy:** route outbound traffic through an allow-list proxy (e.g.
  Squid, [Smokescreen](https://github.com/stripe/smokescreen)) that permits `api.x.ai` and the
  domains your agents need.
- **Least privilege:** give the VM no instance role, or one without access to anything that
  matters.

On a workstation the metadata address usually leads nowhere, but the same rules keep agents off
your LAN.

Report vulnerabilities as described in [SECURITY.md](SECURITY.md).

## Implementation Notes

- **Image generation uses gRPC, not REST.** xAI's REST API endpoint `POST /images/edits` returns `403`. All image generation goes through the `xai_sdk` gRPC client (`client.image.sample()`).
- **Reference images go in `image_urls`.** The gRPC request takes a list of data URIs under `image_urls`, in order. 
- **Text and vision use the OpenAI-compatible API** at `https://api.x.ai/v1`.

## Development

Run tests:

```bash
pdm run pytest
```

Lint and format:

```bash
pdm run ruff check src tests
pdm run ruff format src tests
```

## License

MIT — see [LICENSE](LICENSE).
