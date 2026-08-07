# Security Policy

## Reporting a vulnerability

Please report vulnerabilities privately through the repository's **Security → Report a
vulnerability** form (GitHub private vulnerability reporting). Do not open a public issue.

Include the affected version or commit, the configuration (transport, which tools are
enabled, how the server is exposed) and steps to reproduce. You can expect an acknowledgement
within 7 days.

## Scope

This server is designed to give an AI agent shell access and local file access. The following
are **expected behaviour**, not vulnerabilities:

- `shell` running any command, inside the server's container or user account;
- a prompt-injected agent doing what its enabled tools allow;
- any holder of `MCP_AUTH_TOKEN` reading or deleting every stored conversation and job — the
  server is single-tenant: one token is full access.

The server does keep its own secrets (`XAI_API_KEY`, `MCP_AUTH_TOKEN`, `DATABASE_URL`,
`PGPASSWORD`) away from
the agent: they are removed from the process environment at startup, and in the container the
init holding Docker's environment runs as root while the server drops to its own uid, so neither
commands run by `shell` nor reads of `/proc` can reach them. A way around that is in scope.

Out of scope: secrets an operator places where the agent may read them — a `docker exec`
session opened as the server's uid (it carries the container environment; exec as root), any
file under the mounted directories (the `.env` name guard covers the `read` / `file_paths`
tools only; `shell` reads any file there), or, in stdio mode, the MCP client's own config file holding the keys
(the server runs as that user). Likewise files the agent writes into a mount that later run on
the host (git hooks, `.envrc`, IDE tasks): the write tools refuse `.git/`, but `shell` can
write anywhere in the mounts by design.

In scope, for example: bypassing `MCP_AUTH_TOKEN`, escaping
`GROK_WRITE_ALLOWED_PATHS` / `GROK_READ_ALLOWED_PATHS` through the file tools (with `shell`
enabled the read allow-list is not a confidentiality boundary), writing into `.git/` through
them, reaching non-public addresses through
`fetch`, secrets or prompt content appearing in logs under `GROK_LOG_CONTENT=metadata`, and
stored data outliving its configured retention.

## Supported versions

Only the latest release receives security fixes.
