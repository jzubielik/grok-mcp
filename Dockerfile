# Production image for grok-mcp
#
# Base image pinned by digest (python:3.12-slim); bump it deliberately.
#
# Multi-stage build: installs deps in builder stage, copies venv into slim final image.
#
# Build:
#   docker build -t grok-mcp .
#
# Run (stdio MCP — default):
#   docker run --rm -i -e XAI_API_KEY -e DATABASE_URL grok-mcp
#
# Run (streamable HTTP — a token is required on any non-loopback bind):
#   docker run --rm -p 127.0.0.1:8000:8000 -e XAI_API_KEY -e MCP_AUTH_TOKEN -e DATABASE_URL \
#     -e MCP_TRANSPORT=http -e MCP_HTTP_HOST=0.0.0.0 grok-mcp
#
# Long-running with the agent's shell: see README "Container" — an init as PID 1, started as
# root with GROK_RUN_UID/GROK_RUN_GID, so the init holding the secrets is not the agent's uid.

FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS builder

WORKDIR /app

RUN pip install --no-cache-dir pdm==2.29.0

COPY pyproject.toml pdm.lock ./
RUN touch README.md
COPY src/ src/

RUN pdm install --prod --no-editable

FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

# git for project change detection (git_utils); the rest is a small toolbox for the agent's
# `shell`. Compilers, Node and the like are left out — build FROM this image to add them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ripgrep jq curl make unzip patch file procps less \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

# Non-root runtime user; compose passes DOCKER_UID/DOCKER_GID so files the tools create belong
# to the host user.
ARG UID=1000
ARG GID=1000

# Private home for git and other tools, readable only by the server's uid. Compose mounts a
# tmpfs with the same mode over it.
RUN mkdir -m 700 /tmp/grok-mcp /srv/grok-assets \
    && chown "${UID}:${GID}" /tmp/grok-mcp /srv/grok-assets

# The server's venv stays off PATH: `python3` in the agent's shell is the image's own Python,
# without the server's packages, and `pip install --user` goes to the private HOME (a tmpfs,
# gone on restart).
ENV PATH="/tmp/grok-mcp/.local/bin:$PATH" \
    HOME=/tmp/grok-mcp \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

USER ${UID}:${GID}

ENTRYPOINT ["/app/.venv/bin/grok-mcp"]
