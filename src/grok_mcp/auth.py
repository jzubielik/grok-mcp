"""Server secrets: read once at startup, then removed from the process environment.

Commands the agent runs (`shell`) and files it reads (`/proc/self/environ`) must not see the
xAI key, the MCP bearer token or the database password — one prompt-injected run would
otherwise hand out persistent access and unbounded billing. `capture_secrets()` moves them
out of os.environ into this module; everything reads them through `get_secret()`.
"""

from __future__ import annotations

import ctypes
import os
import re
import sys

SECRET_NAMES = ("XAI_API_KEY", "MCP_AUTH_TOKEN", "DATABASE_URL", "PGPASSWORD")
# Anything that looks like a credential is removed too — compose passes the whole .env into the
# container (POSTGRES_PASSWORD and whatever else an operator adds), and a shell child must see
# none of it. The server itself only needs SECRET_NAMES.
_SECRET_LIKE = re.compile(r"PASSWORD|PASSWD|TOKEN|SECRET|_KEY$|^KEY$|CREDENTIAL|DSN|_URL$", re.I)
_secrets: dict[str, str] = {}


def is_secret_name(name: str) -> bool:
    return name in SECRET_NAMES or bool(_SECRET_LIKE.search(name))


def capture_secrets() -> None:
    """Move the secrets out of os.environ into module state, and hide this process from others.

    Popping from os.environ does not touch the environment block the kernel keeps for the
    process, which /proc/<pid>/environ shows to any process of the same user — a `shell`
    command could just read /proc/1/environ. Marking the process non-dumpable makes its
    /proc entries (environ, mem, …) inaccessible to other processes of the same uid.
    """
    for name in [n for n in os.environ if is_secret_name(n)]:
        value = os.environ.pop(name)
        if name in SECRET_NAMES and value:
            _secrets[name] = value
    _make_non_dumpable()


def drop_privileges() -> None:
    """Started as root (the compose setup), switch to GROK_RUN_UID / GROK_RUN_GID for good.

    Docker hands the container environment, secrets included, to PID 1. With an init as PID 1
    (needed to reap what the agent's shell orphans) that init would hold them, readable through
    /proc/1/environ by every process of its uid. So the container starts as root: the init runs
    as root, out of the agent's reach, and the server drops to the runtime uid before any tool
    can run. The uid change also makes the kernel mark this process non-dumpable.
    """
    if os.getuid() != 0:
        return
    uid, gid = os.environ.get("GROK_RUN_UID"), os.environ.get("GROK_RUN_GID")
    if not uid or not gid:
        raise RuntimeError("running as root: set GROK_RUN_UID and GROK_RUN_GID to drop to")
    if int(uid) == 0 or int(gid) == 0:
        raise RuntimeError("GROK_RUN_UID / GROK_RUN_GID must not be 0")
    os.setgroups([])
    os.setgid(int(gid))
    os.setuid(int(uid))
    if os.getuid() == 0 or os.geteuid() == 0:
        raise RuntimeError("could not drop root privileges")


def _make_non_dumpable() -> None:
    if not sys.platform.startswith("linux"):
        return
    pr_set_dumpable = 4
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(pr_set_dumpable, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_DUMPABLE, 0) failed")


def get_secret(name: str) -> str | None:
    """A captured secret, or — before capture (tests, library use) — the environment value."""
    return _secrets.get(name) or os.environ.get(name) or None


def scrubbed_env() -> dict[str, str]:
    """The environment for child processes, with every secret removed."""
    return {k: v for k, v in os.environ.items() if not is_secret_name(k)}


def load_api_key() -> str:
    """Return the xAI API key, or raise if it is not set."""
    if key := get_secret("XAI_API_KEY"):
        return key
    raise RuntimeError("xAI API key not found. Set the XAI_API_KEY environment variable.")
