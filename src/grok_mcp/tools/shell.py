"""Contained shell execution, and the shell() tool that sends a command's output to Grok."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import math
import os
import signal
import subprocess
import tempfile
from typing import BinaryIO

from ..auth import scrubbed_env
from ..client import get_text_client
from ..config import settings
from ..models_registry import require_model
from ..server import mcp

_MAX_SHELL_OUTPUT = 1_000_000  # bytes kept per stream; the rest is cut


def _read_output(f: BinaryIO) -> str:
    f.seek(0)
    data = f.read(_MAX_SHELL_OUTPUT + 1)
    text = data[:_MAX_SHELL_OUTPUT].decode("utf-8", errors="replace")
    if len(data) > _MAX_SHELL_OUTPUT:
        text += f"\n[output truncated at {_MAX_SHELL_OUTPUT} bytes]"
    return text


def run_shell(cmd: str, timeout: float) -> subprocess.CompletedProcess:
    """Run `cmd` in a shell, contained:

    - stdin is /dev/null — in stdio mode the server's stdin is the MCP channel itself;
    - the environment has no secrets (auth.scrubbed_env);
    - output goes to temp files and only the first _MAX_SHELL_OUTPUT bytes are read back, so
      `cat /dev/zero` cannot exhaust memory; undecodable bytes are replaced, not fatal;
    - on timeout the whole process group is killed, not just `sh`.
    Raises subprocess.TimeoutExpired after the kill.
    """
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            env=scrubbed_env(),
            start_new_session=True,
        )
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise
        return subprocess.CompletedProcess(
            cmd, proc.returncode, _read_output(out), _read_output(err)
        )


# Shell commands run up to GROK_SHELL_TIMEOUT each; give them their own small pool so a few
# long ones never starve the default executor (file reads, image generation, DNS for fetch).
SHELL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="shell")


async def run_shell_async(cmd: str, timeout: float) -> subprocess.CompletedProcess:
    return await asyncio.get_running_loop().run_in_executor(SHELL_EXECUTOR, run_shell, cmd, timeout)


def format_output(proc: subprocess.CompletedProcess) -> str:
    """Exit code, stdout and (when non-empty) stderr, as one block for the model."""
    text = f"[exit code: {proc.returncode}]\n[stdout]\n{proc.stdout}"
    if proc.stderr:
        text += f"\n[stderr]\n{proc.stderr}"
    return text


def _run_shell(cmd: str, timeout: float) -> str:
    """run_shell for the agent() tool loop: a timeout is reported to the model, not raised."""
    try:
        proc = run_shell(cmd, timeout)
    except subprocess.TimeoutExpired:
        return f"[error: command timed out after {timeout:g}s]"
    return format_output(proc)


async def _shell_internal(
    cmd: str,
    question: str,
    model: str,
    timeout_seconds: float | None = None,
) -> dict:
    model = await require_model(model)
    # GROK_SHELL_TIMEOUT is both the default and the ceiling.
    if timeout_seconds is not None and math.isnan(timeout_seconds):
        raise ValueError("timeout_seconds must be a number, got NaN")  # min(nan, x) is nan
    timeout = min(timeout_seconds or settings.shell_timeout, settings.shell_timeout)
    try:
        # Off the event loop: a long command must not stall other sessions or the job heartbeat.
        proc = await run_shell_async(cmd, timeout)
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Command timed out after {timeout}s: {cmd}") from None

    prompt = f"{format_output(proc)}\n\n{question}"

    client = get_text_client()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "answer": response.choices[0].message.content or "",
        "model": model,
        "exit_code": proc.returncode,
    }


@mcp.tool(
    description=(
        "Run a shell command and send its output to Grok for analysis. "
        "Output never enters the caller's context — Grok reads it directly. "
        "cmd: shell command string. question: what to ask Grok about the output. "
        "timeout_seconds: default and max GROK_SHELL_TIMEOUT (240 s). "
        "Returns {answer, model, exit_code}. "
        "model (required): language model that analyses the output — choose deliberately:\n"
        "{models}"
    )
)
async def shell(
    cmd: str,
    question: str,
    model: str,
    timeout_seconds: float | None = None,
) -> dict:
    return await _shell_internal(cmd, question, model=model, timeout_seconds=timeout_seconds)
