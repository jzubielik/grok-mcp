# tests/test_auth.py
import os

import pytest

from grok_mcp import auth

# In pieces, so secret scanners do not flag this test fixture.
_DSN = "postgresql" + "://u:p@h/db"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(auth, "_secrets", {})


def test_load_key_from_env_var(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-env-key")
    assert auth.load_api_key() == "xai-env-key"


def test_load_key_raises_when_missing(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="xAI API key"):
        auth.load_api_key()


def test_capture_removes_secrets_from_the_environment(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-k")
    monkeypatch.setenv("MCP_AUTH_TOKEN", "tok")
    monkeypatch.setenv("DATABASE_URL", _DSN)
    monkeypatch.setenv("PGPASSWORD", "p/w@x")
    auth.capture_secrets()
    for name in auth.SECRET_NAMES:
        assert name not in os.environ
        assert name not in auth.scrubbed_env()
    assert auth.load_api_key() == "xai-k"
    assert auth.get_secret("MCP_AUTH_TOKEN") == "tok"
    assert auth.get_secret("DATABASE_URL") == _DSN
    assert auth.get_secret("PGPASSWORD") == "p/w@x"


def test_capture_hides_process_environ_from_children(monkeypatch, tmp_path):
    """Even the kernel's copy (/proc/<pid>/environ) must be unreadable to a child process."""
    import subprocess
    import sys

    script = tmp_path / "probe.py"
    script.write_text(
        "import os, subprocess\n"
        "from grok_mcp import auth\n"
        "auth.capture_secrets()\n"
        "r = subprocess.run(['cat', f'/proc/{os.getpid()}/environ'], capture_output=True)\n"
        "print('LEAK' if b'SECRET-VALUE' in r.stdout else 'HIDDEN')\n"
    )
    out = subprocess.run(
        [sys.executable, str(script)],
        env={**os.environ, "XAI_API_KEY": "xai-SECRET-VALUE"},
        capture_output=True,
        text=True,
    )
    assert out.stdout.strip() == "HIDDEN", out.stderr


def test_credential_like_variables_are_scrubbed_too(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
    monkeypatch.setenv("SOME_API_KEY", "k")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setenv("WORKSPACE_DIR", "/w")
    env = auth.scrubbed_env()
    assert not {"POSTGRES_PASSWORD", "SOME_API_KEY", "GITHUB_TOKEN"} & env.keys()
    assert env["WORKSPACE_DIR"] == "/w"
    auth.capture_secrets()
    assert "POSTGRES_PASSWORD" not in os.environ


class _FakeOs:
    """Records the privilege calls drop_privileges makes instead of running them."""

    def __init__(self, uid=0):
        self.uid, self.calls = uid, []

    def getuid(self):
        return self.uid

    geteuid = getuid

    def setgroups(self, groups):
        self.calls.append(("setgroups", groups))

    def setgid(self, gid):
        self.calls.append(("setgid", gid))

    def setuid(self, uid):
        self.calls.append(("setuid", uid))
        self.uid = uid


@pytest.fixture
def fake_os(monkeypatch):
    fake = _FakeOs()
    for name in ("getuid", "geteuid", "setgroups", "setgid", "setuid"):
        monkeypatch.setattr(auth.os, name, getattr(fake, name))
    return fake


def test_drop_privileges_switches_root_to_the_run_uid(fake_os, monkeypatch):
    # Docker gives PID 1 (tini) the secrets; it must stay root while the server drops, or
    # /proc/1/environ is readable by the agent's shell.
    monkeypatch.setenv("GROK_RUN_UID", "1000")
    monkeypatch.setenv("GROK_RUN_GID", "1001")
    auth.drop_privileges()
    assert fake_os.calls == [("setgroups", []), ("setgid", 1001), ("setuid", 1000)]


def test_drop_privileges_refuses_root_without_a_target(fake_os, monkeypatch):
    monkeypatch.delenv("GROK_RUN_UID", raising=False)
    monkeypatch.delenv("GROK_RUN_GID", raising=False)
    with pytest.raises(RuntimeError, match="GROK_RUN_UID"):
        auth.drop_privileges()


def test_drop_privileges_refuses_uid_zero(fake_os, monkeypatch):
    monkeypatch.setenv("GROK_RUN_UID", "0")
    monkeypatch.setenv("GROK_RUN_GID", "0")
    with pytest.raises(RuntimeError, match="must not be 0"):
        auth.drop_privileges()


def test_drop_privileges_is_a_no_op_for_a_non_root_start(fake_os):
    fake_os.uid = 1000
    auth.drop_privileges()
    assert fake_os.calls == []


def test_main_drops_privileges_before_capturing_secrets(monkeypatch):
    from grok_mcp import server

    order = []
    monkeypatch.setattr(auth, "drop_privileges", lambda: order.append("drop"))
    monkeypatch.setattr(auth, "capture_secrets", lambda: order.append("capture"))
    monkeypatch.setattr(server, "check_config_outside_mounts", lambda: order.append("check"))
    monkeypatch.setattr(server, "_resolve_transport", lambda: "stdio")
    monkeypatch.setattr(server.mcp, "run", lambda transport: order.append("run"))
    server.main()
    assert order == ["drop", "capture", "check", "run"]
