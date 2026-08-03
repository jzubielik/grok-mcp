"""JSON Lines logging: format, data stripping, tool-call and job-id logging."""

from __future__ import annotations

import json
import logging

import pytest
from mcp.server.fastmcp.exceptions import ToolError

import grok_mcp.server as srv  # configures logging at import — before caplog attaches
from grok_mcp.logjson import JsonFormatter, current_job_id, strip_data


def _format(record_kwargs: dict | None = None, **fields) -> dict:
    record = logging.LogRecord("grok_mcp.x", logging.INFO, __file__, 1, "evt", None, None)
    if fields:
        record.fields = fields
    for k, v in (record_kwargs or {}).items():
        setattr(record, k, v)
    return json.loads(JsonFormatter().format(record))


def test_line_is_json_with_core_fields_and_extra_fields():
    line = _format(tool="agent", args={"prompt": "x" * 5000})
    assert line["event"] == "evt"
    assert line["level"] == "INFO"
    assert line["logger"] == "grok_mcp.x"
    assert line["ts"].endswith("+00:00")
    assert line["args"]["prompt"] == "x" * 5000  # never truncated


def test_job_id_is_attached_while_set():
    token = current_job_id.set("job-42")
    try:
        assert _format()["job_id"] == "job-42"
    finally:
        current_job_id.reset(token)
    assert "job_id" not in _format()


def test_strip_data_keeps_args_but_replaces_file_bodies_with_sizes():
    assert strip_data({"path": "/a", "content": "abcd", "dry_run": False}) == {
        "path": "/a",
        "content_chars": 4,
        "dry_run": False,
    }
    assert strip_data({"files": {"/a": "xx", "/b": "yyy"}}) == {"files_chars": {"/a": 2, "/b": 3}}
    nested = {"result": {"answer": "long answer", "tool_calls": [{"args": {"cmd": "ls -la"}}]}}
    assert strip_data(nested) == nested


@pytest.mark.asyncio
async def test_every_mcp_tool_call_is_logged_with_args_and_result(caplog):
    with caplog.at_level(logging.INFO, logger="grok_mcp.tools"):
        await srv.mcp._tool_manager.call_tool("get_version", {})

    events = [(r.getMessage(), getattr(r, "fields", {})) for r in caplog.records]
    assert ("tool_call", {"tool": "get_version", "args": {}}) in events
    result = [f for e, f in events if e == "tool_result"][0]
    assert result["tool"] == "get_version"
    assert "version" in result["result"]
    assert result["duration_s"] >= 0


@pytest.mark.asyncio
async def test_failing_tool_call_is_logged_as_error(caplog):
    with caplog.at_level(logging.INFO, logger="grok_mcp.tools"), pytest.raises(ToolError):
        await srv.mcp._tool_manager.call_tool("agent", {"prompt": "x", "model": "nope"})

    errors = [r for r in caplog.records if r.getMessage() == "tool_error"]
    assert errors and errors[0].fields["tool"] == "agent"
    assert errors[0].exc_info is not None


@pytest.mark.asyncio
async def test_unknown_arguments_are_rejected_not_silently_dropped(caplog):
    with caplog.at_level(logging.INFO, logger="grok_mcp.tools"), pytest.raises(ToolError) as err:
        await srv.mcp._tool_manager.call_tool(
            "agent", {"prompt": "x", "model": "m", "role": "reasoning"}
        )
    assert "role" in str(err.value) and "valid" in str(err.value)


def test_request_chatter_is_quieted():
    for name in ("mcp.server.lowlevel.server", "uvicorn.access", "httpx"):
        assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING


# --- content policy & redaction ----------------------------------------------------------

from grok_mcp.config import settings  # noqa: E402
from grok_mcp.logjson import error_text, loggable, redact  # noqa: E402

# PEM markers assembled from pieces, so secret scanners do not flag these test fixtures.
BEGIN = "-----BEGIN " + "PRIVATE KEY-----"
END = "-----END " + "PRIVATE KEY-----"
RSA_BEGIN = "-----BEGIN RSA " + "PRIVATE KEY-----"
PG = "postgresql" + "://"  # likewise for DSNs with a password


@pytest.mark.parametrize(
    ("text", "leak"),
    [
        ("key " + "xai-" + "AbCdEfGhIjKlMnOpQrStUv", "AbCdEfGhIjKlMnOp"),
        ("Authorization: Bearer abc.def.ghijklmnop", "abc.def.ghijklmnop"),
        (PG + "grok:hunter22@db:5432/x", "hunter22"),
        ("PASSWORD=hunter22 next", "hunter22"),
        ('{"api_key": "s3cr3t-value"}', "s3cr3t-value"),
        ("aws " + "AKIA" + "ABCDEFGHIJKLMNOP", "ABCDEFGHIJKLMNOP"),
        (RSA_BEGIN + "\nMIIE\n-----END RSA " + "PRIVATE KEY-----", "MIIE"),
    ],
)
def test_redact_masks_secrets(text, leak):
    assert leak not in redact(text)


def test_redact_keeps_ordinary_text():
    assert (
        redact("refactor the token parser in lexer.py") == "refactor the token parser in lexer.py"
    )


def test_formatter_redacts_every_field():
    line = _format(args={"prompt": "use " + "xai-" + "AbCdEfGhIjKlMnOpQrStUv"})
    assert "AbCdEfGhIjKlMnOp" not in json.dumps(line)


def test_metadata_mode_hides_free_text(monkeypatch):
    monkeypatch.setattr(settings, "log_content", "metadata")
    out = loggable(
        {
            "prompt": "my private question",
            "model": "grok-4",
            "max_iterations": 5,
            "file_paths": ["/w/a.py"],
            "url": "https://example.com/a?token=x",
            "result": {"answer": "secret answer", "usage": {"prompt_tokens": 10}},
        }
    )
    assert out["prompt"]["chars"] == len("my private question")
    assert len(out["prompt"]["hmac"]) == 16
    assert out["model"] == "grok-4"
    assert out["max_iterations"] == 5
    assert out["file_paths"] == ["/w/a.py"]
    assert out["url"] == "example.com"
    assert out["result"]["usage"] == {"prompt_tokens": 10}
    assert "secret answer" not in json.dumps(out)


def test_full_mode_keeps_text(monkeypatch):
    monkeypatch.setattr(settings, "log_content", "full")
    assert loggable({"prompt": "my private question"}) == {"prompt": "my private question"}


def test_error_text_and_traceback_follow_mode(monkeypatch):
    prompt = "my private question"  # runtime data, not in the raising source line
    try:
        raise ValueError(prompt)
    except ValueError as exc:
        err, info = exc, (type(exc), exc, exc.__traceback__)
    monkeypatch.setattr(settings, "log_content", "metadata")
    assert error_text(err) == "ValueError"
    tb = _format({"exc_info": info})["traceback"]
    assert "private question" not in tb and tb.endswith("ValueError")
    monkeypatch.setattr(settings, "log_content", "full")
    assert "private question" in error_text(err)
    assert "private question" in _format({"exc_info": info})["traceback"]


def test_exception_args_follow_mode(monkeypatch):
    secret = "row value: my private question"
    record = logging.LogRecord(
        "grok_mcp.x",
        logging.WARNING,
        __file__,
        1,
        "save failed for %s: %s",
        ("conv", RuntimeError(secret)),
        None,
    )
    monkeypatch.setattr(settings, "log_content", "metadata")
    assert (
        json.loads(JsonFormatter().format(record))["event"] == "save failed for conv: RuntimeError"
    )
    monkeypatch.setattr(settings, "log_content", "full")
    assert secret in json.loads(JsonFormatter().format(record))["event"]


def test_digest_is_keyed_not_a_plain_hash(monkeypatch):
    import hashlib

    monkeypatch.setattr(settings, "log_content", "metadata")
    out = loggable({"cmd": "ls -la"})["cmd"]
    assert out["hmac"] != hashlib.sha256(b"ls -la").hexdigest()[:16]
    assert loggable({"cmd": "ls -la"})["cmd"] == out  # stable within the process


def test_third_party_error_text_hidden_in_metadata_mode(monkeypatch):
    record = logging.LogRecord(
        "mcp.server.lowlevel.server",
        logging.ERROR,
        __file__,
        1,
        "Received exception from stream: input_value='my private prompt'",
        None,
        None,
    )
    monkeypatch.setattr(settings, "log_content", "metadata")
    event = json.loads(JsonFormatter().format(record))["event"]
    assert "private prompt" not in event and "hidden" in event
    monkeypatch.setattr(settings, "log_content", "full")
    assert "private prompt" in json.loads(JsonFormatter().format(record))["event"]


def test_private_key_redaction_is_linear():
    import time

    begins = BEGIN * 32000
    # No END (the worst case for a lazy match), and one END far behind every BEGIN.
    for text in (begins, begins + "x" * 100_000 + END):
        t = time.monotonic()
        redact(text)
        assert time.monotonic() - t < 0.5


def test_private_key_without_end_masks_only_the_key_body():
    # Masking to the end of the text used to drop everything after a mere mention of the
    # marker from stored chat history.
    body = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC\nBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKj"
    out = redact("a " + BEGIN + "\n" + body + "\n\nThe rest stays.")
    assert "MIIEvQ" not in out and "BKcwgg" not in out
    assert out.endswith("The rest stays.")
    prose = redact("files start with " + BEGIN + " and then base64 follows")
    assert prose.endswith(" and then base64 follows")


B64 = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKj"


def test_stray_end_line_does_not_unmask_the_next_key():
    # The END tail "PRIVATE KEY-----" also ends every BEGIN marker: an END-less key followed by
    # some other END line used to treat the next key's BEGIN as its close, leaving that key whole.
    text = (
        BEGIN + "\ncut\n-----END CERTIFICATE-----\n" + BEGIN + "\n" + B64 + "\n" + END + "\nafter"
    )
    out = redact(text)
    assert B64 not in out and out.endswith("after")


def test_cut_key_in_a_json_string_is_masked():
    # A key pasted as a JSON string ("\\n" escapes) and cut before its END line.
    text = '{"private_key": "' + BEGIN + "\\n" + B64 + "\\n" + B64[::-1]
    out = redact(text)
    assert B64 not in out and B64[::-1] not in out


def test_cut_encrypted_key_masks_headers_and_body():
    text = (
        RSA_BEGIN + "\nProc-Type: 4,ENCRYPTED\n"
        "DEK-Info: AES-128-CBC,0123456789ABCDEF\n\n" + B64 + "\n\nnext paragraph"
    )
    out = redact(text)
    assert B64 not in out and "DEK-Info" not in out and out.endswith("next paragraph")


def test_begin_before_a_far_end_masks_only_its_body():
    # A mentioned BEGIN marker must not swallow text up to a later key's END.
    text = "see " + BEGIN + " above. " + BEGIN + "\n" + B64 + "\n" + END
    out = redact(text)
    assert "above." in out and B64 not in out


@pytest.mark.parametrize(
    "text",
    [
        "It must start with " + BEGIN + "\nFix: run openssl pkcs8 -in k.pem\n"
        "Note: keep the passphrase\nThen restart.",
        BEGIN + "\napiVersion: v1\nkind: Secret\nThen restart.",
    ],
)
def test_cut_key_mention_keeps_following_prose(text):
    # Only key-looking lines are masked after an END-less BEGIN, not any `Word:` line.
    out = redact(text)
    assert out.endswith("Then restart.") and ("Note: keep" in out or "kind: Secret" in out)


@pytest.mark.parametrize(
    "line",
    ["{n}: {b}", "key.pem:{n}:{b}", "> {b}", "# {b}"],
)
def test_cut_key_with_line_prefixes_is_masked(line):
    # Numbered read output, grep -n, a markdown quote, a comment — each line has a prefix.
    lines = [line.format(n=i, b=b) for i, b in enumerate((B64, B64[::-1]), start=2)]
    out = redact("1: " + BEGIN + "\n" + "\n".join(lines))
    assert B64 not in out and B64[::-1] not in out


def test_cut_key_double_escaped_is_masked():
    # A service-account JSON embedded in another JSON string: `\\\\n` between lines.
    text = BEGIN + "\\\\n" + B64 + "\\\\n" + B64[::-1]
    out = redact(text)
    assert B64 not in out and B64[::-1] not in out


def test_redact_url_password_containing_slash():
    from grok_mcp.logjson import redact

    out = redact(PG + "grok:ab/cd@postgres:5432/db")
    assert "ab/cd" not in out and "cd@" not in out
    assert out == PG + "grok:***@postgres:5432/db"


def test_redact_url_password_containing_at():
    assert redact(PG + "grok:p/a@ss@host/db") == PG + "grok:***@host/db"


def test_url_password_redaction_is_linear():
    import time

    for text in ("a://b:" * 20000, "a://b:" + "@" * 100000, "x" * 100000 + "://u:p@h"):
        t = time.monotonic()
        redact(text)
        assert time.monotonic() - t < 0.5
