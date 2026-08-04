import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from grok_mcp.http_auth import BearerAuthMiddleware, check_bind_allowed, token_matches


def _app(token: str) -> TestClient:
    inner = Starlette(routes=[Route("/mcp", lambda r: PlainTextResponse("ok"), methods=["POST"])])
    return TestClient(BearerAuthMiddleware(inner, token))


def test_token_matches():
    assert token_matches("Bearer s3cret", "s3cret")
    assert not token_matches("Bearer wrong", "s3cret")
    assert not token_matches(None, "s3cret")
    assert not token_matches("s3cret", "s3cret")  # scheme required


def test_token_scheme_is_case_insensitive():
    assert token_matches("bearer s3cret", "s3cret")
    assert token_matches("BEARER s3cret", "s3cret")
    assert not token_matches("Basic s3cret", "s3cret")
    assert not token_matches("bearer S3CRET", "s3cret")  # the token itself stays exact


def test_middleware_rejects_missing_and_wrong_token():
    client = _app("s3cret")
    assert client.post("/mcp").status_code == 401
    assert client.post("/mcp", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_middleware_passes_valid_token():
    client = _app("s3cret")
    resp = client.post("/mcp", headers={"Authorization": "Bearer s3cret"})
    assert resp.status_code == 200
    assert resp.text == "ok"


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_without_token_allowed(host):
    check_bind_allowed(host, None)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10"])
def test_public_bind_without_token_refused(host):
    with pytest.raises(RuntimeError, match="MCP_AUTH_TOKEN"):
        check_bind_allowed(host, None)


def test_public_bind_with_token_allowed():
    check_bind_allowed("0.0.0.0", "x" * 32)


def test_websocket_without_token_is_rejected():
    from starlette.websockets import WebSocket, WebSocketDisconnect

    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_text("hi")
        await websocket.close()

    from starlette.routing import WebSocketRoute

    inner = Starlette(routes=[WebSocketRoute("/ws", ws_endpoint)])
    client = TestClient(BearerAuthMiddleware(inner, "s3cret"))
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws"):
        pass
    with client.websocket_connect("/ws", headers={"Authorization": "Bearer s3cret"}) as ws:
        assert ws.receive_text() == "hi"


def test_short_token_refused():
    with pytest.raises(RuntimeError, match="too short"):
        check_bind_allowed("0.0.0.0", "1")


@pytest.mark.parametrize("token", [" " * 32, "  " + "a" * 64 + "  ", "a" * 20 + " " + "b" * 20])
def test_token_with_whitespace_refused(token):
    with pytest.raises(RuntimeError, match="whitespace.*openssl rand -hex 32"):
        check_bind_allowed("0.0.0.0", token)
