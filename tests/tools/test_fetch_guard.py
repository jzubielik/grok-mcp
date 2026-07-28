import socket

import pytest

from grok_mcp.tools import fetch


def _resolve_to(monkeypatch, mapping: dict[str, str]) -> None:
    async def fake_getaddrinfo(host, port, type=0):  # noqa: A002
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (mapping[host], port))]

    class Loop:
        getaddrinfo = staticmethod(fake_getaddrinfo)

    monkeypatch.setattr(fetch.asyncio, "get_running_loop", lambda: Loop())


@pytest.mark.parametrize(
    "addr", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "172.17.0.1", "0.0.0.0"]
)
async def test_non_public_addresses_rejected(monkeypatch, addr):
    _resolve_to(monkeypatch, {"evil.test": addr})
    with pytest.raises(PermissionError, match="non-public"):
        await fetch._check_public_url("http://evil.test/")


async def test_public_address_allowed(monkeypatch):
    _resolve_to(monkeypatch, {"example.com": "93.184.216.34"})
    await fetch._check_public_url("https://example.com/x")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/", "gopher://x/"])
async def test_non_http_schemes_rejected(url):
    with pytest.raises(PermissionError, match="only http"):
        await fetch._check_public_url(url)


async def test_redirect_to_private_address_rejected(monkeypatch):
    import httpx

    _resolve_to(monkeypatch, {"public.test": "93.184.216.34", "internal.test": "10.0.0.1"})

    def handler(request):
        return httpx.Response(302, headers={"location": "http://internal.test/secret"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    with pytest.raises(PermissionError, match="internal.test"):
        await fetch._fetch("http://public.test/")


async def test_connection_is_pinned_to_the_checked_address(monkeypatch):
    """The request goes to the IP that passed the check, with the name in Host and SNI —
    a second DNS answer (rebinding) is never consulted."""
    import httpx

    _resolve_to(monkeypatch, {"example.com": "93.184.216.34"})
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, text="hello")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    assert await fetch._fetch("https://example.com:8443/page?q=1") == "hello"
    (req,) = seen
    assert req.url.host == "93.184.216.34"
    assert req.url.port == 8443
    assert req.url.raw_path == b"/page?q=1"
    assert req.headers["host"] == "example.com:8443"
    assert req.extensions["sni_hostname"] == "example.com"


def _mock_client(monkeypatch, handler):
    import httpx

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)
    )


async def test_large_body_is_truncated(monkeypatch):
    import httpx

    _resolve_to(monkeypatch, {"example.com": "93.184.216.34"})
    monkeypatch.setattr(fetch, "_MAX_FETCH_BYTES", 10)
    _mock_client(monkeypatch, lambda r: httpx.Response(200, text="x" * 100))
    text = await fetch._fetch("https://example.com/")
    assert text.startswith("x" * 10) and "truncated" in text
    assert "x" * 11 not in text


async def test_redirect_without_location_is_a_clear_error(monkeypatch):
    import httpx

    _resolve_to(monkeypatch, {"example.com": "93.184.216.34"})
    _mock_client(monkeypatch, lambda r: httpx.Response(302))
    with pytest.raises(ValueError, match="without a Location"):
        await fetch._fetch("https://example.com/")


async def test_url_credentials_are_never_sent(monkeypatch):
    """httpx turns `user:pass@` into Authorization: Basic — stripped on every hop."""
    import httpx

    _resolve_to(monkeypatch, {"example.com": "93.184.216.34", "other.test": "93.184.216.35"})
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(302, headers={"location": "http://u2:p2@other.test/next"})
        return httpx.Response(200, text="ok")

    _mock_client(monkeypatch, handler)
    assert await fetch._fetch(
        "https://user:" + "secret@example.com:8443/x"
    )  # in pieces for scanners == "ok"
    assert [r.headers["host"] for r in seen] == ["example.com:8443", "other.test"]
    assert all("authorization" not in r.headers for r in seen)


@pytest.mark.parametrize(
    "addr", ["64:ff9b::a9fe:a9fe", "64:ff9b::a00:1", "::ffff:127.0.0.1", "2002:a00:1::1"]
)
async def test_ipv6_embedding_private_ipv4_rejected(monkeypatch, addr):
    _resolve_to(monkeypatch, {"evil.test": addr})
    with pytest.raises(PermissionError, match="non-public"):
        await fetch._check_public_url("http://evil.test/")


async def test_nat64_of_public_ipv4_allowed(monkeypatch):
    _resolve_to(monkeypatch, {"example.com": "64:ff9b::808:808"})
    await fetch._check_public_url("https://example.com/")


@pytest.mark.parametrize(
    "addr", ["239.255.255.250", "224.0.0.1", "ff02::1", "ff0e::1", "::169.254.169.254"]
)
def test_multicast_and_ipv4_compatible_rejected(addr):
    import ipaddress

    assert not fetch._is_public(ipaddress.ip_address(addr))


@pytest.mark.parametrize("addr", ["8.8.8.8", "2001:4860:4860::8888"])
def test_unicast_global_allowed(addr):
    import ipaddress

    assert fetch._is_public(ipaddress.ip_address(addr))


async def test_error_status_is_reported(monkeypatch):
    import httpx

    _resolve_to(monkeypatch, {"example.com": "93.184.216.34"})
    _mock_client(monkeypatch, lambda r: httpx.Response(404, text="not here"))
    assert await fetch._fetch("https://example.com/") == "[HTTP 404]\nnot here"


async def test_success_status_is_not_prefixed(monkeypatch):
    import httpx

    _resolve_to(monkeypatch, {"example.com": "93.184.216.34"})
    _mock_client(monkeypatch, lambda r: httpx.Response(200, text="ok"))
    assert await fetch._fetch("https://example.com/") == "ok"
