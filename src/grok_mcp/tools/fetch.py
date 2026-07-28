"""fetch for the agent() tool loop: HTTP GET behind an SSRF guard, size and time caps."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Any
from urllib.parse import urljoin, urlsplit

_MAX_REDIRECTS = 5
_MAX_FETCH_BYTES = 5 * 1024 * 1024
_NAT64_PREFIXES = (ipaddress.ip_network("64:ff9b::/96"), ipaddress.ip_network("64:ff9b:1::/48"))


def _is_public(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """is_global, plus IPv6 forms that embed an IPv4 address the gateway would route to —
    `64:ff9b::a9fe:a9fe` reaches 169.254.169.254 through a NAT64 yet counts as global."""
    # is_global is True for most multicast (e.g. ff0e::1), which no fetch has business with.
    if not addr.is_global or addr.is_multicast:
        return False
    if isinstance(addr, ipaddress.IPv6Address):
        embedded = [addr.ipv4_mapped, addr.sixtofour, *(addr.teredo or ())]
        # Deprecated IPv4-compatible `::a.b.c.d` (`::` and `::1` already fail is_global).
        ipv4_compat = addr.packed[:12] == bytes(12)
        if ipv4_compat or any(addr in net for net in _NAT64_PREFIXES):
            embedded.append(ipaddress.IPv4Address(addr.packed[-4:]))
        return all(e is None or e.is_global for e in embedded)
    return True


def _strip_userinfo(url: str) -> str:
    """`url` without `user:pass@` — httpx would turn it into an Authorization header and send
    credentials the model put (or a redirect smuggled) into a URL to whatever host it names."""
    import httpx  # noqa: PLC0415

    return str(httpx.URL(url).copy_with(userinfo=b""))


async def _check_public_url(url: str) -> str:
    """Return the address to connect to for `url`, or raise PermissionError.

    `url` must be http(s) and every address its host resolves to must be public — keeps fetch
    away from localhost, the LAN, the Docker network and cloud metadata (169.254.169.254).
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise PermissionError(f"fetch: only http(s) URLs are allowed: {url}")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    infos = await asyncio.get_running_loop().getaddrinfo(
        parts.hostname, port, type=socket.SOCK_STREAM
    )
    addrs = [ipaddress.ip_address(info[4][0].split("%")[0]) for info in infos]
    for addr in addrs:
        if not _is_public(addr):
            raise PermissionError(f"fetch: {parts.hostname} resolves to non-public {addr}")
    if not addrs:
        raise PermissionError(f"fetch: {parts.hostname} does not resolve")
    return str(addrs[0])


_FETCH_DEADLINE_S = 60


async def _fetch(url: str) -> str:
    """One fetch, at most _FETCH_DEADLINE_S in total — httpx's timeout is per operation, so a
    server dripping a byte every few seconds could otherwise hold the call for hours."""
    async with asyncio.timeout(_FETCH_DEADLINE_S):
        return await _fetch_unbounded(url)


async def _fetch_unbounded(url: str) -> str:
    """HTTP GET following redirects by hand; every hop is checked by _check_public_url.

    The connection goes to the very address that was checked — resolving the name a second
    time at connect would let a DNS server answer "public" for the check and "private" for
    the connection (DNS rebinding). The name still travels in Host and TLS SNI, so virtual
    hosts work and the certificate is verified against the name, not the IP.
    """
    import httpx  # noqa: PLC0415

    async with httpx.AsyncClient(follow_redirects=False, timeout=30) as client:
        for _ in range(_MAX_REDIRECTS + 1):
            url = _strip_userinfo(url)
            ip = await _check_public_url(url)
            target = httpx.URL(url)
            request = client.build_request(
                "GET",
                target.copy_with(host=ip),
                headers={"Host": target.netloc.decode("ascii")},
                extensions={"sni_hostname": target.host},
            )
            resp = await client.send(request, stream=True)
            try:
                if not resp.is_redirect:
                    body = await _read_capped(resp)
                    # Without the status an error page reads like real content to the model.
                    if resp.status_code >= 400:
                        return f"[HTTP {resp.status_code}]\n{body}"
                    return body
                location = resp.headers.get("location")
            finally:
                await resp.aclose()
            if not location:
                raise ValueError(f"fetch: {resp.status_code} redirect without a Location header")
            url = urljoin(url, location)
    raise ValueError(f"fetch: more than {_MAX_REDIRECTS} redirects")


async def _read_capped(resp: Any) -> str:
    """Body as text, at most _MAX_FETCH_BYTES — a huge download must not exhaust memory."""
    chunks: list[bytes] = []
    size = 0
    async for chunk in resp.aiter_bytes():
        chunks.append(chunk)
        size += len(chunk)
        if size > _MAX_FETCH_BYTES:
            body = b"".join(chunks)[:_MAX_FETCH_BYTES].decode(resp.encoding or "utf-8", "replace")
            return body + f"\n[truncated: response larger than {_MAX_FETCH_BYTES} bytes]"
    return b"".join(chunks).decode(resp.encoding or "utf-8", "replace")
