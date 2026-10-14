"""
One keep-alive HTTP session for every Sarvam call.

Every STT, TTS and translate call used to open its own `aiohttp.ClientSession`,
so each paid a DNS lookup, a TCP connect and a TLS handshake to api.sarvam.ai
before sending a byte. A mobile voice turn makes two calls (STT, then TTS), so
it paid that twice. Now one session per event loop holds the connections open
and reuses them.

The trace shows which: the `sarvam.*` span carries `new_connection: true` when
the call had to open one and `false` when it reused one.

One session per loop, not per process: an aiohttp session is bound to the
loop it was made on, and tests and scripts run several loops in a process.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from typing import Optional

import aiohttp

from observability import current_span

logger = logging.getLogger(__name__)

# Sarvam closes idle connections on its side; holding ours a little under
# that avoids sending on a socket the far end has already dropped.
KEEPALIVE_S = 60
MAX_CONNECTIONS = 32

_sessions: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, aiohttp.ClientSession]" = (
    weakref.WeakKeyDictionary())


def _mark(new: bool) -> None:
    span = current_span()
    if span is not None:
        span.set(new_connection=new)


async def _on_create(session, context, params) -> None:
    _mark(True)


async def _on_reuse(session, context, params) -> None:
    _mark(False)


def _trace_config() -> aiohttp.TraceConfig:
    config = aiohttp.TraceConfig()
    config.on_connection_create_end.append(_on_create)
    config.on_connection_reuseconn.append(_on_reuse)
    return config


def shared_session() -> aiohttp.ClientSession:
    """The running loop's Sarvam session, opened on first use."""
    loop = asyncio.get_running_loop()
    session: Optional[aiohttp.ClientSession] = _sessions.get(loop)
    if session is None or session.closed:
        session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                limit=MAX_CONNECTIONS, keepalive_timeout=KEEPALIVE_S, ttl_dns_cache=300),
            # Honour HTTPS_PROXY where a deployment sets one, as requests does.
            trust_env=True,
            trace_configs=[_trace_config()],
        )
        _sessions[loop] = session
    return session


async def close_shared_session() -> None:
    """Close the running loop's session (server shutdown)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    session = _sessions.pop(loop, None)
    if session is not None and not session.closed:
        await session.close()


class borrowed_session:
    """`async with borrowed_session() as session:` in place of a fresh
    `aiohttp.ClientSession()`: the shared session, left open afterwards."""

    async def __aenter__(self) -> aiohttp.ClientSession:
        return shared_session()

    async def __aexit__(self, *exc) -> None:
        return None
