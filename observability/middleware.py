"""
ASGI middleware that opens a root trace for every HTTP request.

The channel comes from the path, so a Bolna call, a mobile request and a
WhatsApp webhook land in separate rows of the trace summary. WebSockets are left
alone: a socket lives for a whole call, so the voice handlers open one trace per
turn instead.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict

from observability.tracing import trace_request

# Longest prefix first.
_CHANNELS = (
    ("/api/mobile", "mobile"),
    ("/api/bolna", "bolna"),
    ("/api/retell", "retell"),
    ("/api/sarvam", "sarvam"),
    ("/api/query", "web"),
    ("/webhook", "whatsapp"),
    ("/api/medication", "medication"),
)
_UNTRACED = ("/health", "/admin", "/static", "/media", "/favicon", "/docs", "/openapi")


def channel_for_path(path: str) -> str:
    for prefix, channel in _CHANNELS:
        if path.startswith(prefix):
            return channel
    return "http"


class TraceMiddleware:
    """Root span per HTTP request: route template, method and status code."""

    def __init__(self, app: Callable[..., Awaitable[None]]):
        self.app = app

    async def __call__(self, scope: Dict[str, Any], receive, send) -> None:
        path = scope.get("path", "")
        if scope.get("type") != "http" or path == "/" or path.startswith(_UNTRACED):
            await self.app(scope, receive, send)
            return

        status = {"code": 0}

        async def send_with_status(message: Dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                status["code"] = message.get("status", 0)
            await send(message)

        with trace_request(channel_for_path(path), name="http",
                           method=scope.get("method", "")) as root:
            try:
                await self.app(scope, receive, send_with_status)
            finally:
                # The route template, never the concrete path: paths can carry ids.
                route = scope.get("route")
                root.set(route=getattr(route, "path", "unmatched"), status_code=status["code"])
