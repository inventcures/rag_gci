"""
One authentication policy for every route on the main server.

Every route declares one of three policies in ``ROUTE_POLICIES``:

  * ``public``      - no token. Health, the release gate, the static web client,
                      and the webhooks and provider sockets, which carry their own
                      verification (the ``note`` says whose).
  * ``participant`` - any authenticated person: a mobile access token, or an
                      admin token.
  * ``admin``       - an authenticated person with the admin role. Admin, study,
                      clinical-data and upload routes.

``enforce_route_policy`` is the one dependency that applies the table. The
server installs it app-wide, so a route added without a row in the table is
treated as admin (fail closed) and the route-table test fails.

Admin tokens are signed with ``ADMIN_JWT_SECRET``, a key the mobile API never
uses, so registering on the app with ``role: "admin"`` cannot make anyone an
administrator. Mint one with ``python scripts/issue_admin_token.py <name>``. The
token's subject is the person: the voice-provider override records it as the
actor. Send it as ``Authorization: Bearer <token>``, or, from a browser, as the
password of the Basic-auth prompt (the user name is ignored).
"""

import base64
import binascii
import functools
import logging
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import jwt
import yaml
from fastapi import HTTPException, WebSocketException, status
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse
from starlette.routing import Mount
from starlette.types import ASGIApp, Receive, Scope, Send

from auth.jwt_handler import ACCESS
from auth.middleware import get_jwt_handler
from auth.models import AuthenticatedUser

logger = logging.getLogger(__name__)

ADMIN_SECRET_ENV = "ADMIN_JWT_SECRET"
ADMIN_ROLE = "admin"
ADMIN_TOKEN_TYPE = "admin"
ADMIN_TOKEN_ISSUER = "palli-sahayak-admin"
ADMIN_SECRET_MIN_LENGTH = 32
CORS_ORIGINS_ENV = "CORS_ALLOWED_ORIGINS"

_CHALLENGE = {"WWW-Authenticate": 'Bearer, Basic realm="Palli Sahayak"'}


class Access(str, Enum):
    """Who may call a route."""

    PUBLIC = "public"
    PARTICIPANT = "participant"
    ADMIN = "admin"


@dataclass(frozen=True)
class RoutePolicy:
    """A route's declared access, and for a public route who verifies it instead."""

    access: Access
    note: str = ""


PUBLIC = RoutePolicy(Access.PUBLIC)
PARTICIPANT = RoutePolicy(Access.PARTICIPANT)
ADMIN = RoutePolicy(Access.ADMIN)


def own_verification(who: str) -> RoutePolicy:
    """A public route that checks its caller itself (webhook signature, provider key)."""
    return RoutePolicy(Access.PUBLIC, f"own verification: {who}")


# Keys are route path templates exactly as FastAPI holds them (with the include
# prefix). A key ending in "*" covers every path under it; an exact key wins over
# a prefix, and a longer prefix over a shorter one.
ROUTE_POLICIES: Dict[str, RoutePolicy] = {
    # -- public --------------------------------------------------------------
    "/": PUBLIC,
    "/health": PUBLIC,
    "/docs": RoutePolicy(Access.PUBLIC, "API schema only"),
    "/docs/oauth2-redirect": RoutePolicy(Access.PUBLIC, "API schema only"),
    "/redoc": RoutePolicy(Access.PUBLIC, "API schema only"),
    "/openapi.json": RoutePolicy(Access.PUBLIC, "API schema only"),
    "/voice": RoutePolicy(Access.PUBLIC, "static web voice client"),
    "/api/voice/models": RoutePolicy(Access.PUBLIC, "read by the web voice client"),
    "/api/voice/languages": RoutePolicy(Access.PUBLIC, "the web voice client's language picker"),
    "/ws/voice": RoutePolicy(Access.PUBLIC, "web Live voice; a participant token on the header "
                                            "names the caller, else the session is ephemeral"),
    "/api/pageindex/health": PUBLIC,
    "/api/retell/health": PUBLIC,
    "/api/sarvam/health": PUBLIC,
    "/api/sarvam/languages": PUBLIC,
    "/api/kg/health": PUBLIC,
    "/api/graphrag/health": PUBLIC,
    "/api/temporal/health": PUBLIC,
    # -- webhooks and provider sockets: their own verification (ticket 07) ---
    "/api/bolna/webhook": own_verification("Bolna signature and key"),
    "/api/bolna/query": own_verification("Bolna signature and key"),
    "/api/medication/voice-reminder/callback": own_verification("phone provider callback"),
    "/api/retell/webhook": own_verification("Retell signature"),
    "/ws/retell/llm/{call_id}": own_verification("Retell custom-LLM socket"),
    "/api/sarvam/webhook": own_verification("Sarvam signature"),
    "/webhook": own_verification("X-Twilio-Signature"),
    "/media/{filename}": RoutePolicy(Access.PUBLIC, "fetched by Twilio; unguessable file name"),
    # -- mobile API: the app's participants ------------------------------------
    "/api/mobile/v1/*": PARTICIPANT,
    "/api/mobile/v1/health": PUBLIC,
    "/api/mobile/v1/release/status": RoutePolicy(Access.PUBLIC, "release gate, read before sign-in"),
    "/api/mobile/v1/auth/register": PUBLIC,
    "/api/mobile/v1/auth/login": RoutePolicy(Access.PUBLIC, "own verification: PIN"),
    "/api/mobile/v1/auth/refresh": RoutePolicy(Access.PUBLIC, "own verification: the refresh token (ticket 08)"),
    "/api/mobile/v1/ws/voice": RoutePolicy(
        Access.PARTICIPANT, "access token on the upgrade request; turns recorded under it (ticket 22)"
    ),
    # -- participant-facing answers outside the app ----------------------------
    "/api/query": PARTICIPANT,
    "/api/sarvam/stt": PARTICIPANT,
    "/api/sarvam/tts": PARTICIPANT,
    "/api/sarvam/translate": PARTICIPANT,
    # -- admin, study, clinical data, upload -------------------------------------
    "/admin": RoutePolicy(Access.ADMIN, "Gradio admin panel, gated as a mount"),
    "/admin/study/*": ADMIN,
    "/api/debug": ADMIN,
    "/api/upload": ADMIN,
    "/api/upload/{job_id}": ADMIN,
    "/api/pageindex/*": ADMIN,
    "/api/bolna/stats": ADMIN,
    "/api/bolna/calls": ADMIN,
    "/api/medication/*": ADMIN,
    "/api/retell/stats": ADMIN,
    "/api/retell/calls": ADMIN,
    "/api/retell/vobiz": ADMIN,
    "/api/sarvam/stats": ADMIN,
    "/api/kg/*": ADMIN,
    "/api/graphrag/*": ADMIN,
    "/api/temporal/*": ADMIN,
    "/api/careteam/*": ADMIN,
    "/api/fhir/*": ADMIN,
}


_STRICTNESS = {Access.PUBLIC: 0, Access.PARTICIPANT: 1, Access.ADMIN: 2}


@functools.lru_cache(maxsize=512)
def _template_regex(template: str) -> "re.Pattern[str]":
    """A regex for a path template: {name} is one segment, {name:path} the rest."""
    pattern, last = "", 0
    for match in re.finditer(r"\{([^}:]+)(?::([^}]+))?\}", template):
        pattern += re.escape(template[last:match.start()])
        pattern += ".*" if match.group(2) == "path" else "[^/]+"
        last = match.end()
    return re.compile("^" + pattern + re.escape(template[last:]) + "$")


def policy_for(path: str, table: Mapping[str, RoutePolicy] = ROUTE_POLICIES) -> Optional[RoutePolicy]:
    """The declared policy for a route template or a request path, or None.

    An exact key wins; then the strictest template the path matches; then the
    longest "*" prefix."""
    if path in table:
        return table[path]
    templated = [policy for key, policy in table.items()
                 if not key.endswith("*") and "{" in key and _template_regex(key).match(path)]
    if templated:
        return max(templated, key=lambda policy: _STRICTNESS[policy.access])
    best: Optional[str] = None
    for key in table:
        if key.endswith("*") and path.startswith(key[:-1]):
            if best is None or len(key) > len(best):
                best = key
    return table[best] if best is not None else None


def iter_route_paths(routes: Iterable[Any], prefix: str = "") -> Iterator[Tuple[str, Any]]:
    """(full path template, route) for every route, mount and socket, with routers
    FastAPI includes lazily (newer releases) expanded under their prefix."""
    for route in routes:
        path = getattr(route, "path", None)
        if path is not None:
            yield prefix + path, route
            continue
        router = getattr(route, "original_router", None)
        context = getattr(route, "include_context", None)
        if router is not None:
            yield from iter_route_paths(router.routes, prefix + getattr(context, "prefix", ""))
        else:
            logger.warning("Route %r has no path; the route table cannot cover it", route)


# -- tokens -------------------------------------------------------------------


def _admin_secret() -> Optional[str]:
    """The admin signing key, or None when it is unset or too short to trust."""
    secret = os.environ.get(ADMIN_SECRET_ENV, "")
    if not secret:
        return None
    if len(secret) < ADMIN_SECRET_MIN_LENGTH:
        logger.error("%s is shorter than %d characters; admin sign-in is disabled",
                     ADMIN_SECRET_ENV, ADMIN_SECRET_MIN_LENGTH)
        return None
    return secret


def issue_admin_token(subject: str, hours: float = 12.0, secret: Optional[str] = None) -> str:
    """Mint an admin token for a named person. The subject becomes the audit actor."""
    if not subject or not subject.strip():
        raise ValueError("an admin token needs the person's name as its subject")
    key = secret or _admin_secret()
    if key is None:
        raise RuntimeError(f"set {ADMIN_SECRET_ENV} (at least {ADMIN_SECRET_MIN_LENGTH} characters) first")
    now = time.time()
    payload = {
        "sub": subject.strip(),
        "role": ADMIN_ROLE,
        "type": ADMIN_TOKEN_TYPE,
        "iss": ADMIN_TOKEN_ISSUER,
        "iat": now,
        "exp": now + hours * 3600,
    }
    return jwt.encode(payload, key, algorithm="HS256")


def _admin_from_token(token: str) -> Optional[AuthenticatedUser]:
    secret = _admin_secret()
    if secret is None:
        return None
    try:
        payload = jwt.decode(
            token, secret, algorithms=["HS256"], issuer=ADMIN_TOKEN_ISSUER,
            options={"require": ["exp", "sub", "iss"]},
        )
    except jwt.InvalidTokenError:
        return None
    if payload.get("type") != ADMIN_TOKEN_TYPE or payload.get("role") != ADMIN_ROLE:
        return None
    return AuthenticatedUser(user_id=str(payload["sub"]), role=ADMIN_ROLE, site_id="", is_admin=True)


def _participant_from_token(token: str) -> Optional[AuthenticatedUser]:
    handler = get_jwt_handler()
    if handler is None:
        return None
    # Only an access token identifies a session; a refresh token is for /auth/refresh.
    payload = handler.verify_token(token, ACCESS)
    if not payload or payload.get("type") != ACCESS or not payload.get("user_id"):
        return None
    return AuthenticatedUser(
        user_id=str(payload["user_id"]),
        role=str(payload.get("role", "")),
        site_id=str(payload.get("site_id", "")),
    )


def authenticate_token(token: str) -> Optional[AuthenticatedUser]:
    """The person a bearer token names: an administrator, a participant, or None."""
    if not token:
        return None
    return _admin_from_token(token) or _participant_from_token(token)


def token_from_headers(headers: Mapping[str, str]) -> Optional[str]:
    """The token in an Authorization header: Bearer, or the password of Basic."""
    value = headers.get("authorization", "")
    scheme, _, credential = value.partition(" ")
    credential = credential.strip()
    if scheme.lower() == "bearer":
        return credential or None
    if scheme.lower() == "basic":
        try:
            decoded = base64.b64decode(credential, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            return None
        _, _, password = decoded.partition(":")
        return password or None
    return None


def authenticate_connection(conn: HTTPConnection) -> Optional[AuthenticatedUser]:
    """The authenticated person behind a request or socket, or None."""
    token = token_from_headers(conn.headers)
    return authenticate_token(token) if token else None


# -- enforcement -------------------------------------------------------------


def _request_path(conn: HTTPConnection) -> str:
    """The request path as the app routes it (without any proxy root_path)."""
    path = conn.scope.get("path", "")
    root = conn.scope.get("root_path", "")
    if root and path.startswith(root):
        path = path[len(root):] or "/"
    return path


def _deny(conn: HTTPConnection, code: int, detail: str) -> Exception:
    if conn.scope["type"] == "websocket":
        return WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason=detail)
    headers = _CHALLENGE if code == status.HTTP_401_UNAUTHORIZED else None
    return HTTPException(status_code=code, detail=detail, headers=headers)


def check_access(conn: HTTPConnection, policy: Optional[RoutePolicy]) -> Optional[AuthenticatedUser]:
    """Apply one policy to one connection. A missing policy is treated as admin."""
    access = policy.access if policy is not None else Access.ADMIN
    if access is Access.PUBLIC:
        return None
    user = authenticate_connection(conn)
    if user is None:
        raise _deny(conn, status.HTTP_401_UNAUTHORIZED, "Authentication required")
    if access is Access.ADMIN and not user.is_admin:
        raise _deny(conn, status.HTTP_403_FORBIDDEN, "Admin role required")
    conn.state.user = user
    return user


async def enforce_route_policy(conn: HTTPConnection) -> Optional[AuthenticatedUser]:
    """The one authentication dependency: applies the route's declared policy.

    Installed app-wide (``FastAPI(dependencies=[Depends(enforce_route_policy)])``).
    Returns the authenticated person (None on a public route)."""
    path = _request_path(conn)
    policy = policy_for(path)
    if policy is None:
        logger.error("Route %s has no declared policy; treating it as admin-only", path)
    return check_access(conn, policy)


async def require_admin(conn: HTTPConnection) -> AuthenticatedUser:
    """Dependency for a route or router that is admin-only wherever it is mounted,
    and that needs to know who the administrator is."""
    user = check_access(conn, ADMIN)
    if user is None:  # unreachable: check_access raises unless an admin signed in
        raise _deny(conn, status.HTTP_401_UNAUTHORIZED, "Authentication required")
    return user


class PolicyGate:
    """Applies a route policy to a mounted ASGI app (the Gradio admin panel),
    which the app-wide dependency does not reach."""

    def __init__(self, app: ASGIApp, policy: RoutePolicy) -> None:
        self.app = app
        self.policy = policy

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        conn = HTTPConnection(scope)
        try:
            check_access(conn, self.policy)
        except HTTPException as exc:
            response = JSONResponse({"status": "error", "error": exc.detail},
                                    status_code=exc.status_code, headers=exc.headers)
            await response(scope, receive, send)
            return
        except WebSocketException as exc:
            await send({"type": "websocket.close", "code": exc.code, "reason": exc.reason})
            return
        await self.app(scope, receive, send)


def guard_mounts(app: Any, table: Mapping[str, RoutePolicy] = ROUTE_POLICIES) -> None:
    """Wrap every mounted sub-app whose policy is not public in a PolicyGate.

    A mount with no declared policy is gated as admin."""
    for route in app.routes:
        if isinstance(route, Mount):
            policy = policy_for(route.path, table)
            if policy is None or policy.access is not Access.PUBLIC:
                route.app = PolicyGate(route.app, policy or ADMIN)


def undeclared_routes(app: Any, table: Mapping[str, RoutePolicy] = ROUTE_POLICIES) -> List[str]:
    """Paths of the app's routes and mounts with no declared policy."""
    return sorted({path for path, _ in iter_route_paths(app.routes) if policy_for(path, table) is None})


# -- CORS ----------------------------------------------------------------------


def cors_allowed_origins(
    environ: Mapping[str, str] = os.environ,
    config: Optional[Mapping[str, Any]] = None,
    config_path: Path = Path("config.yaml"),
) -> List[str]:
    """The origins a browser may call the API from. Default none.

    ``CORS_ALLOWED_ORIGINS`` (comma-separated) wins over config.yaml
    ``server.cors_allowed_origins``. A ``*`` entry is dropped: it would allow
    every site, which is what the policy exists to prevent."""
    raw: Iterable[str]
    if environ.get(CORS_ORIGINS_ENV, "").strip():
        raw = environ[CORS_ORIGINS_ENV].split(",")
    else:
        if config is None:
            config = _load_config(config_path)
        raw = ((config.get("server") or {}).get("cors_allowed_origins") or [])
    origins: List[str] = []
    for origin in raw:
        origin = str(origin).strip().rstrip("/")
        if not origin:
            continue
        if origin == "*":
            logger.error("CORS origin '*' ignored: list the study's own origins instead")
            continue
        origins.append(origin)
    return origins


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}
    except (OSError, yaml.YAMLError) as exc:
        logger.error("Could not read %s for CORS origins (%s); allowing none", path, exc)
        return {}


__all__: List[str] = [
    "Access", "RoutePolicy", "ROUTE_POLICIES", "PUBLIC", "PARTICIPANT", "ADMIN",
    "own_verification", "policy_for", "iter_route_paths", "issue_admin_token", "authenticate_token",
    "authenticate_connection", "token_from_headers", "check_access",
    "enforce_route_policy", "require_admin", "PolicyGate", "guard_mounts",
    "undeclared_routes", "cors_allowed_origins",
]
