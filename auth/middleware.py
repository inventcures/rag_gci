from fastapi import Depends, HTTPException, WebSocket, WebSocketException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from auth.jwt_handler import ACCESS, JWTHandler
from auth.models import AuthenticatedUser

security = HTTPBearer()

_jwt_handler: Optional[JWTHandler] = None


def init_auth(jwt_secret: str, jwt_expiry_hours: int = 72):
    global _jwt_handler
    _jwt_handler = JWTHandler(secret=jwt_secret, expiry_hours=jwt_expiry_hours)


def get_jwt_handler() -> Optional[JWTHandler]:
    return _jwt_handler


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> AuthenticatedUser:
    """The caller of an API route: a valid, unexpired access token, or 401.

    A refresh token is refused here. It lives longer than the access token and
    only belongs on /auth/refresh, so a leaked one must not open the API.
    """
    if _jwt_handler is None:
        raise HTTPException(status_code=500, detail="Auth not initialized")

    payload = _jwt_handler.verify_token(credentials.credentials, expected_type=ACCESS)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    return AuthenticatedUser(
        user_id=payload["user_id"],
        role=payload.get("role", ""),
        site_id=payload.get("site_id", ""),
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> AuthenticatedUser:
    return await require_auth(credentials)


async def require_socket_auth(websocket: WebSocket) -> AuthenticatedUser:
    """The caller of a socket route: an access token on the Authorization header.

    The same rule as require_auth, for a WebSocket: only a valid, unexpired
    access token opens it, sent as ``Authorization: Bearer <token>``. Anything
    else closes the socket with 1008 (policy violation) before it is accepted.
    """
    scheme, _, credential = websocket.headers.get("authorization", "").partition(" ")
    token = credential.strip() if scheme.lower() == "bearer" else ""
    payload = (
        _jwt_handler.verify_token(token, expected_type=ACCESS)
        if token and _jwt_handler is not None else None
    )
    if not payload or not payload.get("user_id"):
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Authentication required",
        )
    return AuthenticatedUser(
        user_id=payload["user_id"],
        role=payload.get("role", ""),
        site_id=payload.get("site_id", ""),
    )
