from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from auth.jwt_handler import JWTHandler
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
    if _jwt_handler is None:
        raise HTTPException(status_code=500, detail="Auth not initialized")

    payload = _jwt_handler.verify_token(credentials.credentials)
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
