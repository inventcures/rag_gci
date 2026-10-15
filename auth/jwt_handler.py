import jwt
import time
import secrets
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Every token carries one of these in its "type" claim, and is accepted only where
# its type belongs: the access token on API routes, the refresh token on
# /auth/refresh. A token with no type, or the other type, is refused.
ACCESS = "access"
REFRESH = "refresh"


@dataclass
class TokenPair:
    access_token: str
    refresh_token: str
    expires_at: float


class JWTHandler:
    """Issues and checks the mobile API's access and refresh tokens.

    A PIN sign-in starts a session, and the refresh token's expiry is where that
    session ends. Renewing issues a new pair but never moves that end, so a
    session lasts at most ``2 * expiry_hours`` after the PIN before the PIN is
    needed again.
    """

    def __init__(self, secret: str, expiry_hours: int = 72):
        self.secret = secret
        self.expiry_seconds = expiry_hours * 3600
        self.algorithm = "HS256"

    def create_token_pair(
        self,
        user_id: str,
        role: str,
        site_id: str,
        session_expires_at: Optional[float] = None,
    ) -> TokenPair:
        """Issue an access and a refresh token.

        ``session_expires_at`` is passed when renewing, to keep the session's end
        where the PIN sign-in put it. Left out, a new session starts now.
        """
        now = time.time()
        if session_expires_at is None:
            session_expires_at = now + self.expiry_seconds * 2
        expires_at = min(now + self.expiry_seconds, session_expires_at)

        access_payload = {
            "user_id": user_id,
            "role": role,
            "site_id": site_id,
            "iat": now,
            "exp": expires_at,
            "type": ACCESS,
        }
        access_token = jwt.encode(access_payload, self.secret, algorithm=self.algorithm)

        refresh_payload = {
            "user_id": user_id,
            "iat": now,
            "exp": session_expires_at,
            "type": REFRESH,
            "jti": secrets.token_hex(16),
        }
        refresh_token = jwt.encode(refresh_payload, self.secret, algorithm=self.algorithm)

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
        )

    def verify_token(self, token: str, expected_type: str = ACCESS) -> Optional[dict]:
        """Return the token's claims, or None when it is invalid, expired or of the wrong type.

        The default is the access type, so a caller that does not say which type
        it wants is never handed a refresh token's claims.
        """
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        if payload.get("type") != expected_type:
            logger.warning("Token of type %r refused where %r is required", payload.get("type"), expected_type)
            return None
        return payload
