import jwt
import time
import secrets
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenPair:
    access_token: str
    refresh_token: str
    expires_at: float


class JWTHandler:
    def __init__(self, secret: str, expiry_hours: int = 72):
        self.secret = secret
        self.expiry_seconds = expiry_hours * 3600
        self.algorithm = "HS256"

    def create_token_pair(self, user_id: str, role: str, site_id: str) -> TokenPair:
        now = time.time()
        expires_at = now + self.expiry_seconds

        access_payload = {
            "user_id": user_id,
            "role": role,
            "site_id": site_id,
            "iat": now,
            "exp": expires_at,
            "type": "access",
        }
        access_token = jwt.encode(access_payload, self.secret, algorithm=self.algorithm)

        refresh_payload = {
            "user_id": user_id,
            "iat": now,
            "exp": now + (self.expiry_seconds * 2),
            "type": "refresh",
            "jti": secrets.token_hex(16),
        }
        refresh_token = jwt.encode(refresh_payload, self.secret, algorithm=self.algorithm)

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
        )

    def verify_token(self, token: str) -> Optional[dict]:
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None

    def refresh_access_token(self, refresh_token: str) -> Optional[TokenPair]:
        payload = self.verify_token(refresh_token)
        if payload and payload.get("type") == "refresh":
            return self.create_token_pair(
                user_id=payload["user_id"],
                role=payload.get("role", ""),
                site_id=payload.get("site_id", ""),
            )
        return None
