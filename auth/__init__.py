"""
Authentication module for Palli Sahayak mobile API.

Provides JWT-based authentication with PIN verification.
Auth is opt-in: only /api/mobile/v1/* endpoints require it.
Existing endpoints remain unauthenticated for backward compatibility.
"""

from auth.jwt_handler import JWTHandler
from auth.pin_auth import PinAuthenticator
from auth.middleware import require_auth, get_current_user
from auth.models import DeviceRegistration, AuthenticatedUser

__all__ = [
    "JWTHandler",
    "PinAuthenticator",
    "require_auth",
    "get_current_user",
    "DeviceRegistration",
    "AuthenticatedUser",
]
