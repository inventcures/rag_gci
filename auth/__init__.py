"""
Authentication for Palli Sahayak.

The mobile API signs participants in with a PIN and a JWT. Every route on the
main server declares a policy (public, participant or admin) in
auth/policy.py, and one dependency applies it.
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
