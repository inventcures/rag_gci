"""PIN-based authentication for the 6 reviewer accounts.

Mirrors the PBKDF2 pattern in ``auth/pin_auth.py`` so that PIN handling stays
consistent across the codebase. Sessions are signed cookies (HMAC-SHA256)
to avoid an extra dependency on ``itsdangerous``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from pathlib import Path
from typing import Optional

APP_DIR = Path(__file__).parent
USERS_FILE = APP_DIR / "users.json"
SECRET_FILE = APP_DIR / ".session_secret"
SESSION_COOKIE = "review_session"
SESSION_TTL_SECONDS = 60 * 60 * 8


def _hash_pin(pin: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        pin.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=100_000,
    ).hex()


def _load_users() -> list[dict]:
    if not USERS_FILE.exists():
        return []
    with USERS_FILE.open() as f:
        return json.load(f)


def _save_users(users: list[dict]) -> None:
    with USERS_FILE.open("w") as f:
        json.dump(users, f, indent=2)


def ensure_users_seeded(seed: list[dict]) -> None:
    """Seed users.json once with hashed PINs. Idempotent."""
    if USERS_FILE.exists():
        existing = _load_users()
        if existing and all("pin_hash" in u for u in existing):
            return

    hashed = []
    for entry in seed:
        salt = secrets.token_hex(16)
        hashed.append(
            {
                "user_id": entry["user_id"],
                "name": entry["name"],
                "site": entry["site"],
                "pin_salt": salt,
                "pin_hash": _hash_pin(entry["pin"], salt),
            }
        )
    _save_users(hashed)


def list_users() -> list[dict]:
    return [
        {"user_id": u["user_id"], "name": u["name"], "site": u["site"]}
        for u in _load_users()
    ]


def get_user(user_id: str) -> Optional[dict]:
    for u in _load_users():
        if u["user_id"] == user_id:
            return {"user_id": u["user_id"], "name": u["name"], "site": u["site"]}
    return None


def verify_pin(user_id: str, pin: str) -> bool:
    for u in _load_users():
        if u["user_id"] == user_id:
            return _hash_pin(pin, u["pin_salt"]) == u["pin_hash"]
    return False


def _secret() -> bytes:
    if not SECRET_FILE.exists():
        SECRET_FILE.write_bytes(secrets.token_bytes(32))
        try:
            SECRET_FILE.chmod(0o600)
        except OSError:
            pass
    return SECRET_FILE.read_bytes()


def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64d(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def make_session_cookie(user_id: str) -> str:
    payload = {"user_id": user_id, "issued_at": int(time.time())}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    body = _b64e(raw)
    sig = hmac.new(_secret(), body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_b64e(sig)}"


def read_session_cookie(cookie_value: Optional[str]) -> Optional[dict]:
    if not cookie_value or "." not in cookie_value:
        return None
    body, sig = cookie_value.split(".", 1)
    expected = hmac.new(_secret(), body.encode("ascii"), hashlib.sha256).digest()
    try:
        provided = _b64d(sig)
    except (ValueError, base64.binascii.Error):
        return None
    if not hmac.compare_digest(expected, provided):
        return None
    try:
        payload = json.loads(_b64d(body))
    except (ValueError, json.JSONDecodeError):
        return None
    if int(time.time()) - int(payload.get("issued_at", 0)) > SESSION_TTL_SECONDS:
        return None
    return payload
