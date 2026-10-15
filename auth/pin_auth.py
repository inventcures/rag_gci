import hashlib
import secrets
import json
import time
import uuid
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

DATA_DIR = Path("./data/auth/devices")

# Wrong PINs allowed before the account locks, and how long the lock lasts. The
# lock lifts on its own after the cooling-off period, so a participant who
# mistyped is not locked out of the study for good. Each wrong PIN after a lock
# has lifted starts a new cooling-off period, so a guesser gets one try per
# period, not a fresh five.
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 15 * 60


def _is_user_id(user_id: str) -> bool:
    """True for a user id in the canonical UUID form registration issues.

    The id names the record's file, and it arrives from the client on
    /auth/login, so anything else (a path such as "../x") is refused before
    any file is touched."""
    if not isinstance(user_id, str):
        return False
    try:
        return str(uuid.UUID(user_id)) == user_id
    except ValueError:
        return False


class PinAuthenticator:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def register_device(
        self,
        user_id: str,
        pin: str,
        name: str,
        phone_hash: str,
        role: str,
        language: str,
        site_id: str,
        abha_id: Optional[str] = None,
        digital_literacy_score: Optional[int] = None,
    ) -> dict:
        filepath = self._record_path(user_id)
        if filepath is None:
            raise ValueError("user_id must be a UUID")
        salt = secrets.token_hex(16)
        pin_hash = self._hash_pin(pin, salt)

        device_record = {
            "user_id": user_id,
            "name": name,
            "phone_hash": phone_hash,
            "role": role,
            "language": language,
            "site_id": site_id,
            "abha_id": abha_id,
            "digital_literacy_score": digital_literacy_score,
            "pin_hash": pin_hash,
            "pin_salt": salt,
            "failed_attempts": 0,
            "created_at": time.time(),
            "last_login_at": None,
        }

        with open(filepath, "w") as f:
            json.dump(device_record, f, indent=2)

        logger.info(f"Registered device for user {user_id} at site {site_id}")
        return {"user_id": user_id, "status": "registered"}

    def verify_pin(self, user_id: str, pin: str) -> bool:
        """True when the PIN matches and the account is not in a cooling-off period.

        A locked account refuses even the correct PIN until the period ends.
        """
        filepath = self._record_path(user_id)
        if filepath is None or not filepath.exists():
            return False

        with open(filepath) as f:
            record = json.load(f)

        failed = record.get("failed_attempts", 0)
        now = time.time()
        if failed >= MAX_FAILED_ATTEMPTS:
            locked_until = record.get("last_failed_at", 0.0) + LOCKOUT_SECONDS
            if now < locked_until:
                logger.warning(
                    f"User {user_id} locked out after {failed} failed attempts, "
                    f"for {int(locked_until - now)}s more"
                )
                return False

        expected_hash = record["pin_hash"]
        salt = record["pin_salt"]
        actual_hash = self._hash_pin(pin, salt)

        if secrets.compare_digest(actual_hash, expected_hash):
            record["failed_attempts"] = 0
            record.pop("last_failed_at", None)
            record["last_login_at"] = now
            with open(filepath, "w") as f:
                json.dump(record, f, indent=2)
            return True
        else:
            record["failed_attempts"] = failed + 1
            record["last_failed_at"] = now
            with open(filepath, "w") as f:
                json.dump(record, f, indent=2)
            return False

    def get_user_record(self, user_id: str) -> Optional[dict]:
        filepath = self._record_path(user_id)
        if filepath is None or not filepath.exists():
            return None
        with open(filepath) as f:
            record = json.load(f)
        record.pop("pin_hash", None)
        record.pop("pin_salt", None)
        return record

    def _record_path(self, user_id: str) -> Optional[Path]:
        """The device record's file for a well-formed user id, else None."""
        if not _is_user_id(user_id):
            logger.warning("Refused a device record lookup for a malformed user id")
            return None
        return DATA_DIR / f"{user_id}.json"

    def _hash_pin(self, pin: str, salt: str) -> str:
        return hashlib.pbkdf2_hmac(
            "sha256",
            pin.encode("utf-8"),
            salt.encode("utf-8"),
            iterations=100_000,
        ).hex()
