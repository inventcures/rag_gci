import hashlib
import secrets
import json
import time
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

DATA_DIR = Path("./data/auth/devices")


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

        filepath = DATA_DIR / f"{user_id}.json"
        with open(filepath, "w") as f:
            json.dump(device_record, f, indent=2)

        logger.info(f"Registered device for user {user_id} at site {site_id}")
        return {"user_id": user_id, "status": "registered"}

    def verify_pin(self, user_id: str, pin: str) -> bool:
        filepath = DATA_DIR / f"{user_id}.json"
        if not filepath.exists():
            return False

        with open(filepath) as f:
            record = json.load(f)

        if record.get("failed_attempts", 0) >= 5:
            logger.warning(f"User {user_id} locked out after 5 failed attempts")
            return False

        expected_hash = record["pin_hash"]
        salt = record["pin_salt"]
        actual_hash = self._hash_pin(pin, salt)

        if actual_hash == expected_hash:
            record["failed_attempts"] = 0
            record["last_login_at"] = time.time()
            with open(filepath, "w") as f:
                json.dump(record, f, indent=2)
            return True
        else:
            record["failed_attempts"] = record.get("failed_attempts", 0) + 1
            with open(filepath, "w") as f:
                json.dump(record, f, indent=2)
            return False

    def get_user_record(self, user_id: str) -> Optional[dict]:
        filepath = DATA_DIR / f"{user_id}.json"
        if not filepath.exists():
            return None
        with open(filepath) as f:
            record = json.load(f)
        record.pop("pin_hash", None)
        record.pop("pin_salt", None)
        return record

    def _hash_pin(self, pin: str, salt: str) -> str:
        return hashlib.pbkdf2_hmac(
            "sha256",
            pin.encode("utf-8"),
            salt.encode("utf-8"),
            iterations=100_000,
        ).hex()
