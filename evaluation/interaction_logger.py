import json
import re
import time
import uuid
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)

LOGS_DIR = Path("./data/evaluation/interaction_logs")
VIGNETTE_RESPONSES_DIR = Path("./data/evaluation/vignette_responses")

# A client-supplied record id names the record's file, so it is restricted to
# characters that cannot form a path.
_RECORD_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _check_record_id(record_id: object) -> str:
    """The id, if it is safe to use as a file name; ValueError otherwise."""
    if not isinstance(record_id, str) or not _RECORD_ID.match(record_id):
        raise ValueError("record id must be 1-64 letters, digits, '-' or '_'")
    return record_id


def _check_not_anothers(filepath: Path, user_id: str) -> None:
    """PermissionError if the file already holds another household's record."""
    if not filepath.exists():
        return
    try:
        with open(filepath) as f:
            owner = json.load(f).get("user_id")
    except (OSError, ValueError) as e:
        raise PermissionError(f"existing record is unreadable: {type(e).__name__}") from e
    if owner != user_id:
        raise PermissionError("record id belongs to another household")


class MobileInteractionLogger:
    """
    Stores structured interaction events from mobile devices.
    Events are append-only and synced in batches.

    A record belongs to the household that wrote it: a push that reuses another
    household's id is refused (PermissionError), and an id that is not a safe
    file name is refused (ValueError). A record that cannot be written raises
    OSError, so the caller can report it instead of dropping it.
    """

    def __init__(self):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    async def ingest_log(self, log_entry: dict, user_id: str) -> str:
        """Store one interaction log for `user_id`; returns its id."""
        log_id = _check_record_id(log_entry.get("log_id") or str(uuid.uuid4()))
        record = {
            "log_id": log_id,
            "user_id": user_id,
            "session_id": log_entry.get("session_id", ""),
            "event_type": log_entry.get("event_type", ""),
            "event_data": log_entry.get("event_data"),
            "timestamp": log_entry.get("timestamp", time.time()),
            "duration_ms": log_entry.get("duration_ms"),
            "language": log_entry.get("language", ""),
            "site_id": log_entry.get("site_id", ""),
            "is_offline": log_entry.get("is_offline", False),
            "ingested_at": time.time(),
        }

        date_str = time.strftime("%Y-%m-%d", time.gmtime(record["timestamp"]))
        day_dir = LOGS_DIR / date_str
        filepath = day_dir / f"{log_id}.json"
        _check_not_anothers(filepath, user_id)
        day_dir.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(record, f)

        return log_id

    async def ingest_batch(self, logs: List[dict], user_id: str) -> int:
        """Store each log; returns how many were saved. A failure is logged, not raised."""
        count = 0
        for index, log_entry in enumerate(logs):
            try:
                await self.ingest_log(log_entry, user_id)
                count += 1
            except (OSError, ValueError, PermissionError, TypeError) as e:
                logger.warning(
                    f"Interaction log {log_entry.get('log_id')!r} (item {index}) from user {user_id} "
                    f"not saved: {type(e).__name__}: {e}"
                )
        return count

    async def ingest_vignette_response(self, response: dict, user_id: str) -> str:
        """Store one vignette response for `user_id`; returns its id."""
        response_id = _check_record_id(response.get("response_id") or str(uuid.uuid4()))
        filepath = VIGNETTE_RESPONSES_DIR / f"{response_id}.json"
        _check_not_anothers(filepath, user_id)
        record = {**response, "response_id": response_id, "user_id": user_id, "ingested_at": time.time()}
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(record, f, indent=2)
        return response_id
