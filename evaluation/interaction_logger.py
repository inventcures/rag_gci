import json
import time
import uuid
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

LOGS_DIR = Path("./data/evaluation/interaction_logs")


class MobileInteractionLogger:
    """
    Stores structured interaction events from mobile devices.
    Events are append-only and synced in batches.
    """

    def __init__(self):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    async def ingest_log(self, log_entry: dict, user_id: str) -> str:
        log_id = log_entry.get("log_id", str(uuid.uuid4()))
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
        day_dir.mkdir(exist_ok=True)

        filepath = day_dir / f"{log_id}.json"
        with open(filepath, "w") as f:
            json.dump(record, f)

        return log_id

    async def ingest_batch(self, logs: List[dict], user_id: str) -> int:
        count = 0
        for log_entry in logs:
            await self.ingest_log(log_entry, user_id)
            count += 1
        return count

    async def ingest_vignette_response(self, response: dict, user_id: str) -> str:
        response_id = response.get("response_id", str(uuid.uuid4()))
        filepath = Path("./data/evaluation/vignette_responses") / f"{response_id}.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        response["user_id"] = user_id
        response["ingested_at"] = time.time()
        with open(filepath, "w") as f:
            json.dump(response, f, indent=2)
        return response_id
