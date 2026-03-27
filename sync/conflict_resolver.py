from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ConflictStrategy(Enum):
    SERVER_WINS = "server_wins"
    CLIENT_WINS = "client_wins"
    APPEND_ONLY = "append_only"


ENTITY_STRATEGIES = {
    "patient": ConflictStrategy.SERVER_WINS,
    "observation": ConflictStrategy.APPEND_ONLY,
    "care_team_member": ConflictStrategy.SERVER_WINS,
    "medication_reminder_schedule": ConflictStrategy.CLIENT_WINS,
    "medication_reminder_status": ConflictStrategy.SERVER_WINS,
    "interaction_log": ConflictStrategy.APPEND_ONLY,
    "vignette_response": ConflictStrategy.CLIENT_WINS,
}


class ConflictResolver:

    def resolve(self, entity_type: str, server_record: dict, client_record: dict) -> dict:
        strategy = ENTITY_STRATEGIES.get(entity_type, ConflictStrategy.SERVER_WINS)

        if strategy == ConflictStrategy.APPEND_ONLY:
            return client_record

        if strategy == ConflictStrategy.CLIENT_WINS:
            return client_record

        if strategy == ConflictStrategy.SERVER_WINS:
            server_ts = server_record.get("updated_at", 0)
            client_ts = client_record.get("updated_at", 0)
            if client_ts > server_ts:
                logger.info(f"Conflict on {entity_type}: client newer but server wins per policy")
            return server_record

        return server_record
