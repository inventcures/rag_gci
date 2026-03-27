import time
from typing import List
import logging

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 50


class BatchProcessor:
    """Processes batch uploads from mobile devices efficiently."""

    def __init__(self, memory_manager, medication_manager, evaluation_logger):
        self.memory_manager = memory_manager
        self.medication_manager = medication_manager
        self.evaluation_logger = evaluation_logger

    async def process_push(self, push_data: dict, user_id: str) -> dict:
        accepted = 0
        rejected = 0
        conflicts = []

        for obs in (push_data.get("observations") or [])[:MAX_BATCH_SIZE]:
            try:
                await self.memory_manager.add_observation_from_sync(obs, source="mobile_sync")
                accepted += 1
            except Exception as e:
                logger.warning(f"Rejected observation {obs.get('observation_id')}: {e}")
                rejected += 1

        for reminder in (push_data.get("medication_reminders") or [])[:MAX_BATCH_SIZE]:
            try:
                await self.medication_manager.upsert_reminder_from_sync(reminder)
                accepted += 1
            except Exception as e:
                rejected += 1

        for log_entry in (push_data.get("interaction_logs") or [])[:MAX_BATCH_SIZE]:
            try:
                await self.evaluation_logger.ingest_log(log_entry, user_id=user_id)
                accepted += 1
            except Exception as e:
                rejected += 1

        for vignette in (push_data.get("vignette_responses") or [])[:MAX_BATCH_SIZE]:
            try:
                await self.evaluation_logger.ingest_vignette_response(vignette, user_id=user_id)
                accepted += 1
            except Exception as e:
                rejected += 1

        return {
            "accepted": accepted,
            "rejected": rejected,
            "conflicts": conflicts,
            "server_timestamp": time.time(),
        }
