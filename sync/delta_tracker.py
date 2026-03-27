import time
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class DeltaTracker:
    """Tracks record modifications and provides delta queries."""

    def __init__(self, memory_manager, medication_manager, user_profile_manager):
        self.memory_manager = memory_manager
        self.medication_manager = medication_manager
        self.user_profile_manager = user_profile_manager

    async def get_changes_since(self, user_id: str, since_timestamp: float) -> dict:
        """
        Returns all records modified after since_timestamp.
        This is the core of the delta pull sync.
        """
        patients = []
        observations = []
        care_team = []
        reminders = []

        if self.memory_manager:
            try:
                patients = await self.memory_manager.get_patients_modified_since(
                    user_id=user_id,
                    since=since_timestamp,
                )
            except Exception as e:
                logger.warning(f"Failed to get patient deltas: {e}")

            try:
                observations = await self.memory_manager.get_observations_modified_since(
                    user_id=user_id,
                    since=since_timestamp,
                )
            except Exception as e:
                logger.warning(f"Failed to get observation deltas: {e}")

            try:
                care_team = await self.memory_manager.get_care_team_modified_since(
                    user_id=user_id,
                    since=since_timestamp,
                )
            except Exception as e:
                logger.warning(f"Failed to get care team deltas: {e}")

        if self.medication_manager:
            try:
                reminders = await self.medication_manager.get_reminders_modified_since(
                    user_id=user_id,
                    since=since_timestamp,
                )
            except Exception as e:
                logger.warning(f"Failed to get reminder deltas: {e}")

        def to_dict_list(items):
            return [
                item.to_dict() if hasattr(item, "to_dict") else item
                for item in items
            ]

        return {
            "patients": to_dict_list(patients),
            "observations": to_dict_list(observations),
            "care_team_members": to_dict_list(care_team),
            "medication_reminders": to_dict_list(reminders),
            "server_timestamp": time.time(),
        }
