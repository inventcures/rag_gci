import time
from typing import Any, Awaitable, Callable, Dict, List, Optional
import logging

from care_role import care_role_value

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 50

ACCEPTED = "accepted"
REJECTED = "rejected"

# Why an item was rejected. The app keeps a rejected item in its queue; these
# tell it whether retrying can help (save_failed, unavailable) or not.
INVALID = "invalid"          # malformed: no id, an unusable id, a bad value
NOT_OWNED = "not_owned"      # another household's patient or record
SAVE_FAILED = "save_failed"  # the server could not store it
UNAVAILABLE = "unavailable"  # the store for this kind is not running
OVER_LIMIT = "over_limit"    # beyond MAX_BATCH_SIZE for its kind; push it again

# (push field, acknowledgement kind, the item's id field)
_KINDS = (
    ("observations", "observation", "observation_id"),
    ("medication_reminders", "medication_reminder", "reminder_id"),
    ("interaction_logs", "interaction_log", "log_id"),
    ("vignette_responses", "vignette_response", "response_id"),
)


def _ack(kind: str, index: int, item_id: Optional[str], status: str, reason: Optional[str] = None) -> Dict[str, Any]:
    return {"kind": kind, "index": index, "id": item_id, "status": status, "reason": reason}


class BatchProcessor:
    """Processes batch uploads from mobile devices.

    Every pushed item gets one acknowledgement, in its kind's order:
    ``{"kind", "index", "id", "status": "accepted"|"rejected", "reason"}``.
    ``index`` is the item's position in its list of the push, and ``id`` is the
    id the server stored it under (None when it has none). Every write is
    scoped to the caller's household: owners come from the token, never from
    the payload.
    """

    def __init__(self, memory_manager, medication_manager, evaluation_logger):
        self.memory_manager = memory_manager
        self.medication_manager = medication_manager
        self.evaluation_logger = evaluation_logger

    async def process_push(self, push_data: dict, user_id: str, role: str) -> dict:
        """Apply one push for `user_id`; returns counts and per-item acknowledgements.

        `role` is the caller's registered Care Role, recorded on a patient this
        push claims. A value outside the vocabulary is recorded as "" rather than
        defaulting to the ASHA Worker."""
        role = care_role_value(role)
        savers: Dict[str, Optional[Callable[[dict], Awaitable[Optional[str]]]]] = {
            "observation": self._save_observation(user_id, role) if self.memory_manager else None,
            "medication_reminder": self._save_reminder(user_id) if self.medication_manager else None,
            "interaction_log": self._save_log(user_id) if self.evaluation_logger else None,
            "vignette_response": self._save_vignette(user_id) if self.evaluation_logger else None,
        }

        acknowledgements: List[Dict[str, Any]] = []
        for field, kind, id_field in _KINDS:
            items = push_data.get(field) or []
            for index, item in enumerate(items):
                item_id = item.get(id_field) if isinstance(item, dict) else None
                if index >= MAX_BATCH_SIZE:
                    acknowledgements.append(_ack(kind, index, item_id, REJECTED, OVER_LIMIT))
                    continue
                acknowledgements.append(await self._apply(kind, index, item, item_id, savers[kind], user_id))

        accepted = sum(1 for a in acknowledgements if a["status"] == ACCEPTED)
        return {
            "accepted": accepted,
            "rejected": len(acknowledgements) - accepted,
            "acknowledgements": acknowledgements,
            "conflicts": [],
            "server_timestamp": time.time(),
        }

    async def _apply(self, kind: str, index: int, item: Any, item_id: Optional[str],
                     save: Optional[Callable[[dict], Awaitable[Optional[str]]]], user_id: str) -> Dict[str, Any]:
        """Save one item and acknowledge it truthfully. Every rejection is logged."""
        def rejected(reason: str, error: Optional[BaseException] = None) -> Dict[str, Any]:
            detail = f": {type(error).__name__}: {error}" if error else ""
            logger.warning(f"Sync push from user {user_id}: {kind} {item_id!r} (item {index}) rejected, {reason}{detail}")
            return _ack(kind, index, item_id, REJECTED, reason)

        if save is None:
            return rejected(UNAVAILABLE)
        if not isinstance(item, dict):
            return rejected(INVALID)
        try:
            stored_id = await save(item)
        except PermissionError as e:
            return rejected(NOT_OWNED, e)
        except (ValueError, TypeError, KeyError) as e:
            return rejected(INVALID, e)
        except OSError as e:
            return rejected(SAVE_FAILED, e)
        except Exception as e:  # an unexpected store failure is still reported, never dropped
            logger.exception(f"Unexpected error saving {kind} {item_id!r} from user {user_id}")
            return rejected(SAVE_FAILED, e)
        if stored_id is None:
            return rejected(INVALID)
        return _ack(kind, index, stored_id, ACCEPTED)

    def _save_observation(self, user_id: str, role: str) -> Callable[[dict], Awaitable[Optional[str]]]:
        async def save(obs: dict) -> Optional[str]:
            # Owner-checked: a new patient becomes this user's; another
            # household's patient raises PermissionError.
            return await self.memory_manager.add_observation_from_sync(
                obs, source="mobile_sync", user_id=user_id, role=role)
        return save

    def _save_reminder(self, user_id: str) -> Callable[[dict], Awaitable[Optional[str]]]:
        async def save(reminder: dict) -> Optional[str]:
            # Synchronous: the reminder store does not await.
            return self.medication_manager.upsert_reminder_from_sync(reminder, user_id=user_id)
        return save

    def _save_log(self, user_id: str) -> Callable[[dict], Awaitable[Optional[str]]]:
        async def save(log_entry: dict) -> Optional[str]:
            return await self.evaluation_logger.ingest_log(log_entry, user_id=user_id)
        return save

    def _save_vignette(self, user_id: str) -> Callable[[dict], Awaitable[Optional[str]]]:
        async def save(vignette: dict) -> Optional[str]:
            return await self.evaluation_logger.ingest_vignette_response(vignette, user_id=user_id)
        return save
