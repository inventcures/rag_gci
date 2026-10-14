"""
Participant ids that belong to one session, not to a person.

A call id or a socket session id lives for one call. Keying longitudinal
records or comprehension profiles on it creates a new record per call that is
never read again. Channels that cannot identify the caller mark the id with
`ephemeral()`; the answer path still logs and traces it, but stores nothing
per person under it.
"""

from __future__ import annotations

from typing import Optional

EPHEMERAL_PREFIX = "session:"
# Work the server does for itself (building the offline bundle), not a caller.
SYSTEM_PREFIX = "system:"


def ephemeral(channel: str, session_id: str) -> str:
    """A per-session participant id, e.g. ephemeral("retell", call_id)."""
    return f"{EPHEMERAL_PREFIX}{channel}:{session_id}"


def is_ephemeral(user_id: Optional[str]) -> bool:
    return bool(user_id) and str(user_id).startswith(EPHEMERAL_PREFIX)


def system(purpose: str) -> str:
    """The id for server-initiated answers, e.g. system("offline_bundle")."""
    return f"{SYSTEM_PREFIX}{purpose}"


def is_system(user_id: Optional[str]) -> bool:
    return bool(user_id) and str(user_id).startswith(SYSTEM_PREFIX)


def is_person(user_id: Optional[str]) -> bool:
    """True when per-person state (longitudinal record, profile) may be kept."""
    return bool(user_id) and not is_ephemeral(user_id) and not is_system(user_id)
