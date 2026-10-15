"""
Care Role: the one vocabulary of the three tiers a person registers under
(ADR 0001): ASHA Worker, Family Caregiver and Patient.

The profile, the mobile schemas, sync and the study log all read this type, and
the Android app's ``CareRole.wireName`` values are these values (pinned by
tests/test_one_care_role_vocabulary.py). A Care Role attributes an interaction
for analysis; it is not an access boundary (ADR 0006).

Rules this module exists to enforce:

  * A role is never guessed from what someone said ("nurse", "my mother"). It is
    the role the person registered with, or the Active Role they switched to.
  * A value outside the three is no Care Role: ``parse_care_role`` returns None,
    and a study row records it as unknown (an empty string) rather than as a
    free string or a default.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional


class CareRole(str, Enum):
    """The three Care Roles, by the value the app sends and the study log records."""

    ASHA_WORKER = "asha_worker"
    FAMILY_CAREGIVER = "family_caregiver"
    PATIENT = "patient"

    @property
    def label(self) -> str:
        """The role's English name, as CONTEXT.md writes it."""
        return _LABELS[self]


_LABELS = {
    CareRole.ASHA_WORKER: "ASHA Worker",
    CareRole.FAMILY_CAREGIVER: "Family Caregiver",
    CareRole.PATIENT: "Patient",
}


def parse_care_role(value: Any) -> Optional[CareRole]:
    """The Care Role `value` names exactly, or None. Never a guess or a default."""
    if isinstance(value, CareRole):
        return value
    if not isinstance(value, str):
        return None
    try:
        return CareRole(value)
    except ValueError:
        return None


def care_role_value(value: Any) -> str:
    """The recorded form of a role: its value, or "" when it is no Care Role."""
    role = parse_care_role(value)
    return role.value if role else ""


def speaker_role(active: Any, registered: Any) -> str:
    """Who is speaking: the Active Role the app sent, else the registered Care Role.

    Returns "" when neither is a Care Role (for example a token minted before the
    vocabulary was enforced), so nothing is attributed to a role nobody chose.
    """
    return care_role_value(active) or care_role_value(registered)
