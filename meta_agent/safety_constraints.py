"""Safety constraints that the meta-agent must never violate."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    MONOTONICALLY_INCREASING = "monotonically_increasing"
    READ_ONLY = "read_only"
    APPEND_ONLY = "append_only"


SAFETY_CONSTRAINTS: Dict[str, str] = {
    "emergency_sensitivity": "monotonically_increasing",
    "dosage_validation": "read_only",
    "handoff_triggers": "append_only",
    "evidence_thresholds": "monotonically_increasing",
}


@dataclass
class Modification:
    target: str
    field: str
    old_value: Any
    new_value: Any
    reason: str


class SafetyConstraintChecker:
    """Validates that proposed modifications respect clinical safety boundaries."""

    def __init__(self, constraints: Optional[Dict[str, str]] = None):
        self._constraints = constraints or SAFETY_CONSTRAINTS

    def validate_modification(self, modification: Modification) -> bool:
        """
        Check whether a proposed modification violates safety constraints.
        Returns True if the modification is safe, False if it violates a constraint.
        """
        constraint_key = modification.field
        if constraint_key not in self._constraints:
            return True

        constraint_type = self._constraints[constraint_key]

        if constraint_type == ConstraintType.READ_ONLY.value:
            logger.warning(
                f"Blocked modification to read-only field '{constraint_key}': "
                f"{modification.old_value} -> {modification.new_value}"
            )
            return False

        if constraint_type == ConstraintType.MONOTONICALLY_INCREASING.value:
            try:
                if float(modification.new_value) < float(modification.old_value):
                    logger.warning(
                        f"Blocked decrease of monotonically-increasing field '{constraint_key}': "
                        f"{modification.old_value} -> {modification.new_value}"
                    )
                    return False
            except (TypeError, ValueError):
                logger.warning(
                    f"Cannot compare values for monotonic check on '{constraint_key}'"
                )
                return False

        if constraint_type == ConstraintType.APPEND_ONLY.value:
            old_set = set(modification.old_value) if isinstance(modification.old_value, (list, set)) else set()
            new_set = set(modification.new_value) if isinstance(modification.new_value, (list, set)) else set()
            removed = old_set - new_set
            if removed:
                logger.warning(
                    f"Blocked removal from append-only field '{constraint_key}': "
                    f"removed items: {removed}"
                )
                return False

        return True

    def validate_batch(self, modifications: list) -> Dict[str, bool]:
        return {
            f"{m.field}:{m.target}": self.validate_modification(m)
            for m in modifications
        }

    def get_constraint(self, field: str) -> Optional[str]:
        return self._constraints.get(field)

    @property
    def constrained_fields(self) -> list:
        return list(self._constraints.keys())
