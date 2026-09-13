"""Versioned clinical knowledge with explicit publication and request policies.

No imported knowledge is clinically approved automatically. The optional API
and UI are enabled separately from the existing research retrieval paths.
"""

from .models import Actor, GovernanceError
from .store import GovernanceStore

__all__ = ["Actor", "GovernanceError", "GovernanceStore"]
