"""
User Personalization Module for Palli Sahayak

Provides personalized experiences through:
1. User profiles with role-based context (patient, caregiver, healthcare worker)
2. Patient context memory (conditions, symptoms, medications)
3. Interaction history tracking for continuity
4. Preference management (language, communication style)
"""

from .user_profile import UserProfile, UserProfileManager, UserRole
from .context_memory import PatientContext, ContextMemory
from .interaction_history import InteractionHistory, ConversationTurn

__all__ = [
    "UserProfile",
    "UserProfileManager",
    "UserRole",
    "PatientContext",
    "ContextMemory",
    "InteractionHistory",
    "ConversationTurn",
]
