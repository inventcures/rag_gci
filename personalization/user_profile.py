"""
User Profile Management

Manages user profiles with:
- The person's Care Role (care_role.py), as registered; never guessed
- Language preferences
- Communication style preferences
- Session continuity
"""

import json
import logging
import hashlib
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import asyncio
import aiofiles

from care_role import CareRole, parse_care_role

logger = logging.getLogger(__name__)


class CommunicationStyle(Enum):
    """Preferred communication style."""
    SIMPLE = "simple"           # Simple language, fewer details
    DETAILED = "detailed"       # Comprehensive explanations
    CLINICAL = "clinical"       # Medical terminology for HCWs
    EMPATHETIC = "empathetic"   # Focus on emotional support


@dataclass
class UserPreferences:
    """User preferences."""
    language: str = "en-IN"
    communication_style: CommunicationStyle = CommunicationStyle.SIMPLE
    voice_speed: str = "normal"  # slow, normal, fast
    send_summaries: bool = False  # SMS summaries after call
    auto_followup: bool = True   # Proactive follow-up prompts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "communication_style": self.communication_style.value,
            "voice_speed": self.voice_speed,
            "send_summaries": self.send_summaries,
            "auto_followup": self.auto_followup
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreferences":
        return cls(
            language=data.get("language", "en-IN"),
            communication_style=CommunicationStyle(
                data.get("communication_style", "simple")
            ),
            voice_speed=data.get("voice_speed", "normal"),
            send_summaries=data.get("send_summaries", False),
            auto_followup=data.get("auto_followup", True)
        )


@dataclass
class UserProfile:
    """Complete user profile."""
    user_id: str
    phone_number: Optional[str] = None
    # The Care Role the person registered with or confirmed; None until then.
    role: Optional[CareRole] = None
    preferences: UserPreferences = field(default_factory=UserPreferences)
    first_interaction: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = field(default_factory=datetime.now)
    total_sessions: int = 0
    total_queries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "phone_number": self.phone_number,
            "role": self.role.value if self.role else "unknown",
            "preferences": self.preferences.to_dict(),
            "first_interaction": self.first_interaction.isoformat(),
            "last_interaction": self.last_interaction.isoformat(),
            "total_sessions": self.total_sessions,
            "total_queries": self.total_queries,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        return cls(
            user_id=data["user_id"],
            phone_number=data.get("phone_number"),
            # A stored value outside the vocabulary (the old "caregiver",
            # "healthcare_worker", "unknown") is no Care Role, not a guess.
            role=parse_care_role(data.get("role")),
            preferences=UserPreferences.from_dict(data.get("preferences", {})),
            first_interaction=datetime.fromisoformat(data["first_interaction"]),
            last_interaction=datetime.fromisoformat(data["last_interaction"]),
            total_sessions=data.get("total_sessions", 0),
            total_queries=data.get("total_queries", 0),
            metadata=data.get("metadata", {})
        )

    def increment_session(self) -> None:
        """Track new session."""
        self.total_sessions += 1
        self.last_interaction = datetime.now()

    def increment_query(self) -> None:
        """Track new query."""
        self.total_queries += 1
        self.last_interaction = datetime.now()


class UserProfileManager:
    """
    Manages user profiles with persistent storage.

    Features:
    - Profile creation and retrieval
    - Preference management
    - Session tracking
    """

    def __init__(
        self,
        storage_path: str = "data/user_profiles",
        cache_size: int = 100
    ):
        """
        Initialize the profile manager.

        Args:
            storage_path: Directory for profile storage
            cache_size: Maximum profiles to cache in memory
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.cache_size = cache_size
        self._cache: Dict[str, UserProfile] = {}
        self._lock = asyncio.Lock()

        logger.info(f"UserProfileManager initialized - path={storage_path}")

    def _get_user_id(self, phone_number: str) -> str:
        """Generate consistent user ID from phone number."""
        # Hash phone number for privacy
        return f"user_{hashlib.sha256(phone_number.encode()).hexdigest()[:12]}"

    def _get_profile_path(self, user_id: str) -> Path:
        """Get file path for user profile."""
        return self.storage_path / f"{user_id}.json"

    async def get_or_create_profile(
        self,
        phone_number: str,
        language: str = "en-IN"
    ) -> UserProfile:
        """
        Get existing profile or create new one.

        Args:
            phone_number: User's phone number
            language: Default language preference

        Returns:
            UserProfile
        """
        user_id = self._get_user_id(phone_number)

        # Check cache first
        if user_id in self._cache:
            profile = self._cache[user_id]
            profile.increment_session()
            await self._save_profile(profile)
            return profile

        # Try to load from storage
        profile = await self._load_profile(user_id)

        if profile:
            profile.increment_session()
        else:
            # Create new profile
            profile = UserProfile(
                user_id=user_id,
                phone_number=self._mask_phone(phone_number),
                preferences=UserPreferences(language=language)
            )
            profile.total_sessions = 1
            logger.info(f"Created new user profile - id={user_id}")

        # Update cache
        self._cache[user_id] = profile
        self._manage_cache()

        await self._save_profile(profile)
        return profile

    def _mask_phone(self, phone_number: str) -> str:
        """Mask phone number for privacy (keep last 4 digits)."""
        if len(phone_number) > 4:
            return "X" * (len(phone_number) - 4) + phone_number[-4:]
        return phone_number

    async def _load_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load profile from storage."""
        file_path = self._get_profile_path(user_id)

        if not file_path.exists():
            return None

        try:
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                data = json.loads(content)
                return UserProfile.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading profile {user_id}: {e}")
            return None

    async def _save_profile(self, profile: UserProfile) -> None:
        """Save profile to storage."""
        async with self._lock:
            file_path = self._get_profile_path(profile.user_id)

            try:
                async with aiofiles.open(file_path, "w") as f:
                    await f.write(json.dumps(profile.to_dict(), indent=2))
            except Exception as e:
                logger.error(f"Error saving profile {profile.user_id}: {e}")

    def _manage_cache(self) -> None:
        """Evict oldest entries if cache is full."""
        if len(self._cache) > self.cache_size:
            # Remove oldest accessed profiles
            sorted_profiles = sorted(
                self._cache.items(),
                key=lambda x: x[1].last_interaction
            )
            for user_id, _ in sorted_profiles[:len(self._cache) - self.cache_size]:
                del self._cache[user_id]

    async def update_role(
        self,
        user_id: str,
        role: CareRole
    ) -> Optional[UserProfile]:
        """
        Update user's Care Role.

        Args:
            user_id: User identifier
            role: The Care Role the person registered with or confirmed

        Returns:
            Updated profile
        """
        profile = self._cache.get(user_id)
        if not profile:
            profile = await self._load_profile(user_id)

        if profile:
            profile.role = role
            await self._save_profile(profile)
            self._cache[user_id] = profile
            logger.info(f"Updated user role - id={user_id}, role={role.value}")
            return profile

        return None

    async def update_preferences(
        self,
        user_id: str,
        **preferences
    ) -> Optional[UserProfile]:
        """
        Update user preferences.

        Args:
            user_id: User identifier
            **preferences: Preference key-value pairs

        Returns:
            Updated profile
        """
        profile = self._cache.get(user_id)
        if not profile:
            profile = await self._load_profile(user_id)

        if profile:
            for key, value in preferences.items():
                if hasattr(profile.preferences, key):
                    if key == "communication_style" and isinstance(value, str):
                        value = CommunicationStyle(value)
                    setattr(profile.preferences, key, value)

            await self._save_profile(profile)
            self._cache[user_id] = profile
            logger.info(f"Updated user preferences - id={user_id}")
            return profile

        return None

    async def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get profile by user ID."""
        if user_id in self._cache:
            return self._cache[user_id]
        return await self._load_profile(user_id)

    async def record_query(self, user_id: str) -> None:
        """Record that user made a query."""
        profile = self._cache.get(user_id)
        if not profile:
            profile = await self._load_profile(user_id)

        if profile:
            profile.increment_query()
            await self._save_profile(profile)

    def get_system_context(self, profile: UserProfile) -> str:
        """
        Generate system context based on user profile.

        Args:
            profile: User profile

        Returns:
            Context string for LLM system prompt
        """
        context_parts = []

        # Role-specific context
        if profile.role == CareRole.PATIENT:
            context_parts.append(
                "The user is a patient. Use empathetic, simple language. "
                "Focus on comfort and practical self-care guidance."
            )
        elif profile.role == CareRole.FAMILY_CAREGIVER:
            context_parts.append(
                "The user is a family caregiver for a patient. Provide practical care guidance, "
                "acknowledge their emotional burden, and include self-care reminders for them."
            )
        elif profile.role == CareRole.ASHA_WORKER:
            context_parts.append(
                "The user is an ASHA Worker visiting the household. Give practical, "
                "step-by-step guidance they can act on in the home."
            )

        # Communication style
        if profile.preferences.communication_style == CommunicationStyle.SIMPLE:
            context_parts.append("Use simple, clear language without medical jargon.")
        elif profile.preferences.communication_style == CommunicationStyle.DETAILED:
            context_parts.append("Provide comprehensive explanations with reasoning.")
        elif profile.preferences.communication_style == CommunicationStyle.CLINICAL:
            context_parts.append("Use appropriate medical terminology.")
        elif profile.preferences.communication_style == CommunicationStyle.EMPATHETIC:
            context_parts.append(
                "Prioritize emotional support and validation before providing information."
            )

        # Returning user context
        if profile.total_sessions > 5:
            context_parts.append(
                f"This is a returning user with {profile.total_sessions} previous sessions. "
                "They are familiar with the service."
            )

        return " ".join(context_parts)

    async def get_statistics(self) -> Dict[str, Any]:
        """Get user profile statistics."""
        profiles = []

        for file_path in self.storage_path.glob("user_*.json"):
            try:
                async with aiofiles.open(file_path, "r") as f:
                    content = await f.read()
                    data = json.loads(content)
                    profiles.append(UserProfile.from_dict(data))
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        if not profiles:
            return {
                "total_users": 0,
                "by_role": {},
                "by_language": {},
                "avg_sessions": 0,
                "avg_queries": 0
            }

        by_role = {}
        for role in CareRole:
            by_role[role.value] = sum(1 for p in profiles if p.role == role)
        by_role["unknown"] = sum(1 for p in profiles if p.role is None)

        by_language = {}
        for p in profiles:
            lang = p.preferences.language
            by_language[lang] = by_language.get(lang, 0) + 1

        return {
            "total_users": len(profiles),
            "by_role": by_role,
            "by_language": by_language,
            "avg_sessions": sum(p.total_sessions for p in profiles) / len(profiles),
            "avg_queries": sum(p.total_queries for p in profiles) / len(profiles),
            "cached_users": len(self._cache)
        }
