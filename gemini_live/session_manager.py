"""
Session Manager for Gemini Live API

Handles session lifecycle management including:
- Session creation and cleanup
- Timeout management (14 minute limit before Gemini's 15 min max)
- Concurrent session limits (one per user)
- Session resumption support
- Background cleanup of expired sessions
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from dataclasses import dataclass, field

from .service import GeminiLiveService, GeminiLiveSession, GeminiLiveError
from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """
    Container for session metadata and state.

    Tracks session lifecycle information for management purposes.
    """
    session_id: str
    user_id: str
    session: GeminiLiveSession
    created_at: datetime
    last_activity: datetime
    language: str
    voice: str
    resumption_handle: Optional[str] = None

    def is_expired(self, timeout: timedelta) -> bool:
        """Check if session has expired based on inactivity timeout."""
        return datetime.now() - self.last_activity > timeout

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "language": self.language,
            "voice": self.voice,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "is_active": self.session.is_active if self.session else False,
            "has_resumption_handle": self.resumption_handle is not None,
        }


class SessionManager:
    """
    Manages lifecycle of Gemini Live sessions.

    Features:
    - Automatic session cleanup after timeout
    - Session resumption support
    - One session per user limit
    - Health monitoring
    - Graceful shutdown

    Usage:
        manager = SessionManager(gemini_service)
        await manager.start()

        # Get or create session for user
        session = await manager.get_or_create_session(
            user_id="user123",
            language="hi-IN"
        )

        # Use session...

        # Cleanup
        await manager.stop()
    """

    def __init__(
        self,
        gemini_service: GeminiLiveService,
        timeout_minutes: Optional[int] = None,
        max_sessions_per_user: int = 1,
        cleanup_interval_seconds: int = 60
    ):
        """
        Initialize SessionManager.

        Args:
            gemini_service: The GeminiLiveService instance
            timeout_minutes: Session timeout (default from config, max 14)
            max_sessions_per_user: Max concurrent sessions per user (default 1)
            cleanup_interval_seconds: How often to run cleanup (default 60s)
        """
        self.service = gemini_service
        self.config = get_config()

        # Session timeout (cap at 14 minutes, Gemini max is 15)
        timeout = timeout_minutes or self.config.session_timeout_minutes
        self.timeout = timedelta(minutes=min(timeout, 14))

        self.max_sessions_per_user = max_sessions_per_user
        self.cleanup_interval = cleanup_interval_seconds

        # Session storage
        self.sessions: Dict[str, SessionInfo] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id

        # Background task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            f"SessionManager initialized - "
            f"timeout={self.timeout}, max_per_user={self.max_sessions_per_user}"
        )

    async def start(self) -> None:
        """Start the session manager and background cleanup task."""
        if self._running:
            logger.warning("SessionManager already running")
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("SessionManager started")

    async def stop(self) -> None:
        """Stop the session manager and cleanup all sessions."""
        self._running = False

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        # Close all active sessions
        session_ids = list(self.sessions.keys())
        for session_id in session_ids:
            try:
                await self._close_session_by_id(session_id)
            except Exception as e:
                logger.error(f"Error closing session {session_id}: {e}")

        logger.info("SessionManager stopped")

    async def get_or_create_session(
        self,
        user_id: str,
        language: str = "en-IN",
        voice: str = "Aoede"
    ) -> GeminiLiveSession:
        """
        Get existing session or create new one for user.

        Ensures only one session per user. If an existing session exists
        and is still valid, returns it. Otherwise creates a new session.

        Args:
            user_id: Unique identifier for the user
            language: Language code (en-IN, hi-IN, mr-IN, ta-IN)
            voice: Voice name (Aoede, Puck, Kore, etc.)

        Returns:
            GeminiLiveSession ready for use

        Raises:
            GeminiLiveError: If session creation fails
        """
        # Check for existing valid session
        if user_id in self.user_sessions:
            session_id = self.user_sessions[user_id]
            if session_id in self.sessions:
                session_info = self.sessions[session_id]

                # Check if still valid
                if not session_info.is_expired(self.timeout):
                    if session_info.session.is_active:
                        session_info.update_activity()
                        logger.debug(
                            f"Reusing existing session {session_id} for user {user_id}"
                        )
                        return session_info.session

                # Session expired or inactive, close it
                logger.info(
                    f"Existing session {session_id} expired/inactive, "
                    f"creating new session for user {user_id}"
                )
                await self._close_session_by_id(session_id)

        # Create new session
        session_id = self._generate_session_id(user_id)

        try:
            session = await self.service.create_session(
                session_id=session_id,
                language=language,
                voice=voice
            )

            # Connect the session
            await session.connect()

            # Store session info
            now = datetime.now()
            session_info = SessionInfo(
                session_id=session_id,
                user_id=user_id,
                session=session,
                created_at=now,
                last_activity=now,
                language=language,
                voice=voice
            )

            self.sessions[session_id] = session_info
            self.user_sessions[user_id] = session_id

            logger.info(
                f"Created new session {session_id} for user {user_id} "
                f"(language={language}, voice={voice})"
            )

            return session

        except Exception as e:
            logger.error(f"Failed to create session for user {user_id}: {e}")
            raise GeminiLiveError(f"Session creation failed: {e}") from e

    async def close_session(self, user_id: str) -> bool:
        """
        Close a user's session.

        Args:
            user_id: User ID whose session to close

        Returns:
            True if session was closed, False if no session existed
        """
        if user_id not in self.user_sessions:
            logger.debug(f"No session found for user {user_id}")
            return False

        session_id = self.user_sessions[user_id]
        await self._close_session_by_id(session_id)
        return True

    async def _close_session_by_id(self, session_id: str) -> None:
        """Close session by session ID."""
        if session_id not in self.sessions:
            return

        session_info = self.sessions[session_id]

        try:
            # Disconnect the session
            if session_info.session:
                await session_info.session.disconnect()

            # Also close from service
            await self.service.close_session(session_id)

        except Exception as e:
            logger.error(f"Error closing session {session_id}: {e}")
        finally:
            # Remove from tracking
            user_id = session_info.user_id
            if user_id in self.user_sessions:
                del self.user_sessions[user_id]
            if session_id in self.sessions:
                del self.sessions[session_id]

            logger.info(f"Closed session {session_id} for user {user_id}")

    def _generate_session_id(self, user_id: str) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{user_id}_{timestamp}"

    async def _cleanup_loop(self) -> None:
        """Background task to cleanup expired sessions."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)

                if not self._running:
                    break

                await self._cleanup_expired_sessions()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup_expired_sessions(self) -> None:
        """Cleanup expired sessions."""
        expired_sessions = []

        for session_id, session_info in self.sessions.items():
            if session_info.is_expired(self.timeout):
                expired_sessions.append(session_id)
            elif not session_info.session.is_active:
                # Session became inactive (disconnected)
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            logger.info(f"Cleaning up expired session: {session_id}")
            try:
                await self._close_session_by_id(session_id)
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    def get_session_for_user(self, user_id: str) -> Optional[GeminiLiveSession]:
        """
        Get active session for a user without creating.

        Args:
            user_id: User ID

        Returns:
            GeminiLiveSession if exists and active, None otherwise
        """
        if user_id not in self.user_sessions:
            return None

        session_id = self.user_sessions[user_id]
        if session_id not in self.sessions:
            return None

        session_info = self.sessions[session_id]
        if session_info.is_expired(self.timeout):
            return None

        if not session_info.session.is_active:
            return None

        return session_info.session

    def get_session_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session info for a user.

        Args:
            user_id: User ID

        Returns:
            Session info dict or None
        """
        if user_id not in self.user_sessions:
            return None

        session_id = self.user_sessions[user_id]
        if session_id not in self.sessions:
            return None

        return self.sessions[session_id].to_dict()

    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get info for all active sessions."""
        return {
            session_id: info.to_dict()
            for session_id, info in self.sessions.items()
        }

    def get_status(self) -> Dict[str, Any]:
        """Get manager status for health checks."""
        active_count = sum(
            1 for info in self.sessions.values()
            if info.session and info.session.is_active
        )

        return {
            "manager": "SessionManager",
            "running": self._running,
            "total_sessions": len(self.sessions),
            "active_sessions": active_count,
            "unique_users": len(self.user_sessions),
            "timeout_minutes": self.timeout.total_seconds() / 60,
            "max_per_user": self.max_sessions_per_user,
            "cleanup_interval_seconds": self.cleanup_interval,
        }

    async def refresh_session(self, user_id: str) -> bool:
        """
        Refresh a session's activity timestamp.

        Args:
            user_id: User ID

        Returns:
            True if session was refreshed, False if no session
        """
        if user_id not in self.user_sessions:
            return False

        session_id = self.user_sessions[user_id]
        if session_id not in self.sessions:
            return False

        self.sessions[session_id].update_activity()
        return True

    async def switch_language(
        self,
        user_id: str,
        new_language: str,
        new_voice: Optional[str] = None
    ) -> GeminiLiveSession:
        """
        Switch language for a user's session.

        This closes the existing session and creates a new one
        with the new language setting.

        Args:
            user_id: User ID
            new_language: New language code
            new_voice: Optional new voice (keeps current if not specified)

        Returns:
            New GeminiLiveSession with updated language
        """
        # Get current voice if not specified
        voice = new_voice
        if not voice and user_id in self.user_sessions:
            session_id = self.user_sessions[user_id]
            if session_id in self.sessions:
                voice = self.sessions[session_id].voice

        voice = voice or "Aoede"

        # Close existing session
        await self.close_session(user_id)

        # Create new session with new language
        return await self.get_or_create_session(
            user_id=user_id,
            language=new_language,
            voice=voice
        )
