"""
Webhook Handler for Sarvam Voice Session Events

Tracks Sarvam-powered voice sessions including:
- Session started/ended events
- Transcription updates
- Session analytics

Follows the pattern of bolna_integration/webhooks.py.
"""

import logging
from typing import Dict, Any, Optional, Callable, Awaitable
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SarvamCallRecord:
    """Record of a Sarvam-powered voice session."""
    session_id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_seconds: int = 0
    transcript: str = ""
    language: str = "hi-IN"
    stt_model: str = "saaras:v3"
    tts_voice: str = "meera"
    status: str = "in_progress"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds,
            "transcript": self.transcript,
            "language": self.language,
            "stt_model": self.stt_model,
            "tts_voice": self.tts_voice,
            "status": self.status,
            "metadata": self.metadata,
        }


class SarvamWebhookHandler:
    """
    Handler for Sarvam-related voice session events.

    Usage:
        handler = SarvamWebhookHandler()
        handler.register_handler("session_ended", my_callback)
        result = await handler.handle_event(event_data)
    """

    def __init__(self):
        self.active_sessions: Dict[str, SarvamCallRecord] = {}
        self.completed_sessions: Dict[str, SarvamCallRecord] = {}
        self.event_handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[None]]] = {}

    def register_handler(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any]], Awaitable[None]]
    ):
        """Register a custom handler for an event type."""
        self.event_handlers[event_type] = handler
        logger.info(f"Registered Sarvam handler for event: {event_type}")

    async def handle_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming webhook/session event."""
        event_type = event_data.get("event", event_data.get("type", "unknown"))
        session_id = event_data.get("session_id", event_data.get("id", "unknown"))

        logger.info(f"Processing Sarvam event: {event_type} for session {session_id}")

        if event_type in self.event_handlers:
            try:
                await self.event_handlers[event_type](event_data)
            except Exception as e:
                logger.error(f"Custom handler error for {event_type}: {e}")

        if event_type == "session_started":
            return await self._handle_session_started(event_data)
        elif event_type == "session_ended":
            return await self._handle_session_ended(event_data)
        elif event_type == "transcription":
            return await self._handle_transcription(event_data)
        else:
            logger.warning(f"Unknown Sarvam event type: {event_type}")
            return {"status": "ignored", "reason": "unknown_event", "event_type": event_type}

    async def _handle_session_started(self, data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = data.get("session_id", data.get("id"))

        record = SarvamCallRecord(
            session_id=session_id,
            started_at=datetime.now(),
            language=data.get("language", "hi-IN"),
            stt_model=data.get("stt_model", "saaras:v3"),
            tts_voice=data.get("tts_voice", "meera"),
            status="in_progress",
        )

        self.active_sessions[session_id] = record
        logger.info(f"Sarvam session started: {session_id}")

        return {"status": "recorded", "session_id": session_id, "event": "session_started"}

    async def _handle_session_ended(self, data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = data.get("session_id", data.get("id"))

        if session_id in self.active_sessions:
            record = self.active_sessions.pop(session_id)
            record.ended_at = datetime.now()
            record.duration_seconds = data.get("duration_seconds", 0)
            record.status = "completed"

            self.completed_sessions[session_id] = record
            logger.info(f"Sarvam session ended: {session_id}, duration: {record.duration_seconds}s")

        return {"status": "recorded", "session_id": session_id, "event": "session_ended"}

    async def _handle_transcription(self, data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = data.get("session_id", data.get("id"))
        text = data.get("text", "")
        role = data.get("role", "user")

        if session_id in self.active_sessions:
            record = self.active_sessions[session_id]
            prefix = "User: " if role == "user" else "Assistant: "
            record.transcript += f"\n{prefix}{text}"

        return {"status": "received", "session_id": session_id, "event": "transcription"}

    def get_session(self, session_id: str) -> Optional[SarvamCallRecord]:
        """Get a session record (active or completed)."""
        return self.active_sessions.get(session_id) or self.completed_sessions.get(session_id)

    def get_session_stats(self) -> Dict[str, Any]:
        total_completed = len(self.completed_sessions)
        total_duration = sum(s.duration_seconds for s in self.completed_sessions.values())

        language_counts: Dict[str, int] = {}
        for session in self.completed_sessions.values():
            lang = session.language
            language_counts[lang] = language_counts.get(lang, 0) + 1

        return {
            "active_sessions": len(self.active_sessions),
            "completed_sessions": total_completed,
            "total_duration_seconds": total_duration,
            "average_duration_seconds": total_duration / total_completed if total_completed > 0 else 0,
            "sessions_by_language": language_counts,
        }

    def get_recent_sessions(self, limit: int = 10) -> list:
        sessions = list(self.completed_sessions.values())
        sessions.sort(key=lambda x: x.ended_at or x.started_at, reverse=True)
        return [s.to_dict() for s in sessions[:limit]]

    def clear_old_sessions(self, hours: int = 24) -> int:
        cutoff = datetime.now()
        to_remove = []

        for session_id, record in self.completed_sessions.items():
            if record.ended_at:
                age = (cutoff - record.ended_at).total_seconds() / 3600
                if age > hours:
                    to_remove.append(session_id)

        for session_id in to_remove:
            del self.completed_sessions[session_id]

        if to_remove:
            logger.info(f"Cleared {len(to_remove)} old Sarvam session records")

        return len(to_remove)
