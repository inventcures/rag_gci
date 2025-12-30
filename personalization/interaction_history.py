"""
Interaction History Tracking

Maintains conversation history for:
- Session continuity
- Follow-up context
- Pattern analysis
- Quality improvement
"""

import json
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import asyncio
import aiofiles
import hashlib

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries."""
    SYMPTOM_INQUIRY = "symptom_inquiry"
    MEDICATION_QUESTION = "medication_question"
    CARE_GUIDANCE = "care_guidance"
    EMOTIONAL_SUPPORT = "emotional_support"
    CLARIFICATION = "clarification"
    GREETING = "greeting"
    FEEDBACK = "feedback"
    OTHER = "other"


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    turn_id: str
    timestamp: datetime
    query: str
    response: str
    query_type: QueryType
    language: str
    used_rag: bool
    rag_sources: List[str] = field(default_factory=list)
    response_time_ms: float = 0
    was_helpful: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "timestamp": self.timestamp.isoformat(),
            "query": self.query,
            "response": self.response,
            "query_type": self.query_type.value,
            "language": self.language,
            "used_rag": self.used_rag,
            "rag_sources": self.rag_sources,
            "response_time_ms": self.response_time_ms,
            "was_helpful": self.was_helpful,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        return cls(
            turn_id=data["turn_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            query=data["query"],
            response=data["response"],
            query_type=QueryType(data.get("query_type", "other")),
            language=data.get("language", "en-IN"),
            used_rag=data.get("used_rag", False),
            rag_sources=data.get("rag_sources", []),
            response_time_ms=data.get("response_time_ms", 0),
            was_helpful=data.get("was_helpful"),
            metadata=data.get("metadata", {})
        )


@dataclass
class Session:
    """A conversation session."""
    session_id: str
    user_id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    turns: List[ConversationTurn] = field(default_factory=list)
    language: str = "en-IN"
    channel: str = "voice"  # voice, whatsapp, web
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "turns": [t.to_dict() for t in self.turns],
            "language": self.language,
            "channel": self.channel,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        return cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            turns=[ConversationTurn.from_dict(t) for t in data.get("turns", [])],
            language=data.get("language", "en-IN"),
            channel=data.get("channel", "voice"),
            metadata=data.get("metadata", {})
        )

    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        end = self.ended_at or datetime.now()
        return (end - self.started_at).total_seconds()

    @property
    def turn_count(self) -> int:
        """Get number of conversation turns."""
        return len(self.turns)


class InteractionHistory:
    """
    Interaction History Manager.

    Features:
    - Session tracking
    - Conversation turn recording
    - History retrieval for context
    - Pattern analysis
    - Follow-up detection
    """

    # Keywords for query type classification
    QUERY_TYPE_KEYWORDS = {
        QueryType.SYMPTOM_INQUIRY: [
            "pain", "nausea", "vomiting", "breathless", "fatigue",
            "दर्द", "मतली", "थकान", "उल्टी"
        ],
        QueryType.MEDICATION_QUESTION: [
            "dose", "dosage", "medicine", "tablet", "morphine",
            "दवा", "खुराक", "गोली"
        ],
        QueryType.CARE_GUIDANCE: [
            "how to", "what should", "tips for", "care for",
            "कैसे करें", "क्या करना चाहिए"
        ],
        QueryType.EMOTIONAL_SUPPORT: [
            "scared", "afraid", "worry", "anxious", "sad", "depressed",
            "डर", "चिंता", "उदास"
        ],
        QueryType.CLARIFICATION: [
            "what do you mean", "can you repeat", "didn't understand",
            "समझ नहीं आया", "फिर से बताओ"
        ],
        QueryType.GREETING: [
            "hello", "hi", "namaste", "good morning",
            "नमस्ते", "हैलो"
        ]
    }

    # Recent context window (turns)
    CONTEXT_WINDOW = 5
    # Session timeout (minutes)
    SESSION_TIMEOUT_MINUTES = 30

    def __init__(
        self,
        storage_path: str = "data/interaction_history",
        max_history_days: int = 30
    ):
        """
        Initialize interaction history.

        Args:
            storage_path: Directory for history storage
            max_history_days: Days of history to retain
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.max_history_days = max_history_days
        self._active_sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()

        logger.info(f"InteractionHistory initialized - path={storage_path}")

    def _generate_id(self, prefix: str, content: str) -> str:
        """Generate unique ID."""
        data = f"{content}:{datetime.now().isoformat()}"
        return f"{prefix}_{hashlib.md5(data.encode()).hexdigest()[:10]}"

    def _get_user_history_path(self, user_id: str) -> Path:
        """Get history file path for user."""
        return self.storage_path / f"{user_id}_history.json"

    def _classify_query_type(self, query: str) -> QueryType:
        """Classify query type based on keywords."""
        query_lower = query.lower()

        for query_type, keywords in self.QUERY_TYPE_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return query_type

        return QueryType.OTHER

    async def start_session(
        self,
        user_id: str,
        language: str = "en-IN",
        channel: str = "voice",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Session:
        """
        Start a new conversation session.

        Args:
            user_id: User identifier
            language: Language code
            channel: Interaction channel
            metadata: Additional metadata

        Returns:
            New Session
        """
        session = Session(
            session_id=self._generate_id("sess", user_id),
            user_id=user_id,
            started_at=datetime.now(),
            language=language,
            channel=channel,
            metadata=metadata or {}
        )

        self._active_sessions[user_id] = session
        logger.info(f"Started session {session.session_id} for user {user_id}")

        return session

    async def get_or_create_session(
        self,
        user_id: str,
        language: str = "en-IN",
        channel: str = "voice"
    ) -> Session:
        """
        Get active session or create new one.

        Args:
            user_id: User identifier
            language: Language code
            channel: Interaction channel

        Returns:
            Active or new Session
        """
        # Check for active session
        if user_id in self._active_sessions:
            session = self._active_sessions[user_id]

            # Check if session timed out
            last_activity = session.turns[-1].timestamp if session.turns else session.started_at
            if datetime.now() - last_activity > timedelta(minutes=self.SESSION_TIMEOUT_MINUTES):
                # End old session and start new
                await self.end_session(user_id)
                return await self.start_session(user_id, language, channel)

            return session

        # Create new session
        return await self.start_session(user_id, language, channel)

    async def add_turn(
        self,
        user_id: str,
        query: str,
        response: str,
        language: str = "en-IN",
        used_rag: bool = False,
        rag_sources: Optional[List[str]] = None,
        response_time_ms: float = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationTurn:
        """
        Add a conversation turn to the session.

        Args:
            user_id: User identifier
            query: User's query
            response: System response
            language: Language code
            used_rag: Whether RAG was used
            rag_sources: Source documents used
            response_time_ms: Response generation time
            metadata: Additional metadata

        Returns:
            ConversationTurn
        """
        session = await self.get_or_create_session(user_id, language)

        turn = ConversationTurn(
            turn_id=self._generate_id("turn", query),
            timestamp=datetime.now(),
            query=query,
            response=response,
            query_type=self._classify_query_type(query),
            language=language,
            used_rag=used_rag,
            rag_sources=rag_sources or [],
            response_time_ms=response_time_ms,
            metadata=metadata or {}
        )

        session.turns.append(turn)
        logger.debug(f"Added turn {turn.turn_id} to session {session.session_id}")

        return turn

    async def record_feedback(
        self,
        user_id: str,
        turn_id: str,
        was_helpful: bool
    ) -> None:
        """
        Record feedback for a turn.

        Args:
            user_id: User identifier
            turn_id: Turn identifier
            was_helpful: Whether response was helpful
        """
        if user_id in self._active_sessions:
            session = self._active_sessions[user_id]
            for turn in session.turns:
                if turn.turn_id == turn_id:
                    turn.was_helpful = was_helpful
                    logger.debug(f"Recorded feedback for turn {turn_id}: {was_helpful}")
                    return

    async def end_session(self, user_id: str) -> Optional[Session]:
        """
        End and save a session.

        Args:
            user_id: User identifier

        Returns:
            Ended session
        """
        if user_id not in self._active_sessions:
            return None

        session = self._active_sessions.pop(user_id)
        session.ended_at = datetime.now()

        await self._save_session(session)
        logger.info(
            f"Ended session {session.session_id} - "
            f"turns={len(session.turns)}, duration={session.duration_seconds:.1f}s"
        )

        return session

    async def _save_session(self, session: Session) -> None:
        """Save session to user's history file."""
        async with self._lock:
            file_path = self._get_user_history_path(session.user_id)

            # Load existing history
            sessions = []
            if file_path.exists():
                try:
                    async with aiofiles.open(file_path, "r") as f:
                        content = await f.read()
                        data = json.loads(content) if content else []
                        sessions = [Session.from_dict(s) for s in data]
                except Exception as e:
                    logger.error(f"Error loading history: {e}")

            # Add new session
            sessions.append(session)

            # Remove old sessions
            cutoff = datetime.now() - timedelta(days=self.max_history_days)
            sessions = [s for s in sessions if s.started_at > cutoff]

            # Save
            try:
                async with aiofiles.open(file_path, "w") as f:
                    await f.write(json.dumps(
                        [s.to_dict() for s in sessions],
                        indent=2
                    ))
            except Exception as e:
                logger.error(f"Error saving history: {e}")

    async def get_recent_context(
        self,
        user_id: str,
        max_turns: int = CONTEXT_WINDOW
    ) -> List[ConversationTurn]:
        """
        Get recent conversation turns for context.

        Args:
            user_id: User identifier
            max_turns: Maximum turns to return

        Returns:
            List of recent turns
        """
        turns = []

        # Get from active session first
        if user_id in self._active_sessions:
            session = self._active_sessions[user_id]
            turns.extend(session.turns)

        # If not enough, load from history
        if len(turns) < max_turns:
            history = await self._load_user_history(user_id)
            for session in reversed(history):
                turns = session.turns + turns
                if len(turns) >= max_turns:
                    break

        return turns[-max_turns:]

    async def _load_user_history(self, user_id: str) -> List[Session]:
        """Load user's session history."""
        file_path = self._get_user_history_path(user_id)

        if not file_path.exists():
            return []

        try:
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                data = json.loads(content) if content else []
                return [Session.from_dict(s) for s in data]
        except Exception as e:
            logger.error(f"Error loading history for {user_id}: {e}")
            return []

    def get_context_summary(
        self,
        turns: List[ConversationTurn],
        max_chars: int = 500
    ) -> str:
        """
        Generate context summary from recent turns.

        Args:
            turns: List of conversation turns
            max_chars: Maximum characters in summary

        Returns:
            Summary string
        """
        if not turns:
            return "No recent conversation history."

        summary_parts = []
        total_chars = 0

        for turn in reversed(turns):
            # Truncate query/response if needed
            query = turn.query[:100] + "..." if len(turn.query) > 100 else turn.query
            response = turn.response[:150] + "..." if len(turn.response) > 150 else turn.response

            part = f"User: {query}\nAssistant: {response}"

            if total_chars + len(part) > max_chars:
                break

            summary_parts.insert(0, part)
            total_chars += len(part)

        return "Recent conversation:\n" + "\n\n".join(summary_parts)

    def is_followup_query(
        self,
        query: str,
        recent_turns: List[ConversationTurn]
    ) -> bool:
        """
        Detect if query is a follow-up to previous conversation.

        Args:
            query: Current query
            recent_turns: Recent conversation turns

        Returns:
            True if likely a follow-up
        """
        if not recent_turns:
            return False

        query_lower = query.lower()

        # Pronouns suggesting reference to previous topic
        followup_indicators = [
            "it", "that", "this", "those", "the same",
            "more about", "what about", "and", "also",
            "वो", "यह", "इसके बारे में", "और"
        ]

        for indicator in followup_indicators:
            if indicator in query_lower:
                return True

        # Short queries often are follow-ups
        if len(query.split()) < 5:
            return True

        return False

    def get_last_topic(
        self,
        recent_turns: List[ConversationTurn]
    ) -> Optional[str]:
        """
        Extract the last discussion topic.

        Args:
            recent_turns: Recent turns

        Returns:
            Topic string or None
        """
        if not recent_turns:
            return None

        last_turn = recent_turns[-1]
        return last_turn.query_type.value

    async def get_user_statistics(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get interaction statistics for a user.

        Args:
            user_id: User identifier

        Returns:
            Statistics dictionary
        """
        history = await self._load_user_history(user_id)

        if not history:
            return {
                "total_sessions": 0,
                "total_turns": 0,
                "avg_session_duration": 0,
                "by_query_type": {},
                "languages_used": []
            }

        total_turns = sum(len(s.turns) for s in history)
        avg_duration = sum(s.duration_seconds for s in history) / len(history)

        by_query_type: Dict[str, int] = {}
        languages: set = set()

        for session in history:
            languages.add(session.language)
            for turn in session.turns:
                qt = turn.query_type.value
                by_query_type[qt] = by_query_type.get(qt, 0) + 1

        return {
            "total_sessions": len(history),
            "total_turns": total_turns,
            "avg_session_duration": round(avg_duration, 1),
            "avg_turns_per_session": round(total_turns / len(history), 1),
            "by_query_type": by_query_type,
            "languages_used": list(languages)
        }

    async def get_global_statistics(self) -> Dict[str, Any]:
        """Get global interaction statistics."""
        all_sessions: List[Session] = []

        for file_path in self.storage_path.glob("*_history.json"):
            try:
                async with aiofiles.open(file_path, "r") as f:
                    content = await f.read()
                    data = json.loads(content) if content else []
                    all_sessions.extend(Session.from_dict(s) for s in data)
            except Exception:
                pass

        if not all_sessions:
            return {
                "total_users": 0,
                "total_sessions": 0,
                "total_turns": 0
            }

        total_turns = sum(len(s.turns) for s in all_sessions)
        unique_users = len(set(s.user_id for s in all_sessions))

        by_channel: Dict[str, int] = {}
        by_language: Dict[str, int] = {}
        by_query_type: Dict[str, int] = {}

        for session in all_sessions:
            by_channel[session.channel] = by_channel.get(session.channel, 0) + 1
            by_language[session.language] = by_language.get(session.language, 0) + 1

            for turn in session.turns:
                qt = turn.query_type.value
                by_query_type[qt] = by_query_type.get(qt, 0) + 1

        rag_turns = sum(
            1 for s in all_sessions for t in s.turns if t.used_rag
        )

        helpful_turns = [
            t for s in all_sessions for t in s.turns
            if t.was_helpful is not None
        ]
        helpful_rate = (
            sum(1 for t in helpful_turns if t.was_helpful) / len(helpful_turns)
            if helpful_turns else None
        )

        return {
            "total_users": unique_users,
            "total_sessions": len(all_sessions),
            "total_turns": total_turns,
            "avg_turns_per_session": round(total_turns / len(all_sessions), 1),
            "rag_usage_rate": round(rag_turns / total_turns * 100, 1) if total_turns else 0,
            "helpful_rate": round(helpful_rate * 100, 1) if helpful_rate else None,
            "by_channel": by_channel,
            "by_language": by_language,
            "by_query_type": by_query_type,
            "active_sessions": len(self._active_sessions)
        }

    async def cleanup_expired(self) -> int:
        """
        Clean up expired history files.

        Returns:
            Number of sessions removed
        """
        removed = 0
        cutoff = datetime.now() - timedelta(days=self.max_history_days)

        for file_path in self.storage_path.glob("*_history.json"):
            try:
                async with aiofiles.open(file_path, "r") as f:
                    content = await f.read()
                    data = json.loads(content) if content else []
                    sessions = [Session.from_dict(s) for s in data]

                original_count = len(sessions)
                sessions = [s for s in sessions if s.started_at > cutoff]
                removed += original_count - len(sessions)

                if sessions:
                    async with aiofiles.open(file_path, "w") as f:
                        await f.write(json.dumps(
                            [s.to_dict() for s in sessions],
                            indent=2
                        ))
                else:
                    file_path.unlink()

            except Exception as e:
                logger.error(f"Error cleaning up {file_path}: {e}")

        logger.info(f"Cleaned up {removed} expired sessions")
        return removed
