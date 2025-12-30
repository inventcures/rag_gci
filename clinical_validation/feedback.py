"""
User Feedback Collection System

Collects and manages user feedback on responses:
- Helpfulness ratings (1-5 stars)
- Issue reporting (incorrect, incomplete, harmful)
- Free-text comments
- Feedback aggregation and analysis
"""

import json
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import asyncio
import aiofiles

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback."""
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"
    INCORRECT = "incorrect"
    INCOMPLETE = "incomplete"
    HARMFUL = "harmful"
    CONFUSING = "confusing"
    OUT_OF_SCOPE = "out_of_scope"
    OTHER = "other"


class FeedbackChannel(Enum):
    """Channel through which feedback was received."""
    VOICE_PROMPT = "voice_prompt"
    WEB_UI = "web_ui"
    WHATSAPP = "whatsapp"
    API = "api"
    ADMIN = "admin"


@dataclass
class UserFeedback:
    """User feedback record."""
    feedback_id: str
    query: str
    response: str
    rating: int  # 1-5 stars
    feedback_type: FeedbackType
    channel: FeedbackChannel
    comment: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    language: str = "en-IN"
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "query": self.query,
            "response": self.response,
            "rating": self.rating,
            "feedback_type": self.feedback_type.value,
            "channel": self.channel.value,
            "comment": self.comment,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "language": self.language,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserFeedback":
        return cls(
            feedback_id=data["feedback_id"],
            query=data["query"],
            response=data["response"],
            rating=data["rating"],
            feedback_type=FeedbackType(data["feedback_type"]),
            channel=FeedbackChannel(data["channel"]),
            comment=data.get("comment"),
            session_id=data.get("session_id"),
            user_id=data.get("user_id"),
            language=data.get("language", "en-IN"),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {})
        )


class FeedbackCollector:
    """
    Collects and manages user feedback.

    Features:
    - Multi-channel feedback collection
    - Persistent storage
    - Aggregation and statistics
    - Issue tracking for negative feedback
    """

    # Voice prompts for feedback collection
    FEEDBACK_PROMPTS = {
        "en-IN": {
            "ask": "Was this information helpful? Say yes or no.",
            "thanks_positive": "Thank you for your feedback!",
            "thanks_negative": "I'm sorry this wasn't helpful. We'll work to improve.",
            "follow_up": "Would you like to tell us more about what was wrong?"
        },
        "hi-IN": {
            "ask": "क्या यह जानकारी उपयोगी थी? हां या नहीं बोलें।",
            "thanks_positive": "आपकी प्रतिक्रिया के लिए धन्यवाद!",
            "thanks_negative": "मुझे खेद है कि यह उपयोगी नहीं था। हम सुधार करने का प्रयास करेंगे।",
            "follow_up": "क्या आप हमें बताना चाहेंगे कि क्या गलत था?"
        },
        "mr-IN": {
            "ask": "ही माहिती उपयुक्त होती का? हो किंवा नाही म्हणा.",
            "thanks_positive": "तुमच्या अभिप्रायाबद्दल धन्यवाद!",
            "thanks_negative": "मला माफ करा हे उपयुक्त नव्हते. आम्ही सुधारण्याचा प्रयत्न करू.",
            "follow_up": "काय चुकीचे होते ते आम्हाला सांगाल का?"
        },
        "ta-IN": {
            "ask": "இந்த தகவல் பயனுள்ளதாக இருந்ததா? ஆம் அல்லது இல்லை என்று சொல்லுங்கள்.",
            "thanks_positive": "உங்கள் கருத்துக்கு நன்றி!",
            "thanks_negative": "இது பயனுள்ளதாக இல்லாததற்கு மன்னிக்கவும். நாங்கள் மேம்படுத்த முயற்சிப்போம்.",
            "follow_up": "என்ன தவறு என்று எங்களிடம் சொல்ல விரும்புகிறீர்களா?"
        }
    }

    def __init__(
        self,
        storage_path: str = "data/feedback",
        auto_prompt_rate: float = 0.2  # 20% of responses prompt for feedback
    ):
        """
        Initialize the feedback collector.

        Args:
            storage_path: Directory to store feedback records
            auto_prompt_rate: Rate at which to auto-prompt for feedback
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.auto_prompt_rate = auto_prompt_rate
        self._lock = asyncio.Lock()
        self._feedback_count = 0

        logger.info(f"FeedbackCollector initialized - path={storage_path}")

    def _generate_feedback_id(self) -> str:
        """Generate unique feedback ID."""
        import hashlib
        content = f"{datetime.now().isoformat()}:{self._feedback_count}"
        self._feedback_count += 1
        return f"fb_{hashlib.md5(content.encode()).hexdigest()[:10]}"

    def _get_storage_file(self) -> Path:
        """Get the storage file for current month."""
        month = datetime.now().strftime("%Y-%m")
        return self.storage_path / f"feedback_{month}.json"

    def should_prompt_feedback(self) -> bool:
        """Determine if we should prompt for feedback."""
        import random
        return random.random() < self.auto_prompt_rate

    def get_feedback_prompt(self, language: str) -> Dict[str, str]:
        """Get feedback prompts for the specified language."""
        return self.FEEDBACK_PROMPTS.get(
            language,
            self.FEEDBACK_PROMPTS["en-IN"]
        )

    async def collect_feedback(
        self,
        query: str,
        response: str,
        rating: int,
        feedback_type: FeedbackType,
        channel: FeedbackChannel,
        comment: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        language: str = "en-IN",
        metadata: Optional[Dict[str, Any]] = None
    ) -> UserFeedback:
        """
        Collect user feedback.

        Args:
            query: Original user query
            response: Response that was given
            rating: User rating (1-5)
            feedback_type: Type of feedback
            channel: Feedback channel
            comment: Optional user comment
            session_id: Session identifier
            user_id: User identifier
            language: Language code
            metadata: Additional metadata

        Returns:
            UserFeedback record
        """
        async with self._lock:
            feedback = UserFeedback(
                feedback_id=self._generate_feedback_id(),
                query=query,
                response=response,
                rating=max(1, min(5, rating)),  # Clamp to 1-5
                feedback_type=feedback_type,
                channel=channel,
                comment=comment,
                session_id=session_id,
                user_id=user_id,
                language=language,
                metadata=metadata or {}
            )

            await self._save_feedback(feedback)

            logger.info(
                f"Feedback collected - id={feedback.feedback_id}, "
                f"rating={rating}, type={feedback_type.value}"
            )

            return feedback

    async def collect_quick_feedback(
        self,
        query: str,
        response: str,
        is_helpful: bool,
        channel: FeedbackChannel,
        session_id: Optional[str] = None,
        language: str = "en-IN"
    ) -> UserFeedback:
        """
        Collect quick yes/no feedback.

        Args:
            query: Original query
            response: Response given
            is_helpful: Whether response was helpful
            channel: Feedback channel
            session_id: Session ID
            language: Language code

        Returns:
            UserFeedback record
        """
        return await self.collect_feedback(
            query=query,
            response=response,
            rating=5 if is_helpful else 2,
            feedback_type=FeedbackType.HELPFUL if is_helpful else FeedbackType.NOT_HELPFUL,
            channel=channel,
            session_id=session_id,
            language=language
        )

    async def report_issue(
        self,
        query: str,
        response: str,
        issue_type: FeedbackType,
        description: str,
        channel: FeedbackChannel,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        language: str = "en-IN"
    ) -> UserFeedback:
        """
        Report an issue with a response.

        Args:
            query: Original query
            response: Problematic response
            issue_type: Type of issue
            description: Issue description
            channel: Report channel
            session_id: Session ID
            user_id: User ID
            language: Language code

        Returns:
            UserFeedback record
        """
        return await self.collect_feedback(
            query=query,
            response=response,
            rating=1,
            feedback_type=issue_type,
            channel=channel,
            comment=description,
            session_id=session_id,
            user_id=user_id,
            language=language,
            metadata={"is_issue_report": True}
        )

    async def _save_feedback(self, feedback: UserFeedback) -> None:
        """Save feedback to storage."""
        file_path = self._get_storage_file()

        # Load existing
        records = []
        if file_path.exists():
            try:
                async with aiofiles.open(file_path, "r") as f:
                    content = await f.read()
                    records = json.loads(content) if content else []
            except Exception as e:
                logger.error(f"Error loading feedback: {e}")
                records = []

        # Add new
        records.append(feedback.to_dict())

        # Save
        async with aiofiles.open(file_path, "w") as f:
            await f.write(json.dumps(records, indent=2))

    async def get_feedback(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        feedback_type: Optional[FeedbackType] = None,
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
        limit: int = 100
    ) -> List[UserFeedback]:
        """
        Query feedback records.

        Args:
            start_date: Filter by start date
            end_date: Filter by end date
            feedback_type: Filter by feedback type
            min_rating: Minimum rating filter
            max_rating: Maximum rating filter
            limit: Maximum records to return

        Returns:
            List of UserFeedback records
        """
        all_feedback = await self._load_all_feedback()

        # Apply filters
        filtered = all_feedback

        if start_date:
            filtered = [f for f in filtered if f.created_at >= start_date]
        if end_date:
            filtered = [f for f in filtered if f.created_at <= end_date]
        if feedback_type:
            filtered = [f for f in filtered if f.feedback_type == feedback_type]
        if min_rating:
            filtered = [f for f in filtered if f.rating >= min_rating]
        if max_rating:
            filtered = [f for f in filtered if f.rating <= max_rating]

        # Sort by date (newest first)
        filtered.sort(key=lambda x: x.created_at, reverse=True)

        return filtered[:limit]

    async def _load_all_feedback(self) -> List[UserFeedback]:
        """Load all feedback from storage."""
        feedback = []

        for file_path in self.storage_path.glob("feedback_*.json"):
            try:
                async with aiofiles.open(file_path, "r") as f:
                    content = await f.read()
                    data = json.loads(content) if content else []
                    for item in data:
                        feedback.append(UserFeedback.from_dict(item))
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        return feedback

    async def get_statistics(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get feedback statistics.

        Args:
            days: Number of days to analyze

        Returns:
            Statistics dictionary
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=days)
        all_feedback = await self._load_all_feedback()
        recent = [f for f in all_feedback if f.created_at >= cutoff]

        if not recent:
            return {
                "total_feedback": 0,
                "period_days": days,
                "average_rating": None,
                "by_type": {},
                "by_channel": {},
                "satisfaction_rate": None
            }

        total = len(recent)

        # Average rating
        avg_rating = sum(f.rating for f in recent) / total

        # By type
        by_type = {}
        for ft in FeedbackType:
            by_type[ft.value] = sum(1 for f in recent if f.feedback_type == ft)

        # By channel
        by_channel = {}
        for ch in FeedbackChannel:
            by_channel[ch.value] = sum(1 for f in recent if f.channel == ch)

        # Satisfaction rate (4+ stars)
        satisfied = sum(1 for f in recent if f.rating >= 4)
        satisfaction_rate = satisfied / total if total > 0 else 0

        # Issues reported
        issues = sum(1 for f in recent if f.feedback_type in [
            FeedbackType.INCORRECT,
            FeedbackType.HARMFUL,
            FeedbackType.CONFUSING
        ])

        # By language
        by_language = {}
        for f in recent:
            lang = f.language
            if lang not in by_language:
                by_language[lang] = {"count": 0, "avg_rating": 0, "total_rating": 0}
            by_language[lang]["count"] += 1
            by_language[lang]["total_rating"] += f.rating

        for lang in by_language:
            by_language[lang]["avg_rating"] = (
                by_language[lang]["total_rating"] / by_language[lang]["count"]
            )
            del by_language[lang]["total_rating"]

        return {
            "total_feedback": total,
            "period_days": days,
            "average_rating": round(avg_rating, 2),
            "satisfaction_rate": round(satisfaction_rate * 100, 1),
            "issues_reported": issues,
            "by_type": by_type,
            "by_channel": by_channel,
            "by_language": by_language
        }

    async def get_negative_feedback(
        self,
        limit: int = 50
    ) -> List[UserFeedback]:
        """Get feedback with low ratings for review."""
        return await self.get_feedback(
            max_rating=2,
            limit=limit
        )

    async def get_issue_reports(
        self,
        limit: int = 50
    ) -> List[UserFeedback]:
        """Get reported issues."""
        all_feedback = await self._load_all_feedback()

        issues = [
            f for f in all_feedback
            if f.feedback_type in [
                FeedbackType.INCORRECT,
                FeedbackType.HARMFUL,
                FeedbackType.INCOMPLETE
            ]
        ]

        issues.sort(key=lambda x: x.created_at, reverse=True)
        return issues[:limit]
