"""
Expert Sampling System for Clinical Validation

Implements random sampling of responses for human expert review:
- Configurable sampling rate (default 5%)
- Priority sampling for flagged responses
- Sample storage and retrieval
- Review status tracking
- Expert feedback integration
"""

import json
import logging
import random
import hashlib
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import asyncio
import aiofiles

logger = logging.getLogger(__name__)


class ReviewStatus(Enum):
    """Status of expert review."""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class SamplingPriority(Enum):
    """Priority levels for sampling."""
    NORMAL = "normal"
    HIGH = "high"       # Validation warnings
    CRITICAL = "critical"  # Validation errors or safety concerns


@dataclass
class ExpertReview:
    """Expert review feedback."""
    reviewer_id: str
    review_date: datetime
    status: ReviewStatus
    accuracy_score: float  # 0-10
    completeness_score: float  # 0-10
    safety_score: float  # 0-10
    comments: str = ""
    corrections: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reviewer_id": self.reviewer_id,
            "review_date": self.review_date.isoformat(),
            "status": self.status.value,
            "accuracy_score": self.accuracy_score,
            "completeness_score": self.completeness_score,
            "safety_score": self.safety_score,
            "comments": self.comments,
            "corrections": self.corrections,
            "tags": self.tags
        }


@dataclass
class SampleRecord:
    """A sampled query-response pair for expert review."""
    sample_id: str
    query: str
    response: str
    language: str
    sources: List[Dict[str, Any]]
    validation_result: Dict[str, Any]
    priority: SamplingPriority
    status: ReviewStatus = ReviewStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    expert_review: Optional[ExpertReview] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "query": self.query,
            "response": self.response,
            "language": self.language,
            "sources": self.sources,
            "validation_result": self.validation_result,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "expert_review": self.expert_review.to_dict() if self.expert_review else None,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SampleRecord":
        """Create SampleRecord from dictionary."""
        expert_review = None
        if data.get("expert_review"):
            er = data["expert_review"]
            expert_review = ExpertReview(
                reviewer_id=er["reviewer_id"],
                review_date=datetime.fromisoformat(er["review_date"]),
                status=ReviewStatus(er["status"]),
                accuracy_score=er["accuracy_score"],
                completeness_score=er["completeness_score"],
                safety_score=er["safety_score"],
                comments=er.get("comments", ""),
                corrections=er.get("corrections"),
                tags=er.get("tags", [])
            )

        return cls(
            sample_id=data["sample_id"],
            query=data["query"],
            response=data["response"],
            language=data["language"],
            sources=data["sources"],
            validation_result=data["validation_result"],
            priority=SamplingPriority(data["priority"]),
            status=ReviewStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            session_id=data.get("session_id"),
            user_id=data.get("user_id"),
            expert_review=expert_review,
            metadata=data.get("metadata", {})
        )


class ExpertSampler:
    """
    Expert Sampling System for collecting review samples.

    Features:
    - Random sampling with configurable rate
    - Priority-based sampling for flagged content
    - Persistent storage with async I/O
    - Sample retrieval by status
    - Statistics and reporting
    """

    DEFAULT_SAMPLE_RATE = 0.05  # 5% sampling rate
    HIGH_PRIORITY_SAMPLE_RATE = 0.50  # 50% for high priority
    CRITICAL_SAMPLE_RATE = 1.0  # 100% for critical

    def __init__(
        self,
        storage_path: str = "data/expert_samples",
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        max_samples_per_day: int = 100
    ):
        """
        Initialize the expert sampler.

        Args:
            storage_path: Directory to store sample records
            sample_rate: Base sampling rate (0.0 to 1.0)
            max_samples_per_day: Maximum samples to collect per day
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.sample_rate = sample_rate
        self.max_samples_per_day = max_samples_per_day

        # In-memory cache for today's samples
        self._today_samples: List[SampleRecord] = []
        self._last_date: Optional[str] = None

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        logger.info(
            f"ExpertSampler initialized - rate={sample_rate:.1%}, "
            f"max_daily={max_samples_per_day}, path={storage_path}"
        )

    def _generate_sample_id(self, query: str, response: str) -> str:
        """Generate unique sample ID."""
        content = f"{query}:{response}:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _get_today_file(self) -> Path:
        """Get the storage file for today's samples."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.storage_path / f"samples_{today}.json"

    def _determine_priority(
        self,
        validation_result: Dict[str, Any]
    ) -> SamplingPriority:
        """Determine sampling priority based on validation result."""
        issues = validation_result.get("issues", [])

        for issue in issues:
            if issue.get("level") == "critical":
                return SamplingPriority.CRITICAL
            if issue.get("level") == "error":
                return SamplingPriority.HIGH

        if validation_result.get("confidence_score", 1.0) < 0.7:
            return SamplingPriority.HIGH

        return SamplingPriority.NORMAL

    def _should_sample(self, priority: SamplingPriority) -> bool:
        """Determine if this response should be sampled."""
        # Check daily limit
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_date != today:
            self._today_samples = []
            self._last_date = today

        if len(self._today_samples) >= self.max_samples_per_day:
            return False

        # Determine rate based on priority
        if priority == SamplingPriority.CRITICAL:
            rate = self.CRITICAL_SAMPLE_RATE
        elif priority == SamplingPriority.HIGH:
            rate = self.HIGH_PRIORITY_SAMPLE_RATE
        else:
            rate = self.sample_rate

        return random.random() < rate

    async def maybe_sample(
        self,
        query: str,
        response: str,
        language: str,
        sources: List[Dict[str, Any]],
        validation_result: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[SampleRecord]:
        """
        Maybe sample this query-response pair for expert review.

        Args:
            query: User's query
            response: Generated response
            language: Response language
            sources: Source documents used
            validation_result: Result from ClinicalValidator
            session_id: Optional session identifier
            user_id: Optional user identifier
            metadata: Additional metadata

        Returns:
            SampleRecord if sampled, None otherwise
        """
        priority = self._determine_priority(validation_result)

        if not self._should_sample(priority):
            return None

        async with self._lock:
            sample = SampleRecord(
                sample_id=self._generate_sample_id(query, response),
                query=query,
                response=response,
                language=language,
                sources=sources,
                validation_result=validation_result,
                priority=priority,
                session_id=session_id,
                user_id=user_id,
                metadata=metadata or {}
            )

            # Save sample
            await self._save_sample(sample)
            self._today_samples.append(sample)

            logger.info(
                f"Sampled response for expert review - id={sample.sample_id}, "
                f"priority={priority.value}"
            )

            return sample

    async def force_sample(
        self,
        query: str,
        response: str,
        language: str,
        sources: List[Dict[str, Any]],
        validation_result: Dict[str, Any],
        reason: str = "manual",
        **kwargs
    ) -> SampleRecord:
        """
        Force sample a response (bypass random sampling).

        Args:
            query: User's query
            response: Generated response
            language: Response language
            sources: Source documents used
            validation_result: Validation result
            reason: Reason for forced sampling
            **kwargs: Additional fields

        Returns:
            SampleRecord
        """
        async with self._lock:
            sample = SampleRecord(
                sample_id=self._generate_sample_id(query, response),
                query=query,
                response=response,
                language=language,
                sources=sources,
                validation_result=validation_result,
                priority=SamplingPriority.CRITICAL,
                metadata={"force_sample_reason": reason, **kwargs.get("metadata", {})}
            )

            await self._save_sample(sample)
            self._today_samples.append(sample)

            logger.info(f"Force sampled response - id={sample.sample_id}, reason={reason}")

            return sample

    async def _save_sample(self, sample: SampleRecord) -> None:
        """Save sample to storage file."""
        file_path = self._get_today_file()

        # Load existing samples
        samples = []
        if file_path.exists():
            try:
                async with aiofiles.open(file_path, "r") as f:
                    content = await f.read()
                    samples = json.loads(content) if content else []
            except Exception as e:
                logger.error(f"Error loading samples: {e}")
                samples = []

        # Add new sample
        samples.append(sample.to_dict())

        # Save back
        async with aiofiles.open(file_path, "w") as f:
            await f.write(json.dumps(samples, indent=2))

    async def get_pending_samples(
        self,
        limit: int = 50,
        priority: Optional[SamplingPriority] = None
    ) -> List[SampleRecord]:
        """
        Get samples pending expert review.

        Args:
            limit: Maximum samples to return
            priority: Filter by priority level

        Returns:
            List of pending SampleRecords
        """
        samples = await self._load_all_samples()

        pending = [
            s for s in samples
            if s.status == ReviewStatus.PENDING
            and (priority is None or s.priority == priority)
        ]

        # Sort by priority (critical first) and date
        pending.sort(
            key=lambda x: (
                x.priority != SamplingPriority.CRITICAL,
                x.priority != SamplingPriority.HIGH,
                x.created_at
            )
        )

        return pending[:limit]

    async def submit_review(
        self,
        sample_id: str,
        reviewer_id: str,
        status: ReviewStatus,
        accuracy_score: float,
        completeness_score: float,
        safety_score: float,
        comments: str = "",
        corrections: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[SampleRecord]:
        """
        Submit expert review for a sample.

        Args:
            sample_id: ID of the sample
            reviewer_id: ID of the reviewing expert
            status: Review status
            accuracy_score: Accuracy rating (0-10)
            completeness_score: Completeness rating (0-10)
            safety_score: Safety rating (0-10)
            comments: Review comments
            corrections: Suggested corrections
            tags: Classification tags

        Returns:
            Updated SampleRecord
        """
        async with self._lock:
            samples = await self._load_all_samples()

            for sample in samples:
                if sample.sample_id == sample_id:
                    sample.expert_review = ExpertReview(
                        reviewer_id=reviewer_id,
                        review_date=datetime.now(),
                        status=status,
                        accuracy_score=accuracy_score,
                        completeness_score=completeness_score,
                        safety_score=safety_score,
                        comments=comments,
                        corrections=corrections,
                        tags=tags or []
                    )
                    sample.status = status

                    # Save updated sample
                    await self._save_all_samples(samples)

                    logger.info(
                        f"Expert review submitted - sample={sample_id}, "
                        f"status={status.value}, scores=({accuracy_score}, {completeness_score}, {safety_score})"
                    )

                    return sample

            logger.warning(f"Sample not found for review: {sample_id}")
            return None

    async def _load_all_samples(self) -> List[SampleRecord]:
        """Load all samples from storage."""
        samples = []

        for file_path in sorted(self.storage_path.glob("samples_*.json")):
            try:
                async with aiofiles.open(file_path, "r") as f:
                    content = await f.read()
                    data = json.loads(content) if content else []
                    for item in data:
                        samples.append(SampleRecord.from_dict(item))
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        return samples

    async def _save_all_samples(self, samples: List[SampleRecord]) -> None:
        """Save all samples back to storage (grouped by date)."""
        # Group by date
        by_date: Dict[str, List[SampleRecord]] = {}
        for sample in samples:
            date_str = sample.created_at.strftime("%Y-%m-%d")
            if date_str not in by_date:
                by_date[date_str] = []
            by_date[date_str].append(sample)

        # Save each date's file
        for date_str, date_samples in by_date.items():
            file_path = self.storage_path / f"samples_{date_str}.json"
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(
                    [s.to_dict() for s in date_samples],
                    indent=2
                ))

    async def get_statistics(self) -> Dict[str, Any]:
        """Get sampling statistics."""
        samples = await self._load_all_samples()

        total = len(samples)
        if total == 0:
            return {
                "total_samples": 0,
                "pending": 0,
                "reviewed": 0,
                "approved": 0,
                "rejected": 0,
                "average_scores": None
            }

        by_status = {}
        for status in ReviewStatus:
            by_status[status.value] = sum(1 for s in samples if s.status == status)

        by_priority = {}
        for priority in SamplingPriority:
            by_priority[priority.value] = sum(1 for s in samples if s.priority == priority)

        # Calculate average scores for reviewed samples
        reviewed = [s for s in samples if s.expert_review]
        avg_scores = None
        if reviewed:
            avg_scores = {
                "accuracy": sum(s.expert_review.accuracy_score for s in reviewed) / len(reviewed),
                "completeness": sum(s.expert_review.completeness_score for s in reviewed) / len(reviewed),
                "safety": sum(s.expert_review.safety_score for s in reviewed) / len(reviewed)
            }

        return {
            "total_samples": total,
            "by_status": by_status,
            "by_priority": by_priority,
            "average_scores": avg_scores,
            "today_samples": len(self._today_samples),
            "daily_limit": self.max_samples_per_day,
            "sample_rate": self.sample_rate
        }
