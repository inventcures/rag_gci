"""
Usage Analytics

Provides analytics on usage patterns:
- Query patterns and trends
- User engagement metrics
- Geographic distribution
- Time-based analysis
- Topic popularity
"""

import json
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import asyncio
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class DailyStats:
    """Daily usage statistics."""
    date: str
    total_queries: int = 0
    unique_users: int = 0
    total_sessions: int = 0
    avg_session_duration: float = 0
    queries_by_hour: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    queries_by_language: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    queries_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    rag_queries: int = 0
    rag_success: int = 0
    validation_pass: int = 0
    validation_fail: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "total_queries": self.total_queries,
            "unique_users": self.unique_users,
            "total_sessions": self.total_sessions,
            "avg_session_duration": self.avg_session_duration,
            "queries_by_hour": dict(self.queries_by_hour),
            "queries_by_language": dict(self.queries_by_language),
            "queries_by_type": dict(self.queries_by_type),
            "rag_queries": self.rag_queries,
            "rag_success": self.rag_success,
            "validation_pass": self.validation_pass,
            "validation_fail": self.validation_fail,
            "positive_feedback": self.positive_feedback,
            "negative_feedback": self.negative_feedback
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DailyStats":
        stats = cls(date=data["date"])
        stats.total_queries = data.get("total_queries", 0)
        stats.unique_users = data.get("unique_users", 0)
        stats.total_sessions = data.get("total_sessions", 0)
        stats.avg_session_duration = data.get("avg_session_duration", 0)
        stats.queries_by_hour = defaultdict(int, data.get("queries_by_hour", {}))
        stats.queries_by_language = defaultdict(int, data.get("queries_by_language", {}))
        stats.queries_by_type = defaultdict(int, data.get("queries_by_type", {}))
        stats.rag_queries = data.get("rag_queries", 0)
        stats.rag_success = data.get("rag_success", 0)
        stats.validation_pass = data.get("validation_pass", 0)
        stats.validation_fail = data.get("validation_fail", 0)
        stats.positive_feedback = data.get("positive_feedback", 0)
        stats.negative_feedback = data.get("negative_feedback", 0)
        return stats


class UsageAnalytics:
    """
    Usage Analytics Engine.

    Features:
    - Daily statistics aggregation
    - Trend analysis
    - Usage pattern detection
    - Report generation
    """

    def __init__(
        self,
        storage_path: str = "data/analytics",
        retention_days: int = 90
    ):
        """
        Initialize usage analytics.

        Args:
            storage_path: Directory for analytics storage
            retention_days: Days of data to retain
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.retention_days = retention_days
        self._today_stats: Optional[DailyStats] = None
        self._today_users: set = set()
        self._lock = asyncio.Lock()

        logger.info(f"UsageAnalytics initialized - path={storage_path}")

    def _get_stats_file(self, date: str) -> Path:
        """Get stats file for a date."""
        return self.storage_path / f"stats_{date}.json"

    async def _get_today_stats(self) -> DailyStats:
        """Get or create today's stats."""
        today = datetime.now().strftime("%Y-%m-%d")

        if self._today_stats and self._today_stats.date == today:
            return self._today_stats

        # Try to load existing
        file_path = self._get_stats_file(today)
        if file_path.exists():
            try:
                async with aiofiles.open(file_path, "r") as f:
                    content = await f.read()
                    data = json.loads(content)
                    self._today_stats = DailyStats.from_dict(data)
                    return self._today_stats
            except Exception as e:
                logger.error(f"Error loading stats: {e}")

        # Create new
        self._today_stats = DailyStats(date=today)
        self._today_users = set()
        return self._today_stats

    async def _save_today_stats(self) -> None:
        """Save today's stats."""
        if not self._today_stats:
            return

        file_path = self._get_stats_file(self._today_stats.date)

        try:
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(self._today_stats.to_dict(), indent=2))
        except Exception as e:
            logger.error(f"Error saving stats: {e}")

    async def record_query(
        self,
        user_id: str,
        language: str = "en-IN",
        query_type: str = "other",
        used_rag: bool = False,
        rag_success: bool = False,
        validation_passed: Optional[bool] = None
    ) -> None:
        """
        Record a query event.

        Args:
            user_id: User identifier
            language: Query language
            query_type: Type of query
            used_rag: Whether RAG was used
            rag_success: Whether RAG found relevant sources
            validation_passed: Whether validation passed
        """
        async with self._lock:
            stats = await self._get_today_stats()

            stats.total_queries += 1
            stats.queries_by_hour[datetime.now().hour] += 1
            stats.queries_by_language[language] += 1
            stats.queries_by_type[query_type] += 1

            if user_id not in self._today_users:
                self._today_users.add(user_id)
                stats.unique_users = len(self._today_users)

            if used_rag:
                stats.rag_queries += 1
                if rag_success:
                    stats.rag_success += 1

            if validation_passed is not None:
                if validation_passed:
                    stats.validation_pass += 1
                else:
                    stats.validation_fail += 1

            await self._save_today_stats()

    async def record_session(
        self,
        user_id: str,
        duration_seconds: float
    ) -> None:
        """
        Record session end.

        Args:
            user_id: User identifier
            duration_seconds: Session duration
        """
        async with self._lock:
            stats = await self._get_today_stats()

            # Update running average
            total_duration = stats.avg_session_duration * stats.total_sessions
            stats.total_sessions += 1
            stats.avg_session_duration = (total_duration + duration_seconds) / stats.total_sessions

            await self._save_today_stats()

    async def record_feedback(
        self,
        is_positive: bool
    ) -> None:
        """
        Record feedback.

        Args:
            is_positive: Whether feedback was positive
        """
        async with self._lock:
            stats = await self._get_today_stats()

            if is_positive:
                stats.positive_feedback += 1
            else:
                stats.negative_feedback += 1

            await self._save_today_stats()

    async def get_daily_stats(self, date: str) -> Optional[DailyStats]:
        """
        Get stats for a specific date.

        Args:
            date: Date string (YYYY-MM-DD)

        Returns:
            DailyStats or None
        """
        if self._today_stats and self._today_stats.date == date:
            return self._today_stats

        file_path = self._get_stats_file(date)
        if not file_path.exists():
            return None

        try:
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                return DailyStats.from_dict(json.loads(content))
        except Exception as e:
            logger.error(f"Error loading stats for {date}: {e}")
            return None

    async def get_date_range_stats(
        self,
        days: int = 7
    ) -> List[DailyStats]:
        """
        Get stats for a date range.

        Args:
            days: Number of days

        Returns:
            List of DailyStats
        """
        stats = []
        end_date = datetime.now()

        for i in range(days):
            date = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
            day_stats = await self.get_daily_stats(date)
            if day_stats:
                stats.append(day_stats)

        stats.reverse()
        return stats

    async def get_trend_analysis(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Analyze usage trends.

        Args:
            days: Number of days to analyze

        Returns:
            Trend analysis results
        """
        stats = await self.get_date_range_stats(days)

        if len(stats) < 2:
            return {
                "status": "insufficient_data",
                "message": "Need at least 2 days of data for trend analysis"
            }

        # Calculate trends
        queries_trend = self._calculate_trend([s.total_queries for s in stats])
        users_trend = self._calculate_trend([s.unique_users for s in stats])

        # Peak hours analysis
        hourly_totals: Dict[int, int] = defaultdict(int)
        for s in stats:
            for hour, count in s.queries_by_hour.items():
                hourly_totals[int(hour)] += count

        peak_hours = sorted(hourly_totals.items(), key=lambda x: -x[1])[:3]

        # Language distribution
        lang_totals: Dict[str, int] = defaultdict(int)
        for s in stats:
            for lang, count in s.queries_by_language.items():
                lang_totals[lang] += count

        # Query type distribution
        type_totals: Dict[str, int] = defaultdict(int)
        for s in stats:
            for qtype, count in s.queries_by_type.items():
                type_totals[qtype] += count

        # RAG performance
        total_rag = sum(s.rag_queries for s in stats)
        total_rag_success = sum(s.rag_success for s in stats)
        rag_success_rate = total_rag_success / total_rag if total_rag else None

        # Validation performance
        total_val_pass = sum(s.validation_pass for s in stats)
        total_val_fail = sum(s.validation_fail for s in stats)
        total_val = total_val_pass + total_val_fail
        validation_rate = total_val_pass / total_val if total_val else None

        # Feedback analysis
        total_positive = sum(s.positive_feedback for s in stats)
        total_negative = sum(s.negative_feedback for s in stats)
        total_feedback = total_positive + total_negative
        satisfaction_rate = total_positive / total_feedback if total_feedback else None

        return {
            "period_days": days,
            "data_points": len(stats),
            "totals": {
                "queries": sum(s.total_queries for s in stats),
                "users": len(set().union(*[
                    set() for s in stats  # Would need actual user tracking
                ])),
                "sessions": sum(s.total_sessions for s in stats)
            },
            "trends": {
                "queries": queries_trend,
                "users": users_trend
            },
            "peak_hours": [{"hour": h, "count": c} for h, c in peak_hours],
            "language_distribution": dict(lang_totals),
            "query_type_distribution": dict(type_totals),
            "performance": {
                "rag_success_rate": round(rag_success_rate, 3) if rag_success_rate else None,
                "validation_rate": round(validation_rate, 3) if validation_rate else None,
                "satisfaction_rate": round(satisfaction_rate, 3) if satisfaction_rate else None
            }
        }

    def _calculate_trend(self, values: List[int]) -> str:
        """Calculate trend direction."""
        if len(values) < 3:
            return "insufficient_data"

        mid = len(values) // 2
        first_avg = sum(values[:mid]) / mid if mid else 0
        second_avg = sum(values[mid:]) / (len(values) - mid) if len(values) - mid else 0

        if first_avg == 0:
            return "growing" if second_avg > 0 else "stable"

        change = (second_avg - first_avg) / first_avg

        if change > 0.1:
            return "growing"
        elif change < -0.1:
            return "declining"
        else:
            return "stable"

    async def get_hourly_distribution(
        self,
        days: int = 7
    ) -> Dict[int, int]:
        """
        Get hourly query distribution.

        Args:
            days: Number of days to analyze

        Returns:
            Hour -> Count mapping
        """
        stats = await self.get_date_range_stats(days)

        hourly: Dict[int, int] = defaultdict(int)
        for s in stats:
            for hour, count in s.queries_by_hour.items():
                hourly[int(hour)] += count

        return dict(hourly)

    async def get_language_stats(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get language usage statistics.

        Args:
            days: Number of days

        Returns:
            Language statistics
        """
        stats = await self.get_date_range_stats(days)

        lang_totals: Dict[str, int] = defaultdict(int)
        for s in stats:
            for lang, count in s.queries_by_language.items():
                lang_totals[lang] += count

        total = sum(lang_totals.values())

        return {
            "total_queries": total,
            "by_language": {
                lang: {
                    "count": count,
                    "percentage": round(count / total * 100, 1) if total else 0
                }
                for lang, count in sorted(lang_totals.items(), key=lambda x: -x[1])
            }
        }

    async def get_summary(self) -> Dict[str, Any]:
        """
        Get usage summary.

        Returns:
            Summary dictionary
        """
        today = await self._get_today_stats()
        week_stats = await self.get_date_range_stats(7)

        week_queries = sum(s.total_queries for s in week_stats)
        week_sessions = sum(s.total_sessions for s in week_stats)

        return {
            "today": {
                "date": today.date,
                "queries": today.total_queries,
                "unique_users": today.unique_users,
                "sessions": today.total_sessions,
                "avg_session_duration": round(today.avg_session_duration, 1),
                "rag_success_rate": round(
                    today.rag_success / today.rag_queries, 3
                ) if today.rag_queries else None,
                "feedback_positive_rate": round(
                    today.positive_feedback / (today.positive_feedback + today.negative_feedback), 3
                ) if (today.positive_feedback + today.negative_feedback) else None
            },
            "week": {
                "total_queries": week_queries,
                "total_sessions": week_sessions,
                "avg_daily_queries": round(week_queries / len(week_stats), 1) if week_stats else 0,
                "data_days": len(week_stats)
            },
            "trends": await self.get_trend_analysis(7)
        }

    async def cleanup_old_data(self) -> int:
        """
        Clean up old analytics data.

        Returns:
            Number of files removed
        """
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        removed = 0

        for file_path in self.storage_path.glob("stats_*.json"):
            try:
                date_str = file_path.stem.replace("stats_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")

                if file_date < cutoff:
                    file_path.unlink()
                    removed += 1
            except Exception:
                pass

        logger.info(f"Cleaned up {removed} old analytics files")
        return removed
