"""
Validation Metrics Tracking

Tracks and reports clinical validation metrics:
- Accuracy scores
- Hallucination rates
- Expert agreement rates
- User satisfaction
- Response quality trends
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
class MetricSnapshot:
    """A point-in-time metric measurement."""
    timestamp: datetime
    metric_name: str
    value: float
    count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "value": self.value,
            "count": self.count,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricSnapshot":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metric_name=data["metric_name"],
            value=data["value"],
            count=data.get("count", 1),
            metadata=data.get("metadata", {})
        )


class ValidationMetrics:
    """
    Tracks and reports clinical validation metrics.

    Metrics tracked:
    - validation_confidence: Average confidence score
    - validation_pass_rate: % of responses passing validation
    - hallucination_rate: % of responses with hallucination warnings
    - safety_issue_rate: % of responses with safety issues
    - expert_accuracy: Average expert accuracy scores
    - expert_agreement: % agreement between validation and expert review
    - user_satisfaction: Average user feedback rating
    - response_time: Average response generation time
    """

    METRIC_NAMES = [
        "validation_confidence",
        "validation_pass_rate",
        "hallucination_rate",
        "safety_issue_rate",
        "citation_rate",
        "expert_accuracy",
        "expert_completeness",
        "expert_safety",
        "expert_agreement",
        "user_satisfaction",
        "response_time_ms",
        "rag_retrieval_success"
    ]

    def __init__(
        self,
        storage_path: str = "data/metrics",
        aggregation_interval_minutes: int = 60
    ):
        """
        Initialize metrics tracker.

        Args:
            storage_path: Directory to store metrics
            aggregation_interval_minutes: Interval for metric aggregation
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.aggregation_interval = timedelta(minutes=aggregation_interval_minutes)
        self._lock = asyncio.Lock()

        # In-memory buffers for current interval
        self._current_interval_start = datetime.now()
        self._buffers: Dict[str, List[float]] = defaultdict(list)

        logger.info(f"ValidationMetrics initialized - path={storage_path}")

    def _get_storage_file(self, date: datetime) -> Path:
        """Get storage file for a specific date."""
        date_str = date.strftime("%Y-%m-%d")
        return self.storage_path / f"metrics_{date_str}.json"

    async def record_validation(
        self,
        validation_result: Dict[str, Any],
        response_time_ms: Optional[float] = None
    ) -> None:
        """
        Record metrics from a validation result.

        Args:
            validation_result: Result from ClinicalValidator
            response_time_ms: Response generation time
        """
        async with self._lock:
            # Confidence score
            confidence = validation_result.get("confidence_score", 0)
            self._buffers["validation_confidence"].append(confidence)

            # Pass rate
            is_valid = 1.0 if validation_result.get("is_valid", False) else 0.0
            self._buffers["validation_pass_rate"].append(is_valid)

            # Issue rates
            issues = validation_result.get("issues", [])

            has_hallucination = any(
                i.get("category") == "hallucination" for i in issues
            )
            self._buffers["hallucination_rate"].append(1.0 if has_hallucination else 0.0)

            has_safety_issue = any(
                i.get("category") == "safety" and i.get("level") in ["error", "critical"]
                for i in issues
            )
            self._buffers["safety_issue_rate"].append(1.0 if has_safety_issue else 0.0)

            has_citation = "citation_check" in validation_result.get("checks_passed", [])
            self._buffers["citation_rate"].append(1.0 if has_citation else 0.0)

            # Response time
            if response_time_ms is not None:
                self._buffers["response_time_ms"].append(response_time_ms)

            # Check if we need to flush
            await self._maybe_flush()

    async def record_expert_review(
        self,
        accuracy_score: float,
        completeness_score: float,
        safety_score: float,
        validation_agreed: bool
    ) -> None:
        """
        Record metrics from an expert review.

        Args:
            accuracy_score: Expert accuracy rating (0-10)
            completeness_score: Expert completeness rating (0-10)
            safety_score: Expert safety rating (0-10)
            validation_agreed: Whether expert agreed with automated validation
        """
        async with self._lock:
            # Normalize to 0-1 scale
            self._buffers["expert_accuracy"].append(accuracy_score / 10.0)
            self._buffers["expert_completeness"].append(completeness_score / 10.0)
            self._buffers["expert_safety"].append(safety_score / 10.0)
            self._buffers["expert_agreement"].append(1.0 if validation_agreed else 0.0)

            await self._maybe_flush()

    async def record_user_feedback(
        self,
        rating: int  # 1-5
    ) -> None:
        """
        Record user feedback rating.

        Args:
            rating: User rating (1-5)
        """
        async with self._lock:
            # Normalize to 0-1 scale
            self._buffers["user_satisfaction"].append((rating - 1) / 4.0)
            await self._maybe_flush()

    async def record_rag_retrieval(
        self,
        success: bool
    ) -> None:
        """
        Record RAG retrieval success.

        Args:
            success: Whether RAG found relevant documents
        """
        async with self._lock:
            self._buffers["rag_retrieval_success"].append(1.0 if success else 0.0)
            await self._maybe_flush()

    async def _maybe_flush(self) -> None:
        """Flush metrics to storage if interval elapsed."""
        now = datetime.now()
        if now - self._current_interval_start >= self.aggregation_interval:
            await self._flush()
            self._current_interval_start = now

    async def _flush(self) -> None:
        """Flush current buffers to storage."""
        if not any(self._buffers.values()):
            return

        snapshots = []
        timestamp = self._current_interval_start

        for metric_name, values in self._buffers.items():
            if values:
                avg_value = sum(values) / len(values)
                snapshots.append(MetricSnapshot(
                    timestamp=timestamp,
                    metric_name=metric_name,
                    value=avg_value,
                    count=len(values)
                ))

        # Save to file
        file_path = self._get_storage_file(timestamp)

        existing = []
        if file_path.exists():
            try:
                async with aiofiles.open(file_path, "r") as f:
                    content = await f.read()
                    existing = json.loads(content) if content else []
            except Exception:
                existing = []

        existing.extend([s.to_dict() for s in snapshots])

        async with aiofiles.open(file_path, "w") as f:
            await f.write(json.dumps(existing, indent=2))

        # Clear buffers
        self._buffers.clear()

        logger.debug(f"Flushed {len(snapshots)} metric snapshots")

    async def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current (real-time) metrics from buffers.

        Returns:
            Dictionary of current metric values
        """
        async with self._lock:
            metrics = {}
            for metric_name in self.METRIC_NAMES:
                values = self._buffers.get(metric_name, [])
                if values:
                    metrics[metric_name] = {
                        "value": sum(values) / len(values),
                        "count": len(values)
                    }
                else:
                    metrics[metric_name] = {"value": None, "count": 0}

            return metrics

    async def get_metrics_history(
        self,
        metric_name: str,
        days: int = 7
    ) -> List[MetricSnapshot]:
        """
        Get historical metrics for a specific metric.

        Args:
            metric_name: Name of the metric
            days: Number of days of history

        Returns:
            List of MetricSnapshots
        """
        snapshots = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        current_date = start_date
        while current_date <= end_date:
            file_path = self._get_storage_file(current_date)
            if file_path.exists():
                try:
                    async with aiofiles.open(file_path, "r") as f:
                        content = await f.read()
                        data = json.loads(content) if content else []
                        for item in data:
                            if item["metric_name"] == metric_name:
                                snapshots.append(MetricSnapshot.from_dict(item))
                except Exception as e:
                    logger.error(f"Error loading metrics from {file_path}: {e}")

            current_date += timedelta(days=1)

        snapshots.sort(key=lambda x: x.timestamp)
        return snapshots

    async def get_summary(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.

        Args:
            days: Number of days to analyze

        Returns:
            Summary dictionary with all metrics
        """
        summary = {
            "period_days": days,
            "generated_at": datetime.now().isoformat(),
            "metrics": {}
        }

        for metric_name in self.METRIC_NAMES:
            history = await self.get_metrics_history(metric_name, days)

            if history:
                values = [s.value for s in history]
                counts = [s.count for s in history]
                total_count = sum(counts)

                # Weighted average
                weighted_avg = sum(v * c for v, c in zip(values, counts)) / total_count

                summary["metrics"][metric_name] = {
                    "current": values[-1] if values else None,
                    "average": weighted_avg,
                    "min": min(values),
                    "max": max(values),
                    "total_count": total_count,
                    "trend": self._calculate_trend(values)
                }
            else:
                summary["metrics"][metric_name] = {
                    "current": None,
                    "average": None,
                    "min": None,
                    "max": None,
                    "total_count": 0,
                    "trend": "insufficient_data"
                }

        # Add derived metrics
        summary["derived"] = await self._calculate_derived_metrics(summary["metrics"])

        return summary

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 3:
            return "insufficient_data"

        # Compare first half to second half
        mid = len(values) // 2
        first_half_avg = sum(values[:mid]) / mid if mid > 0 else 0
        second_half_avg = sum(values[mid:]) / (len(values) - mid)

        diff = second_half_avg - first_half_avg
        threshold = 0.05  # 5% change threshold

        if diff > threshold:
            return "improving"
        elif diff < -threshold:
            return "declining"
        else:
            return "stable"

    async def _calculate_derived_metrics(
        self,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate derived/composite metrics."""
        derived = {}

        # Overall quality score (weighted average of key metrics)
        weights = {
            "validation_pass_rate": 0.3,
            "expert_accuracy": 0.25,
            "user_satisfaction": 0.25,
            "citation_rate": 0.1,
            "rag_retrieval_success": 0.1
        }

        quality_sum = 0
        weight_sum = 0
        for metric_name, weight in weights.items():
            if metrics.get(metric_name, {}).get("average") is not None:
                quality_sum += metrics[metric_name]["average"] * weight
                weight_sum += weight

        if weight_sum > 0:
            derived["overall_quality_score"] = round(quality_sum / weight_sum, 3)
        else:
            derived["overall_quality_score"] = None

        # Safety score (1 - safety_issue_rate)
        if metrics.get("safety_issue_rate", {}).get("average") is not None:
            derived["safety_score"] = round(
                1 - metrics["safety_issue_rate"]["average"], 3
            )
        else:
            derived["safety_score"] = None

        # Hallucination-free rate
        if metrics.get("hallucination_rate", {}).get("average") is not None:
            derived["hallucination_free_rate"] = round(
                1 - metrics["hallucination_rate"]["average"], 3
            )
        else:
            derived["hallucination_free_rate"] = None

        return derived

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data formatted for dashboard display.

        Returns:
            Dashboard-ready data structure
        """
        summary = await self.get_summary(days=7)
        current = await self.get_current_metrics()

        # Format for dashboard
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "realtime": {},
            "weekly_summary": {},
            "trends": {},
            "alerts": []
        }

        # Real-time metrics
        for metric_name in ["validation_pass_rate", "user_satisfaction", "response_time_ms"]:
            if current.get(metric_name, {}).get("value") is not None:
                dashboard["realtime"][metric_name] = {
                    "value": round(current[metric_name]["value"], 3),
                    "count": current[metric_name]["count"]
                }

        # Weekly summary
        for metric_name, data in summary["metrics"].items():
            if data["average"] is not None:
                dashboard["weekly_summary"][metric_name] = round(data["average"], 3)

        # Trends
        for metric_name, data in summary["metrics"].items():
            if data["trend"] != "insufficient_data":
                dashboard["trends"][metric_name] = data["trend"]

        # Alerts
        if summary["derived"].get("safety_score") is not None:
            if summary["derived"]["safety_score"] < 0.95:
                dashboard["alerts"].append({
                    "type": "warning",
                    "metric": "safety_score",
                    "message": f"Safety score below threshold: {summary['derived']['safety_score']:.1%}"
                })

        if summary["derived"].get("hallucination_free_rate") is not None:
            if summary["derived"]["hallucination_free_rate"] < 0.95:
                dashboard["alerts"].append({
                    "type": "warning",
                    "metric": "hallucination_rate",
                    "message": f"Hallucination rate above threshold: {1-summary['derived']['hallucination_free_rate']:.1%}"
                })

        if summary["metrics"].get("user_satisfaction", {}).get("average") is not None:
            if summary["metrics"]["user_satisfaction"]["average"] < 0.7:
                dashboard["alerts"].append({
                    "type": "warning",
                    "metric": "user_satisfaction",
                    "message": f"User satisfaction below target: {summary['metrics']['user_satisfaction']['average']:.1%}"
                })

        dashboard["quality_score"] = summary["derived"].get("overall_quality_score")

        return dashboard

    async def force_flush(self) -> None:
        """Force flush all buffered metrics."""
        async with self._lock:
            await self._flush()
