"""
Real-time Metrics Collection

Collects and aggregates real-time metrics:
- Response latency (P50, P95, P99)
- Throughput (requests/second)
- Error rates
- RAG performance
- System health
"""

import logging
import time
from typing import Optional, List, Dict, Any, Deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import asyncio
import statistics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked."""
    # Latency metrics
    RESPONSE_LATENCY_MS = "response_latency_ms"
    RAG_LATENCY_MS = "rag_latency_ms"
    TTS_LATENCY_MS = "tts_latency_ms"
    STT_LATENCY_MS = "stt_latency_ms"

    # Throughput metrics
    QUERIES_PER_MINUTE = "queries_per_minute"
    SESSIONS_ACTIVE = "sessions_active"

    # Success/Error metrics
    RAG_SUCCESS_RATE = "rag_success_rate"
    VALIDATION_PASS_RATE = "validation_pass_rate"
    ERROR_RATE = "error_rate"

    # Quality metrics
    USER_SATISFACTION = "user_satisfaction"
    SOURCE_QUALITY = "source_quality"


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricWindow:
    """Rolling window of metrics."""
    metric_type: MetricType
    window_size_seconds: int
    points: Deque[MetricPoint] = field(default_factory=lambda: deque())

    def add(self, value: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a data point."""
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            metadata=metadata or {}
        )
        self.points.append(point)
        self._cleanup()

    def _cleanup(self) -> None:
        """Remove expired points."""
        cutoff = datetime.now() - timedelta(seconds=self.window_size_seconds)
        while self.points and self.points[0].timestamp < cutoff:
            self.points.popleft()

    @property
    def values(self) -> List[float]:
        """Get all values in window."""
        self._cleanup()
        return [p.value for p in self.points]

    @property
    def count(self) -> int:
        """Get count of values."""
        self._cleanup()
        return len(self.points)

    def percentile(self, p: float) -> Optional[float]:
        """Calculate percentile."""
        values = self.values
        if not values:
            return None
        if len(values) == 1:
            return values[0]

        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * p
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_values) else f

        return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)

    def average(self) -> Optional[float]:
        """Calculate average."""
        values = self.values
        return sum(values) / len(values) if values else None

    def rate_per_minute(self) -> float:
        """Calculate rate per minute."""
        self._cleanup()
        if not self.points:
            return 0.0

        window_seconds = min(
            self.window_size_seconds,
            (datetime.now() - self.points[0].timestamp).total_seconds()
        )

        if window_seconds < 1:
            return 0.0

        return len(self.points) / window_seconds * 60


class RealtimeMetrics:
    """
    Real-time Metrics Collector.

    Features:
    - Rolling window aggregation
    - Percentile calculations
    - Rate calculations
    - Alert thresholds
    - Async-safe operations
    """

    DEFAULT_WINDOW_SECONDS = 300  # 5 minutes
    ALERT_THRESHOLDS = {
        MetricType.RESPONSE_LATENCY_MS: {"p95": 2000, "p99": 5000},
        MetricType.ERROR_RATE: {"avg": 0.05},
        MetricType.RAG_SUCCESS_RATE: {"avg": 0.90},
        MetricType.VALIDATION_PASS_RATE: {"avg": 0.95}
    }

    def __init__(
        self,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
        enable_alerts: bool = True
    ):
        """
        Initialize metrics collector.

        Args:
            window_seconds: Rolling window size
            enable_alerts: Enable threshold alerts
        """
        self.window_seconds = window_seconds
        self.enable_alerts = enable_alerts

        self._windows: Dict[MetricType, MetricWindow] = {}
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._lock = asyncio.Lock()

        # Initialize windows for all metric types
        for metric_type in MetricType:
            self._windows[metric_type] = MetricWindow(
                metric_type=metric_type,
                window_size_seconds=window_seconds
            )

        logger.info(f"RealtimeMetrics initialized - window={window_seconds}s")

    async def record_latency(
        self,
        metric_type: MetricType,
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a latency measurement.

        Args:
            metric_type: Type of latency metric
            latency_ms: Latency in milliseconds
            metadata: Additional context
        """
        async with self._lock:
            self._windows[metric_type].add(latency_ms, metadata)

            # Check alerts
            if self.enable_alerts:
                await self._check_latency_alert(metric_type)

    async def record_success(
        self,
        metric_type: MetricType,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a success/failure event.

        Args:
            metric_type: Type of success metric
            success: Whether operation succeeded
            metadata: Additional context
        """
        async with self._lock:
            self._windows[metric_type].add(1.0 if success else 0.0, metadata)

    async def increment_counter(self, name: str, amount: int = 1) -> None:
        """Increment a counter."""
        async with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    async def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        async with self._lock:
            self._gauges[name] = value

    async def record_query(self) -> None:
        """Record a query event."""
        async with self._lock:
            self._windows[MetricType.QUERIES_PER_MINUTE].add(1.0)
            self._counters["total_queries"] = self._counters.get("total_queries", 0) + 1

    async def record_error(self, error_type: str = "general") -> None:
        """Record an error event."""
        async with self._lock:
            self._windows[MetricType.ERROR_RATE].add(1.0, {"error_type": error_type})
            self._counters["total_errors"] = self._counters.get("total_errors", 0) + 1

    async def record_rag_query(
        self,
        success: bool,
        latency_ms: float,
        sources_count: int = 0
    ) -> None:
        """
        Record RAG query metrics.

        Args:
            success: Whether RAG found relevant sources
            latency_ms: RAG query latency
            sources_count: Number of sources retrieved
        """
        async with self._lock:
            self._windows[MetricType.RAG_SUCCESS_RATE].add(
                1.0 if success else 0.0,
                {"sources": sources_count}
            )
            self._windows[MetricType.RAG_LATENCY_MS].add(latency_ms)

    async def record_validation(
        self,
        passed: bool,
        confidence: float
    ) -> None:
        """
        Record validation result.

        Args:
            passed: Whether validation passed
            confidence: Confidence score
        """
        async with self._lock:
            self._windows[MetricType.VALIDATION_PASS_RATE].add(
                1.0 if passed else 0.0,
                {"confidence": confidence}
            )

    async def record_user_feedback(self, rating: int) -> None:
        """
        Record user feedback rating.

        Args:
            rating: User rating (1-5)
        """
        # Normalize to 0-1
        normalized = (rating - 1) / 4.0
        async with self._lock:
            self._windows[MetricType.USER_SATISFACTION].add(normalized)

    async def set_active_sessions(self, count: int) -> None:
        """Set active session count."""
        async with self._lock:
            self._gauges["active_sessions"] = count

    async def _check_latency_alert(self, metric_type: MetricType) -> None:
        """Check if latency exceeds threshold."""
        thresholds = self.ALERT_THRESHOLDS.get(metric_type)
        if not thresholds:
            return

        window = self._windows[metric_type]

        if "p95" in thresholds:
            p95 = window.percentile(0.95)
            if p95 and p95 > thresholds["p95"]:
                logger.warning(
                    f"Alert: {metric_type.value} P95 ({p95:.0f}ms) exceeds threshold ({thresholds['p95']}ms)"
                )

        if "p99" in thresholds:
            p99 = window.percentile(0.99)
            if p99 and p99 > thresholds["p99"]:
                logger.warning(
                    f"Alert: {metric_type.value} P99 ({p99:.0f}ms) exceeds threshold ({thresholds['p99']}ms)"
                )

    async def get_latency_stats(
        self,
        metric_type: MetricType
    ) -> Dict[str, Any]:
        """
        Get latency statistics.

        Args:
            metric_type: Latency metric type

        Returns:
            Statistics dictionary
        """
        async with self._lock:
            window = self._windows[metric_type]
            values = window.values

            if not values:
                return {
                    "count": 0,
                    "avg": None,
                    "p50": None,
                    "p95": None,
                    "p99": None,
                    "min": None,
                    "max": None
                }

            return {
                "count": len(values),
                "avg": round(window.average(), 2),
                "p50": round(window.percentile(0.50), 2),
                "p95": round(window.percentile(0.95), 2),
                "p99": round(window.percentile(0.99), 2),
                "min": round(min(values), 2),
                "max": round(max(values), 2)
            }

    async def get_rate_stats(
        self,
        metric_type: MetricType
    ) -> Dict[str, Any]:
        """
        Get rate statistics.

        Args:
            metric_type: Rate metric type

        Returns:
            Statistics dictionary
        """
        async with self._lock:
            window = self._windows[metric_type]

            return {
                "count": window.count,
                "rate_per_minute": round(window.rate_per_minute(), 2),
                "avg": round(window.average(), 4) if window.average() else None
            }

    async def get_success_rate(
        self,
        metric_type: MetricType
    ) -> Dict[str, Any]:
        """
        Get success rate statistics.

        Args:
            metric_type: Success rate metric type

        Returns:
            Statistics dictionary
        """
        async with self._lock:
            window = self._windows[metric_type]
            values = window.values

            if not values:
                return {"count": 0, "rate": None, "successful": 0, "failed": 0}

            successful = sum(1 for v in values if v == 1.0)
            failed = len(values) - successful

            return {
                "count": len(values),
                "rate": round(successful / len(values), 4),
                "successful": successful,
                "failed": failed
            }

    async def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all current metrics.

        Returns:
            Complete metrics snapshot
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "window_seconds": self.window_seconds,
            "latency": {},
            "rates": {},
            "success_rates": {},
            "counters": {},
            "gauges": {}
        }

        # Latency metrics
        for metric_type in [
            MetricType.RESPONSE_LATENCY_MS,
            MetricType.RAG_LATENCY_MS,
            MetricType.TTS_LATENCY_MS,
            MetricType.STT_LATENCY_MS
        ]:
            metrics["latency"][metric_type.value] = await self.get_latency_stats(metric_type)

        # Rate metrics
        metrics["rates"]["queries_per_minute"] = await self.get_rate_stats(
            MetricType.QUERIES_PER_MINUTE
        )

        # Success rates
        for metric_type in [
            MetricType.RAG_SUCCESS_RATE,
            MetricType.VALIDATION_PASS_RATE,
            MetricType.ERROR_RATE
        ]:
            metrics["success_rates"][metric_type.value] = await self.get_success_rate(metric_type)

        # User satisfaction
        sat_window = self._windows[MetricType.USER_SATISFACTION]
        sat_values = sat_window.values
        if sat_values:
            metrics["user_satisfaction"] = {
                "count": len(sat_values),
                "avg": round(sum(sat_values) / len(sat_values), 3),
                "rating_5_star": round((sum(sat_values) / len(sat_values)) * 4 + 1, 2)
            }

        # Counters and gauges
        async with self._lock:
            metrics["counters"] = dict(self._counters)
            metrics["gauges"] = dict(self._gauges)

        return metrics

    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get system health status.

        Returns:
            Health status with alerts
        """
        metrics = await self.get_all_metrics()
        alerts = []
        status = "healthy"

        # Check latency thresholds
        response_latency = metrics["latency"].get("response_latency_ms", {})
        if response_latency.get("p95") and response_latency["p95"] > 2000:
            alerts.append({
                "level": "warning",
                "metric": "response_latency_ms",
                "message": f"High P95 latency: {response_latency['p95']}ms"
            })
            status = "degraded"

        if response_latency.get("p99") and response_latency["p99"] > 5000:
            alerts.append({
                "level": "critical",
                "metric": "response_latency_ms",
                "message": f"Critical P99 latency: {response_latency['p99']}ms"
            })
            status = "unhealthy"

        # Check error rate
        error_stats = metrics["success_rates"].get("error_rate", {})
        if error_stats.get("rate") and error_stats["rate"] > 0.05:
            alerts.append({
                "level": "warning",
                "metric": "error_rate",
                "message": f"High error rate: {error_stats['rate']*100:.1f}%"
            })
            status = "degraded" if status == "healthy" else status

        # Check RAG success
        rag_stats = metrics["success_rates"].get("rag_success_rate", {})
        if rag_stats.get("rate") and rag_stats["rate"] < 0.90:
            alerts.append({
                "level": "warning",
                "metric": "rag_success_rate",
                "message": f"Low RAG success rate: {rag_stats['rate']*100:.1f}%"
            })

        # Check validation
        val_stats = metrics["success_rates"].get("validation_pass_rate", {})
        if val_stats.get("rate") and val_stats["rate"] < 0.95:
            alerts.append({
                "level": "warning",
                "metric": "validation_pass_rate",
                "message": f"Low validation pass rate: {val_stats['rate']*100:.1f}%"
            })

        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "alerts": alerts,
            "summary": {
                "queries_per_minute": metrics["rates"]["queries_per_minute"]["rate_per_minute"],
                "avg_latency_ms": response_latency.get("avg"),
                "error_rate": error_stats.get("rate"),
                "active_sessions": metrics["gauges"].get("active_sessions", 0)
            }
        }

    async def reset(self) -> None:
        """Reset all metrics."""
        async with self._lock:
            for window in self._windows.values():
                window.points.clear()
            self._counters.clear()
            self._gauges.clear()

        logger.info("Metrics reset")
