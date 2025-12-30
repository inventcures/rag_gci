"""
Analytics Dashboard

Provides unified dashboard data combining:
- Real-time metrics
- Usage analytics
- Clinical validation metrics
- System health status
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio

from .realtime_metrics import RealtimeMetrics, MetricType
from .usage_analytics import UsageAnalytics

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    realtime_window_seconds: int = 300
    trend_days: int = 7
    refresh_interval_seconds: int = 10
    enable_alerts: bool = True


class AnalyticsDashboard:
    """
    Unified Analytics Dashboard.

    Combines all analytics sources into a single dashboard API:
    - Real-time system metrics
    - Usage analytics
    - Clinical validation metrics
    - Personalization stats
    - System health
    """

    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        metrics_path: str = "data/metrics",
        analytics_path: str = "data/analytics"
    ):
        """
        Initialize the analytics dashboard.

        Args:
            config: Dashboard configuration
            metrics_path: Path for metrics storage
            analytics_path: Path for analytics storage
        """
        self.config = config or DashboardConfig()

        self.realtime = RealtimeMetrics(
            window_seconds=self.config.realtime_window_seconds,
            enable_alerts=self.config.enable_alerts
        )
        self.usage = UsageAnalytics(storage_path=analytics_path)

        # External components (can be set later)
        self._validation_metrics = None
        self._personalization_stats = None
        self._interaction_history = None

        self._last_refresh = datetime.now()

        logger.info("AnalyticsDashboard initialized")

    def set_validation_metrics(self, validation_metrics) -> None:
        """Set validation metrics component."""
        self._validation_metrics = validation_metrics

    def set_personalization_stats(self, profile_manager, context_memory) -> None:
        """Set personalization stats components."""
        self._personalization_stats = {
            "profile_manager": profile_manager,
            "context_memory": context_memory
        }

    def set_interaction_history(self, interaction_history) -> None:
        """Set interaction history component."""
        self._interaction_history = interaction_history

    async def record_query(
        self,
        user_id: str,
        query: str,
        response: str,
        language: str = "en-IN",
        query_type: str = "other",
        used_rag: bool = False,
        rag_success: bool = False,
        rag_latency_ms: float = 0,
        response_latency_ms: float = 0,
        validation_passed: Optional[bool] = None,
        validation_confidence: Optional[float] = None,
        sources_count: int = 0
    ) -> None:
        """
        Record a complete query event across all metrics.

        Args:
            user_id: User identifier
            query: User query
            response: System response
            language: Language code
            query_type: Type of query
            used_rag: Whether RAG was used
            rag_success: Whether RAG found sources
            rag_latency_ms: RAG query latency
            response_latency_ms: Total response latency
            validation_passed: Validation result
            validation_confidence: Validation confidence
            sources_count: Number of sources used
        """
        # Record in realtime metrics
        await self.realtime.record_query()
        await self.realtime.record_latency(
            MetricType.RESPONSE_LATENCY_MS,
            response_latency_ms
        )

        if used_rag:
            await self.realtime.record_rag_query(
                success=rag_success,
                latency_ms=rag_latency_ms,
                sources_count=sources_count
            )

        if validation_passed is not None:
            await self.realtime.record_validation(
                passed=validation_passed,
                confidence=validation_confidence or 0
            )

        # Record in usage analytics
        await self.usage.record_query(
            user_id=user_id,
            language=language,
            query_type=query_type,
            used_rag=used_rag,
            rag_success=rag_success,
            validation_passed=validation_passed
        )

    async def record_session_end(
        self,
        user_id: str,
        duration_seconds: float
    ) -> None:
        """Record session end."""
        await self.usage.record_session(user_id, duration_seconds)

    async def record_feedback(
        self,
        rating: int,
        is_positive: bool
    ) -> None:
        """
        Record user feedback.

        Args:
            rating: Rating (1-5)
            is_positive: Whether feedback was positive
        """
        await self.realtime.record_user_feedback(rating)
        await self.usage.record_feedback(is_positive)

    async def record_error(self, error_type: str = "general") -> None:
        """Record an error event."""
        await self.realtime.record_error(error_type)

    async def set_active_sessions(self, count: int) -> None:
        """Set active session count."""
        await self.realtime.set_active_sessions(count)

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get complete dashboard data.

        Returns:
            Comprehensive dashboard data
        """
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "refresh_interval_seconds": self.config.refresh_interval_seconds,
            "sections": {}
        }

        # System Health
        dashboard["health"] = await self.realtime.get_health_status()

        # Real-time Metrics
        realtime_data = await self.realtime.get_all_metrics()
        dashboard["sections"]["realtime"] = {
            "latency": {
                "response": realtime_data["latency"].get("response_latency_ms"),
                "rag": realtime_data["latency"].get("rag_latency_ms")
            },
            "throughput": realtime_data["rates"]["queries_per_minute"],
            "success_rates": {
                "rag": realtime_data["success_rates"].get("rag_success_rate"),
                "validation": realtime_data["success_rates"].get("validation_pass_rate")
            },
            "active_sessions": realtime_data["gauges"].get("active_sessions", 0),
            "total_queries": realtime_data["counters"].get("total_queries", 0)
        }

        # Usage Analytics
        usage_summary = await self.usage.get_summary()
        dashboard["sections"]["usage"] = {
            "today": usage_summary["today"],
            "week": usage_summary["week"],
            "trends": usage_summary["trends"]
        }

        # Language Distribution
        dashboard["sections"]["languages"] = await self.usage.get_language_stats(
            days=self.config.trend_days
        )

        # Hourly Distribution (for chart)
        hourly = await self.usage.get_hourly_distribution(days=self.config.trend_days)
        dashboard["sections"]["hourly_distribution"] = [
            {"hour": h, "queries": hourly.get(h, 0)}
            for h in range(24)
        ]

        # Validation Metrics (if available)
        if self._validation_metrics:
            try:
                val_data = await self._validation_metrics.get_dashboard_data()
                dashboard["sections"]["validation"] = val_data
            except Exception as e:
                logger.error(f"Error getting validation metrics: {e}")
                dashboard["sections"]["validation"] = None

        # Personalization Stats (if available)
        if self._personalization_stats:
            try:
                profile_stats = await self._personalization_stats["profile_manager"].get_statistics()
                context_stats = await self._personalization_stats["context_memory"].get_statistics()
                dashboard["sections"]["personalization"] = {
                    "profiles": profile_stats,
                    "patient_context": context_stats
                }
            except Exception as e:
                logger.error(f"Error getting personalization stats: {e}")
                dashboard["sections"]["personalization"] = None

        # Interaction History Stats (if available)
        if self._interaction_history:
            try:
                history_stats = await self._interaction_history.get_global_statistics()
                dashboard["sections"]["interactions"] = history_stats
            except Exception as e:
                logger.error(f"Error getting interaction stats: {e}")
                dashboard["sections"]["interactions"] = None

        self._last_refresh = datetime.now()
        return dashboard

    async def get_realtime_snapshot(self) -> Dict[str, Any]:
        """
        Get lightweight realtime snapshot for frequent updates.

        Returns:
            Minimal realtime data
        """
        metrics = await self.realtime.get_all_metrics()
        health = await self.realtime.get_health_status()

        return {
            "timestamp": datetime.now().isoformat(),
            "status": health["status"],
            "alerts_count": len(health["alerts"]),
            "queries_per_minute": metrics["rates"]["queries_per_minute"]["rate_per_minute"],
            "avg_latency_ms": metrics["latency"]["response_latency_ms"].get("avg"),
            "p95_latency_ms": metrics["latency"]["response_latency_ms"].get("p95"),
            "active_sessions": metrics["gauges"].get("active_sessions", 0),
            "error_rate": metrics["success_rates"]["error_rate"].get("rate"),
            "rag_success_rate": metrics["success_rates"]["rag_success_rate"].get("rate")
        }

    async def get_alerts(self) -> List[Dict[str, Any]]:
        """
        Get current alerts.

        Returns:
            List of active alerts
        """
        health = await self.realtime.get_health_status()
        return health["alerts"]

    async def get_trend_report(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get detailed trend report.

        Args:
            days: Number of days

        Returns:
            Trend analysis report
        """
        return await self.usage.get_trend_analysis(days)

    async def get_performance_report(self) -> Dict[str, Any]:
        """
        Get performance report.

        Returns:
            Performance metrics summary
        """
        metrics = await self.realtime.get_all_metrics()

        return {
            "timestamp": datetime.now().isoformat(),
            "latency": {
                "response": metrics["latency"]["response_latency_ms"],
                "rag": metrics["latency"]["rag_latency_ms"],
                "tts": metrics["latency"]["tts_latency_ms"],
                "stt": metrics["latency"]["stt_latency_ms"]
            },
            "throughput": {
                "queries_per_minute": metrics["rates"]["queries_per_minute"]
            },
            "quality": {
                "rag_success": metrics["success_rates"]["rag_success_rate"],
                "validation": metrics["success_rates"]["validation_pass_rate"],
                "user_satisfaction": metrics.get("user_satisfaction")
            },
            "reliability": {
                "error_rate": metrics["success_rates"]["error_rate"],
                "uptime_estimate": self._estimate_uptime(metrics)
            }
        }

    def _estimate_uptime(self, metrics: Dict[str, Any]) -> float:
        """Estimate uptime based on error rate."""
        error_rate = metrics["success_rates"]["error_rate"].get("rate", 0)
        if error_rate is None:
            return 1.0
        return round(1.0 - error_rate, 4)

    async def export_data(
        self,
        days: int = 7,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Export analytics data.

        Args:
            days: Number of days to export
            format: Export format (json only for now)

        Returns:
            Exported data
        """
        daily_stats = await self.usage.get_date_range_stats(days)

        return {
            "export_timestamp": datetime.now().isoformat(),
            "period_days": days,
            "daily_statistics": [s.to_dict() for s in daily_stats],
            "trends": await self.usage.get_trend_analysis(days),
            "language_distribution": await self.usage.get_language_stats(days),
            "hourly_distribution": await self.usage.get_hourly_distribution(days)
        }

    async def cleanup(self) -> Dict[str, int]:
        """
        Clean up old data.

        Returns:
            Cleanup results
        """
        analytics_cleaned = await self.usage.cleanup_old_data()

        return {
            "analytics_files_removed": analytics_cleaned
        }


# Helper function for creating dashboard with all components
async def create_full_dashboard(
    base_path: str = "data"
) -> AnalyticsDashboard:
    """
    Create dashboard with all analytics components.

    Args:
        base_path: Base path for data storage

    Returns:
        Configured AnalyticsDashboard
    """
    from ..clinical_validation import ValidationMetrics
    from ..personalization import UserProfileManager, ContextMemory, InteractionHistory

    dashboard = AnalyticsDashboard(
        metrics_path=f"{base_path}/metrics",
        analytics_path=f"{base_path}/analytics"
    )

    # Set up validation metrics
    validation_metrics = ValidationMetrics(storage_path=f"{base_path}/metrics")
    dashboard.set_validation_metrics(validation_metrics)

    # Set up personalization components
    profile_manager = UserProfileManager(storage_path=f"{base_path}/user_profiles")
    context_memory = ContextMemory(storage_path=f"{base_path}/patient_context")
    dashboard.set_personalization_stats(profile_manager, context_memory)

    # Set up interaction history
    interaction_history = InteractionHistory(storage_path=f"{base_path}/interaction_history")
    dashboard.set_interaction_history(interaction_history)

    return dashboard
