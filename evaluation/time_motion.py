import json
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

LOGS_DIR = Path("./data/evaluation/interaction_logs")


class TimeMotionAnalyzer:
    """
    Analyzes time-motion data from structured interaction logs.
    Computes task durations, time-to-first-action, and session metrics
    for the EVAH comparative effectiveness study.
    """

    def compute_session_metrics(self, session_logs: List[dict]) -> dict:
        if not session_logs:
            return {}

        sorted_logs = sorted(session_logs, key=lambda l: l.get("timestamp", 0))
        first_ts = sorted_logs[0].get("timestamp", 0)
        last_ts = sorted_logs[-1].get("timestamp", 0)
        total_duration_ms = int((last_ts - first_ts) * 1000)

        query_events = [l for l in sorted_logs if l.get("event_type") == "query"]
        observation_events = [l for l in sorted_logs if l.get("event_type") == "observation_created"]

        return {
            "session_duration_ms": total_duration_ms,
            "event_count": len(sorted_logs),
            "query_count": len(query_events),
            "observation_count": len(observation_events),
            "first_event_at": first_ts,
            "last_event_at": last_ts,
        }

    def compute_task_durations(self, logs: List[dict], task_type: str) -> List[int]:
        """Extract duration_ms for all events of a given task_type."""
        return [
            l["duration_ms"]
            for l in logs
            if l.get("event_type") == task_type and l.get("duration_ms") is not None
        ]

    def aggregate_by_user(self, logs: List[dict]) -> dict:
        """Group logs by user_id and compute per-user session metrics."""
        user_sessions = {}
        for log_entry in logs:
            uid = log_entry.get("user_id", "unknown")
            sid = log_entry.get("session_id", "default")
            key = (uid, sid)
            user_sessions.setdefault(key, []).append(log_entry)

        results = {}
        for (uid, sid), session_logs in user_sessions.items():
            results.setdefault(uid, []).append(
                self.compute_session_metrics(session_logs)
            )
        return results
