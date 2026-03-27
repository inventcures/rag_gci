import csv
import json
import io
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


class EvaluationExporter:
    """Exports evaluation data in formats compatible with R/lme4 analysis."""

    def export_sus_csv(self) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "submission_id", "user_id", "site_id", "language",
            "item_1", "item_2", "item_3", "item_4", "item_5",
            "item_6", "item_7", "item_8", "item_9", "item_10",
            "sus_score", "completed_at"
        ])

        sus_dir = Path("./data/evaluation/sus_scores")
        for filepath in sorted(sus_dir.glob("*.json")):
            with open(filepath) as f:
                record = json.load(f)
            scores = record["scores"]
            writer.writerow([
                record["submission_id"], record["user_id"],
                record["site_id"], record["language"],
                *scores,
                record["sus_score"], record["completed_at"],
            ])

        return output.getvalue()

    def export_vignettes_csv(self) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "response_id", "user_id", "vignette_id", "with_tool",
            "response_text", "started_at", "completed_at", "duration_ms",
            "submitted_at"
        ])

        responses_dir = Path("./data/evaluation/vignette_responses")
        for filepath in sorted(responses_dir.glob("*.json")):
            with open(filepath) as f:
                record = json.load(f)
            writer.writerow([
                record["response_id"], record["user_id"],
                record["vignette_id"], record["with_tool"],
                record.get("response_text", ""),
                record["started_at"], record["completed_at"],
                record.get("duration_ms", ""),
                record.get("submitted_at", ""),
            ])

        return output.getvalue()

    def export_interaction_logs_csv(self, start_date: str = None, end_date: str = None) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "log_id", "user_id", "session_id", "event_type",
            "timestamp", "duration_ms", "language", "site_id", "is_offline"
        ])

        logs_dir = Path("./data/evaluation/interaction_logs")
        if not logs_dir.exists():
            return output.getvalue()

        for day_dir in sorted(logs_dir.iterdir()):
            if not day_dir.is_dir():
                continue
            if start_date and day_dir.name < start_date:
                continue
            if end_date and day_dir.name > end_date:
                continue
            for filepath in sorted(day_dir.glob("*.json")):
                with open(filepath) as f:
                    record = json.load(f)
                writer.writerow([
                    record["log_id"], record["user_id"],
                    record.get("session_id", ""),
                    record["event_type"],
                    record["timestamp"],
                    record.get("duration_ms", ""),
                    record.get("language", ""),
                    record.get("site_id", ""),
                    record.get("is_offline", False),
                ])

        return output.getvalue()
