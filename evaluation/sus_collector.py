import json
import time
import uuid
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

DATA_DIR = Path("./data/evaluation/sus_scores")


class SusCollector:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def submit_score(
        self,
        user_id: str,
        scores: List[int],
        site_id: str,
        language: str,
        completed_at: float,
    ) -> dict:
        if len(scores) != 10 or not all(1 <= s <= 5 for s in scores):
            raise ValueError("SUS requires exactly 10 scores, each 1-5")

        sus_score = self._calculate_sus(scores)
        submission_id = str(uuid.uuid4())

        record = {
            "submission_id": submission_id,
            "user_id": user_id,
            "scores": scores,
            "sus_score": sus_score,
            "site_id": site_id,
            "language": language,
            "completed_at": completed_at,
            "submitted_at": time.time(),
        }

        filepath = DATA_DIR / f"{submission_id}.json"
        with open(filepath, "w") as f:
            json.dump(record, f, indent=2)

        logger.info(f"SUS score submitted: {sus_score:.1f} by user {user_id} at {site_id}")
        return {"submission_id": submission_id, "sus_score": sus_score, "status": "accepted"}

    def _calculate_sus(self, scores: List[int]) -> float:
        """Standard SUS calculation: odd items (score-1), even items (5-score), sum * 2.5"""
        adjusted = []
        for i, score in enumerate(scores):
            if (i + 1) % 2 == 1:
                adjusted.append(score - 1)
            else:
                adjusted.append(5 - score)
        return sum(adjusted) * 2.5

    def get_scores_by_site(self, site_id: str) -> List[dict]:
        results = []
        for filepath in DATA_DIR.glob("*.json"):
            with open(filepath) as f:
                record = json.load(f)
            if record.get("site_id") == site_id:
                results.append(record)
        return sorted(results, key=lambda r: r["submitted_at"])
