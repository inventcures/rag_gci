import json
import time
import uuid
import random
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

VIGNETTES_DIR = Path("./data/evaluation/vignettes")
RESPONSES_DIR = Path("./data/evaluation/vignette_responses")
ASSIGNMENTS_DIR = Path("./data/evaluation/vignette_assignments")


class VignetteManager:
    """
    Manages the EVAH clinical vignette crossover assessment.

    Per EVAH section 6.4:
    - 20 standardized vignettes per participant
    - 10 with Palli Sahayak, 10 without
    - Randomized, counterbalanced assignment
    """

    def __init__(self):
        VIGNETTES_DIR.mkdir(parents=True, exist_ok=True)
        RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
        ASSIGNMENTS_DIR.mkdir(parents=True, exist_ok=True)

    def get_vignettes(self) -> List[dict]:
        vignettes = []
        for filepath in sorted(VIGNETTES_DIR.glob("*.json")):
            with open(filepath) as f:
                vignettes.append(json.load(f))
        return vignettes

    def assign_vignettes(self, user_id: str) -> List[dict]:
        """
        Assign 20 vignettes with randomized counterbalanced with/without-tool split.
        Returns list of {vignette_id, with_tool} assignments.
        """
        assignment_path = ASSIGNMENTS_DIR / f"{user_id}.json"
        if assignment_path.exists():
            with open(assignment_path) as f:
                return json.load(f)["assignments"]

        vignettes = self.get_vignettes()
        if len(vignettes) < 20:
            logger.warning(f"Only {len(vignettes)} vignettes available, expected 20")

        vignette_ids = [v["vignette_id"] for v in vignettes]
        random.shuffle(vignette_ids)

        assignments = []
        for i, vid in enumerate(vignette_ids):
            assignments.append({
                "vignette_id": vid,
                "with_tool": i < 10,
                "order": i + 1,
            })

        random.shuffle(assignments)

        record = {
            "user_id": user_id,
            "assignments": assignments,
            "assigned_at": time.time(),
        }
        with open(assignment_path, "w") as f:
            json.dump(record, f, indent=2)

        return assignments

    def submit_response(
        self,
        user_id: str,
        vignette_id: str,
        with_tool: bool,
        response_text: Optional[str],
        started_at: float,
        completed_at: float,
        metadata: Optional[dict] = None,
    ) -> dict:
        response_id = str(uuid.uuid4())
        record = {
            "response_id": response_id,
            "user_id": user_id,
            "vignette_id": vignette_id,
            "with_tool": with_tool,
            "response_text": response_text,
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_ms": int((completed_at - started_at) * 1000),
            "metadata": metadata or {},
            "submitted_at": time.time(),
        }

        filepath = RESPONSES_DIR / f"{response_id}.json"
        with open(filepath, "w") as f:
            json.dump(record, f, indent=2)

        logger.info(f"Vignette response: {vignette_id} by {user_id} (with_tool={with_tool})")
        return {"submission_id": response_id, "status": "accepted"}
