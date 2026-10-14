"""
Per-interaction retrieval record for the release log (protocol 7.8).

Every call to `SimpleRAGPipeline.query()` appends one line here, whatever the
channel: which retrievers were asked for, which answered, the fusion settings,
the knowledge-graph version, the claim-check outcome and the stage timings,
stamped with the study release id. The rows hold no question or answer text and
the participant id is the same keyed pseudonym the study log uses, so this file
carries no identifiable data and stays on the deployment's own disk.

One JSONL file per UTC day under data/study/retrieval/ (RETRIEVAL_AUDIT_DIR
overrides). Append-only and best effort: a failed write is logged, never raised.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def audit_dir() -> Path:
    return Path(os.getenv("RETRIEVAL_AUDIT_DIR", "./data/study/retrieval"))


def record_retrieval(
    *,
    user_id: Optional[str],
    retrieval: Dict[str, Any],
    kg_check: Optional[Dict[str, Any]] = None,
    status: str = "success",
) -> Optional[Dict[str, Any]]:
    """Append one row. Returns the row written, or None on failure."""
    try:
        from study_logging import pseudonymise
        from study_release import get_study_release

        stamp = get_study_release().stamp()
        now = time.time()
        row = {
            "occurred_at": now,
            "occurred_at_iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "release_id": stamp.get("release_id"),
            "release_name": stamp.get("release_name"),
            "participant_id": pseudonymise(user_id or ""),
            "status": status,
            "retrieval": retrieval,
            "kg_check": kg_check or {},
        }
        directory = audit_dir()
        directory.mkdir(parents=True, exist_ok=True)
        day = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")
        path = directory / f"retrieval-{day}.jsonl"
        with _lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return row
    except Exception:
        logger.exception("Retrieval audit row was not written")
        return None
