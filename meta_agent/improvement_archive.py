"""Persistent archive of successful improvements (stepping stones)."""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

IMPROVEMENTS_DIR = "data/meta_agent/improvements"


@dataclass
class Improvement:
    id: str
    domain: str
    change_type: str
    description: str
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    improvement_delta: float
    source_site: Optional[str] = None
    transferable: bool = False
    created_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = datetime.utcnow().timestamp()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "change_type": self.change_type,
            "description": self.description,
            "before_metrics": self.before_metrics,
            "after_metrics": self.after_metrics,
            "improvement_delta": self.improvement_delta,
            "source_site": self.source_site,
            "transferable": self.transferable,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Improvement":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ImprovementArchive:
    """Persistent archive of successful improvements as stepping stones."""

    def __init__(self, archive_dir: str = IMPROVEMENTS_DIR):
        self._dir = Path(archive_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    async def archive_improvement(self, improvement: Improvement) -> str:
        filepath = self._dir / f"{improvement.id}.json"
        try:
            with open(filepath, "w") as f:
                json.dump(improvement.to_dict(), f, indent=2)
            logger.info(
                f"Archived improvement {improvement.id}: "
                f"delta={improvement.improvement_delta:.4f} "
                f"domain={improvement.domain}"
            )
        except Exception as e:
            logger.error(f"Failed to archive improvement {improvement.id}: {e}")
        return improvement.id

    async def get_improvements(
        self,
        domain: Optional[str] = None,
        limit: int = 100,
    ) -> List[Improvement]:
        results = []
        for filepath in sorted(self._dir.glob("*.json"), reverse=True):
            if len(results) >= limit:
                break
            try:
                with open(filepath) as f:
                    data = json.load(f)
                imp = Improvement.from_dict(data)
                if domain and imp.domain != domain:
                    continue
                results.append(imp)
            except Exception:
                continue
        return results

    async def get_transferable_improvements(
        self,
        source_site: str,
        target_site: str,
    ) -> List[Improvement]:
        """Find improvements from source_site that may transfer to target_site."""
        all_improvements = await self.get_improvements()
        return [
            imp for imp in all_improvements
            if imp.transferable
            and imp.source_site == source_site
            and imp.improvement_delta > 0
        ]

    async def get_best_improvements(
        self,
        top_n: int = 10,
    ) -> List[Improvement]:
        all_improvements = await self.get_improvements()
        all_improvements.sort(key=lambda i: i.improvement_delta, reverse=True)
        return all_improvements[:top_n]

    async def count(self, domain: Optional[str] = None) -> int:
        if domain:
            return len(await self.get_improvements(domain=domain))
        return len(list(self._dir.glob("*.json")))
