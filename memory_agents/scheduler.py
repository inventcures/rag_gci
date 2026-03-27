"""Background consolidation scheduler for memory agents."""

import asyncio
import logging
import time
from typing import Dict, Optional

from memory_agents.memory_store import MemoryStore
from memory_agents.consolidate_agent import PatientConsolidateAgent

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_MINUTES = 30


class ConsolidationScheduler:
    """Runs periodic memory consolidation as an asyncio background task."""

    def __init__(
        self,
        memory_store: MemoryStore,
        interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    ):
        self._store = memory_store
        self._agent = PatientConsolidateAgent(memory_store)
        self._interval = interval_minutes * 60
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_consolidation: Dict[str, float] = {}

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.ensure_future(self._run_loop())
        logger.info(
            f"ConsolidationScheduler started (interval={self._interval // 60}min)"
        )

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None
        logger.info("ConsolidationScheduler stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_consolidation_times(self) -> Dict[str, float]:
        return dict(self._last_consolidation)

    async def _run_loop(self) -> None:
        await asyncio.sleep(5)

        while self._running:
            try:
                await self._consolidation_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consolidation cycle error: {e}")

            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                break

    async def _consolidation_cycle(self) -> None:
        patient_ids = await self._store.get_patients_with_unconsolidated()

        if not patient_ids:
            logger.debug("No patients with unconsolidated memories")
            return

        now = time.time()
        for pid in patient_ids:
            last = self._last_consolidation.get(pid, 0)
            if now - last < self._interval:
                continue

            try:
                insights = await self._agent.consolidate_patient(pid)
                self._last_consolidation[pid] = now
                if insights:
                    logger.info(
                        f"Consolidated patient {pid}: {len(insights)} new insights"
                    )
            except Exception as e:
                logger.error(f"Failed to consolidate patient {pid}: {e}")

    async def force_consolidate(self, patient_id: str) -> int:
        """Force immediate consolidation for a specific patient."""
        insights = await self._agent.consolidate_patient(patient_id)
        self._last_consolidation[patient_id] = time.time()
        return len(insights)
