"""SQLite-backed memory storage for patient observations and consolidated insights."""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

DB_PATH_DEFAULT = "data/memory_agents/patient_memory.db"

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS raw_memories (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    observation_id TEXT,
    summary TEXT NOT NULL,
    entities TEXT NOT NULL DEFAULT '[]',
    topics TEXT NOT NULL DEFAULT '[]',
    importance_score REAL NOT NULL DEFAULT 0.5,
    source_type TEXT NOT NULL DEFAULT 'app',
    timestamp REAL NOT NULL,
    consolidated INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_raw_patient ON raw_memories(patient_id);
CREATE INDEX IF NOT EXISTS idx_raw_consolidated ON raw_memories(consolidated);
CREATE INDEX IF NOT EXISTS idx_raw_timestamp ON raw_memories(timestamp);

CREATE TABLE IF NOT EXISTS consolidated_insights (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    insight_text TEXT NOT NULL,
    source_memory_ids TEXT NOT NULL DEFAULT '[]',
    insight_type TEXT NOT NULL DEFAULT 'trend',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_insight_patient ON consolidated_insights(patient_id);
CREATE INDEX IF NOT EXISTS idx_insight_type ON consolidated_insights(insight_type);

CREATE TABLE IF NOT EXISTS memory_connections (
    id TEXT PRIMARY KEY,
    memory_id_1 TEXT NOT NULL,
    memory_id_2 TEXT NOT NULL,
    connection_type TEXT NOT NULL,
    strength REAL NOT NULL DEFAULT 0.5,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conn_mem1 ON memory_connections(memory_id_1);
CREATE INDEX IF NOT EXISTS idx_conn_mem2 ON memory_connections(memory_id_2);
"""


@dataclass
class MemoryRecord:
    id: str
    patient_id: str
    observation_id: str
    summary: str
    entities: List[str]
    topics: List[str]
    importance_score: float
    source_type: str
    timestamp: float
    consolidated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "observation_id": self.observation_id,
            "summary": self.summary,
            "entities": self.entities,
            "topics": self.topics,
            "importance_score": self.importance_score,
            "source_type": self.source_type,
            "timestamp": self.timestamp,
            "consolidated": self.consolidated,
        }


@dataclass
class ConsolidatedInsight:
    id: str
    patient_id: str
    insight_text: str
    source_memory_ids: List[str]
    insight_type: str
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "insight_text": self.insight_text,
            "source_memory_ids": self.source_memory_ids,
            "insight_type": self.insight_type,
            "created_at": self.created_at,
        }


@dataclass
class MemoryConnection:
    id: str
    memory_id_1: str
    memory_id_2: str
    connection_type: str
    strength: float
    created_at: float


class MemoryStore:
    """SQLite-backed persistent memory store for patient data."""

    def __init__(self, db_path: str = DB_PATH_DEFAULT):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = None

    async def initialize(self) -> None:
        import aiosqlite
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.executescript(CREATE_TABLES_SQL)
        await self._db.commit()
        logger.info(f"MemoryStore initialized at {self._db_path}")

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def _ensure_db(self) -> None:
        if self._db is None:
            await self.initialize()

    async def store_memory(self, record: MemoryRecord) -> str:
        await self._ensure_db()
        await self._db.execute(
            "INSERT OR REPLACE INTO raw_memories "
            "(id, patient_id, observation_id, summary, entities, topics, "
            "importance_score, source_type, timestamp, consolidated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.patient_id,
                record.observation_id,
                record.summary,
                json.dumps(record.entities),
                json.dumps(record.topics),
                record.importance_score,
                record.source_type,
                record.timestamp,
                int(record.consolidated),
            ),
        )
        await self._db.commit()
        return record.id

    async def get_unconsolidated(self, patient_id: str) -> List[MemoryRecord]:
        await self._ensure_db()
        cursor = await self._db.execute(
            "SELECT id, patient_id, observation_id, summary, entities, topics, "
            "importance_score, source_type, timestamp, consolidated "
            "FROM raw_memories WHERE patient_id = ? AND consolidated = 0 "
            "ORDER BY timestamp ASC",
            (patient_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_memory(r) for r in rows]

    async def get_patient_memories(
        self,
        patient_id: str,
        limit: int = 100,
    ) -> List[MemoryRecord]:
        await self._ensure_db()
        cursor = await self._db.execute(
            "SELECT id, patient_id, observation_id, summary, entities, topics, "
            "importance_score, source_type, timestamp, consolidated "
            "FROM raw_memories WHERE patient_id = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (patient_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_memory(r) for r in rows]

    async def mark_consolidated(self, memory_ids: List[str]) -> None:
        await self._ensure_db()
        if not memory_ids:
            return
        placeholders = ",".join("?" for _ in memory_ids)
        await self._db.execute(
            f"UPDATE raw_memories SET consolidated = 1 WHERE id IN ({placeholders})",
            memory_ids,
        )
        await self._db.commit()

    async def store_insight(self, insight: ConsolidatedInsight) -> str:
        await self._ensure_db()
        await self._db.execute(
            "INSERT OR REPLACE INTO consolidated_insights "
            "(id, patient_id, insight_text, source_memory_ids, insight_type, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                insight.id,
                insight.patient_id,
                insight.insight_text,
                json.dumps(insight.source_memory_ids),
                insight.insight_type,
                insight.created_at,
            ),
        )
        await self._db.commit()
        return insight.id

    async def get_patient_insights(
        self,
        patient_id: str,
        insight_type: Optional[str] = None,
    ) -> List[ConsolidatedInsight]:
        await self._ensure_db()
        if insight_type:
            cursor = await self._db.execute(
                "SELECT id, patient_id, insight_text, source_memory_ids, insight_type, created_at "
                "FROM consolidated_insights WHERE patient_id = ? AND insight_type = ? "
                "ORDER BY created_at DESC",
                (patient_id, insight_type),
            )
        else:
            cursor = await self._db.execute(
                "SELECT id, patient_id, insight_text, source_memory_ids, insight_type, created_at "
                "FROM consolidated_insights WHERE patient_id = ? "
                "ORDER BY created_at DESC",
                (patient_id,),
            )
        rows = await cursor.fetchall()
        return [self._row_to_insight(r) for r in rows]

    async def store_connection(self, conn: MemoryConnection) -> str:
        await self._ensure_db()
        await self._db.execute(
            "INSERT OR REPLACE INTO memory_connections "
            "(id, memory_id_1, memory_id_2, connection_type, strength, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (conn.id, conn.memory_id_1, conn.memory_id_2, conn.connection_type, conn.strength, conn.created_at),
        )
        await self._db.commit()
        return conn.id

    async def get_patients_with_unconsolidated(self) -> List[str]:
        await self._ensure_db()
        cursor = await self._db.execute(
            "SELECT DISTINCT patient_id FROM raw_memories WHERE consolidated = 0"
        )
        rows = await cursor.fetchall()
        return [r[0] for r in rows]

    def _row_to_memory(self, row: tuple) -> MemoryRecord:
        return MemoryRecord(
            id=row[0],
            patient_id=row[1],
            observation_id=row[2],
            summary=row[3],
            entities=json.loads(row[4]) if row[4] else [],
            topics=json.loads(row[5]) if row[5] else [],
            importance_score=row[6],
            source_type=row[7],
            timestamp=row[8],
            consolidated=bool(row[9]),
        )

    def _row_to_insight(self, row: tuple) -> ConsolidatedInsight:
        return ConsolidatedInsight(
            id=row[0],
            patient_id=row[1],
            insight_text=row[2],
            source_memory_ids=json.loads(row[3]) if row[3] else [],
            insight_type=row[4],
            created_at=row[5],
        )
