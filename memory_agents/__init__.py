"""
Always-On Memory Agent System for Palli Sahayak.
Three-tier architecture: Ingest -> Consolidate -> Query.
Based on Google's Always-On Memory Agent pattern.
"""

from memory_agents.memory_store import MemoryStore, MemoryRecord, ConsolidatedInsight
from memory_agents.ingest_agent import PatientIngestAgent
from memory_agents.consolidate_agent import PatientConsolidateAgent
from memory_agents.query_agent import PatientQueryAgent, MemoryQueryResult
from memory_agents.scheduler import ConsolidationScheduler

__all__ = [
    "MemoryStore",
    "MemoryRecord",
    "ConsolidatedInsight",
    "PatientIngestAgent",
    "PatientConsolidateAgent",
    "PatientQueryAgent",
    "MemoryQueryResult",
    "ConsolidationScheduler",
]

__version__ = "1.0.0"
