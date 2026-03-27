"""Patient memory query agent - retrieves and synthesizes answers with citations."""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

from memory_agents.memory_store import MemoryStore, MemoryRecord, ConsolidatedInsight

logger = logging.getLogger(__name__)


@dataclass
class MemoryQueryResult:
    answer: str
    cited_memory_ids: List[str]
    cited_insight_ids: List[str]
    raw_memories_used: int
    insights_used: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "cited_memory_ids": self.cited_memory_ids,
            "cited_insight_ids": self.cited_insight_ids,
            "raw_memories_used": self.raw_memories_used,
            "insights_used": self.insights_used,
        }


class PatientQueryAgent:
    """Retrieves and synthesizes answers from patient memory with source citations."""

    def __init__(self, memory_store: MemoryStore):
        self._store = memory_store
        self._groq_api_key = os.getenv("GROQ_API_KEY")
        self._groq_model = "llama-3.1-8b-instant"

    async def query(
        self,
        patient_id: str,
        question: str,
    ) -> MemoryQueryResult:
        """Answer questions about a patient's history with citations."""
        memories = await self._store.get_patient_memories(patient_id, limit=50)
        insights = await self._store.get_patient_insights(patient_id)

        relevant_memories = self._filter_relevant(question, memories)
        relevant_insights = self._filter_relevant_insights(question, insights)

        answer = await self._synthesize(question, relevant_memories, relevant_insights)
        if answer is None:
            answer = self._rule_based_synthesize(question, relevant_memories, relevant_insights)

        return MemoryQueryResult(
            answer=answer,
            cited_memory_ids=[m.id for m in relevant_memories],
            cited_insight_ids=[i.id for i in relevant_insights],
            raw_memories_used=len(relevant_memories),
            insights_used=len(relevant_insights),
        )

    def _filter_relevant(
        self,
        question: str,
        memories: List[MemoryRecord],
        max_results: int = 20,
    ) -> List[MemoryRecord]:
        q_lower = question.lower()
        q_words = set(w for w in q_lower.split() if len(w) > 3)

        scored = []
        for mem in memories:
            score = 0.0
            mem_text = (mem.summary + " " + " ".join(mem.entities)).lower()
            matching_words = sum(1 for w in q_words if w in mem_text)
            score += matching_words / max(len(q_words), 1)
            score += mem.importance_score * 0.3
            if score > 0.1:
                scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:max_results]]

    def _filter_relevant_insights(
        self,
        question: str,
        insights: List[ConsolidatedInsight],
        max_results: int = 10,
    ) -> List[ConsolidatedInsight]:
        q_lower = question.lower()
        q_words = set(w for w in q_lower.split() if len(w) > 3)

        scored = []
        for ins in insights:
            text_lower = ins.insight_text.lower()
            matching = sum(1 for w in q_words if w in text_lower)
            score = matching / max(len(q_words), 1)
            if score > 0.1:
                scored.append((score, ins))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [i for _, i in scored[:max_results]]

    async def _synthesize(
        self,
        question: str,
        memories: List[MemoryRecord],
        insights: List[ConsolidatedInsight],
    ) -> Optional[str]:
        if not self._groq_api_key:
            return None

        if not memories and not insights:
            return "No relevant patient observations or insights found for this question."

        try:
            import aiohttp

            memory_text = "\n".join(
                f"[Observation #{i+1}, {datetime.fromtimestamp(m.timestamp).strftime('%Y-%m-%d')}] {m.summary}"
                for i, m in enumerate(memories[:15])
            )

            insight_text = "\n".join(
                f"[Insight #{i+1}, type={ins.insight_type}] {ins.insight_text}"
                for i, ins in enumerate(insights[:5])
            )

            prompt = (
                "You are a clinical memory assistant. Answer the question about this patient "
                "using ONLY the observations and insights provided. "
                "Cite specific observations by number (e.g., 'Observation #3'). "
                "If information is insufficient, say so.\n\n"
                f"Question: {question}\n\n"
                f"Patient Observations:\n{memory_text}\n\n"
                f"Consolidated Insights:\n{insight_text}\n\n"
                "Answer with citations:"
            )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._groq_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._groq_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 512,
                    },
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.debug(f"LLM synthesis failed: {e}")
            return None

    def _rule_based_synthesize(
        self,
        question: str,
        memories: List[MemoryRecord],
        insights: List[ConsolidatedInsight],
    ) -> str:
        if not memories and not insights:
            return "No relevant patient observations or insights found for this question."

        parts = []

        if insights:
            parts.append("Consolidated insights:")
            for ins in insights[:3]:
                parts.append(f"- {ins.insight_text}")

        if memories:
            parts.append(f"\nBased on {len(memories)} relevant observation(s):")
            for mem in memories[:5]:
                date_str = datetime.fromtimestamp(mem.timestamp).strftime("%Y-%m-%d")
                parts.append(f"- [{date_str}] {mem.summary}")

        return "\n".join(parts)
