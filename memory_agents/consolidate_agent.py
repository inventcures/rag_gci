"""Patient memory consolidation agent - discovers patterns across observations."""

import json
import logging
import os
import uuid
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Optional

from memory_agents.memory_store import (
    MemoryStore,
    MemoryRecord,
    ConsolidatedInsight,
    MemoryConnection,
)

logger = logging.getLogger(__name__)

CONSOLIDATION_INTERVAL_MINUTES = 30
MIN_MEMORIES_FOR_CONSOLIDATION = 3


class PatientConsolidateAgent:
    """
    Background consolidation agent.
    Groups unconsolidated memories by entity and temporal proximity,
    uses LLM to discover trends, correlations, and risk patterns.
    """

    def __init__(self, memory_store: MemoryStore):
        self._store = memory_store
        self._groq_api_key = os.getenv("GROQ_API_KEY")
        self._groq_model = "llama-3.1-8b-instant"

    async def consolidate_patient(self, patient_id: str) -> List[ConsolidatedInsight]:
        raw_memories = await self._store.get_unconsolidated(patient_id)
        if len(raw_memories) < MIN_MEMORIES_FOR_CONSOLIDATION:
            logger.debug(
                f"Patient {patient_id}: {len(raw_memories)} unconsolidated memories "
                f"(need {MIN_MEMORIES_FOR_CONSOLIDATION})"
            )
            return []

        groups = self._group_memories(raw_memories)
        insights: List[ConsolidatedInsight] = []

        for group_key, memories in groups.items():
            if len(memories) < 2:
                continue

            group_insights = await self._generate_insights(patient_id, group_key, memories)
            insights.extend(group_insights)

            await self._create_connections(memories)

        consolidated_ids = [m.id for m in raw_memories]
        await self._store.mark_consolidated(consolidated_ids)

        logger.info(
            f"Consolidated {len(raw_memories)} memories for patient {patient_id}: "
            f"{len(insights)} insights generated"
        )
        return insights

    def _group_memories(
        self,
        memories: List[MemoryRecord],
    ) -> Dict[str, List[MemoryRecord]]:
        entity_groups: Dict[str, List[MemoryRecord]] = defaultdict(list)

        for mem in memories:
            if mem.entities:
                for entity in mem.entities:
                    entity_groups[entity].append(mem)
            else:
                entity_groups["_ungrouped"].append(mem)

        temporal_groups: Dict[str, List[MemoryRecord]] = defaultdict(list)
        for key, group in entity_groups.items():
            group.sort(key=lambda m: m.timestamp)
            current_subgroup: List[MemoryRecord] = []
            last_ts = 0.0

            for mem in group:
                gap_hours = (mem.timestamp - last_ts) / 3600 if last_ts > 0 else 0
                if gap_hours > 72 and current_subgroup:
                    subkey = f"{key}_t{int(current_subgroup[0].timestamp)}"
                    temporal_groups[subkey] = current_subgroup
                    current_subgroup = []

                current_subgroup.append(mem)
                last_ts = mem.timestamp

            if current_subgroup:
                subkey = f"{key}_t{int(current_subgroup[0].timestamp)}"
                temporal_groups[subkey] = current_subgroup

        return temporal_groups

    async def _generate_insights(
        self,
        patient_id: str,
        group_key: str,
        memories: List[MemoryRecord],
    ) -> List[ConsolidatedInsight]:
        llm_insights = await self._llm_consolidate(patient_id, group_key, memories)
        if llm_insights:
            return llm_insights

        return self._rule_based_consolidate(patient_id, group_key, memories)

    async def _llm_consolidate(
        self,
        patient_id: str,
        group_key: str,
        memories: List[MemoryRecord],
    ) -> Optional[List[ConsolidatedInsight]]:
        if not self._groq_api_key:
            return None

        try:
            import aiohttp

            memory_text = "\n".join(
                f"[{datetime.fromtimestamp(m.timestamp).strftime('%Y-%m-%d %H:%M')}] "
                f"(importance: {m.importance_score:.1f}) {m.summary}"
                for m in memories[:20]
            )

            prompt = (
                "Analyze these patient observations and identify clinical patterns. "
                "Return ONLY valid JSON with an array of insights:\n"
                '[{"text": "insight description", "type": "trend|correlation|risk"}]\n\n'
                "Look for:\n"
                "- Symptom severity trends (improving/worsening)\n"
                "- Medication-symptom temporal correlations\n"
                "- Emerging risk factors\n\n"
                f"Observations for entity group '{group_key}':\n{memory_text}\n\nJSON:"
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
                        "temperature": 0.2,
                        "max_tokens": 512,
                    },
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    if resp.status != 200:
                        return None

                    data = await resp.json()
                    raw = data["choices"][0]["message"]["content"].strip()
                    start = raw.find("[")
                    end = raw.rfind("]") + 1
                    if start >= 0 and end > start:
                        parsed = json.loads(raw[start:end])
                    elif raw.startswith("{"):
                        parsed = [json.loads(raw)]
                    else:
                        return None

                    now = datetime.utcnow().timestamp()
                    source_ids = [m.id for m in memories]
                    insights = []
                    for item in parsed:
                        insight = ConsolidatedInsight(
                            id=str(uuid.uuid4()),
                            patient_id=patient_id,
                            insight_text=item.get("text", ""),
                            source_memory_ids=source_ids,
                            insight_type=item.get("type", "trend"),
                            created_at=now,
                        )
                        await self._store.store_insight(insight)
                        insights.append(insight)
                    return insights

        except Exception as e:
            logger.debug(f"LLM consolidation failed: {e}")
            return None

    def _rule_based_consolidate(
        self,
        patient_id: str,
        group_key: str,
        memories: List[MemoryRecord],
    ) -> List[ConsolidatedInsight]:
        insights = []
        now = datetime.utcnow().timestamp()
        source_ids = [m.id for m in memories]

        importance_values = [m.importance_score for m in memories]
        if len(importance_values) >= 3:
            first_half = importance_values[: len(importance_values) // 2]
            second_half = importance_values[len(importance_values) // 2:]
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)

            if avg_second > avg_first + 0.15:
                entity = group_key.split("_t")[0]
                insight = ConsolidatedInsight(
                    id=str(uuid.uuid4()),
                    patient_id=patient_id,
                    insight_text=(
                        f"Worsening trend detected for {entity}: "
                        f"average severity increased from {avg_first:.1f} to {avg_second:.1f} "
                        f"over {len(memories)} observations"
                    ),
                    source_memory_ids=source_ids,
                    insight_type="trend",
                    created_at=now,
                )
                insights.append(insight)
            elif avg_first > avg_second + 0.15:
                entity = group_key.split("_t")[0]
                insight = ConsolidatedInsight(
                    id=str(uuid.uuid4()),
                    patient_id=patient_id,
                    insight_text=(
                        f"Improving trend detected for {entity}: "
                        f"average severity decreased from {avg_first:.1f} to {avg_second:.1f} "
                        f"over {len(memories)} observations"
                    ),
                    source_memory_ids=source_ids,
                    insight_type="trend",
                    created_at=now,
                )
                insights.append(insight)

        all_entities = set()
        for m in memories:
            all_entities.update(m.entities)
        med_entities = {e for e in all_entities if e in _KNOWN_MEDICATIONS}
        symptom_entities = {e for e in all_entities if e in _KNOWN_SYMPTOMS}
        if med_entities and symptom_entities:
            insight = ConsolidatedInsight(
                id=str(uuid.uuid4()),
                patient_id=patient_id,
                insight_text=(
                    f"Temporal correlation detected: medications ({', '.join(sorted(med_entities))}) "
                    f"and symptoms ({', '.join(sorted(symptom_entities))}) "
                    f"appear together in {len(memories)} observations"
                ),
                source_memory_ids=source_ids,
                insight_type="correlation",
                created_at=now,
            )
            insights.append(insight)

        for ins in insights:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._store.store_insight(ins))
            except RuntimeError:
                pass

        return insights

    async def _create_connections(self, memories: List[MemoryRecord]) -> None:
        for i in range(len(memories)):
            for j in range(i + 1, min(i + 5, len(memories))):
                shared_entities = set(memories[i].entities) & set(memories[j].entities)
                if shared_entities:
                    strength = len(shared_entities) / max(
                        len(set(memories[i].entities) | set(memories[j].entities)), 1
                    )
                    conn = MemoryConnection(
                        id=str(uuid.uuid4()),
                        memory_id_1=memories[i].id,
                        memory_id_2=memories[j].id,
                        connection_type="shared_entity",
                        strength=strength,
                        created_at=datetime.utcnow().timestamp(),
                    )
                    await self._store.store_connection(conn)


_KNOWN_MEDICATIONS = {
    "morphine", "fentanyl", "oxycodone", "paracetamol", "ibuprofen",
    "gabapentin", "pregabalin", "amitriptyline", "dexamethasone",
    "ondansetron", "metoclopramide", "haloperidol", "lorazepam",
    "lactulose", "senna", "bisacodyl", "diclofenac", "tramadol",
}

_KNOWN_SYMPTOMS = {
    "pain", "nausea", "vomiting", "constipation", "breathlessness", "dyspnea",
    "fatigue", "weakness", "anxiety", "depression", "insomnia", "delirium",
    "agitation", "appetite", "cough", "fever", "swelling", "edema",
}
