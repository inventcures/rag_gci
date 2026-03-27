"""Patient observation ingest agent - extracts structured memory from raw observations."""

import json
import logging
import os
import re
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from memory_agents.memory_store import MemoryStore, MemoryRecord

logger = logging.getLogger(__name__)

SYMPTOM_KEYWORDS = [
    "pain", "nausea", "vomiting", "constipation", "breathlessness", "dyspnea",
    "fatigue", "weakness", "anxiety", "depression", "insomnia", "delirium",
    "agitation", "appetite", "cough", "fever", "swelling", "edema",
    "bleeding", "wound", "ulcer", "itch", "rash",
]

MEDICATION_KEYWORDS = [
    "morphine", "fentanyl", "oxycodone", "paracetamol", "ibuprofen",
    "gabapentin", "pregabalin", "amitriptyline", "dexamethasone",
    "ondansetron", "metoclopramide", "haloperidol", "lorazepam",
    "lactulose", "senna", "bisacodyl", "diclofenac", "tramadol",
]

TOPIC_MAP = {
    "pain": "symptom_management",
    "nausea": "symptom_management",
    "vomiting": "symptom_management",
    "constipation": "symptom_management",
    "breathlessness": "symptom_management",
    "dyspnea": "symptom_management",
    "morphine": "medication",
    "fentanyl": "medication",
    "oxycodone": "medication",
    "paracetamol": "medication",
    "anxiety": "psychosocial",
    "depression": "psychosocial",
    "insomnia": "psychosocial",
    "appetite": "nutrition",
    "wound": "wound_care",
    "fever": "vital_signs",
}


class PatientIngestAgent:
    """Processes incoming observations and extracts structured memory records."""

    def __init__(self, memory_store: MemoryStore):
        self._store = memory_store
        self._groq_api_key = os.getenv("GROQ_API_KEY")
        self._groq_model = "llama-3.1-8b-instant"

    async def ingest_observation(self, observation: Dict[str, Any]) -> MemoryRecord:
        """
        Extract a structured memory record from a raw observation.
        Uses LLM extraction with rule-based fallback.
        """
        patient_id = observation.get("patient_id", "unknown")
        observation_id = observation.get("observation_id", str(uuid.uuid4()))
        source_type = observation.get("source_type", "app")
        timestamp = observation.get("timestamp", datetime.utcnow().timestamp())

        obs_text = self._observation_to_text(observation)

        extraction = await self._llm_extract(obs_text)
        if extraction is None:
            extraction = self._rule_based_extract(obs_text, observation)

        record = MemoryRecord(
            id=str(uuid.uuid4()),
            patient_id=patient_id,
            observation_id=observation_id,
            summary=extraction["summary"],
            entities=extraction["entities"],
            topics=extraction["topics"],
            importance_score=extraction["importance_score"],
            source_type=source_type,
            timestamp=timestamp,
        )

        await self._store.store_memory(record)
        logger.info(
            f"Ingested observation {observation_id} for patient {patient_id}: "
            f"entities={record.entities}, importance={record.importance_score:.2f}"
        )
        return record

    def _observation_to_text(self, observation: Dict[str, Any]) -> str:
        parts = []
        if observation.get("category"):
            parts.append(f"Category: {observation['category']}")
        if observation.get("entity_name"):
            parts.append(f"Entity: {observation['entity_name']}")
        if observation.get("value"):
            parts.append(f"Value: {observation['value']}")
        if observation.get("value_text"):
            parts.append(f"Details: {observation['value_text']}")
        if observation.get("severity") is not None:
            parts.append(f"Severity: {observation['severity']}/10")
        if observation.get("transcript"):
            parts.append(f"Transcript: {observation['transcript']}")
        return ". ".join(parts) if parts else str(observation)

    async def _llm_extract(self, obs_text: str) -> Optional[Dict[str, Any]]:
        if not self._groq_api_key:
            return None

        try:
            import aiohttp

            prompt = (
                "Extract structured information from this clinical observation. "
                "Return ONLY valid JSON with these fields:\n"
                '- "summary": one-sentence summary\n'
                '- "entities": list of medical entity strings (symptoms, medications, conditions)\n'
                '- "topics": list of topic strings (symptom_management, medication, psychosocial, nutrition, vital_signs, wound_care)\n'
                '- "importance_score": float 0-1 (1=critical, 0.7+=high, 0.4+=moderate, <0.4=routine)\n\n'
                f"Observation: {obs_text}\n\nJSON:"
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
                        "temperature": 0.1,
                        "max_tokens": 256,
                    },
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
                    raw = data["choices"][0]["message"]["content"].strip()
                    start = raw.find("{")
                    end = raw.rfind("}") + 1
                    if start >= 0 and end > start:
                        parsed = json.loads(raw[start:end])
                        return {
                            "summary": parsed.get("summary", obs_text[:200]),
                            "entities": parsed.get("entities", []),
                            "topics": parsed.get("topics", []),
                            "importance_score": min(1.0, max(0.0, float(parsed.get("importance_score", 0.5)))),
                        }
        except Exception as e:
            logger.debug(f"LLM extraction failed: {e}")

        return None

    def _rule_based_extract(
        self,
        obs_text: str,
        observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        text_lower = obs_text.lower()

        entities = []
        for kw in SYMPTOM_KEYWORDS:
            if kw in text_lower:
                entities.append(kw)
        for kw in MEDICATION_KEYWORDS:
            if kw in text_lower:
                entities.append(kw)

        if observation.get("entity_name"):
            name = observation["entity_name"].lower()
            if name not in entities:
                entities.append(name)

        topics = list({TOPIC_MAP[e] for e in entities if e in TOPIC_MAP})
        if observation.get("category"):
            cat = observation["category"].lower()
            if cat not in topics:
                topics.append(cat)
        if not topics:
            topics = ["general"]

        severity = observation.get("severity")
        if severity is not None:
            importance = min(1.0, severity / 10.0)
        elif any(kw in text_lower for kw in ["emergency", "severe", "critical", "urgent"]):
            importance = 0.9
        elif any(kw in text_lower for kw in ["moderate", "worsening", "increasing"]):
            importance = 0.6
        else:
            importance = 0.4

        summary = obs_text[:200] if len(obs_text) > 200 else obs_text

        return {
            "summary": summary,
            "entities": entities[:10],
            "topics": topics[:5],
            "importance_score": importance,
        }
