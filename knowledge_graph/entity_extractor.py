"""
Entity Extractor for Palliative Care Knowledge Graph

Extracts medical entities and relationships from documents using LLM.
Adapted from OncoGraph's entity extraction patterns for palliative care domain.
"""

import os
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities in palliative care domain."""
    SYMPTOM = "Symptom"
    MEDICATION = "Medication"
    CONDITION = "Condition"
    TREATMENT = "Treatment"
    PROCEDURE = "Procedure"
    CAREGIVER_TASK = "CaregiverTask"
    SIDE_EFFECT = "SideEffect"
    BODY_PART = "BodyPart"
    DOSAGE = "Dosage"
    FREQUENCY = "Frequency"


class RelationshipType(Enum):
    """Types of relationships in palliative care domain."""
    TREATS = "TREATS"                      # Medication -> Symptom
    CAUSES = "CAUSES"                      # Condition -> Symptom
    SIDE_EFFECT_OF = "SIDE_EFFECT_OF"     # SideEffect -> Medication
    MANAGES = "MANAGES"                    # Treatment -> Condition
    REQUIRES = "REQUIRES"                  # Procedure -> Medication
    PERFORMED_BY = "PERFORMED_BY"          # Task -> Caregiver
    AFFECTS = "AFFECTS"                    # Condition -> BodyPart
    DOSAGE_FOR = "DOSAGE_FOR"             # Dosage -> Medication
    ALLEVIATES = "ALLEVIATES"             # Medication -> Symptom
    CONTRAINDICATES = "CONTRAINDICATES"   # Condition -> Medication


@dataclass
class Entity:
    """Represents an extracted entity."""
    name: str
    type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    source_text: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "properties": self.properties,
            "source_text": self.source_text,
            "confidence": self.confidence
        }


@dataclass
class Relationship:
    """Represents an extracted relationship between entities."""
    source: str
    source_type: EntityType
    target: str
    target_type: EntityType
    relationship: RelationshipType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "source_type": self.source_type.value,
            "target": self.target,
            "target_type": self.target_type.value,
            "relationship": self.relationship.value,
            "properties": self.properties,
            "confidence": self.confidence
        }


# Extraction prompt for LLM
ENTITY_EXTRACTION_PROMPT = """You are a medical entity extractor specializing in palliative care.
Extract entities and relationships from the given text.

Entity Types:
- Symptom: Pain, nausea, fatigue, breathlessness, anxiety, depression, etc.
- Medication: Morphine, paracetamol, ondansetron, dexamethasone, etc.
- Condition: Cancer, heart failure, COPD, dementia, etc.
- Treatment: Chemotherapy, radiation, palliative care, hospice care, etc.
- Procedure: Blood transfusion, paracentesis, wound care, etc.
- CaregiverTask: Bathing, feeding, medication administration, etc.
- SideEffect: Constipation, drowsiness, nausea (when caused by medication)
- BodyPart: Chest, abdomen, limbs, etc.
- Dosage: 10mg, 5ml, etc.

Relationship Types:
- TREATS: Medication treats symptom
- CAUSES: Condition causes symptom
- SIDE_EFFECT_OF: Side effect is caused by medication
- MANAGES: Treatment manages condition
- ALLEVIATES: Medication alleviates symptom
- AFFECTS: Condition affects body part
- CONTRAINDICATES: Condition contraindicates medication

TEXT TO ANALYZE:
{text}

Respond in JSON format:
{
  "entities": [
    {"name": "entity name", "type": "EntityType", "properties": {}}
  ],
  "relationships": [
    {"source": "entity1", "source_type": "Type1", "target": "entity2", "target_type": "Type2", "relationship": "RELATIONSHIP_TYPE"}
  ]
}
"""


class EntityExtractor:
    """
    Extracts medical entities and relationships from text using LLM.

    Supports multiple LLM providers:
    - Groq (default, uses llama models)
    - OpenAI
    - Local patterns (no LLM required)

    Usage:
        extractor = EntityExtractor()
        entities, relationships = await extractor.extract(text)
    """

    def __init__(
        self,
        llm_provider: str = "groq",
        use_patterns: bool = True
    ):
        """
        Initialize entity extractor.

        Args:
            llm_provider: LLM provider ("groq", "openai", or "none")
            use_patterns: Whether to also use pattern matching
        """
        self.llm_provider = llm_provider
        self.use_patterns = use_patterns

        # Initialize LLM client
        self._groq_client = None
        self._openai_client = None

        if llm_provider == "groq" and os.getenv("GROQ_API_KEY"):
            try:
                from groq import AsyncGroq
                self._groq_client = AsyncGroq()
            except ImportError:
                logger.warning("groq package not installed")

        elif llm_provider == "openai" and os.getenv("OPENAI_API_KEY"):
            try:
                from openai import AsyncOpenAI
                self._openai_client = AsyncOpenAI()
            except ImportError:
                logger.warning("openai package not installed")

        # Common medical terms for pattern matching
        self._symptom_patterns = [
            r'\b(pain|ache|discomfort|soreness)\b',
            r'\b(nausea|vomiting|emesis)\b',
            r'\b(fatigue|tiredness|weakness|exhaustion)\b',
            r'\b(breathlessness|dyspnea|shortness of breath)\b',
            r'\b(anxiety|depression|distress)\b',
            r'\b(constipation|diarrhea)\b',
            r'\b(insomnia|sleeplessness)\b',
            r'\b(loss of appetite|anorexia)\b',
            r'\b(fever|pyrexia)\b',
            r'\b(cough|dysphagia|hiccups)\b',
        ]

        self._medication_patterns = [
            r'\b(morphine|oxycodone|fentanyl|hydromorphone)\b',
            r'\b(paracetamol|acetaminophen|ibuprofen|aspirin)\b',
            r'\b(ondansetron|metoclopramide|domperidone)\b',
            r'\b(dexamethasone|prednisone|prednisolone)\b',
            r'\b(haloperidol|midazolam|lorazepam)\b',
            r'\b(gabapentin|pregabalin|amitriptyline)\b',
            r'\b(laxative|bisacodyl|lactulose|senna)\b',
        ]

        self._condition_patterns = [
            r'\b(cancer|carcinoma|malignancy|tumor|tumour)\b',
            r'\b(heart failure|cardiac failure)\b',
            r'\b(copd|chronic obstructive pulmonary disease)\b',
            r'\b(dementia|alzheimer)\b',
            r'\b(kidney failure|renal failure)\b',
            r'\b(liver failure|hepatic failure)\b',
            r'\b(stroke|cerebrovascular)\b',
        ]

    async def extract(
        self,
        text: str,
        use_llm: bool = True
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from text.

        Args:
            text: Text to analyze
            use_llm: Whether to use LLM extraction

        Returns:
            Tuple of (entities, relationships)
        """
        entities = []
        relationships = []

        # Pattern-based extraction (fast, always available)
        if self.use_patterns:
            pattern_entities = self._extract_with_patterns(text)
            entities.extend(pattern_entities)

        # LLM-based extraction (more accurate, requires API)
        if use_llm and (self._groq_client or self._openai_client):
            try:
                llm_entities, llm_relationships = await self._extract_with_llm(text)
                entities.extend(llm_entities)
                relationships.extend(llm_relationships)
            except Exception as e:
                logger.error(f"LLM extraction failed: {e}")

        # Deduplicate entities
        entities = self._deduplicate_entities(entities)

        return entities, relationships

    def _extract_with_patterns(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns."""
        entities = []
        text_lower = text.lower()

        # Extract symptoms
        for pattern in self._symptom_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                entities.append(Entity(
                    name=match.title(),
                    type=EntityType.SYMPTOM,
                    source_text=text[:100],
                    confidence=0.8
                ))

        # Extract medications
        for pattern in self._medication_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                entities.append(Entity(
                    name=match.title(),
                    type=EntityType.MEDICATION,
                    source_text=text[:100],
                    confidence=0.9
                ))

        # Extract conditions
        for pattern in self._condition_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                entities.append(Entity(
                    name=match.title(),
                    type=EntityType.CONDITION,
                    source_text=text[:100],
                    confidence=0.85
                ))

        return entities

    async def _extract_with_llm(
        self,
        text: str
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities using LLM."""
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text[:4000])

        try:
            if self._groq_client:
                response = await self._groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2000,
                )
                content = response.choices[0].message.content

            elif self._openai_client:
                response = await self._openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2000,
                )
                content = response.choices[0].message.content

            else:
                return [], []

            return self._parse_llm_response(content)

        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            return [], []

    def _parse_llm_response(
        self,
        content: str
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Parse LLM JSON response into entities and relationships."""
        entities = []
        relationships = []

        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                return [], []

            data = json.loads(json_match.group())

            # Parse entities
            for e in data.get("entities", []):
                try:
                    entity_type = EntityType(e.get("type", "Symptom"))
                    entities.append(Entity(
                        name=e.get("name", ""),
                        type=entity_type,
                        properties=e.get("properties", {}),
                        confidence=0.95
                    ))
                except ValueError:
                    continue

            # Parse relationships
            for r in data.get("relationships", []):
                try:
                    rel_type = RelationshipType(r.get("relationship", "TREATS"))
                    source_type = EntityType(r.get("source_type", "Medication"))
                    target_type = EntityType(r.get("target_type", "Symptom"))

                    relationships.append(Relationship(
                        source=r.get("source", ""),
                        source_type=source_type,
                        target=r.get("target", ""),
                        target_type=target_type,
                        relationship=rel_type,
                        properties=r.get("properties", {}),
                        confidence=0.9
                    ))
                except ValueError:
                    continue

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")

        return entities, relationships

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities, keeping highest confidence."""
        seen = {}
        for entity in entities:
            key = (entity.name.lower(), entity.type)
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity

        return list(seen.values())

    def extract_from_chunks(
        self,
        chunks: List[str]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Synchronously extract entities from multiple text chunks.
        Uses pattern matching only (no LLM).

        Args:
            chunks: List of text chunks

        Returns:
            Tuple of (entities, relationships)
        """
        all_entities = []
        for chunk in chunks:
            entities = self._extract_with_patterns(chunk)
            all_entities.extend(entities)

        return self._deduplicate_entities(all_entities), []
