"""
Patient Context Memory

Maintains persistent patient context including:
- Primary condition and diagnosis
- Current symptoms being managed
- Active medications
- Care preferences
- Previous advice given
"""

import json
import logging
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class Medication:
    """Active medication record."""
    name: str
    dosage: str
    frequency: str
    purpose: str
    started: datetime = field(default_factory=datetime.now)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dosage": self.dosage,
            "frequency": self.frequency,
            "purpose": self.purpose,
            "started": self.started.isoformat(),
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Medication":
        return cls(
            name=data["name"],
            dosage=data["dosage"],
            frequency=data["frequency"],
            purpose=data["purpose"],
            started=datetime.fromisoformat(data["started"]),
            notes=data.get("notes", "")
        )


@dataclass
class Symptom:
    """Tracked symptom."""
    name: str
    severity: str  # mild, moderate, severe
    first_reported: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity,
            "first_reported": self.first_reported.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Symptom":
        return cls(
            name=data["name"],
            severity=data["severity"],
            first_reported=datetime.fromisoformat(data["first_reported"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            notes=data.get("notes", "")
        )


@dataclass
class PatientContext:
    """Complete patient context."""
    user_id: str
    primary_condition: Optional[str] = None
    condition_stage: Optional[str] = None
    symptoms: List[Symptom] = field(default_factory=list)
    medications: List[Medication] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    care_location: str = "home"  # home, hospital, hospice
    has_caregiver: bool = True
    emergency_contact: Optional[str] = None
    previous_advice_topics: Set[str] = field(default_factory=set)
    important_notes: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "primary_condition": self.primary_condition,
            "condition_stage": self.condition_stage,
            "symptoms": [s.to_dict() for s in self.symptoms],
            "medications": [m.to_dict() for m in self.medications],
            "allergies": self.allergies,
            "care_location": self.care_location,
            "has_caregiver": self.has_caregiver,
            "emergency_contact": self.emergency_contact,
            "previous_advice_topics": list(self.previous_advice_topics),
            "important_notes": self.important_notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatientContext":
        return cls(
            user_id=data["user_id"],
            primary_condition=data.get("primary_condition"),
            condition_stage=data.get("condition_stage"),
            symptoms=[Symptom.from_dict(s) for s in data.get("symptoms", [])],
            medications=[Medication.from_dict(m) for m in data.get("medications", [])],
            allergies=data.get("allergies", []),
            care_location=data.get("care_location", "home"),
            has_caregiver=data.get("has_caregiver", True),
            emergency_contact=data.get("emergency_contact"),
            previous_advice_topics=set(data.get("previous_advice_topics", [])),
            important_notes=data.get("important_notes", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )

    def add_symptom(self, name: str, severity: str = "moderate") -> None:
        """Add or update a symptom."""
        # Check if symptom exists
        for symptom in self.symptoms:
            if symptom.name.lower() == name.lower():
                symptom.severity = severity
                symptom.last_updated = datetime.now()
                self.updated_at = datetime.now()
                return

        # Add new symptom
        self.symptoms.append(Symptom(name=name, severity=severity))
        self.updated_at = datetime.now()

    def add_medication(
        self,
        name: str,
        dosage: str,
        frequency: str,
        purpose: str
    ) -> None:
        """Add a medication."""
        # Check if medication exists
        for med in self.medications:
            if med.name.lower() == name.lower():
                med.dosage = dosage
                med.frequency = frequency
                self.updated_at = datetime.now()
                return

        self.medications.append(Medication(
            name=name,
            dosage=dosage,
            frequency=frequency,
            purpose=purpose
        ))
        self.updated_at = datetime.now()

    def record_advice_topic(self, topic: str) -> None:
        """Record that advice was given on a topic."""
        self.previous_advice_topics.add(topic.lower())
        self.updated_at = datetime.now()

    def get_active_symptoms(self) -> List[str]:
        """Get list of active symptom names."""
        return [s.name for s in self.symptoms]

    def get_current_medications(self) -> List[str]:
        """Get list of current medication names."""
        return [m.name for m in self.medications]


class ContextMemory:
    """
    Patient Context Memory Manager.

    Features:
    - Persistent storage of patient context
    - Entity extraction from conversations
    - Context injection for personalized responses
    - Continuity across sessions
    """

    # Keywords for entity extraction
    CONDITION_KEYWORDS = {
        "cancer", "कैंसर", "carcinoma", "tumor", "tumour",
        "heart failure", "दिल की विफलता",
        "copd", "lung disease", "फेफड़ों की बीमारी",
        "kidney disease", "गुर्दे की बीमारी",
        "liver disease", "जिगर की बीमारी",
        "dementia", "alzheimer", "मनोभ्रंश",
        "motor neuron", "als", "mnd",
        "parkinson", "पार्किंसन",
        "stroke", "आघात"
    }

    SYMPTOM_KEYWORDS = {
        "pain": "दर्द",
        "nausea": "मतली",
        "vomiting": "उल्टी",
        "breathlessness": "सांस फूलना",
        "fatigue": "थकान",
        "constipation": "कब्ज",
        "anxiety": "चिंता",
        "depression": "अवसाद",
        "insomnia": "अनिद्रा",
        "appetite loss": "भूख न लगना",
        "weakness": "कमजोरी",
        "swelling": "सूजन",
        "confusion": "भ्रम",
        "restlessness": "बेचैनी"
    }

    MEDICATION_PATTERNS = [
        # Common palliative care medications
        "morphine", "मॉर्फिन",
        "oxycodone", "fentanyl",
        "tramadol", "ट्रामाडोल",
        "paracetamol", "पेरासिटामोल",
        "ibuprofen", "diclofenac",
        "ondansetron", "metoclopramide",
        "haloperidol", "lorazepam",
        "dexamethasone", "prednisolone",
        "lactulose", "bisacodyl",
        "omeprazole", "pantoprazole"
    ]

    def __init__(
        self,
        storage_path: str = "data/patient_context",
        context_expiry_days: int = 90
    ):
        """
        Initialize context memory.

        Args:
            storage_path: Directory for context storage
            context_expiry_days: Days after which inactive context expires
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.context_expiry_days = context_expiry_days
        self._cache: Dict[str, PatientContext] = {}
        self._lock = asyncio.Lock()

        logger.info(f"ContextMemory initialized - path={storage_path}")

    def _get_context_path(self, user_id: str) -> Path:
        """Get file path for user's context."""
        return self.storage_path / f"{user_id}_context.json"

    async def get_or_create_context(self, user_id: str) -> PatientContext:
        """
        Get existing context or create new one.

        Args:
            user_id: User identifier

        Returns:
            PatientContext
        """
        # Check cache
        if user_id in self._cache:
            return self._cache[user_id]

        # Try to load from storage
        context = await self._load_context(user_id)

        if not context:
            context = PatientContext(user_id=user_id)
            logger.info(f"Created new patient context - user={user_id}")

        self._cache[user_id] = context
        return context

    async def _load_context(self, user_id: str) -> Optional[PatientContext]:
        """Load context from storage."""
        file_path = self._get_context_path(user_id)

        if not file_path.exists():
            return None

        try:
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                data = json.loads(content)
                context = PatientContext.from_dict(data)

                # Check if expired
                expiry = datetime.now() - timedelta(days=self.context_expiry_days)
                if context.updated_at < expiry:
                    logger.info(f"Context expired for user {user_id}")
                    return None

                return context
        except Exception as e:
            logger.error(f"Error loading context for {user_id}: {e}")
            return None

    async def save_context(self, context: PatientContext) -> None:
        """Save context to storage."""
        async with self._lock:
            file_path = self._get_context_path(context.user_id)
            context.updated_at = datetime.now()

            try:
                async with aiofiles.open(file_path, "w") as f:
                    await f.write(json.dumps(context.to_dict(), indent=2))
                self._cache[context.user_id] = context
            except Exception as e:
                logger.error(f"Error saving context: {e}")

    def extract_entities(
        self,
        text: str,
        language: str = "en-IN"
    ) -> Dict[str, List[str]]:
        """
        Extract medical entities from text.

        Args:
            text: User message
            language: Language code

        Returns:
            Dictionary of entity types to values
        """
        text_lower = text.lower()
        entities: Dict[str, List[str]] = {
            "conditions": [],
            "symptoms": [],
            "medications": []
        }

        # Extract conditions
        for condition in self.CONDITION_KEYWORDS:
            if condition.lower() in text_lower:
                entities["conditions"].append(condition)

        # Extract symptoms
        for symptom_en, symptom_hi in self.SYMPTOM_KEYWORDS.items():
            if symptom_en.lower() in text_lower or symptom_hi in text_lower:
                entities["symptoms"].append(symptom_en)

        # Extract medications
        for med in self.MEDICATION_PATTERNS:
            if med.lower() in text_lower:
                entities["medications"].append(med)

        return entities

    async def update_from_conversation(
        self,
        user_id: str,
        text: str,
        language: str = "en-IN"
    ) -> PatientContext:
        """
        Update patient context from conversation text.

        Args:
            user_id: User identifier
            text: Conversation text
            language: Language code

        Returns:
            Updated PatientContext
        """
        context = await self.get_or_create_context(user_id)
        entities = self.extract_entities(text, language)

        # Update conditions (only if not already set)
        if entities["conditions"] and not context.primary_condition:
            context.primary_condition = entities["conditions"][0]
            logger.info(f"Detected condition for {user_id}: {context.primary_condition}")

        # Update symptoms
        for symptom in entities["symptoms"]:
            context.add_symptom(symptom)
            logger.debug(f"Added symptom for {user_id}: {symptom}")

        # Log medication mentions (don't auto-add, need confirmation)
        if entities["medications"]:
            logger.info(f"Medication mentions for {user_id}: {entities['medications']}")

        await self.save_context(context)
        return context

    def get_context_summary(self, context: PatientContext) -> str:
        """
        Generate a text summary of patient context for LLM.

        Args:
            context: Patient context

        Returns:
            Summary string for injection into prompt
        """
        parts = []

        if context.primary_condition:
            parts.append(f"Patient has {context.primary_condition}")
            if context.condition_stage:
                parts[-1] += f" ({context.condition_stage})"

        if context.symptoms:
            symptom_list = ", ".join(s.name for s in context.symptoms[:5])
            parts.append(f"Current symptoms: {symptom_list}")

        if context.medications:
            med_list = ", ".join(m.name for m in context.medications[:5])
            parts.append(f"Current medications: {med_list}")

        if context.allergies:
            parts.append(f"Allergies: {', '.join(context.allergies)}")

        if context.care_location != "home":
            parts.append(f"Care setting: {context.care_location}")

        if context.previous_advice_topics:
            recent_topics = list(context.previous_advice_topics)[:3]
            parts.append(f"Previously discussed: {', '.join(recent_topics)}")

        if not parts:
            return "No previous patient context available."

        return "Patient context: " + ". ".join(parts) + "."

    async def record_advice_given(
        self,
        user_id: str,
        topic: str
    ) -> None:
        """
        Record that advice was given on a topic.

        Args:
            user_id: User identifier
            topic: Topic of advice given
        """
        context = await self.get_or_create_context(user_id)
        context.record_advice_topic(topic)
        await self.save_context(context)

    async def add_medication(
        self,
        user_id: str,
        name: str,
        dosage: str,
        frequency: str,
        purpose: str
    ) -> PatientContext:
        """
        Add a medication to patient context.

        Args:
            user_id: User identifier
            name: Medication name
            dosage: Dosage amount
            frequency: Dosing frequency
            purpose: Why medication is taken

        Returns:
            Updated context
        """
        context = await self.get_or_create_context(user_id)
        context.add_medication(name, dosage, frequency, purpose)
        await self.save_context(context)
        logger.info(f"Added medication for {user_id}: {name}")
        return context

    async def add_symptom(
        self,
        user_id: str,
        name: str,
        severity: str = "moderate"
    ) -> PatientContext:
        """
        Add or update a symptom.

        Args:
            user_id: User identifier
            name: Symptom name
            severity: Symptom severity

        Returns:
            Updated context
        """
        context = await self.get_or_create_context(user_id)
        context.add_symptom(name, severity)
        await self.save_context(context)
        return context

    async def set_condition(
        self,
        user_id: str,
        condition: str,
        stage: Optional[str] = None
    ) -> PatientContext:
        """
        Set primary condition.

        Args:
            user_id: User identifier
            condition: Primary condition
            stage: Condition stage

        Returns:
            Updated context
        """
        context = await self.get_or_create_context(user_id)
        context.primary_condition = condition
        context.condition_stage = stage
        await self.save_context(context)
        logger.info(f"Set condition for {user_id}: {condition}")
        return context

    async def add_allergy(
        self,
        user_id: str,
        allergy: str
    ) -> PatientContext:
        """
        Add an allergy.

        Args:
            user_id: User identifier
            allergy: Allergy to add

        Returns:
            Updated context
        """
        context = await self.get_or_create_context(user_id)
        if allergy.lower() not in [a.lower() for a in context.allergies]:
            context.allergies.append(allergy)
            await self.save_context(context)
            logger.info(f"Added allergy for {user_id}: {allergy}")
        return context

    async def add_important_note(
        self,
        user_id: str,
        note: str
    ) -> PatientContext:
        """
        Add an important note to context.

        Args:
            user_id: User identifier
            note: Note to add

        Returns:
            Updated context
        """
        context = await self.get_or_create_context(user_id)
        timestamped_note = f"[{datetime.now().strftime('%Y-%m-%d')}] {note}"
        context.important_notes.append(timestamped_note)

        # Keep only last 10 notes
        context.important_notes = context.important_notes[-10:]

        await self.save_context(context)
        return context

    async def clear_context(self, user_id: str) -> None:
        """
        Clear patient context.

        Args:
            user_id: User identifier
        """
        file_path = self._get_context_path(user_id)
        if file_path.exists():
            file_path.unlink()

        if user_id in self._cache:
            del self._cache[user_id]

        logger.info(f"Cleared context for {user_id}")

    async def get_statistics(self) -> Dict[str, Any]:
        """Get context memory statistics."""
        contexts = []

        for file_path in self.storage_path.glob("*_context.json"):
            try:
                async with aiofiles.open(file_path, "r") as f:
                    content = await f.read()
                    data = json.loads(content)
                    contexts.append(PatientContext.from_dict(data))
            except Exception:
                pass

        if not contexts:
            return {
                "total_contexts": 0,
                "with_condition": 0,
                "with_symptoms": 0,
                "with_medications": 0
            }

        conditions: Dict[str, int] = {}
        for ctx in contexts:
            if ctx.primary_condition:
                cond = ctx.primary_condition.lower()
                conditions[cond] = conditions.get(cond, 0) + 1

        symptoms: Dict[str, int] = {}
        for ctx in contexts:
            for s in ctx.symptoms:
                sym = s.name.lower()
                symptoms[sym] = symptoms.get(sym, 0) + 1

        return {
            "total_contexts": len(contexts),
            "with_condition": sum(1 for c in contexts if c.primary_condition),
            "with_symptoms": sum(1 for c in contexts if c.symptoms),
            "with_medications": sum(1 for c in contexts if c.medications),
            "top_conditions": dict(sorted(conditions.items(), key=lambda x: -x[1])[:5]),
            "top_symptoms": dict(sorted(symptoms.items(), key=lambda x: -x[1])[:10]),
            "cached_contexts": len(self._cache)
        }
