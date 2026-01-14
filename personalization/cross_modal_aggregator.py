"""
Cross-Modal Data Aggregator

Combines patient data from multiple sources:
- Voice calls (Bolna, Gemini Live)
- WhatsApp conversations
- Uploaded documents (PDFs, images of prescriptions)
- Caregiver reports
- Future: FHIR/EHR integration

Each source provides data in different formats. This module extracts
structured observations from all sources and unifies them in the
longitudinal patient record.
"""

import json
import logging
import hashlib
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import asyncio
import aiofiles

from .longitudinal_memory import (
    TimestampedObservation,
    DataSourceType,
    LongitudinalPatientRecord,
    LongitudinalMemoryManager,
    SymptomObservation,
    MedicationEvent,
    VitalSignObservation,
    EmotionalObservation,
    SeverityLevel,
    MedicationAction,
)

logger = logging.getLogger(__name__)


# ============================================================================
# ENTITY EXTRACTION PATTERNS
# ============================================================================

# Palliative care symptom keywords (multi-language)
SYMPTOM_PATTERNS = {
    "pain": {
        "keywords": ["pain", "ache", "hurt", "sore", "दर्द", "पीड़ा", "ব্যথা", "வலி", "నొప్పి", "दुखणे", "પીડા", "ನೋವು"],
        "severity_keywords": {
            "mild": ["mild", "slight", "minor", "हल्का", "সামান্য", "லேசமான"],
            "moderate": ["moderate", "medium", "मध्यम", "মাঝারি"],
            "severe": ["severe", "bad", "terrible", "भयानक", "गंभीर", "গুরুতর", "மிகவும் மோசமான", "తీవ్రమైన"],
            "very_severe": ["unbearable", "extreme", "excruciating", "असहनीय", "অসহনীয"]
        }
    },
    "breathlessness": {
        "keywords": ["breathless", "short of breath", "difficult breathing", "सांस फूलना", "শ্বাসকষ্ট", "மூச்சம்", "శ్వాస తీవ్రత"],
        "severity_keywords": {}
    },
    "nausea": {
        "keywords": ["nausea", "vomiting", "feeling sick", "मतली", "उल्टी", "বমি বমি ভাব", "மயக்கம்", "వికారం"],
        "severity_keywords": {}
    },
    "fatigue": {
        "keywords": ["tired", "fatigue", "weakness", "exhausted", "थकान", "दुर्बलता", "ক্লান্তি", "சோர்வு", "అలసత", "थकवा"],
        "severity_keywords": {}
    },
    "constipation": {
        "keywords": ["constipation", "no bowel movement", "कब्ज", "કબજિયા", "മലബന്ധം", "மலச்சிக்கல்"],
        "severity_keywords": {}
    },
    "anxiety": {
        "keywords": ["anxious", "worried", "nervous", "panic", "चिंता", "घबराहट", "উদ্বেগ", "பதற்றம்", "ఆందోళన"],
        "severity_keywords": {}
    },
    "depression": {
        "keywords": ["depressed", "sad", "low", "hopeless", "अवसाद", "उदासीनता", "বিষণ্ণতা", "மனச்சோர்வு"],
        "severity_keywords": {}
    },
    "insomnia": {
        "keywords": ["insomnia", "sleepless", "can't sleep", "अनिद्रा", "নিদ्राहीनतা", "অনিদ্রা", "உறக்கம் இல்லாத"],
        "severity_keywords": {}
    },
    "appetite_loss": {
        "keywords": ["no appetite", "not eating", "loss of appetite", "भूख न लगना", "ভোজন কম খাওয়া", "பசி இல்லாமல்"],
        "severity_keywords": {}
    },
    "swelling": {
        "keywords": ["swelling", "edema", "fluid retention", "सूजन", "ফোলা", "வீக்கம்", "వాపు"],
        "severity_keywords": {}
    },
    "fever": {
        "keywords": ["fever", "temperature", "hot", "बुखार", "জ্বর", "காய்ச்சி", "జ్వరం"],
        "severity_keywords": {}
    },
    "cough": {
        "keywords": ["cough", "कफ", "কাশি", "இருமல்", "దగ్గు"],
        "severity_keywords": {}
    }
}

# Medication patterns
MEDICATION_PATTERNS = {
    "opioids": {
        "keywords": ["morphine", "मॉर्फिन", "fentanyl", "oxycodone", "tramadol", "ट्रामाडोल", "methadone"],
        "class": "opioid_analgesic"
    },
    "nsaids": {
        "keywords": ["paracetamol", "acetaminophen", "एसिटामिनोफेन", "ibuprofen", "diclofenac"],
        "class": "nsaid"
    },
    "antiemetics": {
        "keywords": ["ondansetron", "metoclopramide", "অনদানসেট্রন"],
        "class": "antiemetic"
    },
    "laxatives": {
        "keywords": ["lactulose", "bisacodyl", "senna", "ল্যাকটুলোজ"],
        "class": "laxative"
    },
    "anxiolytics": {
        "keywords": ["lorazepam", "alprazolam", "diazepam"],
        "class": "anxiolytic"
    },
    "corticosteroids": {
        "keywords": ["dexamethasone", "prednisolone", "methylprednisolone"],
        "class": "corticosteroid"
    },
    "ppi": {
        "keywords": ["omeprazole", "pantoprazole", "rabeprazole"],
        "class": "proton_pump_inhibitor"
    }
}

# Emotional state patterns
EMOTION_PATTERNS = {
    "anxiety": ["anxious", "worried", "nervous", "scared", "panicking", "चिंतित", "डरा हुआ"],
    "depression": ["depressed", "sad", "hopeless", "suicidal", "low", "उदास", "निराश"],
    "fear": ["afraid", "fear", "scared", "terrified", "डर", "भय"],
    "anger": ["angry", "frustrated", "irritated", "annoyed", "गुस्सा"],
    "peace": ["calm", "peaceful", "relaxed", "at ease", "शांत", "सुकून"],
    "gratitude": ["grateful", "thankful", "blessed", "कृतज्ञ", "आभारी"]
}


# ============================================================================
# EXTRACTORS
# ============================================================================

class VoiceDataExtractor:
    """
    Extract medical observations from voice call transcripts.

    Voice calls have specific characteristics:
    - Spoken language patterns
    - Emotional tone indicators
    - Longer, more conversational format
    - May include caregiver participation
    """

    def __init__(self):
        """Initialize the voice data extractor."""
        self.symptom_patterns = SYMPTOM_PATTERNS
        self.medication_patterns = MEDICATION_PATTERNS
        self.emotion_patterns = EMOTION_PATTERNS

    async def extract(
        self,
        transcript: str,
        metadata: Dict[str, Any]
    ) -> List[TimestampedObservation]:
        """
        Extract observations from voice call transcript.

        Args:
            transcript: The conversation transcript
            metadata: Call metadata (speaker_role, language, etc.)

        Returns:
            List of extracted observations
        """
        observations = []

        try:
            # Detect language
            language = metadata.get("language", "en-IN")
            speaker_role = metadata.get("speaker_role", "patient")

            # Extract symptoms
            symptom_obs = self._extract_symptoms(transcript, language, speaker_role)
            observations.extend(symptom_obs)

            # Extract medications
            medication_obs = self._extract_medications(transcript, language, speaker_role)
            observations.extend(medication_obs)

            # Extract emotional state
            emotional_obs = self._extract_emotions(transcript, language, speaker_role)
            observations.extend(emotional_obs)

            # Extract vital signs (if mentioned)
            vital_obs = self._extract_vitals(transcript, language, speaker_role)
            observations.extend(vital_obs)

            logger.debug(f"Extracted {len(observations)} observations from voice transcript")

        except Exception as e:
            logger.error(f"Error extracting from voice transcript: {e}")

        return observations

    def _extract_symptoms(
        self,
        text: str,
        language: str,
        speaker_role: str
    ) -> List[SymptomObservation]:
        """Extract symptom observations."""
        observations = []
        text_lower = text.lower()

        for symptom_name, pattern_data in self.symptom_patterns.items():
            # Check if symptom is mentioned
            mentioned = False
            matched_keyword = None

            for keyword in pattern_data["keywords"]:
                if keyword.lower() in text_lower:
                    mentioned = True
                    matched_keyword = keyword
                    break

            if not mentioned:
                continue

            # Determine severity
            severity = SeverityLevel.MODERATE
            severity_keywords = pattern_data.get("severity_keywords", {})

            for severity_level, keywords in severity_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        if severity_level == "mild":
                            severity = SeverityLevel.MILD
                        elif severity_level == "moderate":
                            severity = SeverityLevel.MODERATE
                        elif severity_level == "severe":
                            severity = SeverityLevel.SEVERE
                        elif severity_level == "very_severe":
                            severity = SeverityLevel.VERY_SEVERE
                        break

            # Extract additional context
            location = self._extract_location(text_lower, symptom_name)
            duration = self._extract_duration(text_lower)

            obs = SymptomObservation(
                observation_id="",  # Will be set by aggregator
                timestamp=None,  # Will be set by aggregator
                source_type=DataSourceType.VOICE_CALL,
                source_id="",
                reported_by=speaker_role,
                category="symptom",
                entity_name=symptom_name,
                value=severity,
                value_text=f"{matched_keyword or symptom_name} reported",
                symptom_name=symptom_name,
                severity=severity,
                location=location,
                duration=duration
            )

            observations.append(obs)

        return observations

    def _extract_medications(
        self,
        text: str,
        language: str,
        speaker_role: str
    ) -> List[MedicationEvent]:
        """Extract medication observations."""
        observations = []
        text_lower = text.lower()

        for med_class, pattern_data in self.medication_patterns.items():
            for med_name in pattern_data["keywords"]:
                if med_name.lower() in text_lower:
                    # Determine action
                    action = MedicationAction.TAKEN

                    # Check for specific action keywords
                    if any(word in text_lower for word in ["stopped", "stop", "no longer", "बंद"]):
                        action = MedicationAction.STOPPED
                    elif any(word in text_lower for word in ["started", "start", "new", "शुरू"]):
                        action = MedicationAction.STARTED
                    elif any(word in text_lower for word in ["missed", "forgot", "भूल गया"]):
                        action = MedicationAction.MISSED
                    elif any(word in text_lower for word in ["increased", "higher dose", "बढ़ा"]):
                        action = MedicationAction.DOSE_CHANGED

                    # Extract dosage if available
                    dosage = self._extract_dosage(text, med_name)

                    obs = MedicationEvent(
                        observation_id="",
                        timestamp=None,
                        source_type=DataSourceType.VOICE_CALL,
                        source_id="",
                        reported_by=speaker_role,
                        category="medication",
                        entity_name=med_name,
                        value=action,
                        value_text=f"{med_name} - {action.value}",
                        medication_name=med_name,
                        dosage=dosage,
                        action=action
                    )

                    observations.append(obs)

        return observations

    def _extract_emotions(
        self,
        text: str,
        language: str,
        speaker_role: str
    ) -> List[EmotionalObservation]:
        """Extract emotional state observations."""
        observations = []
        text_lower = text.lower()

        for emotion_type, keywords in EMOTION_PATTERNS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    # Determine intensity
                    intensity = SeverityLevel.MODERATE

                    # Look for intensifiers
                    if any(word in text_lower for word in ["very", "extremely", "बहुत", "अत्यधिक"]):
                        intensity = SeverityLevel.SEVERE
                    elif any(word in text_lower for word in ["little", "somewhat", "थोड़ा"]):
                        intensity = SeverityLevel.MILD

                    obs = EmotionalObservation(
                        observation_id="",
                        timestamp=None,
                        source_type=DataSourceType.VOICE_CALL,
                        source_id="",
                        reported_by=speaker_role,
                        category="emotional",
                        entity_name=emotion_type,
                        value=intensity,
                        value_text=f"Patient reports feeling {emotion_type}",
                        emotion_type=emotion_type,
                        intensity=intensity
                    )

                    observations.append(obs)
                    break

        return observations

    def _extract_vitals(
        self,
        text: str,
        language: str,
        speaker_role: str
    ) -> List[VitalSignObservation]:
        """Extract vital sign observations."""
        observations = []

        # Blood pressure patterns
        bp_pattern = r'(?:bp|blood pressure|pressure|बीपी|रक्त दाब)[\s:]*([\d/]+)\s*(?:mmhg)?'
        bp_match = re.search(bp_pattern, text.lower())
        if bp_match:
            bp_value = bp_match.group(1)
            observations.append(VitalSignObservation(
                observation_id="",
                timestamp=None,
                source_type=DataSourceType.VOICE_CALL,
                source_id="",
                reported_by=speaker_role,
                category="vital_sign",
                entity_name="blood_pressure",
                value=bp_value,
                value_text=f"Blood pressure: {bp_value}",
                vital_name="blood_pressure",
                value_numeric=0.0,
                unit="mmHg"
            ))

        # Temperature patterns
        temp_pattern = r'(?:temperature|temp|fever|बुखार|तापमान)[\s:]*([\d.]+)\s*(?:degree|°|f)?'
        temp_match = re.search(temp_pattern, text.lower())
        if temp_match:
            temp_value = float(temp_match.group(1))
            observations.append(VitalSignObservation(
                observation_id="",
                timestamp=None,
                source_type=DataSourceType.VOICE_CALL,
                source_id="",
                reported_by=speaker_role,
                category="vital_sign",
                entity_name="temperature",
                value=temp_value,
                value_text=f"Temperature: {temp_value}°F",
                vital_name="temperature",
                value_numeric=temp_value,
                unit="°F",
                within_normal_range=temp_value < 100.4
            ))

        return observations

    def _extract_location(self, text: str, symptom: str) -> Optional[str]:
        """Extract symptom location from text."""
        location_patterns = {
            "pain": ["head", "stomach", "back", "chest", "abdomen", "legs", "arms", "throat"],
            "swelling": ["legs", "feet", "hands", "face", "abdomen"]
        }

        if symptom in location_patterns:
            for location in location_patterns[symptom]:
                if location in text.lower():
                    return location

        return None

    def _extract_duration(self, text: str) -> Optional[str]:
        """Extract symptom duration from text."""
        duration_patterns = [
            r'(\d+)\s*days?',
            r'(\d+)\s*weeks?',
            r'(\d+)\s*months?',
            r'since\s+(\w+\s+\d+)',
            r'from\s+(\w+\s+\d+)'
        ]

        for pattern in duration_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(0)

        return None

    def _extract_dosage(self, text: str, medication: str) -> str:
        """Extract medication dosage from text."""
        # Look for dosage patterns near medication name
        dosage_pattern = r'(\d+(?:\.\d+)?)\s*(?:mg|ml|tablet|capsule|times?\s*a?\s*day)'

        matches = re.finditer(dosage_pattern, text.lower())
        for match in matches:
            return match.group(1)

        return ""


class WhatsAppDataExtractor:
    """
    Extract medical observations from WhatsApp conversations.

    WhatsApp messages have specific characteristics:
    - Text-based, shorter than voice
    - May be multiple messages in a session
    - Often asynchronous
    - May include images/documents
    """

    def __init__(self):
        """Initialize the WhatsApp data extractor."""
        self.voice_extractor = VoiceDataExtractor()  # Reuse same patterns

    async def extract(
        self,
        messages: str,
        metadata: Dict[str, Any]
    ) -> List[TimestampedObservation]:
        """
        Extract observations from WhatsApp message(s).

        Args:
            messages: The message content (can be multiple messages)
            metadata: Message metadata

        Returns:
            List of extracted observations
        """
        observations = []

        try:
            language = metadata.get("language", "en-IN")
            sender_role = metadata.get("sender_role", "patient")

            # Treat as text similar to voice but shorter
            symptom_obs = self.voice_extractor._extract_symptoms(
                messages, language, sender_role
            )
            observations.extend(symptom_obs)

            medication_obs = self.voice_extractor._extract_medications(
                messages, language, sender_role
            )
            observations.extend(medication_obs)

            emotional_obs = self.voice_extractor._extract_emotions(
                messages, language, sender_role
            )
            observations.extend(emotional_obs)

        except Exception as e:
            logger.error(f"Error extracting from WhatsApp messages: {e}")

        return observations


class DocumentDataExtractor:
    """
    Extract medical observations from uploaded documents.

    Documents can be:
    - PDFs (discharge summaries, prescriptions, lab reports)
    - Images (prescriptions, reports)
    - Text files

    Uses OCR for images and PDF parsing for documents.
    """

    def __init__(self):
        """Initialize the document data extractor."""
        pass

    async def extract(
        self,
        file_path: str,
        metadata: Dict[str, Any]
    ) -> List[TimestampedObservation]:
        """
        Extract observations from an uploaded document.

        Args:
            file_path: Path to the document file
            metadata: Document metadata (type, upload_date, etc.)

        Returns:
            List of extracted observations
        """
        observations = []
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return observations

        try:
            if path.suffix.lower() == ".pdf":
                observations = await self._extract_from_pdf(path, metadata)
            elif path.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif"]:
                observations = await self._extract_from_image(path, metadata)
            elif path.suffix.lower() in [".txt", ".md", ".json"]:
                observations = await self._extract_from_text(path, metadata)

        except Exception as e:
            logger.error(f"Error extracting from document {file_path}: {e}")

        return observations

    async def _extract_from_pdf(
        self,
        file_path: Path,
        metadata: Dict[str, Any]
    ) -> List[TimestampedObservation]:
        """Extract observations from a PDF document."""
        observations = []

        try:
            # Try to extract text from PDF
            # Note: This requires PyPDF2 or similar library
            # For now, basic implementation
            import PyPDF2

            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"

            # Use voice extractor patterns on extracted text
            extractor = VoiceDataExtractor()

            # Detect document type
            doc_type = metadata.get("document_type", "unknown")

            if "prescription" in doc_type.lower() or "prescription" in text.lower():
                observations.extend(self._extract_prescription_data(text, metadata))
            elif "discharge" in doc_type.lower() or "discharge" in text.lower():
                observations.extend(self._extract_discharge_summary(text, metadata))
            elif "lab" in doc_type.lower() or "report" in text.lower():
                observations.extend(self._extract_lab_report(text, metadata))
            else:
                # General extraction
                observations.extend(extractor._extract_symptoms(text, "en-IN", "document"))
                observations.extend(extractor._extract_medications(text, "en-IN", "document"))

        except ImportError:
            logger.warning("PyPDF2 not installed, skipping PDF extraction")
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")

        return observations

    async def _extract_from_image(
        self,
        file_path: Path,
        metadata: Dict[str, Any]
    ) -> List[TimestampedObservation]:
        """Extract observations from an image (OCR)."""
        observations = []

        try:
            # Try OCR
            # Note: This requires pytesseract or similar
            # For now, placeholder
            import pytesseract
            from PIL import Image

            text = pytesseract.image_to_string(Image.open(file_path))

            # Use voice extractor patterns
            extractor = VoiceDataExtractor()

            observations.extend(extractor._extract_medications(text, "en-IN", "document"))

        except ImportError:
            logger.warning("OCR libraries not installed, skipping image extraction")
        except Exception as e:
            logger.error(f"Error processing image: {e}")

        return observations

    async def _extract_from_text(
        self,
        file_path: Path,
        metadata: Dict[str, Any]
    ) -> List[TimestampedObservation]:
        """Extract observations from a text file."""
        observations = []

        try:
            async with aiofiles.open(file_path, "r") as f:
                text = await f.read()

            extractor = VoiceDataExtractor()
            observations.extend(extractor._extract_symptoms(text, "en-IN", "document"))
            observations.extend(extractor._extract_medications(text, "en-IN", "document"))

        except Exception as e:
            logger.error(f"Error processing text file: {e}")

        return observations

    def _extract_prescription_data(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[MedicationEvent]:
        """Extract structured medication data from prescription."""
        observations = []

        # Look for medication + dosage patterns
        med_pattern = r'([A-Z][a-z]+)\s+(\d+(?:\.\d+)?)\s*(?:mg|ml|tablet|capsule)'

        for match in re.finditer(med_pattern, text):
            med_name = match.group(1)
            dosage = match.group(2)

            obs = MedicationEvent(
                observation_id="",
                timestamp=None,
                source_type=DataSourceType.UPLOADED_DOCUMENT,
                source_id=metadata.get("document_id", ""),
                reported_by="document",
                category="medication",
                entity_name=med_name,
                value=MedicationAction.STARTED,
                value_text=f"{med_name} {dosage} mg prescribed",
                medication_name=med_name,
                dosage=f"{dosage} mg",
                action=MedicationAction.STARTED,
                prescribed_by=metadata.get("doctor_name", "prescribing_doctor")
            )

            observations.append(obs)

        return observations

    def _extract_discharge_summary(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[TimestampedObservation]:
        """Extract observations from discharge summary."""
        observations = []

        # Extract diagnosis
        diagnosis_pattern = r'(?:diagnosis|diagnosed with)[\s:]+([^\n.]+)'
        diagnosis_match = re.search(diagnosis_pattern, text, re.IGNORECASE)
        if diagnosis_match:
            # This would be stored as metadata on the patient record
            pass

        # Extract symptoms mentioned
        extractor = VoiceDataExtractor()
        observations.extend(extractor._extract_symptoms(text, "en-IN", "document"))

        # Extract medications
        observations.extend(extractor._extract_medications(text, "en-IN", "document"))

        return observations

    def _extract_lab_report(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[VitalSignObservation]:
        """Extract vital signs from lab report."""
        observations = []

        # Common lab values
        lab_patterns = {
            "hemoglobin": r'(?:hemoglobin|hb)[\s:]+([\d.]+)\s*(?:g/dl)?',
            "blood_pressure_systolic": r'(?:bp|blood pressure)[\s:]*(\d+)/\d+',
            "blood_pressure_diastolic": r'(?:bp|blood pressure)[\s:]*\d+/(\d+)',
            "heart_rate": r'(?:heart rate|pulse)[\s:]+(\d+)\s*(?:bpm)?'
        }

        for vital_name, pattern in lab_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))

                obs = VitalSignObservation(
                    observation_id="",
                    timestamp=None,
                    source_type=DataSourceType.UPLOADED_DOCUMENT,
                    source_id=metadata.get("document_id", ""),
                    reported_by="document",
                    category="vital_sign",
                    entity_name=vital_name,
                    value=value,
                    value_text=f"{vital_name}: {value}",
                    vital_name=vital_name,
                    value_numeric=value,
                    unit=""
                )

                observations.append(obs)

        return observations


# ============================================================================
# CROSS-MODAL AGGREGATOR
# ============================================================================

class CrossModalAggregator:
    """
    Aggregates patient data from multiple modalities.

    Each source provides data in different formats. This module:
    1. Routes data to appropriate extractor
    2. Normalizes observations to common format
    3. Stores in longitudinal patient record
    4. Triggers monitoring checks
    """

    def __init__(
        self,
        longitudinal_manager: LongitudinalMemoryManager,
        storage_path: str = "data/cross_modal_cache"
    ):
        """
        Initialize the cross-modal aggregator.

        Args:
            longitudinal_manager: Manager for longitudinal records
            storage_path: Cache for extracted data
        """
        self.longitudinal = longitudinal_manager
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize extractors
        self.voice_extractor = VoiceDataExtractor()
        self.whatsapp_extractor = WhatsAppDataExtractor()
        self.document_extractor = DocumentDataExtractor()

        logger.info("CrossModalAggregator initialized")

    def _generate_observation_id(self, patient_id: str, source_id: str, index: int) -> str:
        """Generate unique observation ID."""
        content = f"{patient_id}:{source_id}:{index}:{datetime.now().isoformat()}"
        return f"obs_{hashlib.md5(content.encode()).hexdigest()[:12]}"

    async def process_conversation(
        self,
        patient_id: str,
        conversation_id: str,
        transcript: str,
        source_type: DataSourceType,
        metadata: Dict[str, Any]
    ) -> List[TimestampedObservation]:
        """
        Process a conversation and extract observations.

        Args:
            patient_id: Patient identifier
            conversation_id: Unique conversation identifier
            transcript: The conversation text
            source_type: Type of source (VOICE_CALL, WHATSAPP, etc.)
            metadata: Additional metadata

        Returns:
            List of extracted and stored observations
        """
        observations = []

        try:
            # Select extractor based on source
            if source_type == DataSourceType.VOICE_CALL:
                observations = await self.voice_extractor.extract(transcript, metadata)
            elif source_type == DataSourceType.WHATSAPP:
                observations = await self.whatsapp_extractor.extract(transcript, metadata)
            elif source_type == DataSourceType.WEB_CHAT:
                # Treat web chat similar to WhatsApp
                observations = await self.whatsapp_extractor.extract(transcript, metadata)
            else:
                logger.warning(f"Unsupported source type for conversation: {source_type}")
                return []

            # Add metadata and store
            timestamp = datetime.now()
            for i, obs in enumerate(observations):
                obs.observation_id = self._generate_observation_id(patient_id, conversation_id, i)
                obs.source_id = conversation_id
                obs.source_type = source_type
                if not obs.timestamp:
                    obs.timestamp = timestamp

            # Add to longitudinal record
            record = await self.longitudinal.get_or_create_record(patient_id)

            for obs in observations:
                record.add_observation(obs)

            await self.longitudinal.save_record(record)

            # Run monitoring checks
            await self.longitudinal.run_monitoring_checks(patient_id)

            # Cache extraction results
            await self._cache_extraction(conversation_id, observations, metadata)

            logger.info(
                f"Extracted {len(observations)} observations from "
                f"{source_type.value} conversation {conversation_id}"
            )

        except Exception as e:
            logger.error(f"Error processing conversation {conversation_id}: {e}")

        return observations

    async def process_document(
        self,
        patient_id: str,
        document_id: str,
        file_path: str,
        metadata: Dict[str, Any]
    ) -> List[TimestampedObservation]:
        """
        Process an uploaded document and extract observations.

        Args:
            patient_id: Patient identifier
            document_id: Unique document identifier
            file_path: Path to the document file
            metadata: Document metadata

        Returns:
            List of extracted and stored observations
        """
        observations = []

        try:
            observations = await self.document_extractor.extract(file_path, metadata)

            # Add metadata and store
            timestamp = datetime.now()
            for i, obs in enumerate(observations):
                obs.observation_id = self._generate_observation_id(patient_id, document_id, i)
                obs.source_id = document_id
                obs.source_type = DataSourceType.UPLOADED_DOCUMENT
                obs.reported_by = "document"
                if not obs.timestamp:
                    obs.timestamp = timestamp

            # Add to longitudinal record
            record = await self.longitudinal.get_or_create_record(patient_id)

            for obs in observations:
                record.add_observation(obs)

            await self.longitudinal.save_record(record)

            logger.info(
                f"Extracted {len(observations)} observations from document {document_id}"
            )

        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")

        return observations

    async def process_manual_entry(
        self,
        patient_id: str,
        entry_data: Dict[str, Any],
        entered_by: str
    ) -> List[TimestampedObservation]:
        """
        Process a manual entry from a caregiver or provider.

        Args:
            patient_id: Patient identifier
            entry_data: Structured entry data
            entered_by: Who made the entry

        Returns:
            List of created observations
        """
        observations = []

        try:
            timestamp = entry_data.get("timestamp", datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)

            entry_type = entry_data.get("type", "symptom")

            if entry_type == "symptom":
                obs = SymptomObservation(
                    observation_id=self._generate_observation_id(
                        patient_id, f"manual_{entry_data.get('type')}", 0
                    ),
                    timestamp=timestamp,
                    source_type=DataSourceType.CAREGIVER_REPORT,
                    source_id=f"manual_{timestamp.isoformat()}",
                    reported_by=entered_by,
                    category="symptom",
                    entity_name=entry_data.get("symptom_name", ""),
                    value=SeverityLevel.from_string(entry_data.get("severity", "moderate")),
                    value_text=entry_data.get("notes", ""),
                    symptom_name=entry_data.get("symptom_name", ""),
                    severity=SeverityLevel.from_string(entry_data.get("severity", "moderate")),
                    location=entry_data.get("location"),
                    duration=entry_data.get("duration")
                )
                observations.append(obs)

            elif entry_type == "medication":
                obs = MedicationEvent(
                    observation_id=self._generate_observation_id(
                        patient_id, f"manual_medication", 0
                    ),
                    timestamp=timestamp,
                    source_type=DataSourceType.CAREGIVER_REPORT,
                    source_id=f"manual_{timestamp.isoformat()}",
                    reported_by=entered_by,
                    category="medication",
                    entity_name=entry_data.get("medication_name", ""),
                    value=MedicationAction(entry_data.get("action", "taken")),
                    value_text=entry_data.get("notes", ""),
                    medication_name=entry_data.get("medication_name", ""),
                    dosage=entry_data.get("dosage", ""),
                    action=MedicationAction(entry_data.get("action", "taken"))
                )
                observations.append(obs)

            elif entry_type == "vital_sign":
                obs = VitalSignObservation(
                    observation_id=self._generate_observation_id(
                        patient_id, f"manual_vital", 0
                    ),
                    timestamp=timestamp,
                    source_type=DataSourceType.CAREGIVER_REPORT,
                    source_id=f"manual_{timestamp.isoformat()}",
                    reported_by=entered_by,
                    category="vital_sign",
                    entity_name=entry_data.get("vital_name", ""),
                    value=entry_data.get("value", 0),
                    value_text=entry_data.get("notes", ""),
                    vital_name=entry_data.get("vital_name", ""),
                    value_numeric=float(entry_data.get("value", 0)),
                    unit=entry_data.get("unit", "")
                )
                observations.append(obs)

            # Store observations
            record = await self.longitudinal.get_or_create_record(patient_id)

            for obs in observations:
                record.add_observation(obs)

            await self.longitudinal.save_record(record)

            logger.info(f"Created {len(observations)} manual observations for {patient_id}")

        except Exception as e:
            logger.error(f"Error processing manual entry: {e}")

        return observations

    async def _cache_extraction(
        self,
        source_id: str,
        observations: List[TimestampedObservation],
        metadata: Dict[str, Any]
    ) -> None:
        """Cache extraction results for future reference."""
        try:
            cache_file = self.storage_path / f"{source_id}_extracted.json"

            cache_data = {
                "source_id": source_id,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata,
                "observations": [o.to_dict() for o in observations]
            }

            async with aiofiles.open(cache_file, "w") as f:
                await f.write(json.dumps(cache_data, indent=2))

        except Exception as e:
            logger.error(f"Error caching extraction: {e}")

    async def get_cached_extraction(
        self,
        source_id: str
    ) -> Optional[List[TimestampedObservation]]:
        """Retrieve cached extraction results."""
        try:
            cache_file = self.storage_path / f"{source_id}_extracted.json"

            if not cache_file.exists():
                return None

            async with aiofiles.open(cache_file, "r") as f:
                content = await f.read()
                data = json.loads(content)

            observations = []
            for obs_data in data.get("observations", []):
                category = obs_data.get("category", "")
                if category == "symptom":
                    observations.append(SymptomObservation.from_dict(obs_data))
                elif category == "medication":
                    observations.append(MedicationEvent.from_dict(obs_data))
                elif category == "vital_sign":
                    observations.append(VitalSignObservation.from_dict(obs_data))
                elif category == "emotional":
                    observations.append(EmotionalObservation.from_dict(obs_data))
                else:
                    observations.append(TimestampedObservation.from_dict(obs_data))

            return observations

        except Exception as e:
            logger.error(f"Error retrieving cached extraction: {e}")
            return None
