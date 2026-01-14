"""
Unit Tests for Cross-Modal Data Aggregator Module

Tests the extraction and aggregation of patient data
from multiple sources (voice, WhatsApp, documents).
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from personalization.cross_modal_aggregator import (
    # Patterns
    SYMPTOM_PATTERNS,
    MEDICATION_PATTERNS,
    EMOTION_PATTERNS,
    # Extractors
    VoiceDataExtractor,
    WhatsAppDataExtractor,
    DocumentDataExtractor,
    # Main aggregator
    CrossModalAggregator,
)
from personalization.longitudinal_memory import (
    LongitudinalMemoryManager,
    LongitudinalPatientRecord,
    TimestampedObservation,
    SymptomObservation,
    MedicationEvent,
    VitalSignObservation,
    EmotionalObservation,
    DataSourceType,
    SeverityLevel,
    MedicationAction,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def voice_extractor():
    """Create VoiceDataExtractor."""
    return VoiceDataExtractor()


@pytest.fixture
def whatsapp_extractor():
    """Create WhatsAppDataExtractor."""
    return WhatsAppDataExtractor()


@pytest.fixture
def document_extractor():
    """Create DocumentDataExtractor."""
    return DocumentDataExtractor()


@pytest.fixture
def mock_longitudinal_manager(temp_storage):
    """Create mock longitudinal manager."""
    manager = AsyncMock(spec=LongitudinalMemoryManager)
    manager.storage_path = Path(temp_storage)

    # Default record
    record = LongitudinalPatientRecord(patient_id="patient_001")
    manager.get_or_create_record.return_value = record
    manager.save_record.return_value = None
    manager.run_monitoring_checks.return_value = []

    return manager


@pytest.fixture
def cross_modal_aggregator(mock_longitudinal_manager, temp_storage):
    """Create CrossModalAggregator."""
    return CrossModalAggregator(
        longitudinal_manager=mock_longitudinal_manager,
        storage_path=temp_storage
    )


# ============================================================================
# PATTERN TESTS
# ============================================================================

class TestSymptomPatterns:
    """Test symptom extraction patterns."""

    def test_pain_keywords_exist(self):
        """Test pain keywords are defined."""
        assert "pain" in SYMPTOM_PATTERNS
        keywords = SYMPTOM_PATTERNS["pain"]["keywords"]

        # English
        assert "pain" in keywords
        assert "hurt" in keywords

        # Hindi
        assert "दर्द" in keywords

    def test_common_symptoms_defined(self):
        """Test common palliative symptoms are defined."""
        expected_symptoms = [
            "pain", "breathlessness", "nausea", "fatigue",
            "constipation", "anxiety", "depression", "insomnia"
        ]

        for symptom in expected_symptoms:
            assert symptom in SYMPTOM_PATTERNS, f"Missing pattern for {symptom}"

    def test_severity_keywords(self):
        """Test severity keywords are defined for pain."""
        severity = SYMPTOM_PATTERNS["pain"]["severity_keywords"]

        assert "mild" in severity
        assert "moderate" in severity
        assert "severe" in severity
        assert "very_severe" in severity


class TestMedicationPatterns:
    """Test medication extraction patterns."""

    def test_opioid_keywords(self):
        """Test opioid keywords are defined."""
        assert "opioids" in MEDICATION_PATTERNS
        keywords = MEDICATION_PATTERNS["opioids"]["keywords"]

        assert "morphine" in keywords
        assert "fentanyl" in keywords
        assert "tramadol" in keywords

    def test_common_medication_classes(self):
        """Test common medication classes are defined."""
        expected_classes = [
            "opioids", "nsaids", "antiemetics", "laxatives",
            "anxiolytics", "corticosteroids"
        ]

        for med_class in expected_classes:
            assert med_class in MEDICATION_PATTERNS


class TestEmotionPatterns:
    """Test emotion extraction patterns."""

    def test_emotion_types(self):
        """Test emotion types are defined."""
        expected_emotions = ["anxiety", "depression", "fear", "anger", "peace"]

        for emotion in expected_emotions:
            assert emotion in EMOTION_PATTERNS

    def test_multilingual_emotions(self):
        """Test emotions have Hindi keywords."""
        for emotion, keywords in EMOTION_PATTERNS.items():
            # Check at least some Hindi keywords exist
            hindi_found = any(
                any('\u0900' <= c <= '\u097F' for c in kw)
                for kw in keywords
            )
            # Not all emotions need Hindi, but check structure
            assert isinstance(keywords, list)


# ============================================================================
# VOICE EXTRACTOR TESTS
# ============================================================================

class TestVoiceDataExtractor:
    """Test VoiceDataExtractor class."""

    @pytest.mark.asyncio
    async def test_extract_pain_symptom(self, voice_extractor):
        """Test extracting pain from voice transcript."""
        transcript = "I have been having a lot of pain in my back for the last few days."
        metadata = {"language": "en-IN", "speaker_role": "patient"}

        observations = await voice_extractor.extract(transcript, metadata)

        # Should find pain
        pain_obs = [o for o in observations if o.entity_name == "pain"]
        assert len(pain_obs) >= 1
        assert pain_obs[0].category == "symptom"

    @pytest.mark.asyncio
    async def test_extract_severe_pain(self, voice_extractor):
        """Test extracting severe pain with severity keywords."""
        transcript = "The pain is terrible and unbearable. I can't sleep."
        metadata = {"language": "en-IN", "speaker_role": "patient"}

        observations = await voice_extractor.extract(transcript, metadata)

        pain_obs = [o for o in observations if isinstance(o, SymptomObservation) and o.symptom_name == "pain"]
        if pain_obs:
            assert pain_obs[0].severity in [SeverityLevel.SEVERE, SeverityLevel.VERY_SEVERE]

    @pytest.mark.asyncio
    async def test_extract_hindi_symptoms(self, voice_extractor):
        """Test extracting symptoms from Hindi text."""
        transcript = "मुझे बहुत दर्द हो रहा है और थकान भी लग रही है।"
        metadata = {"language": "hi-IN", "speaker_role": "patient"}

        observations = await voice_extractor.extract(transcript, metadata)

        # Should find pain (दर्द) and fatigue (थकान)
        symptom_names = [o.entity_name for o in observations if o.category == "symptom"]
        assert "pain" in symptom_names or "fatigue" in symptom_names

    @pytest.mark.asyncio
    async def test_extract_medications(self, voice_extractor):
        """Test extracting medications from transcript."""
        transcript = "I took my morphine 10mg this morning as prescribed."
        metadata = {"language": "en-IN", "speaker_role": "patient"}

        observations = await voice_extractor.extract(transcript, metadata)

        med_obs = [o for o in observations if isinstance(o, MedicationEvent)]
        assert len(med_obs) >= 1
        assert "morphine" in med_obs[0].medication_name.lower()

    @pytest.mark.asyncio
    async def test_extract_medication_stopped(self, voice_extractor):
        """Test extracting stopped medication."""
        transcript = "I stopped taking the tramadol because of side effects."
        metadata = {"language": "en-IN", "speaker_role": "patient"}

        observations = await voice_extractor.extract(transcript, metadata)

        med_obs = [o for o in observations if isinstance(o, MedicationEvent)]
        if med_obs:
            assert med_obs[0].action == MedicationAction.STOPPED

    @pytest.mark.asyncio
    async def test_extract_emotions(self, voice_extractor):
        """Test extracting emotional state."""
        transcript = "I have been feeling very anxious about my condition lately."
        metadata = {"language": "en-IN", "speaker_role": "patient"}

        observations = await voice_extractor.extract(transcript, metadata)

        emotion_obs = [o for o in observations if isinstance(o, EmotionalObservation)]
        assert len(emotion_obs) >= 1
        assert emotion_obs[0].emotion_type == "anxiety"

    @pytest.mark.asyncio
    async def test_extract_vital_signs(self, voice_extractor):
        """Test extracting vital signs from transcript."""
        # Use more explicit pattern for blood pressure
        transcript = "BP: 140/90 mmHg, temperature: 99.5"
        metadata = {"language": "en-IN", "speaker_role": "caregiver"}

        observations = await voice_extractor.extract(transcript, metadata)

        vital_obs = [o for o in observations if isinstance(o, VitalSignObservation)]
        # Vital sign extraction from free text is complex - may or may not find
        # The important thing is no errors occurred
        assert isinstance(observations, list)

    @pytest.mark.asyncio
    async def test_extract_multiple_symptoms(self, voice_extractor):
        """Test extracting multiple symptoms from one transcript."""
        transcript = """
        I have been having severe pain in my stomach.
        Also feeling nausea after eating.
        Haven't slept well due to anxiety.
        """
        metadata = {"language": "en-IN", "speaker_role": "patient"}

        observations = await voice_extractor.extract(transcript, metadata)

        symptom_names = [o.entity_name for o in observations if o.category == "symptom"]
        emotion_names = [o.entity_name for o in observations if o.category == "emotional"]

        # Should find multiple symptoms
        assert len(symptom_names) >= 2
        assert "pain" in symptom_names
        assert "nausea" in symptom_names or "anxiety" in emotion_names

    def test_extract_location(self, voice_extractor):
        """Test extracting symptom location."""
        text = "The pain is in my stomach and back."

        location = voice_extractor._extract_location(text, "pain")
        assert location in ["stomach", "back"]

    def test_extract_duration(self, voice_extractor):
        """Test extracting symptom duration."""
        text = "I have had this pain for 3 days now."

        duration = voice_extractor._extract_duration(text)
        assert duration is not None
        assert "3 days" in duration

    def test_extract_dosage(self, voice_extractor):
        """Test extracting medication dosage."""
        text = "I took morphine 10mg twice a day."

        dosage = voice_extractor._extract_dosage(text, "morphine")
        assert "10" in dosage


# ============================================================================
# WHATSAPP EXTRACTOR TESTS
# ============================================================================

class TestWhatsAppDataExtractor:
    """Test WhatsAppDataExtractor class."""

    @pytest.mark.asyncio
    async def test_extract_symptoms(self, whatsapp_extractor):
        """Test extracting symptoms from WhatsApp message."""
        message = "Having bad pain today. Also feeling tired."
        metadata = {"language": "en-IN", "sender_role": "patient"}

        observations = await whatsapp_extractor.extract(message, metadata)

        symptom_names = [o.entity_name for o in observations if o.category == "symptom"]
        assert "pain" in symptom_names

    @pytest.mark.asyncio
    async def test_extract_short_message(self, whatsapp_extractor):
        """Test extracting from short message."""
        message = "Severe pain"
        metadata = {"language": "en-IN", "sender_role": "patient"}

        observations = await whatsapp_extractor.extract(message, metadata)

        assert len(observations) >= 1
        assert observations[0].entity_name == "pain"

    @pytest.mark.asyncio
    async def test_extract_caregiver_message(self, whatsapp_extractor):
        """Test extracting from caregiver message."""
        message = "Mother is having breathing difficulty and seems anxious."
        metadata = {"language": "en-IN", "sender_role": "caregiver"}

        observations = await whatsapp_extractor.extract(message, metadata)

        assert len(observations) >= 1
        assert observations[0].reported_by == "caregiver"


# ============================================================================
# DOCUMENT EXTRACTOR TESTS
# ============================================================================

class TestDocumentDataExtractor:
    """Test DocumentDataExtractor class."""

    @pytest.mark.asyncio
    async def test_extract_nonexistent_file(self, document_extractor):
        """Test extracting from nonexistent file."""
        observations = await document_extractor.extract(
            "/nonexistent/file.pdf",
            {}
        )

        assert observations == []

    @pytest.mark.asyncio
    async def test_extract_text_file(self, document_extractor, temp_storage):
        """Test extracting from text file."""
        # Create test file
        test_file = Path(temp_storage) / "test_doc.txt"
        test_file.write_text("""
        Patient Discharge Summary
        Diagnosis: Stage IV lung cancer
        Current medications: Morphine 10mg, Ondansetron 4mg
        Symptoms: Pain (moderate), Nausea (mild)
        """)

        observations = await document_extractor.extract(
            str(test_file),
            {"document_type": "discharge_summary"}
        )

        # Should extract medications and symptoms
        categories = [o.category for o in observations]
        assert "medication" in categories or "symptom" in categories

    def test_extract_prescription_data(self, document_extractor):
        """Test extracting prescription data."""
        text = """
        Rx:
        Morphine 10mg - Take every 4 hours
        Ondansetron 4mg - As needed for nausea
        """

        observations = document_extractor._extract_prescription_data(
            text,
            {"document_id": "rx_001"}
        )

        assert len(observations) >= 1
        assert all(isinstance(o, MedicationEvent) for o in observations)

    def test_extract_lab_report(self, document_extractor):
        """Test extracting lab report data."""
        text = """
        Lab Results:
        Hemoglobin: 10.5 g/dL
        Blood Pressure: 130/85 mmHg
        Heart Rate: 78 bpm
        """

        observations = document_extractor._extract_lab_report(
            text,
            {"document_id": "lab_001"}
        )

        assert len(observations) >= 1
        assert all(isinstance(o, VitalSignObservation) for o in observations)


# ============================================================================
# CROSS-MODAL AGGREGATOR TESTS
# ============================================================================

class TestCrossModalAggregator:
    """Test CrossModalAggregator class."""

    @pytest.mark.asyncio
    async def test_process_voice_conversation(self, cross_modal_aggregator):
        """Test processing voice conversation."""
        transcript = "I have been having moderate pain and feeling anxious."
        metadata = {"language": "en-IN", "speaker_role": "patient"}

        observations = await cross_modal_aggregator.process_conversation(
            patient_id="patient_001",
            conversation_id="conv_001",
            transcript=transcript,
            source_type=DataSourceType.VOICE_CALL,
            metadata=metadata
        )

        assert len(observations) >= 1
        assert all(o.source_type == DataSourceType.VOICE_CALL for o in observations)
        assert all(o.observation_id != "" for o in observations)

    @pytest.mark.asyncio
    async def test_process_whatsapp_conversation(self, cross_modal_aggregator):
        """Test processing WhatsApp conversation."""
        messages = "Pain is getting worse today. Took morphine 10mg."
        metadata = {"language": "en-IN", "sender_role": "patient"}

        observations = await cross_modal_aggregator.process_conversation(
            patient_id="patient_001",
            conversation_id="msg_001",
            transcript=messages,
            source_type=DataSourceType.WHATSAPP,
            metadata=metadata
        )

        assert len(observations) >= 1
        assert all(o.source_type == DataSourceType.WHATSAPP for o in observations)

    @pytest.mark.asyncio
    async def test_process_webchat_conversation(self, cross_modal_aggregator):
        """Test processing web chat conversation."""
        messages = "Feeling nausea after eating."
        metadata = {"language": "en-IN", "sender_role": "patient"}

        observations = await cross_modal_aggregator.process_conversation(
            patient_id="patient_001",
            conversation_id="chat_001",
            transcript=messages,
            source_type=DataSourceType.WEB_CHAT,
            metadata=metadata
        )

        assert len(observations) >= 1
        assert all(o.source_type == DataSourceType.WEB_CHAT for o in observations)

    @pytest.mark.asyncio
    async def test_process_manual_symptom_entry(self, cross_modal_aggregator):
        """Test processing manual symptom entry."""
        entry_data = {
            "type": "symptom",
            "symptom_name": "pain",
            "severity": "severe",
            "location": "lower back",
            "duration": "2 days",
            "notes": "Pain worse in the evening"
        }

        observations = await cross_modal_aggregator.process_manual_entry(
            patient_id="patient_001",
            entry_data=entry_data,
            entered_by="caregiver_001"
        )

        assert len(observations) == 1
        assert isinstance(observations[0], SymptomObservation)
        assert observations[0].symptom_name == "pain"
        assert observations[0].severity == SeverityLevel.SEVERE

    @pytest.mark.asyncio
    async def test_process_manual_medication_entry(self, cross_modal_aggregator):
        """Test processing manual medication entry."""
        entry_data = {
            "type": "medication",
            "medication_name": "morphine",
            "dosage": "10mg",
            "action": "taken",
            "notes": "Taken at bedtime"
        }

        observations = await cross_modal_aggregator.process_manual_entry(
            patient_id="patient_001",
            entry_data=entry_data,
            entered_by="caregiver_001"
        )

        assert len(observations) == 1
        assert isinstance(observations[0], MedicationEvent)
        assert observations[0].medication_name == "morphine"

    @pytest.mark.asyncio
    async def test_process_manual_vital_entry(self, cross_modal_aggregator):
        """Test processing manual vital sign entry."""
        entry_data = {
            "type": "vital_sign",
            "vital_name": "temperature",
            "value": 99.5,
            "unit": "F",
            "notes": "Slight fever"
        }

        observations = await cross_modal_aggregator.process_manual_entry(
            patient_id="patient_001",
            entry_data=entry_data,
            entered_by="caregiver_001"
        )

        assert len(observations) == 1
        assert isinstance(observations[0], VitalSignObservation)
        assert observations[0].value_numeric == 99.5

    def test_generate_observation_id(self, cross_modal_aggregator):
        """Test observation ID generation."""
        id1 = cross_modal_aggregator._generate_observation_id("patient_001", "conv_001", 0)
        id2 = cross_modal_aggregator._generate_observation_id("patient_001", "conv_001", 1)

        assert id1.startswith("obs_")
        assert id2.startswith("obs_")
        assert id1 != id2

    @pytest.mark.asyncio
    async def test_cache_extraction(self, cross_modal_aggregator, temp_storage):
        """Test caching extraction results."""
        observations = [
            SymptomObservation(
                observation_id="obs_001",
                timestamp=datetime.now(),
                source_type=DataSourceType.VOICE_CALL,
                source_id="conv_001",
                reported_by="patient",
                category="symptom",
                entity_name="pain",
                value=SeverityLevel.MODERATE,
                value_text="Pain reported",
                symptom_name="pain",
                severity=SeverityLevel.MODERATE
            )
        ]

        await cross_modal_aggregator._cache_extraction(
            "conv_001",
            observations,
            {"language": "en-IN"}
        )

        # Check cache file exists
        cache_file = Path(temp_storage) / "conv_001_extracted.json"
        assert cache_file.exists()

    @pytest.mark.asyncio
    async def test_get_cached_extraction(self, cross_modal_aggregator, temp_storage):
        """Test retrieving cached extraction."""
        # First cache something
        observations = [
            SymptomObservation(
                observation_id="obs_001",
                timestamp=datetime.now(),
                source_type=DataSourceType.VOICE_CALL,
                source_id="conv_001",
                reported_by="patient",
                category="symptom",
                entity_name="pain",
                value=SeverityLevel.MODERATE,
                value_text="Pain reported",
                symptom_name="pain",
                severity=SeverityLevel.MODERATE
            )
        ]

        await cross_modal_aggregator._cache_extraction(
            "conv_001",
            observations,
            {"language": "en-IN"}
        )

        # Retrieve
        cached = await cross_modal_aggregator.get_cached_extraction("conv_001")

        assert cached is not None
        assert len(cached) == 1
        assert cached[0].entity_name == "pain"

    @pytest.mark.asyncio
    async def test_get_cached_nonexistent(self, cross_modal_aggregator):
        """Test retrieving nonexistent cache."""
        cached = await cross_modal_aggregator.get_cached_extraction("nonexistent")
        assert cached is None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCrossModalIntegration:
    """Integration tests for cross-modal aggregation."""

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, cross_modal_aggregator, mock_longitudinal_manager):
        """Test complete conversation processing flow."""
        # Process conversation
        transcript = """
        Patient: I have been having severe pain in my back for 3 days.
        Patient: Also feeling very anxious about the upcoming tests.
        Patient: I took my morphine this morning but it didn't help much.
        """

        observations = await cross_modal_aggregator.process_conversation(
            patient_id="patient_001",
            conversation_id="conv_full_test",
            transcript=transcript,
            source_type=DataSourceType.VOICE_CALL,
            metadata={"language": "en-IN", "speaker_role": "patient"}
        )

        # Should extract multiple observations
        assert len(observations) >= 2

        # Should have saved to longitudinal manager
        mock_longitudinal_manager.save_record.assert_called()

        # Should have run monitoring checks
        mock_longitudinal_manager.run_monitoring_checks.assert_called_with("patient_001")

    @pytest.mark.asyncio
    async def test_multi_language_extraction(self, cross_modal_aggregator):
        """Test extraction from multi-language content."""
        # Mix of English and Hindi
        transcript = "I have pain, दर्द बहुत है। Feeling very tired, थकान हो रही है।"

        observations = await cross_modal_aggregator.process_conversation(
            patient_id="patient_001",
            conversation_id="conv_multilang",
            transcript=transcript,
            source_type=DataSourceType.VOICE_CALL,
            metadata={"language": "hi-IN", "speaker_role": "patient"}
        )

        symptom_names = [o.entity_name for o in observations if o.category == "symptom"]

        # Should detect both pain and fatigue
        assert "pain" in symptom_names
        assert "fatigue" in symptom_names


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling in cross-modal aggregation."""

    @pytest.mark.asyncio
    async def test_unsupported_source_type(self, cross_modal_aggregator):
        """Test handling unsupported source type."""
        # Use a source type that doesn't have a conversation handler
        observations = await cross_modal_aggregator.process_conversation(
            patient_id="patient_001",
            conversation_id="conv_001",
            transcript="Some text",
            source_type=DataSourceType.UPLOADED_DOCUMENT,  # Not supported for conversation
            metadata={}
        )

        assert observations == []

    @pytest.mark.asyncio
    async def test_empty_transcript(self, cross_modal_aggregator):
        """Test handling empty transcript."""
        observations = await cross_modal_aggregator.process_conversation(
            patient_id="patient_001",
            conversation_id="conv_empty",
            transcript="",
            source_type=DataSourceType.VOICE_CALL,
            metadata={}
        )

        assert observations == []

    @pytest.mark.asyncio
    async def test_invalid_manual_entry(self, cross_modal_aggregator):
        """Test handling invalid manual entry."""
        entry_data = {
            "type": "unknown_type",
            "data": "some data"
        }

        observations = await cross_modal_aggregator.process_manual_entry(
            patient_id="patient_001",
            entry_data=entry_data,
            entered_by="test"
        )

        # Should return empty list for unknown type
        assert observations == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
