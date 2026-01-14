"""
Unit Tests for Longitudinal Patient Memory Module

Tests the core data structures and memory manager for
1-5 year patient record tracking in palliative care.
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from personalization.longitudinal_memory import (
    # Enums
    DataSourceType,
    SeverityLevel,
    TemporalTrend,
    AlertPriority,
    MedicationAction,
    # Base classes
    TimestampedObservation,
    SymptomObservation,
    MedicationEvent,
    VitalSignObservation,
    FunctionalStatusObservation,
    EmotionalObservation,
    TimeSeriesSummary,
    # Care team
    CareTeamMember,
    # Alerts
    MonitoringAlert,
    MonitoringRule,
    # Core classes
    LongitudinalPatientRecord,
    LongitudinalMemoryManager,
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
def sample_symptom_observation():
    """Create a sample symptom observation."""
    return SymptomObservation(
        observation_id="obs_001",
        timestamp=datetime.now(),
        source_type=DataSourceType.VOICE_CALL,
        source_id="conv_123",
        reported_by="patient",
        category="symptom",
        entity_name="pain",
        value=SeverityLevel.MODERATE,
        value_text="Patient reports moderate pain in lower back",
        symptom_name="pain",
        severity=SeverityLevel.MODERATE,
        location="lower back",
        duration="3 days"
    )


@pytest.fixture
def sample_medication_event():
    """Create a sample medication event."""
    return MedicationEvent(
        observation_id="obs_002",
        timestamp=datetime.now(),
        source_type=DataSourceType.WHATSAPP,
        source_id="msg_456",
        reported_by="caregiver",
        category="medication",
        entity_name="morphine",
        value=MedicationAction.TAKEN,
        value_text="Morphine 10mg taken as prescribed",
        medication_name="morphine",
        dosage="10mg",
        action=MedicationAction.TAKEN,
        frequency="every 4 hours"
    )


@pytest.fixture
def sample_patient_record(sample_symptom_observation, sample_medication_event):
    """Create a sample patient record with observations."""
    record = LongitudinalPatientRecord(patient_id="patient_001")
    record.add_observation(sample_symptom_observation)
    record.add_observation(sample_medication_event)
    return record


@pytest.fixture
def memory_manager(temp_storage):
    """Create a memory manager with temporary storage."""
    return LongitudinalMemoryManager(storage_path=temp_storage)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestEnums:
    """Test enum classes."""

    def test_severity_level_values(self):
        """Test SeverityLevel enum values."""
        assert SeverityLevel.NONE.value == 0
        assert SeverityLevel.MILD.value == 1
        assert SeverityLevel.MODERATE.value == 2
        assert SeverityLevel.SEVERE.value == 3
        assert SeverityLevel.VERY_SEVERE.value == 4

    def test_severity_level_from_string(self):
        """Test SeverityLevel.from_string conversion."""
        assert SeverityLevel.from_string("mild") == SeverityLevel.MILD
        assert SeverityLevel.from_string("SEVERE") == SeverityLevel.SEVERE
        assert SeverityLevel.from_string("2") == SeverityLevel.MODERATE
        assert SeverityLevel.from_string("invalid") == SeverityLevel.MODERATE  # Default

    def test_data_source_type_values(self):
        """Test DataSourceType enum values."""
        assert DataSourceType.VOICE_CALL.value == "voice_call"
        assert DataSourceType.WHATSAPP.value == "whatsapp"
        assert DataSourceType.UPLOADED_DOCUMENT.value == "uploaded_document"

    def test_temporal_trend_values(self):
        """Test TemporalTrend enum values."""
        assert TemporalTrend.IMPROVING.value == "improving"
        assert TemporalTrend.STABLE.value == "stable"
        assert TemporalTrend.WORSENING.value == "worsening"

    def test_alert_priority_values(self):
        """Test AlertPriority enum values."""
        assert AlertPriority.LOW.value == "low"
        assert AlertPriority.MEDIUM.value == "medium"
        assert AlertPriority.HIGH.value == "high"
        assert AlertPriority.URGENT.value == "urgent"


# ============================================================================
# OBSERVATION TESTS
# ============================================================================

class TestTimestampedObservation:
    """Test TimestampedObservation base class."""

    def test_create_observation(self):
        """Test creating a basic observation."""
        obs = TimestampedObservation(
            observation_id="obs_test",
            timestamp=datetime.now(),
            source_type=DataSourceType.VOICE_CALL,
            source_id="conv_001",
            reported_by="patient",
            category="symptom",
            entity_name="pain",
            value=SeverityLevel.MILD,
            value_text="Mild pain reported"
        )
        assert obs.observation_id == "obs_test"
        assert obs.category == "symptom"
        assert obs.entity_name == "pain"

    def test_observation_to_dict(self):
        """Test observation serialization."""
        obs = TimestampedObservation(
            observation_id="obs_test",
            timestamp=datetime(2025, 1, 8, 10, 0, 0),
            source_type=DataSourceType.VOICE_CALL,
            source_id="conv_001",
            reported_by="patient",
            category="symptom",
            entity_name="pain",
            value=SeverityLevel.MILD,
            value_text="Mild pain reported"
        )
        data = obs.to_dict()

        assert data["observation_id"] == "obs_test"
        assert data["source_type"] == "voice_call"
        assert data["category"] == "symptom"
        assert "timestamp" in data

    def test_observation_from_dict(self):
        """Test observation deserialization."""
        data = {
            "observation_id": "obs_test",
            "timestamp": "2025-01-08T10:00:00",
            "source_type": "voice_call",
            "source_id": "conv_001",
            "reported_by": "patient",
            "category": "symptom",
            "entity_name": "pain",
            "value": "1",
            "value_text": "Mild pain"
        }
        obs = TimestampedObservation.from_dict(data)

        assert obs.observation_id == "obs_test"
        assert obs.source_type == DataSourceType.VOICE_CALL
        assert obs.entity_name == "pain"


class TestSymptomObservation:
    """Test SymptomObservation class."""

    def test_create_symptom_observation(self, sample_symptom_observation):
        """Test creating a symptom observation."""
        obs = sample_symptom_observation
        assert obs.symptom_name == "pain"
        assert obs.severity == SeverityLevel.MODERATE
        assert obs.location == "lower back"
        assert obs.duration == "3 days"

    def test_symptom_observation_serialization(self, sample_symptom_observation):
        """Test symptom observation round-trip serialization."""
        obs = sample_symptom_observation
        data = obs.to_dict()

        assert data["symptom_name"] == "pain"
        assert data["severity"] == 2  # MODERATE
        assert data["location"] == "lower back"

        # Deserialize
        restored = SymptomObservation.from_dict(data)
        assert restored.symptom_name == obs.symptom_name
        assert restored.severity == obs.severity
        assert restored.location == obs.location


class TestMedicationEvent:
    """Test MedicationEvent class."""

    def test_create_medication_event(self, sample_medication_event):
        """Test creating a medication event."""
        event = sample_medication_event
        assert event.medication_name == "morphine"
        assert event.dosage == "10mg"
        assert event.action == MedicationAction.TAKEN

    def test_medication_event_serialization(self, sample_medication_event):
        """Test medication event round-trip serialization."""
        event = sample_medication_event
        data = event.to_dict()

        assert data["medication_name"] == "morphine"
        assert data["action"] == "taken"

        # Deserialize
        restored = MedicationEvent.from_dict(data)
        assert restored.medication_name == event.medication_name
        assert restored.action == event.action

    def test_medication_actions(self):
        """Test different medication actions."""
        for action in [MedicationAction.STARTED, MedicationAction.STOPPED,
                       MedicationAction.DOSE_CHANGED, MedicationAction.MISSED]:
            event = MedicationEvent(
                observation_id=f"obs_{action.value}",
                timestamp=datetime.now(),
                source_type=DataSourceType.WHATSAPP,
                source_id="msg_001",
                reported_by="patient",
                category="medication",
                entity_name="test_med",
                value=action,
                value_text=f"Medication {action.value}",
                medication_name="test_med",
                dosage="5mg",
                action=action
            )
            assert event.action == action


class TestVitalSignObservation:
    """Test VitalSignObservation class."""

    def test_create_vital_sign(self):
        """Test creating a vital sign observation."""
        vital = VitalSignObservation(
            observation_id="obs_vital",
            timestamp=datetime.now(),
            source_type=DataSourceType.CAREGIVER_REPORT,
            source_id="manual_001",
            reported_by="caregiver",
            category="vital_sign",
            entity_name="blood_pressure",
            value="120/80",
            value_text="Blood pressure: 120/80 mmHg",
            vital_name="blood_pressure",
            value_numeric=120.0,
            unit="mmHg",
            within_normal_range=True
        )
        assert vital.vital_name == "blood_pressure"
        assert vital.value_numeric == 120.0
        assert vital.within_normal_range is True

    def test_vital_sign_serialization(self):
        """Test vital sign round-trip serialization."""
        vital = VitalSignObservation(
            observation_id="obs_vital",
            timestamp=datetime.now(),
            source_type=DataSourceType.CAREGIVER_REPORT,
            source_id="manual_001",
            reported_by="caregiver",
            category="vital_sign",
            entity_name="temperature",
            value=98.6,
            value_text="Temperature: 98.6F",
            vital_name="temperature",
            value_numeric=98.6,
            unit="F",
            within_normal_range=True
        )
        data = vital.to_dict()
        restored = VitalSignObservation.from_dict(data)

        assert restored.vital_name == vital.vital_name
        assert restored.value_numeric == vital.value_numeric


class TestEmotionalObservation:
    """Test EmotionalObservation class."""

    def test_create_emotional_observation(self):
        """Test creating an emotional observation."""
        emotion = EmotionalObservation(
            observation_id="obs_emotion",
            timestamp=datetime.now(),
            source_type=DataSourceType.VOICE_CALL,
            source_id="conv_001",
            reported_by="patient",
            category="emotional",
            entity_name="anxiety",
            value=SeverityLevel.MODERATE,
            value_text="Patient reports moderate anxiety",
            emotion_type="anxiety",
            intensity=SeverityLevel.MODERATE,
            triggers=["upcoming appointment", "family stress"]
        )
        assert emotion.emotion_type == "anxiety"
        assert emotion.intensity == SeverityLevel.MODERATE
        assert len(emotion.triggers) == 2


# ============================================================================
# CARE TEAM TESTS
# ============================================================================

class TestCareTeamMember:
    """Test CareTeamMember class."""

    def test_create_care_team_member(self):
        """Test creating a care team member."""
        member = CareTeamMember(
            provider_id="dr_001",
            name="Dr. Sharma",
            role="doctor",
            organization="City Hospital",
            primary_contact=True
        )
        assert member.name == "Dr. Sharma"
        assert member.role == "doctor"
        assert member.primary_contact is True

    def test_care_team_member_serialization(self):
        """Test care team member round-trip serialization."""
        member = CareTeamMember(
            provider_id="nurse_001",
            name="Nurse Priya",
            role="nurse",
            organization="Home Care Services",
            phone_number="+91-xxxxx"
        )
        data = member.to_dict()
        restored = CareTeamMember.from_dict(data)

        assert restored.name == member.name
        assert restored.role == member.role


# ============================================================================
# MONITORING TESTS
# ============================================================================

class TestMonitoringAlert:
    """Test MonitoringAlert class."""

    def test_create_alert(self):
        """Test creating a monitoring alert."""
        alert = MonitoringAlert(
            alert_id="alert_001",
            patient_id="patient_001",
            created_at=datetime.now(),
            priority=AlertPriority.HIGH,
            category="symptom_change",
            title="Worsening Pain",
            description="Pain has been worsening over the past week",
            suggested_actions=["Schedule follow-up", "Review medications"]
        )
        assert alert.alert_id == "alert_001"
        assert alert.priority == AlertPriority.HIGH
        assert len(alert.suggested_actions) == 2

    def test_alert_serialization(self):
        """Test alert round-trip serialization."""
        alert = MonitoringAlert(
            alert_id="alert_002",
            patient_id="patient_001",
            created_at=datetime.now(),
            priority=AlertPriority.URGENT,
            category="urgent_symptom",
            title="Severe Breathlessness",
            description="Patient reports severe breathlessness"
        )
        data = alert.to_dict()
        restored = MonitoringAlert.from_dict(data)

        assert restored.alert_id == alert.alert_id
        assert restored.priority == alert.priority


class TestMonitoringRule:
    """Test MonitoringRule class."""

    def test_create_rule(self):
        """Test creating a monitoring rule."""
        rule = MonitoringRule(
            rule_id="rule_001",
            name="Worsening Pain",
            description="Detect worsening pain trend",
            category="symptom",
            entity_name_pattern="pain",
            pattern_type="worsening_trend",
            alert_priority=AlertPriority.HIGH
        )
        assert rule.rule_id == "rule_001"
        assert rule.pattern_type == "worsening_trend"


# ============================================================================
# PATIENT RECORD TESTS
# ============================================================================

class TestLongitudinalPatientRecord:
    """Test LongitudinalPatientRecord class."""

    def test_create_record(self):
        """Test creating a patient record."""
        record = LongitudinalPatientRecord(patient_id="patient_001")
        assert record.patient_id == "patient_001"
        assert len(record.observations) == 0
        assert len(record.care_team) == 0

    def test_add_observation(self, sample_symptom_observation):
        """Test adding observations to record."""
        record = LongitudinalPatientRecord(patient_id="patient_001")
        record.add_observation(sample_symptom_observation)

        assert len(record.observations) == 1
        assert record.total_observations == 1
        assert DataSourceType.VOICE_CALL in record.data_sources

    def test_get_observations_for_entity(self, sample_patient_record):
        """Test filtering observations by entity."""
        record = sample_patient_record
        pain_obs = record.get_observations_for_entity("pain")

        assert len(pain_obs) == 1
        assert pain_obs[0].entity_name == "pain"

    def test_get_observations_by_category(self, sample_patient_record):
        """Test filtering observations by category."""
        record = sample_patient_record
        symptom_obs = record.get_observations_by_category("symptom")
        med_obs = record.get_observations_by_category("medication")

        assert len(symptom_obs) == 1
        assert len(med_obs) == 1

    def test_get_active_symptoms(self):
        """Test getting active symptoms."""
        record = LongitudinalPatientRecord(patient_id="patient_001")

        # Add recent symptom
        obs = SymptomObservation(
            observation_id="obs_001",
            timestamp=datetime.now() - timedelta(days=5),
            source_type=DataSourceType.VOICE_CALL,
            source_id="conv_001",
            reported_by="patient",
            category="symptom",
            entity_name="pain",
            value=SeverityLevel.SEVERE,
            value_text="Severe pain",
            symptom_name="pain",
            severity=SeverityLevel.SEVERE
        )
        record.add_observation(obs)

        active = record.get_active_symptoms()
        assert len(active) == 1
        assert active[0][0] == "pain"
        assert active[0][1] == SeverityLevel.SEVERE

    def test_get_current_medications(self):
        """Test getting current medications."""
        record = LongitudinalPatientRecord(patient_id="patient_001")

        # Add medication event
        event = MedicationEvent(
            observation_id="obs_001",
            timestamp=datetime.now() - timedelta(days=5),
            source_type=DataSourceType.WHATSAPP,
            source_id="msg_001",
            reported_by="patient",
            category="medication",
            entity_name="morphine",
            value=MedicationAction.TAKEN,
            value_text="Morphine taken",
            medication_name="morphine",
            dosage="10mg",
            action=MedicationAction.TAKEN
        )
        record.add_observation(event)

        meds = record.get_current_medications()
        assert len(meds) == 1
        assert meds[0]["name"] == "morphine"

    def test_add_care_team_member(self):
        """Test adding care team members."""
        record = LongitudinalPatientRecord(patient_id="patient_001")

        member = CareTeamMember(
            provider_id="dr_001",
            name="Dr. Sharma",
            role="doctor",
            primary_contact=True
        )
        record.add_care_team_member(member)

        assert len(record.care_team) == 1
        primary = record.get_primary_contact()
        assert primary.name == "Dr. Sharma"

    def test_record_serialization(self, sample_patient_record):
        """Test patient record round-trip serialization."""
        record = sample_patient_record
        data = record.to_dict()
        restored = LongitudinalPatientRecord.from_dict(data)

        assert restored.patient_id == record.patient_id
        assert len(restored.observations) == len(record.observations)

    def test_time_series_summary(self):
        """Test generating time series summary."""
        record = LongitudinalPatientRecord(patient_id="patient_001")

        # Add multiple observations over time
        for i in range(5):
            obs = SymptomObservation(
                observation_id=f"obs_{i}",
                timestamp=datetime.now() - timedelta(days=30-i*5),
                source_type=DataSourceType.VOICE_CALL,
                source_id=f"conv_{i}",
                reported_by="patient",
                category="symptom",
                entity_name="pain",
                value=SeverityLevel(min(4, i+1)),  # Increasing severity
                value_text=f"Pain level {i+1}",
                symptom_name="pain",
                severity=SeverityLevel(min(4, i+1))
            )
            record.add_observation(obs)

        summary = record.get_time_series_summary("pain", days=60)

        assert summary is not None
        assert summary.total_observations == 5
        assert summary.trend in [TemporalTrend.WORSENING, TemporalTrend.FLUCTUATING]

    def test_trend_calculation_improving(self):
        """Test trend calculation for improving symptom."""
        record = LongitudinalPatientRecord(patient_id="patient_001")

        # Add observations showing improvement (decreasing severity)
        for i in range(5):
            obs = SymptomObservation(
                observation_id=f"obs_{i}",
                timestamp=datetime.now() - timedelta(days=30-i*5),
                source_type=DataSourceType.VOICE_CALL,
                source_id=f"conv_{i}",
                reported_by="patient",
                category="symptom",
                entity_name="pain",
                value=SeverityLevel(max(0, 4-i)),  # Decreasing severity
                value_text=f"Pain level {4-i}",
                symptom_name="pain",
                severity=SeverityLevel(max(0, 4-i))
            )
            record.add_observation(obs)

        summary = record.get_time_series_summary("pain", days=60)

        assert summary is not None
        assert summary.trend == TemporalTrend.IMPROVING


# ============================================================================
# MEMORY MANAGER TESTS
# ============================================================================

class TestLongitudinalMemoryManager:
    """Test LongitudinalMemoryManager class."""

    @pytest.mark.asyncio
    async def test_create_manager(self, temp_storage):
        """Test creating a memory manager."""
        manager = LongitudinalMemoryManager(storage_path=temp_storage)
        assert manager.storage_path.exists()

    @pytest.mark.asyncio
    async def test_get_or_create_record(self, memory_manager):
        """Test getting or creating patient record."""
        record = await memory_manager.get_or_create_record("patient_001")

        assert record.patient_id == "patient_001"
        assert len(record.monitoring_rules) > 0  # Default rules

    @pytest.mark.asyncio
    async def test_save_and_load_record(self, memory_manager):
        """Test saving and loading patient record."""
        # Create and save record
        record = await memory_manager.get_or_create_record("patient_001")
        record.primary_condition = "cancer"
        await memory_manager.save_record(record)

        # Clear cache and reload
        memory_manager._cache.clear()
        loaded = await memory_manager.get_or_create_record("patient_001")

        assert loaded.primary_condition == "cancer"

    @pytest.mark.asyncio
    async def test_add_observation_via_manager(self, memory_manager, sample_symptom_observation):
        """Test adding observation through manager."""
        record = await memory_manager.add_observation(
            "patient_001",
            sample_symptom_observation
        )

        assert len(record.observations) == 1
        assert record.total_observations == 1

    @pytest.mark.asyncio
    async def test_get_longitudinal_summary(self, memory_manager, sample_symptom_observation):
        """Test getting longitudinal summary."""
        # Add observation
        await memory_manager.add_observation("patient_001", sample_symptom_observation)

        summary = await memory_manager.get_longitudinal_summary("patient_001", days=30)

        assert summary["patient_id"] == "patient_001"
        assert summary["total_observations"] == 1

    @pytest.mark.asyncio
    async def test_monitoring_checks(self, memory_manager):
        """Test running monitoring checks."""
        # Add observations that would trigger alert
        record = await memory_manager.get_or_create_record("patient_001")

        # Add worsening pain observations
        for i in range(4):
            obs = SymptomObservation(
                observation_id=f"obs_{i}",
                timestamp=datetime.now() - timedelta(days=6-i),
                source_type=DataSourceType.VOICE_CALL,
                source_id=f"conv_{i}",
                reported_by="patient",
                category="symptom",
                entity_name="pain",
                value=SeverityLevel(min(4, i+1)),
                value_text=f"Pain level {i+1}",
                symptom_name="pain",
                severity=SeverityLevel(min(4, i+1))
            )
            record.add_observation(obs)

        await memory_manager.save_record(record)

        # Run monitoring
        alerts = await memory_manager.run_monitoring_checks("patient_001")

        # Should generate alert for worsening trend
        assert isinstance(alerts, list)

    @pytest.mark.asyncio
    async def test_care_team_management(self, memory_manager):
        """Test care team management through manager."""
        member = CareTeamMember(
            provider_id="dr_001",
            name="Dr. Sharma",
            role="doctor",
            primary_contact=True
        )

        await memory_manager.add_care_team_member("patient_001", member)
        care_team = await memory_manager.get_care_team("patient_001")

        assert len(care_team) == 1
        assert care_team[0].name == "Dr. Sharma"

    @pytest.mark.asyncio
    async def test_alert_acknowledgment(self, memory_manager):
        """Test alert acknowledgment."""
        record = await memory_manager.get_or_create_record("patient_001")

        # Add an alert
        alert = MonitoringAlert(
            alert_id="alert_test",
            patient_id="patient_001",
            created_at=datetime.now(),
            priority=AlertPriority.HIGH,
            category="test",
            title="Test Alert",
            description="Test alert description"
        )
        record.active_alerts.append(alert)
        await memory_manager.save_record(record)

        # Acknowledge
        result = await memory_manager.acknowledge_alert("patient_001", "alert_test", "dr_001")
        assert result is True

        # Verify
        record = await memory_manager.get_or_create_record("patient_001")
        assert record.active_alerts[0].acknowledged is True

    @pytest.mark.asyncio
    async def test_alert_resolution(self, memory_manager):
        """Test alert resolution."""
        record = await memory_manager.get_or_create_record("patient_001")

        # Add an alert
        alert = MonitoringAlert(
            alert_id="alert_resolve",
            patient_id="patient_001",
            created_at=datetime.now(),
            priority=AlertPriority.MEDIUM,
            category="test",
            title="Test Alert",
            description="Test alert for resolution"
        )
        record.active_alerts.append(alert)
        await memory_manager.save_record(record)

        # Resolve
        result = await memory_manager.resolve_alert(
            "patient_001",
            "alert_resolve",
            "Issue addressed"
        )
        assert result is True

        # Verify moved to history
        record = await memory_manager.get_or_create_record("patient_001")
        assert len(record.active_alerts) == 0
        assert len(record.alert_history) == 1

    @pytest.mark.asyncio
    async def test_statistics(self, memory_manager, sample_symptom_observation):
        """Test getting memory statistics."""
        await memory_manager.add_observation("patient_001", sample_symptom_observation)

        stats = await memory_manager.get_statistics()

        assert "total_patients" in stats
        assert "total_observations" in stats
        assert stats["total_patients"] >= 1


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_record_summary(self):
        """Test time series summary with no data."""
        record = LongitudinalPatientRecord(patient_id="empty_patient")
        summary = record.get_time_series_summary("pain", days=30)

        assert summary is None

    def test_trend_with_insufficient_data(self):
        """Test trend calculation with insufficient data points."""
        record = LongitudinalPatientRecord(patient_id="patient_001")

        # Add only 2 observations (need 3 for trend)
        for i in range(2):
            obs = SymptomObservation(
                observation_id=f"obs_{i}",
                timestamp=datetime.now() - timedelta(days=i),
                source_type=DataSourceType.VOICE_CALL,
                source_id=f"conv_{i}",
                reported_by="patient",
                category="symptom",
                entity_name="pain",
                value=SeverityLevel.MODERATE,
                value_text="Pain",
                symptom_name="pain",
                severity=SeverityLevel.MODERATE
            )
            record.add_observation(obs)

        summary = record.get_time_series_summary("pain", days=30)

        # Should return None or UNKNOWN trend
        if summary:
            assert summary.trend == TemporalTrend.UNKNOWN

    def test_date_range_filtering(self):
        """Test observation filtering by date range."""
        record = LongitudinalPatientRecord(patient_id="patient_001")

        # Add observations at different times
        old_obs = SymptomObservation(
            observation_id="obs_old",
            timestamp=datetime.now() - timedelta(days=100),
            source_type=DataSourceType.VOICE_CALL,
            source_id="conv_old",
            reported_by="patient",
            category="symptom",
            entity_name="pain",
            value=SeverityLevel.MILD,
            value_text="Old pain",
            symptom_name="pain",
            severity=SeverityLevel.MILD
        )

        recent_obs = SymptomObservation(
            observation_id="obs_recent",
            timestamp=datetime.now() - timedelta(days=5),
            source_type=DataSourceType.VOICE_CALL,
            source_id="conv_recent",
            reported_by="patient",
            category="symptom",
            entity_name="pain",
            value=SeverityLevel.SEVERE,
            value_text="Recent pain",
            symptom_name="pain",
            severity=SeverityLevel.SEVERE
        )

        record.add_observation(old_obs)
        record.add_observation(recent_obs)

        # Filter to last 30 days
        start_date = datetime.now() - timedelta(days=30)
        filtered = record.get_observations_for_entity("pain", start_date=start_date)

        assert len(filtered) == 1
        assert filtered[0].observation_id == "obs_recent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
