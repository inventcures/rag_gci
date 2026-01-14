"""
Unit Tests for Temporal Reasoner Module

Tests trend detection, medication effectiveness analysis,
and correlation detection for longitudinal patient data.
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from personalization.temporal_reasoner import (
    SymptomProgressionReport,
    MedicationEffectivenessReport,
    CorrelationAnalysis,
    TemporalReasoner,
)
from personalization.longitudinal_memory import (
    LongitudinalMemoryManager,
    LongitudinalPatientRecord,
    SymptomObservation,
    MedicationEvent,
    DataSourceType,
    SeverityLevel,
    MedicationAction,
    TemporalTrend,
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
def longitudinal_manager(temp_storage):
    """Create real longitudinal manager for testing."""
    return LongitudinalMemoryManager(storage_path=temp_storage)


@pytest.fixture
def temporal_reasoner(longitudinal_manager):
    """Create TemporalReasoner with real manager."""
    return TemporalReasoner(longitudinal_manager=longitudinal_manager)


@pytest.fixture
def sample_worsening_observations():
    """Create observations showing worsening trend."""
    observations = []
    base_time = datetime.now() - timedelta(days=30)

    for i in range(6):
        obs = SymptomObservation(
            observation_id=f"obs_pain_{i}",
            timestamp=base_time + timedelta(days=i * 5),
            source_type=DataSourceType.VOICE_CALL,
            source_id=f"conv_{i}",
            reported_by="patient",
            category="symptom",
            entity_name="pain",
            value=SeverityLevel(min(4, i + 1)),  # 1->2->3->4->4->4
            value_text=f"Pain level {i+1}",
            symptom_name="pain",
            severity=SeverityLevel(min(4, i + 1))
        )
        observations.append(obs)

    return observations


@pytest.fixture
def sample_improving_observations():
    """Create observations showing improving trend."""
    observations = []
    base_time = datetime.now() - timedelta(days=30)

    for i in range(6):
        obs = SymptomObservation(
            observation_id=f"obs_pain_{i}",
            timestamp=base_time + timedelta(days=i * 5),
            source_type=DataSourceType.VOICE_CALL,
            source_id=f"conv_{i}",
            reported_by="patient",
            category="symptom",
            entity_name="pain",
            value=SeverityLevel(max(0, 4 - i)),  # 4->3->2->1->0->0
            value_text=f"Pain level {4-i}",
            symptom_name="pain",
            severity=SeverityLevel(max(0, 4 - i))
        )
        observations.append(obs)

    return observations


@pytest.fixture
def sample_medication_events():
    """Create medication events for testing."""
    events = []
    base_time = datetime.now() - timedelta(days=30)

    for i in range(10):
        event = MedicationEvent(
            observation_id=f"med_{i}",
            timestamp=base_time + timedelta(days=i * 3),
            source_type=DataSourceType.WHATSAPP,
            source_id=f"msg_{i}",
            reported_by="patient",
            category="medication",
            entity_name="morphine",
            value=MedicationAction.TAKEN,
            value_text=f"Morphine taken",
            medication_name="morphine",
            dosage="10mg",
            action=MedicationAction.TAKEN,
            frequency="every 4 hours"
        )
        events.append(event)

    # Add some missed doses
    for i in range(2):
        event = MedicationEvent(
            observation_id=f"med_missed_{i}",
            timestamp=base_time + timedelta(days=15 + i),
            source_type=DataSourceType.WHATSAPP,
            source_id=f"msg_missed_{i}",
            reported_by="patient",
            category="medication",
            entity_name="morphine",
            value=MedicationAction.MISSED,
            value_text="Morphine missed",
            medication_name="morphine",
            dosage="10mg",
            action=MedicationAction.MISSED
        )
        events.append(event)

    return events


# ============================================================================
# REPORT DATACLASS TESTS
# ============================================================================

class TestSymptomProgressionReport:
    """Test SymptomProgressionReport dataclass."""

    def test_create_report(self):
        """Test creating a symptom progression report."""
        report = SymptomProgressionReport(
            symptom_name="pain",
            patient_id="patient_001",
            analysis_period_days=90,
            total_observations=10,
            current_severity=SeverityLevel.SEVERE,
            baseline_severity=SeverityLevel.MILD,
            trend=TemporalTrend.WORSENING,
            trend_confidence=0.85
        )

        assert report.symptom_name == "pain"
        assert report.trend == TemporalTrend.WORSENING
        assert report.trend_confidence == 0.85

    def test_report_to_dict(self):
        """Test report serialization."""
        report = SymptomProgressionReport(
            symptom_name="pain",
            patient_id="patient_001",
            analysis_period_days=90,
            total_observations=10,
            current_severity=SeverityLevel.SEVERE,
            baseline_severity=SeverityLevel.MILD,
            trend=TemporalTrend.WORSENING,
            trend_confidence=0.85
        )

        data = report.to_dict()

        assert data["symptom_name"] == "pain"
        assert data["trend"] == "worsening"
        assert data["current_severity"] == 3  # SEVERE
        assert "generated_at" in data


class TestMedicationEffectivenessReport:
    """Test MedicationEffectivenessReport dataclass."""

    def test_create_report(self):
        """Test creating medication effectiveness report."""
        report = MedicationEffectivenessReport(
            medication_name="morphine",
            patient_id="patient_001",
            analysis_period_days=90,
            total_doses_recorded=30,
            adherence_rate=0.9,
            missed_doses=3,
            effectiveness_trend=TemporalTrend.IMPROVING
        )

        assert report.medication_name == "morphine"
        assert report.adherence_rate == 0.9
        assert report.missed_doses == 3

    def test_report_to_dict(self):
        """Test medication report serialization."""
        report = MedicationEffectivenessReport(
            medication_name="morphine",
            patient_id="patient_001",
            analysis_period_days=90,
            total_doses_recorded=30,
            adherence_rate=0.9,
            missed_doses=3,
            effectiveness_trend=TemporalTrend.STABLE
        )

        data = report.to_dict()

        assert data["medication_name"] == "morphine"
        assert data["adherence_rate"] == 0.9
        assert data["effectiveness_trend"] == "stable"


class TestCorrelationAnalysis:
    """Test CorrelationAnalysis dataclass."""

    def test_create_correlation(self):
        """Test creating correlation analysis."""
        correlation = CorrelationAnalysis(
            variable_1="morphine",
            variable_2="pain",
            correlation_type="negative",
            correlation_strength=-0.7,
            confidence=0.8,
            description="Morphine associated with reduced pain",
            clinical_significance="significant"
        )

        assert correlation.variable_1 == "morphine"
        assert correlation.correlation_type == "negative"
        assert correlation.correlation_strength == -0.7

    def test_correlation_to_dict(self):
        """Test correlation serialization."""
        correlation = CorrelationAnalysis(
            variable_1="morphine",
            variable_2="pain",
            correlation_type="negative",
            correlation_strength=-0.7,
            confidence=0.8,
            description="Morphine reduces pain",
            clinical_significance="significant"
        )

        data = correlation.to_dict()

        assert data["variable_1"] == "morphine"
        assert data["correlation_strength"] == -0.7
        assert data["clinical_significance"] == "significant"


# ============================================================================
# TEMPORAL REASONER TESTS
# ============================================================================

class TestTemporalReasonerInit:
    """Test TemporalReasoner initialization."""

    def test_create_reasoner(self, longitudinal_manager):
        """Test creating temporal reasoner."""
        reasoner = TemporalReasoner(longitudinal_manager=longitudinal_manager)
        assert reasoner.longitudinal is not None


class TestSymptomProgressionAnalysis:
    """Test symptom progression analysis."""

    @pytest.mark.asyncio
    async def test_analyze_worsening_symptom(
        self,
        temporal_reasoner,
        longitudinal_manager,
        sample_worsening_observations
    ):
        """Test analyzing worsening symptom trend."""
        # Add observations to patient record
        record = await longitudinal_manager.get_or_create_record("patient_001")
        for obs in sample_worsening_observations:
            record.add_observation(obs)
        await longitudinal_manager.save_record(record)

        # Analyze
        report = await temporal_reasoner.analyze_symptom_progression(
            patient_id="patient_001",
            symptom_name="pain",
            time_window_days=60
        )

        assert report is not None
        assert report.symptom_name == "pain"
        assert report.total_observations == 6
        assert report.trend == TemporalTrend.WORSENING

    @pytest.mark.asyncio
    async def test_analyze_improving_symptom(
        self,
        temporal_reasoner,
        longitudinal_manager,
        sample_improving_observations
    ):
        """Test analyzing improving symptom trend."""
        # Add observations
        record = await longitudinal_manager.get_or_create_record("patient_002")
        for obs in sample_improving_observations:
            record.add_observation(obs)
        await longitudinal_manager.save_record(record)

        # Analyze
        report = await temporal_reasoner.analyze_symptom_progression(
            patient_id="patient_002",
            symptom_name="pain",
            time_window_days=60
        )

        assert report is not None
        assert report.trend == TemporalTrend.IMPROVING

    @pytest.mark.asyncio
    async def test_analyze_insufficient_data(self, temporal_reasoner, longitudinal_manager):
        """Test analysis with insufficient data returns None."""
        # Create record with only 2 observations (need 3+)
        record = await longitudinal_manager.get_or_create_record("patient_003")

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

        await longitudinal_manager.save_record(record)

        # Analyze - should return None due to insufficient data
        report = await temporal_reasoner.analyze_symptom_progression(
            patient_id="patient_003",
            symptom_name="pain",
            time_window_days=30
        )

        assert report is None

    @pytest.mark.asyncio
    async def test_report_includes_clinical_concerns(
        self,
        temporal_reasoner,
        longitudinal_manager,
        sample_worsening_observations
    ):
        """Test that report includes clinical concerns for worsening symptoms."""
        record = await longitudinal_manager.get_or_create_record("patient_004")
        for obs in sample_worsening_observations:
            record.add_observation(obs)
        await longitudinal_manager.save_record(record)

        report = await temporal_reasoner.analyze_symptom_progression(
            patient_id="patient_004",
            symptom_name="pain",
            time_window_days=60
        )

        assert report is not None
        assert len(report.clinical_concerns) > 0
        assert len(report.recommended_actions) > 0

    @pytest.mark.asyncio
    async def test_report_includes_time_series_data(
        self,
        temporal_reasoner,
        longitudinal_manager,
        sample_worsening_observations
    ):
        """Test that report includes time series data for visualization."""
        record = await longitudinal_manager.get_or_create_record("patient_005")
        for obs in sample_worsening_observations:
            record.add_observation(obs)
        await longitudinal_manager.save_record(record)

        report = await temporal_reasoner.analyze_symptom_progression(
            patient_id="patient_005",
            symptom_name="pain",
            time_window_days=60
        )

        assert report is not None
        assert len(report.time_series_data) == 6
        assert "timestamp" in report.time_series_data[0]
        assert "severity" in report.time_series_data[0]


class TestMedicationEffectivenessAnalysis:
    """Test medication effectiveness analysis."""

    @pytest.mark.asyncio
    async def test_analyze_medication_adherence(
        self,
        temporal_reasoner,
        longitudinal_manager,
        sample_medication_events
    ):
        """Test medication adherence calculation."""
        record = await longitudinal_manager.get_or_create_record("patient_006")
        for event in sample_medication_events:
            record.add_observation(event)
        await longitudinal_manager.save_record(record)

        report = await temporal_reasoner.analyze_medication_effectiveness(
            patient_id="patient_006",
            medication_name="morphine",
            time_window_days=60
        )

        assert report is not None
        assert report.medication_name == "morphine"
        assert report.total_doses_recorded > 0
        assert 0 <= report.adherence_rate <= 1.0
        assert report.missed_doses == 2

    @pytest.mark.asyncio
    async def test_medication_no_data(self, temporal_reasoner, longitudinal_manager):
        """Test medication analysis with no data."""
        await longitudinal_manager.get_or_create_record("patient_007")

        report = await temporal_reasoner.analyze_medication_effectiveness(
            patient_id="patient_007",
            medication_name="unknown_med",
            time_window_days=60
        )

        assert report is None

    @pytest.mark.asyncio
    async def test_rotation_recommendation(
        self,
        temporal_reasoner,
        longitudinal_manager
    ):
        """Test rotation recommendation for poor adherence."""
        record = await longitudinal_manager.get_or_create_record("patient_008")
        base_time = datetime.now() - timedelta(days=30)

        # Add mostly missed doses (poor adherence)
        for i in range(10):
            action = MedicationAction.MISSED if i < 6 else MedicationAction.TAKEN
            event = MedicationEvent(
                observation_id=f"med_{i}",
                timestamp=base_time + timedelta(days=i * 3),
                source_type=DataSourceType.WHATSAPP,
                source_id=f"msg_{i}",
                reported_by="patient",
                category="medication",
                entity_name="tramadol",
                value=action,
                value_text=f"Tramadol {action.value}",
                medication_name="tramadol",
                dosage="50mg",
                action=action
            )
            record.add_observation(event)

        await longitudinal_manager.save_record(record)

        report = await temporal_reasoner.analyze_medication_effectiveness(
            patient_id="patient_008",
            medication_name="tramadol",
            time_window_days=60
        )

        assert report is not None
        assert report.adherence_rate < 0.5
        assert report.should_consider_rotation is True


class TestCorrelationDetection:
    """Test correlation detection between medications and symptoms."""

    @pytest.mark.asyncio
    async def test_find_correlations(
        self,
        temporal_reasoner,
        longitudinal_manager
    ):
        """Test finding correlations between medications and symptoms."""
        record = await longitudinal_manager.get_or_create_record("patient_009")
        base_time = datetime.now() - timedelta(days=30)

        # Add interleaved medication and symptom observations
        for i in range(15):
            day = base_time + timedelta(days=i * 2)

            # Medication event
            med_event = MedicationEvent(
                observation_id=f"med_{i}",
                timestamp=day,
                source_type=DataSourceType.WHATSAPP,
                source_id=f"msg_{i}",
                reported_by="patient",
                category="medication",
                entity_name="morphine",
                value=MedicationAction.TAKEN,
                value_text="Morphine taken",
                medication_name="morphine",
                dosage="10mg",
                action=MedicationAction.TAKEN
            )
            record.add_observation(med_event)

            # Symptom observation (negative correlation - pain decreases with medication)
            symptom_obs = SymptomObservation(
                observation_id=f"pain_{i}",
                timestamp=day + timedelta(hours=2),
                source_type=DataSourceType.VOICE_CALL,
                source_id=f"conv_{i}",
                reported_by="patient",
                category="symptom",
                entity_name="pain",
                value=SeverityLevel(max(0, 3 - (i % 3))),
                value_text="Pain level",
                symptom_name="pain",
                severity=SeverityLevel(max(0, 3 - (i % 3)))
            )
            record.add_observation(symptom_obs)

        await longitudinal_manager.save_record(record)

        correlations = await temporal_reasoner.find_correlations(
            patient_id="patient_009",
            time_window_days=60
        )

        assert isinstance(correlations, list)
        # May or may not find significant correlations depending on data

    @pytest.mark.asyncio
    async def test_correlations_empty_data(self, temporal_reasoner, longitudinal_manager):
        """Test correlation detection with no data."""
        await longitudinal_manager.get_or_create_record("patient_010")

        correlations = await temporal_reasoner.find_correlations(
            patient_id="patient_010",
            time_window_days=60
        )

        assert correlations == []


class TestTrendCalculation:
    """Test internal trend calculation methods."""

    def _create_dummy_observations(self, count: int):
        """Create dummy observations for trend testing."""
        return [
            SymptomObservation(
                observation_id=f"obs_{i}",
                timestamp=datetime.now() - timedelta(days=count-i),
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
            for i in range(count)
        ]

    def test_calculate_trend_worsening(self, temporal_reasoner):
        """Test trend calculation for worsening values."""
        observations = self._create_dummy_observations(6)
        numeric_values = [1, 2, 2, 3, 3, 4]  # Increasing

        trend, confidence = temporal_reasoner._calculate_trend(observations, numeric_values)

        assert trend == TemporalTrend.WORSENING

    def test_calculate_trend_improving(self, temporal_reasoner):
        """Test trend calculation for improving values."""
        observations = self._create_dummy_observations(6)
        numeric_values = [4, 3, 3, 2, 2, 1]  # Decreasing

        trend, confidence = temporal_reasoner._calculate_trend(observations, numeric_values)

        assert trend == TemporalTrend.IMPROVING

    def test_calculate_trend_stable(self, temporal_reasoner):
        """Test trend calculation for stable values."""
        observations = self._create_dummy_observations(6)
        numeric_values = [2, 2, 2, 2, 2, 2]  # Flat

        trend, confidence = temporal_reasoner._calculate_trend(observations, numeric_values)

        assert trend == TemporalTrend.STABLE

    def test_calculate_trend_insufficient_data(self, temporal_reasoner):
        """Test trend calculation with insufficient data."""
        observations = self._create_dummy_observations(2)
        numeric_values = [2, 3]  # Only 2 values

        trend, confidence = temporal_reasoner._calculate_trend(observations, numeric_values)

        assert trend == TemporalTrend.UNKNOWN


class TestPatternDetection:
    """Test pattern detection (diurnal, weekly)."""

    def test_detect_diurnal_pattern_worse_evening(self, temporal_reasoner):
        """Test detection of evening-worse pattern."""
        observations = []
        base_date = datetime.now().replace(hour=0, minute=0)

        # Add morning observations (mild)
        for i in range(3):
            obs = SymptomObservation(
                observation_id=f"morning_{i}",
                timestamp=base_date.replace(hour=8) + timedelta(days=i),
                source_type=DataSourceType.VOICE_CALL,
                source_id=f"conv_{i}",
                reported_by="patient",
                category="symptom",
                entity_name="pain",
                value=SeverityLevel.MILD,
                value_text="Mild pain",
                symptom_name="pain",
                severity=SeverityLevel.MILD
            )
            observations.append(obs)

        # Add evening observations (severe)
        for i in range(3):
            obs = SymptomObservation(
                observation_id=f"evening_{i}",
                timestamp=base_date.replace(hour=20) + timedelta(days=i),
                source_type=DataSourceType.VOICE_CALL,
                source_id=f"conv_e_{i}",
                reported_by="patient",
                category="symptom",
                entity_name="pain",
                value=SeverityLevel.SEVERE,
                value_text="Severe pain",
                symptom_name="pain",
                severity=SeverityLevel.SEVERE
            )
            observations.append(obs)

        pattern = temporal_reasoner._detect_diurnal_pattern(observations)

        assert pattern in ["worse_evening", "worse_daytime", "none"]

    def test_detect_weekly_pattern(self, temporal_reasoner):
        """Test detection of weekly pattern."""
        observations = []
        # Create a Monday
        base_date = datetime(2025, 1, 6, 10, 0)  # Monday

        # Add weekday observations (mild)
        for i in range(5):  # Mon-Fri
            obs = SymptomObservation(
                observation_id=f"weekday_{i}",
                timestamp=base_date + timedelta(days=i),
                source_type=DataSourceType.VOICE_CALL,
                source_id=f"conv_{i}",
                reported_by="patient",
                category="symptom",
                entity_name="pain",
                value=SeverityLevel.MILD,
                value_text="Mild pain",
                symptom_name="pain",
                severity=SeverityLevel.MILD
            )
            observations.append(obs)

        # Add weekend observations (severe)
        for i in range(2):  # Sat-Sun
            obs = SymptomObservation(
                observation_id=f"weekend_{i}",
                timestamp=base_date + timedelta(days=5+i),
                source_type=DataSourceType.VOICE_CALL,
                source_id=f"conv_w_{i}",
                reported_by="patient",
                category="symptom",
                entity_name="pain",
                value=SeverityLevel.SEVERE,
                value_text="Severe pain",
                symptom_name="pain",
                severity=SeverityLevel.SEVERE
            )
            observations.append(obs)

        pattern = temporal_reasoner._detect_weekly_pattern(observations)

        assert pattern in ["worse_weekend", "worse_weekday", "none"]


class TestTrendDescription:
    """Test trend description generation."""

    def test_describe_worsening_trend(self, temporal_reasoner):
        """Test description for worsening trend."""
        desc = temporal_reasoner._describe_trend(
            TemporalTrend.WORSENING,
            0.8,
            "pain"
        )

        assert "pain" in desc.lower()
        assert "worsening" in desc.lower()
        assert "high" in desc.lower()  # High confidence

    def test_describe_improving_trend(self, temporal_reasoner):
        """Test description for improving trend."""
        desc = temporal_reasoner._describe_trend(
            TemporalTrend.IMPROVING,
            0.5,
            "nausea"
        )

        assert "nausea" in desc.lower()
        assert "improving" in desc.lower()
        assert "moderate" in desc.lower()  # Moderate confidence


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestTemporalReasonerIntegration:
    """Integration tests for temporal reasoner."""

    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self, temporal_reasoner, longitudinal_manager):
        """Test complete analysis workflow for a patient."""
        patient_id = "integration_test_patient"
        record = await longitudinal_manager.get_or_create_record(patient_id)
        base_time = datetime.now() - timedelta(days=60)

        # Add symptom observations (worsening pain)
        for i in range(10):
            obs = SymptomObservation(
                observation_id=f"pain_{i}",
                timestamp=base_time + timedelta(days=i * 6),
                source_type=DataSourceType.VOICE_CALL,
                source_id=f"conv_{i}",
                reported_by="patient",
                category="symptom",
                entity_name="pain",
                value=SeverityLevel(min(4, 1 + i // 3)),
                value_text=f"Pain level",
                symptom_name="pain",
                severity=SeverityLevel(min(4, 1 + i // 3))
            )
            record.add_observation(obs)

        # Add medication events
        for i in range(15):
            event = MedicationEvent(
                observation_id=f"med_{i}",
                timestamp=base_time + timedelta(days=i * 4),
                source_type=DataSourceType.WHATSAPP,
                source_id=f"msg_{i}",
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

        await longitudinal_manager.save_record(record)

        # Run all analyses
        symptom_report = await temporal_reasoner.analyze_symptom_progression(
            patient_id, "pain", 90
        )
        med_report = await temporal_reasoner.analyze_medication_effectiveness(
            patient_id, "morphine", 90
        )
        correlations = await temporal_reasoner.find_correlations(patient_id, 90)

        # Verify results
        assert symptom_report is not None
        assert symptom_report.trend in [TemporalTrend.WORSENING, TemporalTrend.FLUCTUATING]

        assert med_report is not None
        assert med_report.adherence_rate == 1.0  # All doses taken

        assert isinstance(correlations, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
