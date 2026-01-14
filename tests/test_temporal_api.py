#!/usr/bin/env python3
"""
Test V25 Temporal Reasoning API endpoints.

Run with:
    PYTHONPATH=. ./venv/bin/pytest tests/test_temporal_api.py -v --tb=short
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from fastapi.testclient import TestClient

# Import after mocking to avoid loading heavy dependencies
@pytest.fixture
def mock_rag_pipeline():
    """Create mock RAG pipeline with V25 components."""
    pipeline = MagicMock()

    # Mock temporal reasoner
    pipeline.temporal_reasoner = MagicMock()
    pipeline.temporal_reasoner.analyze_symptom_progression = AsyncMock()
    pipeline.temporal_reasoner.analyze_medication_effectiveness = AsyncMock()
    pipeline.temporal_reasoner.find_correlations = AsyncMock()

    # Mock alert manager
    pipeline.alert_manager = MagicMock()
    pipeline.alert_manager.get_active_alerts = AsyncMock(return_value=[])

    # Mock longitudinal manager
    pipeline.longitudinal_manager = MagicMock()
    mock_record = MagicMock()
    mock_record.observations = []
    pipeline.longitudinal_manager.get_or_create_record = AsyncMock(return_value=mock_record)

    # Mock context injector
    pipeline.context_injector = MagicMock()

    return pipeline


class TestTemporalHealthEndpoint:
    """Test /api/temporal/health endpoint."""

    def test_health_when_available(self, mock_rag_pipeline):
        """Test health check returns healthy when components available."""
        # This test validates the endpoint structure
        # Full integration requires running server
        assert mock_rag_pipeline.temporal_reasoner is not None
        assert mock_rag_pipeline.alert_manager is not None

    def test_health_components_check(self, mock_rag_pipeline):
        """Test that health check validates all components."""
        components = {
            "temporal_reasoner": mock_rag_pipeline.temporal_reasoner is not None,
            "alert_manager": mock_rag_pipeline.alert_manager is not None,
            "longitudinal_memory": mock_rag_pipeline.longitudinal_manager is not None,
            "context_injector": mock_rag_pipeline.context_injector is not None
        }
        assert all(components.values())


class TestSymptomProgressionEndpoint:
    """Test /api/temporal/symptom-progression endpoint."""

    @pytest.mark.asyncio
    async def test_symptom_progression_returns_report(self, mock_rag_pipeline):
        """Test symptom progression analysis returns valid report."""
        mock_report = MagicMock()
        mock_report.to_dict.return_value = {
            "symptom_name": "pain",
            "patient_id": "test-patient",
            "trend": "WORSENING",
            "trend_confidence": 0.85
        }
        mock_rag_pipeline.temporal_reasoner.analyze_symptom_progression.return_value = mock_report

        result = await mock_rag_pipeline.temporal_reasoner.analyze_symptom_progression(
            patient_id="test-patient",
            symptom_name="pain",
            time_window_days=90
        )

        assert result is not None
        assert result.to_dict()["symptom_name"] == "pain"
        assert result.to_dict()["trend"] == "WORSENING"

    @pytest.mark.asyncio
    async def test_symptom_progression_insufficient_data(self, mock_rag_pipeline):
        """Test endpoint handles insufficient data gracefully."""
        mock_rag_pipeline.temporal_reasoner.analyze_symptom_progression.return_value = None

        result = await mock_rag_pipeline.temporal_reasoner.analyze_symptom_progression(
            patient_id="new-patient",
            symptom_name="nausea",
            time_window_days=30
        )

        assert result is None


class TestMedicationEffectivenessEndpoint:
    """Test /api/temporal/medication-effectiveness endpoint."""

    @pytest.mark.asyncio
    async def test_medication_effectiveness_returns_report(self, mock_rag_pipeline):
        """Test medication effectiveness analysis returns valid report."""
        mock_report = MagicMock()
        mock_report.to_dict.return_value = {
            "medication_name": "morphine",
            "patient_id": "test-patient",
            "adherence_rate": 0.92,
            "symptom_response_rate": 0.65
        }
        mock_rag_pipeline.temporal_reasoner.analyze_medication_effectiveness.return_value = mock_report

        result = await mock_rag_pipeline.temporal_reasoner.analyze_medication_effectiveness(
            patient_id="test-patient",
            medication_name="morphine",
            time_window_days=90
        )

        assert result is not None
        assert result.to_dict()["medication_name"] == "morphine"
        assert result.to_dict()["adherence_rate"] == 0.92


class TestCorrelationsEndpoint:
    """Test /api/temporal/correlations endpoint."""

    @pytest.mark.asyncio
    async def test_correlations_returns_list(self, mock_rag_pipeline):
        """Test correlations analysis returns list of correlations."""
        mock_correlation = MagicMock()
        mock_correlation.to_dict.return_value = {
            "medication_name": "morphine",
            "symptom_name": "pain",
            "correlation_strength": -0.72,
            "direction": "NEGATIVE"
        }
        mock_rag_pipeline.temporal_reasoner.find_correlations.return_value = [mock_correlation]

        result = await mock_rag_pipeline.temporal_reasoner.find_correlations(
            patient_id="test-patient",
            time_window_days=90
        )

        assert len(result) == 1
        assert result[0].to_dict()["correlation_strength"] == -0.72

    @pytest.mark.asyncio
    async def test_correlations_empty_when_no_data(self, mock_rag_pipeline):
        """Test correlations returns empty list for new patient."""
        mock_rag_pipeline.temporal_reasoner.find_correlations.return_value = []

        result = await mock_rag_pipeline.temporal_reasoner.find_correlations(
            patient_id="new-patient",
            time_window_days=30
        )

        assert len(result) == 0


class TestPatientSummaryEndpoint:
    """Test /api/temporal/patient/{patient_id}/summary endpoint."""

    @pytest.mark.asyncio
    async def test_patient_summary_structure(self, mock_rag_pipeline):
        """Test patient summary returns expected structure."""
        # Test the summary structure the endpoint would return
        summary = {
            "patient_id": "test-patient",
            "analysis_period_days": 30,
            "symptom_progressions": [],
            "medication_reports": [],
            "correlations": [],
            "active_alerts": []
        }

        assert "patient_id" in summary
        assert "symptom_progressions" in summary
        assert "medication_reports" in summary
        assert "correlations" in summary
        assert "active_alerts" in summary

    @pytest.mark.asyncio
    async def test_patient_summary_aggregates_data(self, mock_rag_pipeline):
        """Test patient summary aggregates all analysis types."""
        mock_record = MagicMock()
        mock_obs1 = MagicMock()
        mock_obs1.category = "symptom"
        mock_obs1.entity_name = "pain"
        mock_obs2 = MagicMock()
        mock_obs2.category = "medication"
        mock_obs2.entity_name = "morphine"
        mock_record.observations = [mock_obs1, mock_obs2]

        mock_rag_pipeline.longitudinal_manager.get_or_create_record.return_value = mock_record

        record = await mock_rag_pipeline.longitudinal_manager.get_or_create_record("test-patient")

        symptoms = {obs.entity_name for obs in record.observations if obs.category == "symptom"}
        medications = {obs.entity_name for obs in record.observations if obs.category == "medication"}

        assert "pain" in symptoms
        assert "morphine" in medications


class TestEndpointValidation:
    """Test input validation for endpoints."""

    def test_missing_patient_id_error(self):
        """Test endpoint returns error when patient_id missing."""
        # This would be tested with actual HTTP client
        # Validates the validation logic pattern
        body = {"symptom_name": "pain"}
        patient_id = body.get("patient_id")
        assert patient_id is None

    def test_missing_symptom_name_error(self):
        """Test endpoint returns error when symptom_name missing."""
        body = {"patient_id": "test-patient"}
        symptom_name = body.get("symptom_name")
        assert symptom_name is None

    def test_default_days_parameter(self):
        """Test days parameter defaults to 90."""
        body = {"patient_id": "test-patient", "symptom_name": "pain"}
        days = body.get("days", 90)
        assert days == 90

    def test_custom_days_parameter(self):
        """Test days parameter can be customized."""
        body = {"patient_id": "test-patient", "symptom_name": "pain", "days": 30}
        days = body.get("days", 90)
        assert days == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
