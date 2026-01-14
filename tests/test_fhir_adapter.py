#!/usr/bin/env python3
"""
Test V25 FHIR Adapter for interoperability with EHR systems.

Run with:
    PYTHONPATH=. ./venv/bin/pytest tests/test_fhir_adapter.py -v --tb=short
"""
import pytest
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

from personalization.fhir_adapter import (
    FHIRAdapter,
    FHIRBundle,
    FHIRPatient,
    FHIRObservation,
    FHIRMedicationStatement,
    FHIRCareTeam,
    FHIRCodeableConcept,
    FHIRCoding,
    FHIRReference,
    FHIRIdentifier,
    export_to_file,
    import_from_file,
)
from personalization.longitudinal_memory import (
    LongitudinalMemoryManager,
    LongitudinalPatientRecord,
    SymptomObservation,
    MedicationEvent,
    CareTeamMember,
    DataSourceType,
    SeverityLevel,
    MedicationAction,
)


class TestFHIRDataClasses:
    """Test FHIR data class serialization."""

    def test_fhir_coding(self):
        """Test FHIRCoding serialization."""
        coding = FHIRCoding(
            system="http://snomed.info/sct",
            code="22253000",
            display="Pain"
        )
        data = coding.to_dict()

        assert data["system"] == "http://snomed.info/sct"
        assert data["code"] == "22253000"
        assert data["display"] == "Pain"

    def test_fhir_codeable_concept(self):
        """Test FHIRCodeableConcept serialization."""
        concept = FHIRCodeableConcept(
            coding=[FHIRCoding(
                system="http://snomed.info/sct",
                code="22253000",
                display="Pain"
            )],
            text="Moderate pain"
        )
        data = concept.to_dict()

        assert len(data["coding"]) == 1
        assert data["text"] == "Moderate pain"

    def test_fhir_patient(self):
        """Test FHIRPatient serialization."""
        patient = FHIRPatient(
            id="patient-123",
            identifier=[FHIRIdentifier(
                system="urn:uuid/patient-id",
                value="internal-123"
            )],
            active=True
        )
        data = patient.to_dict()

        assert data["resourceType"] == "Patient"
        assert data["id"] == "patient-123"
        assert data["active"] is True

    def test_fhir_observation(self):
        """Test FHIRObservation serialization."""
        obs = FHIRObservation(
            id="obs-123",
            status="final",
            code=FHIRCodeableConcept(
                coding=[FHIRCoding(
                    system="http://snomed.info/sct",
                    code="22253000",
                    display="Pain"
                )]
            ),
            subject=FHIRReference(reference="Patient/patient-123"),
            effectiveDateTime="2024-01-15T10:30:00Z"
        )
        data = obs.to_dict()

        assert data["resourceType"] == "Observation"
        assert data["status"] == "final"
        assert "code" in data
        assert data["subject"]["reference"] == "Patient/patient-123"

    def test_fhir_medication_statement(self):
        """Test FHIRMedicationStatement serialization."""
        med = FHIRMedicationStatement(
            id="med-123",
            status="active",
            medicationCodeableConcept=FHIRCodeableConcept(
                text="Morphine 10mg"
            ),
            subject=FHIRReference(reference="Patient/patient-123"),
            dosage=[{"text": "10mg twice daily"}]
        )
        data = med.to_dict()

        assert data["resourceType"] == "MedicationStatement"
        assert data["status"] == "active"
        assert data["medicationCodeableConcept"]["text"] == "Morphine 10mg"


class TestFHIRBundle:
    """Test FHIR Bundle operations."""

    def test_create_bundle(self):
        """Test creating a FHIR bundle."""
        from personalization.fhir_adapter import FHIRBundleEntry

        bundle = FHIRBundle(
            id="bundle-123",
            type="collection",
            entry=[
                FHIRBundleEntry(
                    fullUrl="urn:uuid:patient-123",
                    resource={"resourceType": "Patient", "id": "patient-123"}
                )
            ]
        )
        data = bundle.to_dict()

        assert data["resourceType"] == "Bundle"
        assert data["type"] == "collection"
        assert len(data["entry"]) == 1

    def test_bundle_to_json(self):
        """Test bundle JSON serialization."""
        bundle = FHIRBundle(id="bundle-123", type="collection")
        json_str = bundle.to_json()

        parsed = json.loads(json_str)
        assert parsed["resourceType"] == "Bundle"
        assert parsed["id"] == "bundle-123"


class TestFHIRAdapterExport:
    """Test FHIR export functionality."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def adapter(self):
        """Create FHIR adapter."""
        return FHIRAdapter()

    def test_observation_to_fhir(self, adapter):
        """Test converting internal observation to FHIR."""
        obs = SymptomObservation(
            observation_id="obs-123",
            timestamp=datetime.now(),
            source_type=DataSourceType.VOICE_CALL,
            source_id="call-456",
            reported_by="patient",
            category="symptom",
            entity_name="pain",
            value=SeverityLevel.MODERATE,
            value_text="moderate pain in lower back",
            symptom_name="pain",
            severity=SeverityLevel.MODERATE
        )

        fhir_obs = adapter.observation_to_fhir(obs, "patient-fhir-id")

        assert fhir_obs.status == "final"
        assert fhir_obs.subject.reference == "Patient/patient-fhir-id"
        assert fhir_obs.code is not None

    def test_medication_to_fhir(self, adapter):
        """Test converting medication event to FHIR."""
        med = MedicationEvent(
            observation_id="med-123",
            timestamp=datetime.now(),
            source_type=DataSourceType.WHATSAPP,
            source_id="msg-789",
            reported_by="patient",
            category="medication",
            entity_name="morphine",
            value=MedicationAction.TAKEN,
            value_text="took morphine 10mg",
            medication_name="morphine",
            action=MedicationAction.TAKEN,
            dosage="10mg"
        )

        fhir_med = adapter.medication_event_to_fhir(med, "patient-fhir-id")

        assert fhir_med.status == "active"
        assert fhir_med.subject.reference == "Patient/patient-fhir-id"

    def test_care_team_to_fhir(self, adapter):
        """Test converting care team member to FHIR."""
        member = CareTeamMember(
            provider_id="dr_sharma",
            name="Dr. Sharma",
            role="doctor",
            organization="City Hospital",
            phone_number="+919876543210",
            primary_contact=True,
            first_contact=datetime.now(),
            last_contact=datetime.now(),
            total_interactions=5,
            attributed_observations=[]
        )

        participant = adapter.care_team_member_to_fhir(member, "patient-fhir-id")

        assert participant.member.reference == "Practitioner/dr_sharma"
        assert participant.member.display == "Dr. Sharma"

    @pytest.mark.asyncio
    async def test_export_patient_bundle(self, adapter, temp_storage):
        """Test exporting complete patient bundle."""
        manager = LongitudinalMemoryManager(storage_path=temp_storage)

        # Add patient with observations
        obs = SymptomObservation(
            observation_id="obs-test",
            timestamp=datetime.now(),
            source_type=DataSourceType.VOICE_CALL,
            source_id="call-test",
            reported_by="patient",
            category="symptom",
            entity_name="pain",
            value=SeverityLevel.MODERATE,
            value_text="moderate pain",
            symptom_name="pain",
            severity=SeverityLevel.MODERATE
        )
        await manager.add_observation("patient-export", obs)

        # Add care team member
        member = CareTeamMember(
            provider_id="dr_test",
            name="Dr. Test",
            role="doctor",
            organization=None,
            phone_number=None,
            primary_contact=True,
            first_contact=datetime.now(),
            last_contact=datetime.now(),
            total_interactions=0,
            attributed_observations=[]
        )
        await manager.add_care_team_member("patient-export", member)

        # Export bundle
        bundle = await adapter.export_patient_bundle("patient-export", manager)

        assert bundle.type == "collection"
        assert len(bundle.entry) >= 2  # At least Patient + Observation

        # Verify resource types
        resource_types = [e.resource["resourceType"] for e in bundle.entry]
        assert "Patient" in resource_types
        assert "Observation" in resource_types


class TestFHIRAdapterImport:
    """Test FHIR import functionality."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def adapter(self):
        """Create FHIR adapter."""
        return FHIRAdapter()

    @pytest.fixture
    def sample_bundle(self):
        """Create sample FHIR bundle for import."""
        return {
            "resourceType": "Bundle",
            "id": "test-bundle",
            "type": "collection",
            "entry": [
                {
                    "fullUrl": "urn:uuid:patient-1",
                    "resource": {
                        "resourceType": "Patient",
                        "id": "patient-1",
                        "identifier": [
                            {"system": "urn:uuid/patient-id", "value": "import-patient"}
                        ]
                    }
                },
                {
                    "fullUrl": "urn:uuid:obs-1",
                    "resource": {
                        "resourceType": "Observation",
                        "id": "obs-1",
                        "status": "final",
                        "code": {
                            "coding": [
                                {"system": "http://snomed.info/sct", "code": "22253000", "display": "Pain"}
                            ],
                            "text": "pain"
                        },
                        "subject": {"reference": "Patient/patient-1"},
                        "effectiveDateTime": "2024-01-15T10:30:00Z",
                        "valueCodeableConcept": {
                            "coding": [
                                {"system": "http://snomed.info/sct", "code": "6736007", "display": "Moderate"}
                            ],
                            "text": "moderate pain"
                        }
                    }
                },
                {
                    "fullUrl": "urn:uuid:med-1",
                    "resource": {
                        "resourceType": "MedicationStatement",
                        "id": "med-1",
                        "status": "active",
                        "medicationCodeableConcept": {"text": "Morphine 10mg"},
                        "subject": {"reference": "Patient/patient-1"},
                        "effectiveDateTime": "2024-01-15T08:00:00Z",
                        "dosage": [{"text": "10mg twice daily"}]
                    }
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_import_bundle(self, adapter, temp_storage, sample_bundle):
        """Test importing a FHIR bundle."""
        manager = LongitudinalMemoryManager(storage_path=temp_storage)

        result = await adapter.import_bundle(sample_bundle, manager)

        assert result["patients_imported"] == 1
        assert result["observations_imported"] == 1
        assert result["medications_imported"] == 1
        assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_import_creates_patient_record(self, adapter, temp_storage, sample_bundle):
        """Test that import creates patient record."""
        manager = LongitudinalMemoryManager(storage_path=temp_storage)

        await adapter.import_bundle(sample_bundle, manager)

        # Verify patient record exists
        record = await manager.get_or_create_record("import-patient")
        assert len(record.observations) >= 1

    @pytest.mark.asyncio
    async def test_import_json_string(self, adapter, temp_storage, sample_bundle):
        """Test importing from JSON string."""
        manager = LongitudinalMemoryManager(storage_path=temp_storage)
        json_str = json.dumps(sample_bundle)

        result = await adapter.import_bundle(json_str, manager)

        assert result["patients_imported"] == 1


class TestFHIRValidation:
    """Test FHIR resource validation."""

    @pytest.fixture
    def adapter(self):
        """Create FHIR adapter."""
        return FHIRAdapter()

    def test_validate_valid_patient(self, adapter):
        """Test validating a valid Patient resource."""
        resource = {
            "resourceType": "Patient",
            "id": "patient-123"
        }
        errors = adapter.validate_resource(resource)
        assert len(errors) == 0

    def test_validate_missing_resource_type(self, adapter):
        """Test validation catches missing resourceType."""
        resource = {"id": "test-123"}
        errors = adapter.validate_resource(resource)
        assert "Missing resourceType" in errors

    def test_validate_observation_missing_status(self, adapter):
        """Test validation catches missing Observation status."""
        resource = {
            "resourceType": "Observation",
            "id": "obs-123",
            "code": {"text": "test"}
        }
        errors = adapter.validate_resource(resource)
        assert any("Missing status" in e for e in errors)

    def test_validate_bundle(self, adapter):
        """Test validating a complete bundle."""
        bundle = {
            "resourceType": "Bundle",
            "id": "bundle-123",
            "type": "collection",
            "entry": [
                {
                    "fullUrl": "urn:uuid:patient-1",
                    "resource": {
                        "resourceType": "Patient",
                        "id": "patient-1"
                    }
                }
            ]
        }
        result = adapter.validate_bundle(bundle)

        assert result["valid"] is True
        assert result["resources_validated"] == 1

    def test_validate_invalid_json(self, adapter):
        """Test validation handles invalid JSON."""
        result = adapter.validate_bundle("not valid json {")
        assert result["valid"] is False
        assert any("Invalid JSON" in e for e in result["errors"])


class TestFHIRFileOperations:
    """Test FHIR file import/export."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_export_to_file(self, temp_dir):
        """Test exporting bundle to file."""
        bundle = FHIRBundle(id="file-test", type="collection")
        file_path = f"{temp_dir}/test_bundle.json"

        export_to_file(bundle, file_path)

        assert Path(file_path).exists()

        # Verify content
        with open(file_path) as f:
            data = json.load(f)
        assert data["resourceType"] == "Bundle"

    def test_import_from_file(self, temp_dir):
        """Test importing bundle from file."""
        # Create test file
        bundle_data = {
            "resourceType": "Bundle",
            "id": "import-test",
            "type": "collection",
            "entry": []
        }
        file_path = f"{temp_dir}/import_bundle.json"
        with open(file_path, "w") as f:
            json.dump(bundle_data, f)

        # Import
        imported = import_from_file(file_path)

        assert imported["resourceType"] == "Bundle"
        assert imported["id"] == "import-test"


class TestSNOMEDCodeMapping:
    """Test SNOMED CT code mapping for symptoms."""

    def test_pain_maps_correctly(self):
        """Test pain symptom maps to correct SNOMED code."""
        from personalization.fhir_adapter import SYMPTOM_SNOMED_CODES

        assert "pain" in SYMPTOM_SNOMED_CODES
        assert SYMPTOM_SNOMED_CODES["pain"]["code"] == "22253000"

    def test_common_symptoms_mapped(self):
        """Test common palliative care symptoms are mapped."""
        from personalization.fhir_adapter import SYMPTOM_SNOMED_CODES

        expected_symptoms = ["pain", "nausea", "fatigue", "breathlessness", "anxiety"]
        for symptom in expected_symptoms:
            assert symptom in SYMPTOM_SNOMED_CODES, f"Missing SNOMED mapping for {symptom}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
