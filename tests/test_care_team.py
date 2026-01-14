#!/usr/bin/env python3
"""
Test V25 Care Team Coordination features.

Run with:
    PYTHONPATH=. ./venv/bin/pytest tests/test_care_team.py -v --tb=short
"""
import pytest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

from personalization.longitudinal_memory import (
    CareTeamMember,
    LongitudinalPatientRecord,
    LongitudinalMemoryManager,
)


class TestCareTeamMember:
    """Test CareTeamMember dataclass."""

    def test_create_care_team_member(self):
        """Test creating a care team member."""
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
            attributed_observations=["obs_1", "obs_2"]
        )

        assert member.provider_id == "dr_sharma"
        assert member.name == "Dr. Sharma"
        assert member.role == "doctor"
        assert member.primary_contact is True
        assert member.total_interactions == 5

    def test_care_team_member_to_dict(self):
        """Test serializing care team member to dict."""
        member = CareTeamMember(
            provider_id="nurse_priya",
            name="Priya",
            role="nurse",
            organization=None,
            phone_number=None,
            primary_contact=False,
            first_contact=datetime.now(),
            last_contact=datetime.now(),
            total_interactions=0,
            attributed_observations=[]
        )

        data = member.to_dict()

        assert data["provider_id"] == "nurse_priya"
        assert data["name"] == "Priya"
        assert data["role"] == "nurse"
        assert data["primary_contact"] is False

    def test_care_team_member_from_dict(self):
        """Test deserializing care team member from dict."""
        data = {
            "provider_id": "asha_lakshmi",
            "name": "Lakshmi",
            "role": "asha_worker",
            "organization": "PHC Anantapur",
            "phone_number": "+919123456789",
            "primary_contact": False,
            "first_contact": datetime.now().isoformat(),
            "last_contact": datetime.now().isoformat(),
            "total_interactions": 10,
            "attributed_observations": ["obs_a", "obs_b"]
        }

        member = CareTeamMember.from_dict(data)

        assert member.provider_id == "asha_lakshmi"
        assert member.role == "asha_worker"
        assert member.total_interactions == 10
        assert len(member.attributed_observations) == 2


class TestPatientRecordCareTeam:
    """Test care team functionality in LongitudinalPatientRecord."""

    def test_add_care_team_member(self):
        """Test adding a care team member to patient record."""
        record = LongitudinalPatientRecord(patient_id="test-patient")

        member = CareTeamMember(
            provider_id="dr_test",
            name="Dr. Test",
            role="doctor",
            organization="Test Hospital",
            phone_number=None,
            primary_contact=True,
            first_contact=datetime.now(),
            last_contact=datetime.now(),
            total_interactions=0,
            attributed_observations=[]
        )

        record.add_care_team_member(member)

        assert len(record.care_team) == 1
        assert record.care_team[0].provider_id == "dr_test"

    def test_add_duplicate_member_updates(self):
        """Test that adding duplicate member updates existing entry."""
        record = LongitudinalPatientRecord(patient_id="test-patient")

        member1 = CareTeamMember(
            provider_id="dr_test",
            name="Dr. Test",
            role="doctor",
            organization="Hospital A",
            phone_number=None,
            primary_contact=False,
            first_contact=datetime.now(),
            last_contact=datetime.now(),
            total_interactions=1,
            attributed_observations=[]
        )

        member2 = CareTeamMember(
            provider_id="dr_test",  # Same provider_id
            name="Dr. Test Updated",
            role="doctor",
            organization="Hospital B",  # Updated org
            phone_number="+911234567890",
            primary_contact=True,  # Now primary
            first_contact=datetime.now(),
            last_contact=datetime.now(),
            total_interactions=5,
            attributed_observations=["obs_1"]
        )

        record.add_care_team_member(member1)
        record.add_care_team_member(member2)

        # Should still have only 1 member (updated)
        assert len(record.care_team) == 1
        assert record.care_team[0].organization == "Hospital B"
        assert record.care_team[0].primary_contact is True

    def test_get_primary_contact(self):
        """Test getting primary contact from care team."""
        record = LongitudinalPatientRecord(patient_id="test-patient")

        # Add non-primary member
        member1 = CareTeamMember(
            provider_id="nurse_1",
            name="Nurse One",
            role="nurse",
            organization=None,
            phone_number=None,
            primary_contact=False,
            first_contact=datetime.now(),
            last_contact=datetime.now(),
            total_interactions=0,
            attributed_observations=[]
        )

        # Add primary member
        member2 = CareTeamMember(
            provider_id="dr_primary",
            name="Dr. Primary",
            role="doctor",
            organization=None,
            phone_number=None,
            primary_contact=True,
            first_contact=datetime.now(),
            last_contact=datetime.now(),
            total_interactions=0,
            attributed_observations=[]
        )

        record.add_care_team_member(member1)
        record.add_care_team_member(member2)

        primary = record.get_primary_contact()

        assert primary is not None
        assert primary.provider_id == "dr_primary"

    def test_get_primary_contact_fallback(self):
        """Test primary contact fallback to most recent contact."""
        record = LongitudinalPatientRecord(patient_id="test-patient")

        # Add members without primary flag
        from datetime import timedelta

        member1 = CareTeamMember(
            provider_id="old_contact",
            name="Old Contact",
            role="nurse",
            organization=None,
            phone_number=None,
            primary_contact=False,
            first_contact=datetime.now() - timedelta(days=30),
            last_contact=datetime.now() - timedelta(days=10),
            total_interactions=5,
            attributed_observations=[]
        )

        member2 = CareTeamMember(
            provider_id="recent_contact",
            name="Recent Contact",
            role="asha_worker",
            organization=None,
            phone_number=None,
            primary_contact=False,
            first_contact=datetime.now() - timedelta(days=5),
            last_contact=datetime.now() - timedelta(days=1),  # More recent
            total_interactions=2,
            attributed_observations=[]
        )

        record.add_care_team_member(member1)
        record.add_care_team_member(member2)

        primary = record.get_primary_contact()

        # Should return most recent contact
        assert primary is not None
        assert primary.provider_id == "recent_contact"


class TestCareTeamPersistence:
    """Test care team persistence with LongitudinalMemoryManager."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_save_and_load_care_team(self, temp_storage):
        """Test saving and loading care team."""
        manager = LongitudinalMemoryManager(storage_path=temp_storage)

        # Create a member
        member = CareTeamMember(
            provider_id="dr_persist",
            name="Dr. Persist",
            role="doctor",
            organization="Test Hospital",
            phone_number="+919999999999",
            primary_contact=True,
            first_contact=datetime.now(),
            last_contact=datetime.now(),
            total_interactions=3,
            attributed_observations=["obs_1"]
        )

        # Add to patient
        await manager.add_care_team_member("patient-persist", member)

        # Load and verify
        care_team = await manager.get_care_team("patient-persist")

        assert len(care_team) == 1
        assert care_team[0].provider_id == "dr_persist"
        assert care_team[0].primary_contact is True

    @pytest.mark.asyncio
    async def test_multiple_care_team_members(self, temp_storage):
        """Test adding multiple care team members."""
        manager = LongitudinalMemoryManager(storage_path=temp_storage)

        members = [
            CareTeamMember(
                provider_id="dr_1",
                name="Doctor One",
                role="doctor",
                organization="Hospital A",
                phone_number=None,
                primary_contact=True,
                first_contact=datetime.now(),
                last_contact=datetime.now(),
                total_interactions=0,
                attributed_observations=[]
            ),
            CareTeamMember(
                provider_id="nurse_1",
                name="Nurse One",
                role="nurse",
                organization="Hospital A",
                phone_number=None,
                primary_contact=False,
                first_contact=datetime.now(),
                last_contact=datetime.now(),
                total_interactions=0,
                attributed_observations=[]
            ),
            CareTeamMember(
                provider_id="asha_1",
                name="ASHA Worker",
                role="asha_worker",
                organization="PHC",
                phone_number=None,
                primary_contact=False,
                first_contact=datetime.now(),
                last_contact=datetime.now(),
                total_interactions=0,
                attributed_observations=[]
            ),
        ]

        for member in members:
            await manager.add_care_team_member("patient-multi", member)

        care_team = await manager.get_care_team("patient-multi")

        assert len(care_team) == 3
        roles = {m.role for m in care_team}
        assert "doctor" in roles
        assert "nurse" in roles
        assert "asha_worker" in roles


class TestCareTeamRoles:
    """Test different care team roles."""

    def test_valid_roles(self):
        """Test all valid care team roles."""
        valid_roles = ["doctor", "nurse", "asha_worker", "caregiver", "volunteer", "social_worker"]

        for role in valid_roles:
            member = CareTeamMember(
                provider_id=f"{role}_1",
                name=f"Test {role}",
                role=role,
                organization=None,
                phone_number=None,
                primary_contact=False,
                first_contact=datetime.now(),
                last_contact=datetime.now(),
                total_interactions=0,
                attributed_observations=[]
            )
            assert member.role == role


class TestProviderAttribution:
    """Test observation attribution to providers."""

    def test_attribute_observation(self):
        """Test attributing observation to care team member."""
        record = LongitudinalPatientRecord(patient_id="test-patient")

        # Add care team member
        member = CareTeamMember(
            provider_id="dr_attr",
            name="Dr. Attribution",
            role="doctor",
            organization=None,
            phone_number=None,
            primary_contact=True,
            first_contact=datetime.now(),
            last_contact=datetime.now(),
            total_interactions=0,
            attributed_observations=[]
        )
        record.add_care_team_member(member)

        # Add an observation
        from personalization.longitudinal_memory import SymptomObservation, SeverityLevel, DataSourceType

        obs = SymptomObservation(
            observation_id="obs_test",
            timestamp=datetime.now(),
            source_type=DataSourceType.VOICE_CALL,
            source_id="call_123",
            reported_by="patient",
            category="symptom",
            entity_name="pain",
            value=SeverityLevel.MODERATE,
            value_text="moderate pain",
            symptom_name="pain",
            severity=SeverityLevel.MODERATE
        )
        record.observations.append(obs)

        # Attribute observation to provider
        for o in record.observations:
            if o.observation_id == "obs_test":
                o.reported_by = "dr_attr"

        # Update provider stats
        for m in record.care_team:
            if m.provider_id == "dr_attr":
                m.total_interactions += 1
                m.attributed_observations.append("obs_test")

        # Verify
        assert record.observations[0].reported_by == "dr_attr"
        assert "obs_test" in record.care_team[0].attributed_observations


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
