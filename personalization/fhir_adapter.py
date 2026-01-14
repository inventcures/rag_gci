"""
FHIR Adapter for V25 Longitudinal Patient Context Memory System.

Provides conversion between internal data models and FHIR R4 resources,
enabling interoperability with Electronic Health Record (EHR) systems.

Supported FHIR Resources:
- Patient: Demographics and identifiers
- Observation: Symptoms, vital signs, assessments
- MedicationStatement: Medication history and adherence
- Condition: Diagnoses and problems
- CareTeam: Care team members and roles

Usage:
    from personalization.fhir_adapter import FHIRAdapter, FHIRBundle

    # Export patient data to FHIR
    adapter = FHIRAdapter()
    bundle = await adapter.export_patient_bundle(patient_id, longitudinal_manager)

    # Import FHIR bundle
    await adapter.import_bundle(bundle_json, longitudinal_manager)
"""
import json
import uuid
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# FHIR Code Systems
# =============================================================================

class FHIRCodeSystem(Enum):
    """Standard FHIR code systems."""
    SNOMED_CT = "http://snomed.info/sct"
    LOINC = "http://loinc.org"
    ICD10 = "http://hl7.org/fhir/sid/icd-10"
    RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"


# Symptom to SNOMED CT mapping (common palliative care symptoms)
SYMPTOM_SNOMED_CODES = {
    "pain": {"code": "22253000", "display": "Pain"},
    "nausea": {"code": "422587007", "display": "Nausea"},
    "vomiting": {"code": "422400008", "display": "Vomiting"},
    "fatigue": {"code": "84229001", "display": "Fatigue"},
    "breathlessness": {"code": "267036007", "display": "Dyspnea"},
    "dyspnea": {"code": "267036007", "display": "Dyspnea"},
    "anxiety": {"code": "48694002", "display": "Anxiety"},
    "depression": {"code": "35489007", "display": "Depression"},
    "constipation": {"code": "14760008", "display": "Constipation"},
    "diarrhea": {"code": "62315008", "display": "Diarrhea"},
    "insomnia": {"code": "193462001", "display": "Insomnia"},
    "appetite_loss": {"code": "79890006", "display": "Loss of appetite"},
    "weakness": {"code": "13791008", "display": "Weakness"},
}

# Severity to FHIR severity codes
SEVERITY_FHIR_CODES = {
    0: {"code": "255604002", "display": "Mild"},
    1: {"code": "6736007", "display": "Moderate"},
    2: {"code": "24484000", "display": "Severe"},
    3: {"code": "442452003", "display": "Life-threatening"},
    4: {"code": "442452003", "display": "Life-threatening"},
}


# =============================================================================
# FHIR Resource Dataclasses
# =============================================================================

@dataclass
class FHIRCoding:
    """FHIR Coding element."""
    system: str
    code: str
    display: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"system": self.system, "code": self.code}
        if self.display:
            result["display"] = self.display
        return result


@dataclass
class FHIRCodeableConcept:
    """FHIR CodeableConcept element."""
    coding: List[FHIRCoding] = field(default_factory=list)
    text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.coding:
            result["coding"] = [c.to_dict() for c in self.coding]
        if self.text:
            result["text"] = self.text
        return result


@dataclass
class FHIRReference:
    """FHIR Reference element."""
    reference: str
    display: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"reference": self.reference}
        if self.display:
            result["display"] = self.display
        return result


@dataclass
class FHIRIdentifier:
    """FHIR Identifier element."""
    system: str
    value: str
    use: str = "usual"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "use": self.use,
            "system": self.system,
            "value": self.value
        }


@dataclass
class FHIRHumanName:
    """FHIR HumanName element."""
    family: Optional[str] = None
    given: List[str] = field(default_factory=list)
    text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.family:
            result["family"] = self.family
        if self.given:
            result["given"] = self.given
        if self.text:
            result["text"] = self.text
        return result


@dataclass
class FHIRContactPoint:
    """FHIR ContactPoint element."""
    system: str  # phone, email, etc.
    value: str
    use: str = "mobile"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system": self.system,
            "value": self.value,
            "use": self.use
        }


# =============================================================================
# FHIR Resources
# =============================================================================

@dataclass
class FHIRPatient:
    """FHIR Patient resource."""
    id: str
    identifier: List[FHIRIdentifier] = field(default_factory=list)
    name: List[FHIRHumanName] = field(default_factory=list)
    telecom: List[FHIRContactPoint] = field(default_factory=list)
    gender: Optional[str] = None
    birthDate: Optional[str] = None
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resourceType": "Patient",
            "id": self.id,
            "identifier": [i.to_dict() for i in self.identifier],
            "name": [n.to_dict() for n in self.name],
            "telecom": [t.to_dict() for t in self.telecom],
            "gender": self.gender,
            "birthDate": self.birthDate,
            "active": self.active
        }


@dataclass
class FHIRObservation:
    """FHIR Observation resource."""
    id: str
    status: str = "final"
    category: List[FHIRCodeableConcept] = field(default_factory=list)
    code: Optional[FHIRCodeableConcept] = None
    subject: Optional[FHIRReference] = None
    effectiveDateTime: Optional[str] = None
    valueCodeableConcept: Optional[FHIRCodeableConcept] = None
    valueString: Optional[str] = None
    valueQuantity: Optional[Dict[str, Any]] = None
    note: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "resourceType": "Observation",
            "id": self.id,
            "status": self.status,
        }
        if self.category:
            result["category"] = [c.to_dict() for c in self.category]
        if self.code:
            result["code"] = self.code.to_dict()
        if self.subject:
            result["subject"] = self.subject.to_dict()
        if self.effectiveDateTime:
            result["effectiveDateTime"] = self.effectiveDateTime
        if self.valueCodeableConcept:
            result["valueCodeableConcept"] = self.valueCodeableConcept.to_dict()
        if self.valueString:
            result["valueString"] = self.valueString
        if self.valueQuantity:
            result["valueQuantity"] = self.valueQuantity
        if self.note:
            result["note"] = self.note
        return result


@dataclass
class FHIRMedicationStatement:
    """FHIR MedicationStatement resource."""
    id: str
    status: str = "active"
    medicationCodeableConcept: Optional[FHIRCodeableConcept] = None
    subject: Optional[FHIRReference] = None
    effectiveDateTime: Optional[str] = None
    effectivePeriod: Optional[Dict[str, str]] = None
    dateAsserted: Optional[str] = None
    dosage: List[Dict[str, Any]] = field(default_factory=list)
    note: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "resourceType": "MedicationStatement",
            "id": self.id,
            "status": self.status,
        }
        if self.medicationCodeableConcept:
            result["medicationCodeableConcept"] = self.medicationCodeableConcept.to_dict()
        if self.subject:
            result["subject"] = self.subject.to_dict()
        if self.effectiveDateTime:
            result["effectiveDateTime"] = self.effectiveDateTime
        if self.effectivePeriod:
            result["effectivePeriod"] = self.effectivePeriod
        if self.dateAsserted:
            result["dateAsserted"] = self.dateAsserted
        if self.dosage:
            result["dosage"] = self.dosage
        if self.note:
            result["note"] = self.note
        return result


@dataclass
class FHIRCondition:
    """FHIR Condition resource."""
    id: str
    clinicalStatus: Optional[FHIRCodeableConcept] = None
    verificationStatus: Optional[FHIRCodeableConcept] = None
    category: List[FHIRCodeableConcept] = field(default_factory=list)
    severity: Optional[FHIRCodeableConcept] = None
    code: Optional[FHIRCodeableConcept] = None
    subject: Optional[FHIRReference] = None
    onsetDateTime: Optional[str] = None
    recordedDate: Optional[str] = None
    note: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "resourceType": "Condition",
            "id": self.id,
        }
        if self.clinicalStatus:
            result["clinicalStatus"] = self.clinicalStatus.to_dict()
        if self.verificationStatus:
            result["verificationStatus"] = self.verificationStatus.to_dict()
        if self.category:
            result["category"] = [c.to_dict() for c in self.category]
        if self.severity:
            result["severity"] = self.severity.to_dict()
        if self.code:
            result["code"] = self.code.to_dict()
        if self.subject:
            result["subject"] = self.subject.to_dict()
        if self.onsetDateTime:
            result["onsetDateTime"] = self.onsetDateTime
        if self.recordedDate:
            result["recordedDate"] = self.recordedDate
        if self.note:
            result["note"] = self.note
        return result


@dataclass
class FHIRCareTeamParticipant:
    """FHIR CareTeam participant."""
    role: List[FHIRCodeableConcept] = field(default_factory=list)
    member: Optional[FHIRReference] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.role:
            result["role"] = [r.to_dict() for r in self.role]
        if self.member:
            result["member"] = self.member.to_dict()
        return result


@dataclass
class FHIRCareTeam:
    """FHIR CareTeam resource."""
    id: str
    status: str = "active"
    name: Optional[str] = None
    subject: Optional[FHIRReference] = None
    participant: List[FHIRCareTeamParticipant] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "resourceType": "CareTeam",
            "id": self.id,
            "status": self.status,
        }
        if self.name:
            result["name"] = self.name
        if self.subject:
            result["subject"] = self.subject.to_dict()
        if self.participant:
            result["participant"] = [p.to_dict() for p in self.participant]
        return result


@dataclass
class FHIRBundleEntry:
    """FHIR Bundle entry."""
    fullUrl: str
    resource: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fullUrl": self.fullUrl,
            "resource": self.resource
        }


@dataclass
class FHIRBundle:
    """FHIR Bundle resource."""
    id: str
    type: str = "collection"
    timestamp: Optional[str] = None
    entry: List[FHIRBundleEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resourceType": "Bundle",
            "id": self.id,
            "type": self.type,
            "timestamp": self.timestamp or datetime.now().isoformat(),
            "entry": [e.to_dict() for e in self.entry]
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert bundle to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# =============================================================================
# FHIR Adapter
# =============================================================================

class FHIRAdapter:
    """
    Adapter for converting between internal data models and FHIR R4 resources.

    Supports:
    - Export: Convert LongitudinalPatientRecord to FHIR Bundle
    - Import: Parse FHIR Bundle and create/update patient records
    - Validation: Validate FHIR resources against R4 schema
    """

    def __init__(self, base_url: str = "urn:uuid"):
        """
        Initialize FHIR adapter.

        Args:
            base_url: Base URL for FHIR resource references
        """
        self.base_url = base_url

    # =========================================================================
    # Export Methods (Internal -> FHIR)
    # =========================================================================

    def observation_to_fhir(
        self,
        observation: Any,
        patient_fhir_id: str
    ) -> FHIRObservation:
        """
        Convert internal observation to FHIR Observation.

        Args:
            observation: Internal TimestampedObservation
            patient_fhir_id: FHIR Patient resource ID

        Returns:
            FHIRObservation resource
        """
        obs_id = str(uuid.uuid4())

        # Determine category
        category_code = "survey"  # Default
        if observation.category == "symptom":
            category_code = "survey"
        elif observation.category == "vital_signs":
            category_code = "vital-signs"

        category = FHIRCodeableConcept(
            coding=[FHIRCoding(
                system="http://terminology.hl7.org/CodeSystem/observation-category",
                code=category_code,
                display=category_code.replace("-", " ").title()
            )]
        )

        # Map symptom to SNOMED code
        entity_name = observation.entity_name.lower()
        snomed_info = SYMPTOM_SNOMED_CODES.get(entity_name, {
            "code": "404684003",
            "display": observation.entity_name
        })

        code = FHIRCodeableConcept(
            coding=[FHIRCoding(
                system=FHIRCodeSystem.SNOMED_CT.value,
                code=snomed_info["code"],
                display=snomed_info["display"]
            )],
            text=observation.entity_name
        )

        # Create value
        value_concept = None
        value_string = None

        if hasattr(observation, 'severity') and observation.severity is not None:
            severity_val = observation.severity.value if hasattr(observation.severity, 'value') else observation.severity
            severity_info = SEVERITY_FHIR_CODES.get(severity_val, {
                "code": "6736007",
                "display": "Moderate"
            })
            value_concept = FHIRCodeableConcept(
                coding=[FHIRCoding(
                    system=FHIRCodeSystem.SNOMED_CT.value,
                    code=severity_info["code"],
                    display=severity_info["display"]
                )],
                text=observation.value_text if hasattr(observation, 'value_text') else None
            )
        elif hasattr(observation, 'value_text'):
            value_string = observation.value_text

        return FHIRObservation(
            id=obs_id,
            status="final",
            category=[category],
            code=code,
            subject=FHIRReference(
                reference=f"Patient/{patient_fhir_id}"
            ),
            effectiveDateTime=observation.timestamp.isoformat() if hasattr(observation.timestamp, 'isoformat') else str(observation.timestamp),
            valueCodeableConcept=value_concept,
            valueString=value_string,
            note=[{"text": f"Source: {observation.source_type.value if hasattr(observation.source_type, 'value') else observation.source_type}"}] if hasattr(observation, 'source_type') else []
        )

    def medication_event_to_fhir(
        self,
        med_event: Any,
        patient_fhir_id: str
    ) -> FHIRMedicationStatement:
        """
        Convert internal MedicationEvent to FHIR MedicationStatement.
        """
        med_id = str(uuid.uuid4())

        # Status mapping
        status = "active"
        if hasattr(med_event, 'action'):
            action = med_event.action.value if hasattr(med_event.action, 'value') else med_event.action
            if action == "stopped":
                status = "stopped"
            elif action == "missed":
                status = "not-taken"

        medication = FHIRCodeableConcept(
            text=med_event.medication_name if hasattr(med_event, 'medication_name') else med_event.entity_name
        )

        dosage = []
        if hasattr(med_event, 'dosage') and med_event.dosage:
            dosage.append({
                "text": med_event.dosage
            })

        return FHIRMedicationStatement(
            id=med_id,
            status=status,
            medicationCodeableConcept=medication,
            subject=FHIRReference(reference=f"Patient/{patient_fhir_id}"),
            effectiveDateTime=med_event.timestamp.isoformat() if hasattr(med_event.timestamp, 'isoformat') else str(med_event.timestamp),
            dosage=dosage
        )

    def care_team_member_to_fhir(
        self,
        member: Any,
        patient_fhir_id: str
    ) -> FHIRCareTeamParticipant:
        """
        Convert internal CareTeamMember to FHIR CareTeam participant.
        """
        # Map role to FHIR care team role codes
        role_mapping = {
            "doctor": {"code": "doctor", "display": "Doctor"},
            "nurse": {"code": "nurse", "display": "Nurse"},
            "asha_worker": {"code": "224535009", "display": "Community Health Worker"},
            "caregiver": {"code": "133932002", "display": "Caregiver"},
            "volunteer": {"code": "volunteer", "display": "Volunteer"},
            "social_worker": {"code": "106328005", "display": "Social Worker"},
        }

        role_info = role_mapping.get(member.role, {"code": member.role, "display": member.role})

        return FHIRCareTeamParticipant(
            role=[FHIRCodeableConcept(
                coding=[FHIRCoding(
                    system=FHIRCodeSystem.SNOMED_CT.value,
                    code=role_info["code"],
                    display=role_info["display"]
                )]
            )],
            member=FHIRReference(
                reference=f"Practitioner/{member.provider_id}",
                display=member.name
            )
        )

    async def export_patient_bundle(
        self,
        patient_id: str,
        longitudinal_manager: Any,
        include_observations: bool = True,
        include_medications: bool = True,
        include_care_team: bool = True
    ) -> FHIRBundle:
        """
        Export patient data as a FHIR Bundle.

        Args:
            patient_id: Internal patient ID
            longitudinal_manager: LongitudinalMemoryManager instance
            include_observations: Include symptom/vital observations
            include_medications: Include medication history
            include_care_team: Include care team

        Returns:
            FHIRBundle containing all patient resources
        """
        record = await longitudinal_manager.get_or_create_record(patient_id)

        bundle_id = str(uuid.uuid4())
        patient_fhir_id = str(uuid.uuid4())
        entries = []

        # Create Patient resource
        patient = FHIRPatient(
            id=patient_fhir_id,
            identifier=[FHIRIdentifier(
                system=f"{self.base_url}/patient-id",
                value=patient_id
            )],
            active=True
        )
        entries.append(FHIRBundleEntry(
            fullUrl=f"{self.base_url}:Patient/{patient_fhir_id}",
            resource=patient.to_dict()
        ))

        # Add observations
        if include_observations:
            for obs in record.observations:
                if obs.category in ["symptom", "vital_signs"]:
                    fhir_obs = self.observation_to_fhir(obs, patient_fhir_id)
                    entries.append(FHIRBundleEntry(
                        fullUrl=f"{self.base_url}:Observation/{fhir_obs.id}",
                        resource=fhir_obs.to_dict()
                    ))

        # Add medications
        if include_medications:
            for obs in record.observations:
                if obs.category == "medication":
                    fhir_med = self.medication_event_to_fhir(obs, patient_fhir_id)
                    entries.append(FHIRBundleEntry(
                        fullUrl=f"{self.base_url}:MedicationStatement/{fhir_med.id}",
                        resource=fhir_med.to_dict()
                    ))

        # Add care team
        if include_care_team and record.care_team:
            care_team_id = str(uuid.uuid4())
            participants = [
                self.care_team_member_to_fhir(member, patient_fhir_id)
                for member in record.care_team
            ]

            care_team = FHIRCareTeam(
                id=care_team_id,
                status="active",
                name=f"Care Team for {patient_id}",
                subject=FHIRReference(reference=f"Patient/{patient_fhir_id}"),
                participant=participants
            )
            entries.append(FHIRBundleEntry(
                fullUrl=f"{self.base_url}:CareTeam/{care_team_id}",
                resource=care_team.to_dict()
            ))

        return FHIRBundle(
            id=bundle_id,
            type="collection",
            timestamp=datetime.now().isoformat(),
            entry=entries
        )

    # =========================================================================
    # Import Methods (FHIR -> Internal)
    # =========================================================================

    async def import_bundle(
        self,
        bundle_json: Union[str, Dict],
        longitudinal_manager: Any
    ) -> Dict[str, Any]:
        """
        Import a FHIR Bundle and create/update patient records.

        Args:
            bundle_json: FHIR Bundle as JSON string or dict
            longitudinal_manager: LongitudinalMemoryManager instance

        Returns:
            Import result with counts and any errors
        """
        if isinstance(bundle_json, str):
            bundle_data = json.loads(bundle_json)
        else:
            bundle_data = bundle_json

        result = {
            "patients_imported": 0,
            "observations_imported": 0,
            "medications_imported": 0,
            "errors": []
        }

        # Find Patient resources first
        patient_map = {}  # FHIR ID -> internal patient_id

        for entry in bundle_data.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Patient":
                fhir_id = resource.get("id")
                # Look for our internal ID in identifiers
                internal_id = None
                for identifier in resource.get("identifier", []):
                    if "patient-id" in identifier.get("system", ""):
                        internal_id = identifier.get("value")
                        break

                if not internal_id:
                    internal_id = fhir_id or str(uuid.uuid4())

                patient_map[fhir_id] = internal_id
                result["patients_imported"] += 1

        # Import observations
        for entry in bundle_data.get("entry", []):
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")

            if resource_type == "Observation":
                try:
                    await self._import_observation(resource, patient_map, longitudinal_manager)
                    result["observations_imported"] += 1
                except Exception as e:
                    result["errors"].append(f"Observation import error: {str(e)}")

            elif resource_type == "MedicationStatement":
                try:
                    await self._import_medication(resource, patient_map, longitudinal_manager)
                    result["medications_imported"] += 1
                except Exception as e:
                    result["errors"].append(f"Medication import error: {str(e)}")

        return result

    async def _import_observation(
        self,
        resource: Dict,
        patient_map: Dict[str, str],
        longitudinal_manager: Any
    ) -> None:
        """Import a single FHIR Observation."""
        from personalization.longitudinal_memory import (
            SymptomObservation,
            DataSourceType,
            SeverityLevel
        )

        # Get patient ID
        subject_ref = resource.get("subject", {}).get("reference", "")
        fhir_patient_id = subject_ref.replace("Patient/", "")
        patient_id = patient_map.get(fhir_patient_id, fhir_patient_id)

        # Parse observation
        code = resource.get("code", {})
        code_text = code.get("text", "")
        for coding in code.get("coding", []):
            if not code_text:
                code_text = coding.get("display", coding.get("code", "unknown"))

        # Parse severity from valueCodeableConcept
        severity = SeverityLevel.MODERATE
        value_text = ""
        if "valueCodeableConcept" in resource:
            value_text = resource["valueCodeableConcept"].get("text", "")
            for coding in resource["valueCodeableConcept"].get("coding", []):
                display = coding.get("display", "").lower()
                if "mild" in display:
                    severity = SeverityLevel.MILD
                elif "severe" in display or "life" in display:
                    severity = SeverityLevel.SEVERE
                elif "moderate" in display:
                    severity = SeverityLevel.MODERATE
        elif "valueString" in resource:
            value_text = resource["valueString"]

        # Parse timestamp
        timestamp = datetime.now()
        if "effectiveDateTime" in resource:
            try:
                timestamp = datetime.fromisoformat(resource["effectiveDateTime"].replace("Z", "+00:00"))
            except:
                pass

        obs = SymptomObservation(
            observation_id=resource.get("id", str(uuid.uuid4())),
            timestamp=timestamp,
            source_type=DataSourceType.FHIR_IMPORT,
            source_id=f"fhir:{resource.get('id', 'unknown')}",
            reported_by="fhir_import",
            category="symptom",
            entity_name=code_text,
            value=severity,
            value_text=value_text,
            symptom_name=code_text,
            severity=severity
        )

        await longitudinal_manager.add_observation(patient_id, obs)

    async def _import_medication(
        self,
        resource: Dict,
        patient_map: Dict[str, str],
        longitudinal_manager: Any
    ) -> None:
        """Import a single FHIR MedicationStatement."""
        from personalization.longitudinal_memory import (
            MedicationEvent,
            MedicationAction,
            DataSourceType
        )

        # Get patient ID
        subject_ref = resource.get("subject", {}).get("reference", "")
        fhir_patient_id = subject_ref.replace("Patient/", "")
        patient_id = patient_map.get(fhir_patient_id, fhir_patient_id)

        # Parse medication name
        med_concept = resource.get("medicationCodeableConcept", {})
        med_name = med_concept.get("text", "")
        for coding in med_concept.get("coding", []):
            if not med_name:
                med_name = coding.get("display", coding.get("code", "unknown"))

        # Parse status to event type
        status = resource.get("status", "active")
        event_type = MedicationAction.TAKEN
        if status == "stopped":
            event_type = MedicationAction.STOPPED
        elif status == "not-taken":
            event_type = MedicationAction.MISSED

        # Parse timestamp
        timestamp = datetime.now()
        if "effectiveDateTime" in resource:
            try:
                timestamp = datetime.fromisoformat(resource["effectiveDateTime"].replace("Z", "+00:00"))
            except:
                pass

        # Parse dosage
        dosage = ""
        for dose in resource.get("dosage", []):
            dosage = dose.get("text", "")
            break

        event = MedicationEvent(
            observation_id=resource.get("id", str(uuid.uuid4())),
            timestamp=timestamp,
            source_type=DataSourceType.FHIR_IMPORT,
            source_id=f"fhir:{resource.get('id', 'unknown')}",
            reported_by="fhir_import",
            category="medication",
            entity_name=med_name,
            value=event_type,
            value_text=f"{med_name} {dosage}".strip(),
            medication_name=med_name,
            action=event_type,
            dosage=dosage
        )

        await longitudinal_manager.add_observation(patient_id, event)

    # =========================================================================
    # Validation Methods
    # =========================================================================

    def validate_resource(self, resource: Dict) -> List[str]:
        """
        Validate a FHIR resource.

        Args:
            resource: FHIR resource as dict

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        resource_type = resource.get("resourceType")
        if not resource_type:
            errors.append("Missing resourceType")
            return errors

        # Validate required fields based on resource type
        if resource_type == "Patient":
            if not resource.get("id"):
                errors.append("Patient: Missing id")

        elif resource_type == "Observation":
            if not resource.get("id"):
                errors.append("Observation: Missing id")
            if not resource.get("status"):
                errors.append("Observation: Missing status")
            if not resource.get("code"):
                errors.append("Observation: Missing code")

        elif resource_type == "MedicationStatement":
            if not resource.get("id"):
                errors.append("MedicationStatement: Missing id")
            if not resource.get("status"):
                errors.append("MedicationStatement: Missing status")

        elif resource_type == "Bundle":
            if not resource.get("type"):
                errors.append("Bundle: Missing type")
            for i, entry in enumerate(resource.get("entry", [])):
                if not entry.get("resource"):
                    errors.append(f"Bundle entry {i}: Missing resource")

        return errors

    def validate_bundle(self, bundle: Union[str, Dict]) -> Dict[str, Any]:
        """
        Validate a FHIR Bundle and all its resources.

        Args:
            bundle: FHIR Bundle as JSON string or dict

        Returns:
            Validation result with errors by resource
        """
        if isinstance(bundle, str):
            try:
                bundle_data = json.loads(bundle)
            except json.JSONDecodeError as e:
                return {"valid": False, "errors": [f"Invalid JSON: {str(e)}"]}
        else:
            bundle_data = bundle

        result = {
            "valid": True,
            "errors": [],
            "resources_validated": 0,
            "resources_with_errors": 0
        }

        # Validate bundle itself
        bundle_errors = self.validate_resource(bundle_data)
        if bundle_errors:
            result["valid"] = False
            result["errors"].extend(bundle_errors)

        # Validate each entry
        for entry in bundle_data.get("entry", []):
            resource = entry.get("resource", {})
            resource_errors = self.validate_resource(resource)
            result["resources_validated"] += 1

            if resource_errors:
                result["valid"] = False
                result["resources_with_errors"] += 1
                result["errors"].extend(resource_errors)

        return result


# =============================================================================
# Utility Functions
# =============================================================================

def export_to_file(bundle: FHIRBundle, file_path: str) -> None:
    """Export FHIR Bundle to JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(bundle.to_json())
    logger.info(f"Exported FHIR Bundle to {file_path}")


def import_from_file(file_path: str) -> Dict[str, Any]:
    """Import FHIR Bundle from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)
