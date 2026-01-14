"""
Longitudinal Patient Memory

Maintains patient records spanning 1-5 years with:
- Time-stamped observations from multiple sources
- Temporal trend analysis and progression tracking
- Care team coordination and provider attribution
- Proactive monitoring and alert generation
- Cross-modal data aggregation (voice, WhatsApp, documents)

Inspired by MedAgentBenchV2's longitudinal patient record architecture,
adapted for compassionate palliative care.
"""

import json
import logging
import hashlib
from typing import Optional, List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum
import asyncio
import aiofiles
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class DataSourceType(Enum):
    """Source of patient data."""
    VOICE_CALL = "voice_call"
    WHATSAPP = "whatsapp"
    UPLOADED_DOCUMENT = "uploaded_document"
    CAREGIVER_REPORT = "caregiver_report"
    CLINICAL_ENTRY = "clinical_entry"  # Future: from EHR/FHIR
    PATIENT_REPORTED = "patient_reported"
    WEB_CHAT = "web_chat"


class SeverityLevel(Enum):
    """Standardized severity levels for palliative care."""
    NONE = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3
    VERY_SEVERE = 4

    @classmethod
    def from_string(cls, value: str) -> "SeverityLevel":
        """Convert string to SeverityLevel."""
        value_map = {
            "none": cls.NONE,
            "mild": cls.MILD,
            "moderate": cls.MODERATE,
            "severe": cls.SEVERE,
            "very_severe": cls.VERY_SEVERE,
            "0": cls.NONE,
            "1": cls.MILD,
            "2": cls.MODERATE,
            "3": cls.SEVERE,
            "4": cls.VERY_SEVERE,
        }
        return value_map.get(str(value).lower(), cls.MODERATE)


class TemporalTrend(Enum):
    """Trend direction for time-series data."""
    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"
    FLUCTUATING = "fluctuating"
    UNKNOWN = "unknown"


class AlertPriority(Enum):
    """Alert severity for proactive monitoring."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class MedicationAction(Enum):
    """Medication action types."""
    STARTED = "started"
    STOPPED = "stopped"
    DOSE_CHANGED = "dose_changed"
    TAKEN = "taken"
    MISSED = "missed"
    SIDE_EFFECT = "side_effect"


# ============================================================================
# BASE OBSERVATION CLASS
# ============================================================================

@dataclass
class TimestampedObservation:
    """
    A single observation at a point in time.

    This is the core primitive for longitudinal tracking - all patient data
    (symptoms, medications, vital signs) is stored as timestamped observations.
    """
    observation_id: str
    timestamp: datetime
    source_type: DataSourceType
    source_id: str  # Reference to conversation_id, document_id, etc.
    reported_by: str  # "patient", "caregiver", "system", "provider", "doctor"

    # Observation content
    category: str  # "symptom", "medication", "vital_sign", "functional_status", "emotional"
    entity_name: str  # "pain", "morphine", "blood_pressure", "anxiety"
    value: Any  # SeverityLevel, numeric value, or text
    value_text: str  # Human-readable description
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "observation_id": self.observation_id,
            "timestamp": self.timestamp.isoformat(),
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "reported_by": self.reported_by,
            "category": self.category,
            "entity_name": self.entity_name,
            "value": str(self.value) if not isinstance(self.value, Enum) else self.value.value,
            "value_text": self.value_text,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimestampedObservation":
        """Deserialize from dictionary."""
        # Convert source_type string to enum
        source_type = DataSourceType(data.get("source_type", "patient_reported"))

        # Convert value if it's a SeverityLevel
        value = data.get("value")
        if isinstance(value, dict) and "value" in value:
            # Handle nested enum serialization
            value = value["value"]
        try:
            value_int = int(value)
            if 0 <= value_int <= 4:
                value = SeverityLevel(value_int)
        except (ValueError, TypeError):
            pass

        return cls(
            observation_id=data["observation_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source_type=source_type,
            source_id=data["source_id"],
            reported_by=data["reported_by"],
            category=data["category"],
            entity_name=data["entity_name"],
            value=value,
            value_text=data["value_text"],
            metadata=data.get("metadata", {})
        )


# ============================================================================
# SPECIALIZED OBSERVATION TYPES
# ============================================================================

@dataclass
class SymptomObservation(TimestampedObservation):
    """
    Symptom-specific observation with palliative care attributes.

    Tracks not just severity, but also:
    - Location (e.g., "lower back", "head")
    - Duration patterns
    - Aggravating and relieving factors
    - Impact on daily functioning
    """
    symptom_name: str = ""
    severity: SeverityLevel = SeverityLevel.MODERATE
    location: Optional[str] = None  # "lower back", "head", "abdomen"
    duration: Optional[str] = None  # "2 days", "chronic", "intermittent"
    aggravating_factors: List[str] = field(default_factory=list)
    relieving_factors: List[str] = field(default_factory=list)
    impact_on_function: Optional[SeverityLevel] = None  # How it affects daily life
    body_site: Optional[str] = None  # More specific SNOMED-style location

    def __post_init__(self):
        """Set defaults based on parent fields."""
        if not self.symptom_name:
            self.symptom_name = self.entity_name
        if isinstance(self.value, SeverityLevel):
            self.severity = self.value

    def to_dict(self) -> Dict[str, Any]:
        """Serialize with symptom-specific fields."""
        base = super().to_dict()
        base.update({
            "symptom_name": self.symptom_name,
            "severity": self.severity.value,
            "location": self.location,
            "duration": self.duration,
            "aggravating_factors": self.aggravating_factors,
            "relieving_factors": self.relieving_factors,
            "impact_on_function": self.impact_on_function.value if self.impact_on_function else None,
            "body_site": self.body_site
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymptomObservation":
        """Deserialize with symptom-specific fields."""
        base = TimestampedObservation.from_dict(data)

        impact = None
        if data.get("impact_on_function") is not None:
            impact = SeverityLevel(data["impact_on_function"])

        return cls(
            observation_id=data["observation_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source_type=DataSourceType(data.get("source_type", "patient_reported")),
            source_id=data["source_id"],
            reported_by=data["reported_by"],
            category=data.get("category", "symptom"),
            entity_name=data.get("entity_name", data.get("symptom_name", "")),
            value=SeverityLevel(data.get("severity", 2)),
            value_text=data["value_text"],
            metadata=data.get("metadata", {}),
            symptom_name=data.get("symptom_name", data.get("entity_name", "")),
            severity=SeverityLevel(data.get("severity", 2)),
            location=data.get("location"),
            duration=data.get("duration"),
            aggravating_factors=data.get("aggravating_factors", []),
            relieving_factors=data.get("relieving_factors", []),
            impact_on_function=impact,
            body_site=data.get("body_site")
        )


@dataclass
class MedicationEvent(TimestampedObservation):
    """
    Medication-specific observation with adherence tracking.

    Tracks:
    - Medication starts, stops, dose changes
    - Adherence (taken vs missed)
    - Effectiveness over time
    - Side effects
    """
    medication_name: str = ""
    dosage: str = ""
    action: MedicationAction = MedicationAction.TAKEN
    effectiveness: Optional[SeverityLevel] = None  # How well it's working
    side_effects: List[str] = field(default_factory=list)
    adherence_rate: Optional[float] = None  # 0.0 to 1.0
    route: Optional[str] = None  # "oral", "IV", "subcutaneous"
    frequency: Optional[str] = None  # "twice daily", "every 4 hours"
    prescribed_by: Optional[str] = None  # Provider name/ID

    def __post_init__(self):
        """Set defaults based on parent fields."""
        if not self.medication_name:
            self.medication_name = self.entity_name
        if isinstance(self.action, str):
            try:
                self.action = MedicationAction(self.action)
            except ValueError:
                self.action = MedicationAction.TAKEN

    def to_dict(self) -> Dict[str, Any]:
        """Serialize with medication-specific fields."""
        base = super().to_dict()
        base.update({
            "medication_name": self.medication_name,
            "dosage": self.dosage,
            "action": self.action.value,
            "effectiveness": self.effectiveness.value if self.effectiveness else None,
            "side_effects": self.side_effects,
            "adherence_rate": self.adherence_rate,
            "route": self.route,
            "frequency": self.frequency,
            "prescribed_by": self.prescribed_by
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MedicationEvent":
        """Deserialize with medication-specific fields."""
        base = TimestampedObservation.from_dict(data)

        effectiveness = None
        if data.get("effectiveness") is not None:
            effectiveness = SeverityLevel(data["effectiveness"])

        action = MedicationAction.TAKEN
        if isinstance(data.get("action"), str):
            try:
                action = MedicationAction(data["action"])
            except ValueError:
                pass

        return cls(
            observation_id=data["observation_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source_type=DataSourceType(data.get("source_type", "patient_reported")),
            source_id=data["source_id"],
            reported_by=data["reported_by"],
            category=data.get("category", "medication"),
            entity_name=data.get("entity_name", data.get("medication_name", "")),
            value=action,
            value_text=data["value_text"],
            metadata=data.get("metadata", {}),
            medication_name=data.get("medication_name", data.get("entity_name", "")),
            dosage=data.get("dosage", ""),
            action=action,
            effectiveness=effectiveness,
            side_effects=data.get("side_effects", []),
            adherence_rate=data.get("adherence_rate"),
            route=data.get("route"),
            frequency=data.get("frequency"),
            prescribed_by=data.get("prescribed_by")
        )


@dataclass
class VitalSignObservation(TimestampedObservation):
    """
    Vital sign measurement.

    Tracks:
    - Blood pressure, heart rate, O2 saturation, temperature
    - Whether values are within normal range
    - Units of measurement
    """
    vital_name: str = ""
    value_numeric: float = 0.0
    unit: str = ""  # "mmHg", "bpm", "%", "°C"
    within_normal_range: Optional[bool] = None
    interpretation: Optional[str] = None  # Clinical interpretation

    def __post_init__(self):
        """Set defaults based on parent fields."""
        if not self.vital_name:
            self.vital_name = self.entity_name

    def to_dict(self) -> Dict[str, Any]:
        """Serialize with vital-specific fields."""
        base = super().to_dict()
        base.update({
            "vital_name": self.vital_name,
            "value_numeric": self.value_numeric,
            "unit": self.unit,
            "within_normal_range": self.within_normal_range,
            "interpretation": self.interpretation
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VitalSignObservation":
        """Deserialize with vital-specific fields."""
        return cls(
            observation_id=data["observation_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source_type=DataSourceType(data.get("source_type", "patient_reported")),
            source_id=data["source_id"],
            reported_by=data["reported_by"],
            category=data.get("category", "vital_sign"),
            entity_name=data.get("entity_name", data.get("vital_name", "")),
            value=data.get("value_numeric", 0.0),
            value_text=data["value_text"],
            metadata=data.get("metadata", {}),
            vital_name=data.get("vital_name", data.get("entity_name", "")),
            value_numeric=data.get("value_numeric", 0.0),
            unit=data.get("unit", ""),
            within_normal_range=data.get("within_normal_range"),
            interpretation=data.get("interpretation")
        )


@dataclass
class FunctionalStatusObservation(TimestampedObservation):
    """
    Functional status assessment.

    Tracks patient's ability to perform daily activities:
    - Mobility
    - Self-care
    - Ability to work/engage in activities
    """
    domain: str = ""  # "mobility", "self_care", "work", "social"
    score: Optional[float] = None  # Numeric score (e.g., ECOG, Karnofsky)
    scale: Optional[str] = None  # "ECOG", "Karnofsky", "custom"
    limitations: List[str] = field(default_factory=list)
    assistance_required: List[str] = field(default_factory=list)  # "walking", "bathing"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize with functional status fields."""
        base = super().to_dict()
        base.update({
            "domain": self.domain,
            "score": self.score,
            "scale": self.scale,
            "limitations": self.limitations,
            "assistance_required": self.assistance_required
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FunctionalStatusObservation":
        """Deserialize with functional status fields."""
        return cls(
            observation_id=data["observation_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source_type=DataSourceType(data.get("source_type", "patient_reported")),
            source_id=data["source_id"],
            reported_by=data["reported_by"],
            category=data.get("category", "functional_status"),
            entity_name=data.get("entity_name", data.get("domain", "")),
            value=data.get("score"),
            value_text=data["value_text"],
            metadata=data.get("metadata", {}),
            domain=data.get("domain", ""),
            score=data.get("score"),
            scale=data.get("scale"),
            limitations=data.get("limitations", []),
            assistance_required=data.get("assistance_required", [])
        )


@dataclass
class EmotionalObservation(TimestampedObservation):
    """
    Emotional/psychological state observation.

    Tracks:
    - Mood (anxious, depressed, peaceful, etc.)
    - Stress level
    - Spiritual concerns
    """
    emotion_type: str = ""  # "anxiety", "depression", "peace", "fear", "anger"
    intensity: SeverityLevel = SeverityLevel.MODERATE
    triggers: List[str] = field(default_factory=list)
    coping_strategies: List[str] = field(default_factory=list)
    spiritual_concerns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize with emotional fields."""
        base = super().to_dict()
        base.update({
            "emotion_type": self.emotion_type,
            "intensity": self.intensity.value,
            "triggers": self.triggers,
            "coping_strategies": self.coping_strategies,
            "spiritual_concerns": self.spiritual_concerns
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmotionalObservation":
        """Deserialize with emotional fields."""
        return cls(
            observation_id=data["observation_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source_type=DataSourceType(data.get("source_type", "patient_reported")),
            source_id=data["source_id"],
            reported_by=data["reported_by"],
            category=data.get("category", "emotional"),
            entity_name=data.get("entity_name", data.get("emotion_type", "")),
            value=SeverityLevel(data.get("intensity", 2)),
            value_text=data["value_text"],
            metadata=data.get("metadata", {}),
            emotion_type=data.get("emotion_type", ""),
            intensity=SeverityLevel(data.get("intensity", 2)),
            triggers=data.get("triggers", []),
            coping_strategies=data.get("coping_strategies", []),
            spiritual_concerns=data.get("spiritual_concerns", [])
        )


# ============================================================================
# TEMPORAL AGGREGATIONS
# ============================================================================

@dataclass
class TimeSeriesSummary:
    """
    Aggregated view of observations over a time period.

    Enables temporal reasoning about trends - e.g., "pain has been worsening
    over the past 7 days" or "appetite has been stable for 2 weeks."
    """
    entity_name: str
    category: str
    start_date: date
    end_date: date
    total_observations: int

    # Raw values
    latest_value: Any
    earliest_value: Any
    values_list: List[Any] = field(default_factory=list)

    # Statistics
    average_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    std_dev: Optional[float] = None

    # Temporal reasoning
    trend: TemporalTrend = TemporalTrend.UNKNOWN
    trend_confidence: float = 0.0  # 0.0 to 1.0 (R-squared)

    # Change rate
    change_per_week: Optional[float] = None

    # Contextual notes
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "entity_name": self.entity_name,
            "category": self.category,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_observations": self.total_observations,
            "latest_value": str(self.latest_value) if not isinstance(self.latest_value, (int, float)) else self.latest_value,
            "earliest_value": str(self.earliest_value) if not isinstance(self.earliest_value, (int, float)) else self.earliest_value,
            "average_value": self.average_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "trend": self.trend.value,
            "trend_confidence": self.trend_confidence,
            "change_per_week": self.change_per_week,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeSeriesSummary":
        """Deserialize from dictionary."""
        return cls(
            entity_name=data["entity_name"],
            category=data["category"],
            start_date=date.fromisoformat(data["start_date"]),
            end_date=date.fromisoformat(data["end_date"]),
            total_observations=data["total_observations"],
            latest_value=data["latest_value"],
            earliest_value=data["earliest_value"],
            average_value=data.get("average_value"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            trend=TemporalTrend(data.get("trend", "unknown")),
            trend_confidence=data.get("trend_confidence", 0.0),
            change_per_week=data.get("change_per_week"),
            notes=data.get("notes", [])
        )


# ============================================================================
# CARE TEAM TRACKING
# ============================================================================

@dataclass
class CareTeamMember:
    """Member of the patient's care team."""
    provider_id: str
    name: str
    role: str  # "doctor", "nurse", "asha_worker", "caregiver", "volunteer", "social_worker"
    organization: Optional[str] = None
    phone_number: Optional[str] = None  # Hashed for privacy
    email: Optional[str] = None
    primary_contact: bool = False

    # Tracking
    first_contact: datetime = field(default_factory=datetime.now)
    last_contact: datetime = field(default_factory=datetime.now)
    total_interactions: int = 0

    # Attribution
    attributed_observations: List[str] = field(default_factory=list)

    # Availability
    available_hours: Optional[str] = None  # "9-5 weekdays"
    timezone: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "provider_id": self.provider_id,
            "name": self.name,
            "role": self.role,
            "organization": self.organization,
            "phone_number": self.phone_number,
            "email": self.email,
            "primary_contact": self.primary_contact,
            "first_contact": self.first_contact.isoformat(),
            "last_contact": self.last_contact.isoformat(),
            "total_interactions": self.total_interactions,
            "attributed_observations": self.attributed_observations,
            "available_hours": self.available_hours,
            "timezone": self.timezone
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CareTeamMember":
        """Deserialize from dictionary."""
        return cls(
            provider_id=data["provider_id"],
            name=data["name"],
            role=data["role"],
            organization=data.get("organization"),
            phone_number=data.get("phone_number"),
            email=data.get("email"),
            primary_contact=data.get("primary_contact", False),
            first_contact=datetime.fromisoformat(data["first_contact"]),
            last_contact=datetime.fromisoformat(data["last_contact"]),
            total_interactions=data.get("total_interactions", 0),
            attributed_observations=data.get("attributed_observations", []),
            available_hours=data.get("available_hours"),
            timezone=data.get("timezone")
        )


# ============================================================================
# ALERTS AND PROACTIVE MONITORING
# ============================================================================

@dataclass
class MonitoringAlert:
    """Proactive alert generated from pattern detection."""
    alert_id: str
    patient_id: str
    created_at: datetime
    priority: AlertPriority
    category: str  # "symptom_change", "medication_issue", "missed_checkin", "care_gap"
    title: str
    description: str

    # Trigger information
    trigger_observation_ids: List[str] = field(default_factory=list)
    pattern_description: str = ""
    rule_id: Optional[str] = None

    # State
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None

    # Actions
    suggested_actions: List[str] = field(default_factory=list)
    assigned_to: Optional[str] = None  # care_team_member_id

    # Delivery tracking
    delivery_channels: List[str] = field(default_factory=list)  # "whatsapp", "email", "dashboard"
    delivery_status: Dict[str, str] = field(default_factory=dict)  # {"whatsapp": "delivered"}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "alert_id": self.alert_id,
            "patient_id": self.patient_id,
            "created_at": self.created_at.isoformat(),
            "priority": self.priority.value,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "trigger_observation_ids": self.trigger_observation_ids,
            "pattern_description": self.pattern_description,
            "rule_id": self.rule_id,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_notes": self.resolution_notes,
            "suggested_actions": self.suggested_actions,
            "assigned_to": self.assigned_to,
            "delivery_channels": self.delivery_channels,
            "delivery_status": self.delivery_status
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MonitoringAlert":
        """Deserialize from dictionary."""
        return cls(
            alert_id=data["alert_id"],
            patient_id=data["patient_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            priority=AlertPriority(data["priority"]),
            category=data["category"],
            title=data["title"],
            description=data["description"],
            trigger_observation_ids=data.get("trigger_observation_ids", []),
            pattern_description=data.get("pattern_description", ""),
            rule_id=data.get("rule_id"),
            acknowledged=data.get("acknowledged", False),
            acknowledged_by=data.get("acknowledged_by"),
            acknowledged_at=datetime.fromisoformat(data["acknowledged_at"]) if data.get("acknowledged_at") else None,
            resolved=data.get("resolved", False),
            resolved_at=datetime.fromisoformat(data["resolved_at"]) if data.get("resolved_at") else None,
            resolution_notes=data.get("resolution_notes"),
            suggested_actions=data.get("suggested_actions", []),
            assigned_to=data.get("assigned_to"),
            delivery_channels=data.get("delivery_channels", []),
            delivery_status=data.get("delivery_status", {})
        )


@dataclass
class MonitoringRule:
    """Rule for generating proactive alerts."""
    rule_id: str
    name: str
    description: str
    enabled: bool = True

    # Trigger conditions
    category: Optional[str] = None  # Filter by observation category
    entity_name_pattern: Optional[str] = None  # "pain*", "*morphine*"
    min_observations: int = 3
    time_window_days: int = 7

    # Pattern to detect
    pattern_type: str = "worsening_trend"  # "worsening_trend", "severe_value", "missed_doses", "gap_in_care"
    threshold_value: Optional[Any] = None

    # Alert configuration
    alert_priority: AlertPriority = AlertPriority.MEDIUM
    alert_message_template: str = ""
    suggested_actions: List[str] = field(default_factory=list)

    # Cooldown
    cooldown_hours: int = 24  # Minimum time between same alert

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "category": self.category,
            "entity_name_pattern": self.entity_name_pattern,
            "min_observations": self.min_observations,
            "time_window_days": self.time_window_days,
            "pattern_type": self.pattern_type,
            "threshold_value": str(self.threshold_value) if self.threshold_value else None,
            "alert_priority": self.alert_priority.value,
            "alert_message_template": self.alert_message_template,
            "suggested_actions": self.suggested_actions,
            "cooldown_hours": self.cooldown_hours
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MonitoringRule":
        """Deserialize from dictionary."""
        return cls(
            rule_id=data["rule_id"],
            name=data["name"],
            description=data["description"],
            enabled=data.get("enabled", True),
            category=data.get("category"),
            entity_name_pattern=data.get("entity_name_pattern"),
            min_observations=data.get("min_observations", 3),
            time_window_days=data.get("time_window_days", 7),
            pattern_type=data.get("pattern_type", "worsening_trend"),
            threshold_value=data.get("threshold_value"),
            alert_priority=AlertPriority(data.get("alert_priority", "medium")),
            alert_message_template=data.get("alert_message_template", ""),
            suggested_actions=data.get("suggested_actions", []),
            cooldown_hours=data.get("cooldown_hours", 24)
        )


# ============================================================================
# LONGITUDINAL PATIENT RECORD
# ============================================================================

@dataclass
class LongitudinalPatientRecord:
    """
    Complete longitudinal patient record supporting 1-5 year tracking.

    This extends the existing PatientContext (90-day) with:
    - Time-series observations (not just current state)
    - Temporal trend analysis
    - Care team tracking
    - Cross-modal data aggregation
    - Proactive monitoring/alerts
    """
    patient_id: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # === Longitudinal observations (1-5 years) ===
    observations: List[TimestampedObservation] = field(default_factory=list)

    # === Care team ===
    care_team: List[CareTeamMember] = field(default_factory=list)

    # === Alerts ===
    active_alerts: List[MonitoringAlert] = field(default_factory=list)
    alert_history: List[MonitoringAlert] = field(default_factory=list)
    last_alert_check: Optional[datetime] = None

    # === Monitoring configuration ===
    monitoring_rules: List[MonitoringRule] = field(default_factory=list)

    # === Cross-modal aggregation ===
    conversations_index: Dict[str, List[str]] = field(default_factory=dict)
    documents_index: Dict[str, List[str]] = field(default_factory=dict)

    # === Metadata ===
    data_sources: Set[DataSourceType] = field(default_factory=set)
    total_observations: int = 0
    observation_date_range: Optional[Tuple[datetime, datetime]] = None

    # === Clinical metadata ===
    primary_condition: Optional[str] = None
    condition_stage: Optional[str] = None
    allergies: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Convert sets to lists for serialization."""
        if isinstance(self.data_sources, set):
            self.data_sources_list = [ds.value for ds in self.data_sources]

    def add_observation(self, observation: TimestampedObservation) -> None:
        """Add a new observation to the longitudinal record."""
        self.observations.append(observation)

        if isinstance(observation.source_type, str):
            observation.source_type = DataSourceType(observation.source_type)

        self.data_sources.add(observation.source_type)
        self.total_observations += 1
        self.updated_at = datetime.now()

        # Update index
        if observation.source_type in [DataSourceType.VOICE_CALL, DataSourceType.WHATSAPP, DataSourceType.WEB_CHAT]:
            self.conversations_index.setdefault(observation.source_id, []).append(
                observation.observation_id
            )
        elif observation.source_type == DataSourceType.UPLOADED_DOCUMENT:
            self.documents_index.setdefault(observation.source_id, []).append(
                observation.observation_id
            )

        # Update date range
        timestamps = [o.timestamp for o in self.observations]
        self.observation_date_range = (min(timestamps), max(timestamps))

    def get_observations_for_entity(
        self,
        entity_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[TimestampedObservation]:
        """Get all observations for a specific entity within date range."""
        observations = [
            o for o in self.observations
            if o.entity_name.lower() == entity_name.lower()
        ]

        if start_date:
            observations = [o for o in observations if o.timestamp >= start_date]
        if end_date:
            observations = [o for o in observations if o.timestamp <= end_date]

        return observations

    def get_observations_by_category(
        self,
        category: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[TimestampedObservation]:
        """Get all observations for a category within date range."""
        observations = [
            o for o in self.observations
            if o.category == category
        ]

        if start_date:
            observations = [o for o in observations if o.timestamp >= start_date]
        if end_date:
            observations = [o for o in observations if o.timestamp <= end_date]

        return sorted(observations, key=lambda o: o.timestamp)

    def get_time_series_summary(
        self,
        entity_name: str,
        days: int = 30
    ) -> Optional[TimeSeriesSummary]:
        """Generate time-series summary for an entity."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        observations = self.get_observations_for_entity(entity_name, start_date, end_date)

        if not observations:
            return None

        # Sort by timestamp
        observations.sort(key=lambda o: o.timestamp)

        # Calculate trend
        trend, confidence = self._calculate_trend(observations)

        # Extract numeric values for statistics
        numeric_values = []
        for obs in observations:
            if isinstance(obs.value, (int, float)):
                numeric_values.append(float(obs.value))
            elif isinstance(obs.value, SeverityLevel):
                numeric_values.append(float(obs.value.value))
            elif isinstance(obs, SymptomObservation):
                numeric_values.append(float(obs.severity.value))
            elif isinstance(obs, MedicationEvent):
                # For medications, use effectiveness if available
                if obs.effectiveness:
                    numeric_values.append(float(obs.effectiveness.value))

        summary = TimeSeriesSummary(
            entity_name=entity_name,
            category=observations[0].category,
            start_date=start_date.date(),
            end_date=end_date.date(),
            total_observations=len(observations),
            latest_value=observations[-1].value,
            earliest_value=observations[0].value,
            values_list=[o.value for o in observations],
            trend=trend,
            trend_confidence=confidence
        )

        # Calculate statistics if we have numeric values
        if numeric_values:
            summary.average_value = sum(numeric_values) / len(numeric_values)
            summary.min_value = min(numeric_values)
            summary.max_value = max(numeric_values)

            # Calculate change per week
            if len(numeric_values) >= 2:
                change = numeric_values[-1] - numeric_values[0]
                summary.change_per_week = change / (days / 7)

        return summary

    def _calculate_trend(
        self,
        observations: List[TimestampedObservation]
    ) -> Tuple[TemporalTrend, float]:
        """
        Calculate trend from observations using linear regression.

        Returns:
            (trend_direction, confidence_r_squared)
        """
        if len(observations) < 3:
            return TemporalTrend.UNKNOWN, 0.0

        # Try to extract numeric values from severity
        numeric_values = []
        for obs in observations:
            if isinstance(obs.value, (int, float)):
                numeric_values.append(float(obs.value))
            elif isinstance(obs.value, SeverityLevel):
                numeric_values.append(float(obs.value.value))
            elif isinstance(obs, SymptomObservation):
                numeric_values.append(float(obs.severity.value))
            elif isinstance(obs, MedicationEvent) and obs.effectiveness:
                numeric_values.append(float(obs.effectiveness.value))

        if len(numeric_values) < 3:
            return TemporalTrend.UNKNOWN, 0.0

        # Simple linear regression for trend
        n = len(numeric_values)
        x = list(range(n))
        y = numeric_values

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return TemporalTrend.STABLE, 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # Calculate R-squared for confidence
        y_mean = sum_y / n
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)

        if ss_tot == 0:
            return TemporalTrend.STABLE, 1.0

        y_pred = [slope * xi + (sum_y - slope * sum_x) / n for xi in x]
        ss_res = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred))

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r_squared = max(0.0, min(1.0, r_squared))

        # Determine trend
        # For symptoms, higher = worse (worsening trend)
        # For effectiveness, higher = better (improving trend)
        # We need to consider the category
        category = observations[0].category if observations else ""

        if category == "medication":
            # For medications, positive slope = improving effectiveness
            if slope > 0.3:
                trend = TemporalTrend.IMPROVING
            elif slope < -0.3:
                trend = TemporalTrend.WORSENING
            elif abs(slope) < 0.15:
                trend = TemporalTrend.STABLE
            else:
                trend = TemporalTrend.FLUCTUATING
        else:
            # For symptoms, higher = worse
            if slope > 0.3:
                trend = TemporalTrend.WORSENING
            elif slope < -0.3:
                trend = TemporalTrend.IMPROVING
            elif abs(slope) < 0.15:
                trend = TemporalTrend.STABLE
            else:
                trend = TemporalTrend.FLUCTUATING

        return trend, r_squared

    def add_care_team_member(self, member: CareTeamMember) -> None:
        """Add or update a care team member."""
        # Check for existing member
        for existing in self.care_team:
            if existing.provider_id == member.provider_id:
                # Update existing
                existing.name = member.name
                existing.role = member.role
                existing.organization = member.organization
                existing.last_contact = datetime.now()
                existing.total_interactions += 1
                if member.primary_contact:
                    existing.primary_contact = True
                self.updated_at = datetime.now()
                return

        # Add new member
        self.care_team.append(member)
        logger.info(f"Added care team member: {member.name} ({member.role})")
        self.updated_at = datetime.now()

    def get_primary_contact(self) -> Optional[CareTeamMember]:
        """Get primary care contact."""
        for member in self.care_team:
            if member.primary_contact:
                return member

        # Fallback to most recent contact
        if self.care_team:
            return max(self.care_team, key=lambda m: m.last_contact)

        return None

    def get_recent_observations(
        self,
        days: int = 7,
        category: Optional[str] = None
    ) -> List[TimestampedObservation]:
        """Get recent observations within time window."""
        cutoff = datetime.now() - timedelta(days=days)

        observations = [o for o in self.observations if o.timestamp >= cutoff]

        if category:
            observations = [o for o in observations if o.category == category]

        return sorted(observations, key=lambda o: o.timestamp, reverse=True)

    def get_active_symptoms(self) -> List[Tuple[str, SeverityLevel, datetime]]:
        """Get list of currently active symptoms with severity."""
        symptom_obs = [o for o in self.observations if o.category == "symptom"]

        # Group by symptom name, get most recent
        symptom_map = {}
        for obs in symptom_obs:
            if isinstance(obs, SymptomObservation):
                symptom_map[obs.symptom_name] = (obs.symptom_name, obs.severity, obs.timestamp)
            elif obs.entity_name:
                severity = SeverityLevel.MODERATE
                if isinstance(obs.value, SeverityLevel):
                    severity = obs.value
                symptom_map[obs.entity_name] = (obs.entity_name, severity, obs.timestamp)

        # Return symptoms reported in last 30 days
        cutoff = datetime.now() - timedelta(days=30)
        return [(name, sev, ts) for name, sev, ts in symptom_map.values() if ts > cutoff]

    def get_current_medications(self) -> List[Dict[str, Any]]:
        """Get list of currently active medications."""
        med_obs = [o for o in self.observations if o.category == "medication"]

        # Track most recent action for each medication
        med_map = {}
        for obs in med_obs:
            if isinstance(obs, MedicationEvent):
                med_map[obs.medication_name] = {
                    "name": obs.medication_name,
                    "dosage": obs.dosage,
                    "action": obs.action,
                    "timestamp": obs.timestamp,
                    "frequency": obs.frequency,
                    "route": obs.route
                }
            elif obs.entity_name:
                action = obs.value if isinstance(obs.value, MedicationAction) else MedicationAction.TAKEN
                med_map[obs.entity_name] = {
                    "name": obs.entity_name,
                    "dosage": obs.metadata.get("dosage", ""),
                    "action": action,
                    "timestamp": obs.timestamp
                }

        # Filter for medications that haven't been stopped
        cutoff = datetime.now() - timedelta(days=90)
        current_meds = []
        for name, info in med_map.items():
            if info["timestamp"] > cutoff:
                if info["action"] != MedicationAction.STOPPED:
                    current_meds.append(info)

        return current_meds

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        # Convert observations
        observations_dict = [o.to_dict() for o in self.observations]

        # Convert care team
        care_team_dict = [ct.to_dict() for ct in self.care_team]

        # Convert alerts
        active_alerts_dict = [a.to_dict() for a in self.active_alerts]
        alert_history_dict = [a.to_dict() for a in self.alert_history]
        monitoring_rules_dict = [r.to_dict() for r in self.monitoring_rules]

        # Convert data_sources set to list
        data_sources_list = [ds.value for ds in self.data_sources]

        # Convert date range
        date_range = None
        if self.observation_date_range:
            date_range = [
                self.observation_date_range[0].isoformat(),
                self.observation_date_range[1].isoformat()
            ]

        return {
            "patient_id": self.patient_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "observations": observations_dict,
            "care_team": care_team_dict,
            "active_alerts": active_alerts_dict,
            "alert_history": alert_history_dict,
            "last_alert_check": self.last_alert_check.isoformat() if self.last_alert_check else None,
            "monitoring_rules": monitoring_rules_dict,
            "conversations_index": self.conversations_index,
            "documents_index": self.documents_index,
            "data_sources": data_sources_list,
            "total_observations": self.total_observations,
            "observation_date_range": date_range,
            "primary_condition": self.primary_condition,
            "condition_stage": self.condition_stage,
            "allergies": self.allergies
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LongitudinalPatientRecord":
        """Deserialize from dictionary."""
        # Convert observations back to objects
        observations = []
        for obs_data in data.get("observations", []):
            category = obs_data.get("category", "")
            if category == "symptom":
                observations.append(SymptomObservation.from_dict(obs_data))
            elif category == "medication":
                observations.append(MedicationEvent.from_dict(obs_data))
            elif category == "vital_sign":
                observations.append(VitalSignObservation.from_dict(obs_data))
            elif category == "functional_status":
                observations.append(FunctionalStatusObservation.from_dict(obs_data))
            elif category == "emotional":
                observations.append(EmotionalObservation.from_dict(obs_data))
            else:
                observations.append(TimestampedObservation.from_dict(obs_data))

        # Convert care team
        care_team = [CareTeamMember.from_dict(ct) for ct in data.get("care_team", [])]

        # Convert alerts
        active_alerts = [MonitoringAlert.from_dict(a) for a in data.get("active_alerts", [])]
        alert_history = [MonitoringAlert.from_dict(a) for a in data.get("alert_history", [])]

        # Convert monitoring rules
        monitoring_rules = [MonitoringRule.from_dict(r) for r in data.get("monitoring_rules", [])]

        # Convert data sources back to set
        data_sources_set = set()
        for ds in data.get("data_sources", []):
            try:
                data_sources_set.add(DataSourceType(ds))
            except ValueError:
                pass

        # Convert date range
        date_range = None
        if data.get("observation_date_range"):
            dr = data["observation_date_range"]
            date_range = (
                datetime.fromisoformat(dr[0]),
                datetime.fromisoformat(dr[1])
            )

        # Convert last alert check
        last_alert_check = None
        if data.get("last_alert_check"):
            last_alert_check = datetime.fromisoformat(data["last_alert_check"])

        record = cls(
            patient_id=data["patient_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            observations=observations,
            care_team=care_team,
            active_alerts=active_alerts,
            alert_history=alert_history,
            last_alert_check=last_alert_check,
            monitoring_rules=monitoring_rules,
            conversations_index=data.get("conversations_index", {}),
            documents_index=data.get("documents_index", {}),
            data_sources=data_sources_set,
            total_observations=data.get("total_observations", 0),
            observation_date_range=date_range,
            primary_condition=data.get("primary_condition"),
            condition_stage=data.get("condition_stage"),
            allergies=data.get("allergies", [])
        )

        return record


# ============================================================================
# LONGITUDINAL MEMORY MANAGER
# ============================================================================

class LongitudinalMemoryManager:
    """
    Manages longitudinal patient records with file-based storage.

    Features:
    - 1-5 year retention
    - Temporal trend analysis
    - Cross-modal aggregation
    - Proactive monitoring/alerts
    - Care team coordination

    Inspired by MedAgentBenchV2's memory system, adapted for palliative care.
    """

    def __init__(
        self,
        storage_path: str = "data/longitudinal_memory",
        retention_years: int = 5,
        monitoring_check_interval_hours: int = 24
    ):
        """
        Initialize the longitudinal memory manager.

        Args:
            storage_path: Directory for longitudinal record storage
            retention_years: Years of data to retain (default 5)
            monitoring_check_interval_hours: Hours between automatic monitoring checks
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Create alerts subdirectory
        (self.storage_path / "alerts").mkdir(exist_ok=True)

        self.retention_years = retention_years
        self.monitoring_interval = timedelta(hours=monitoring_check_interval_hours)

        # Default monitoring rules for palliative care
        self.default_rules = self._get_default_monitoring_rules()

        self._cache: Dict[str, LongitudinalPatientRecord] = {}
        self._lock = asyncio.Lock()

        logger.info(
            f"LongitudinalMemoryManager initialized - "
            f"path={storage_path}, retention={retention_years}y"
        )

    def _get_patient_path(self, patient_id: str) -> Path:
        """Get file path for patient's longitudinal record."""
        # Use patient_id directly for filename
        safe_id = patient_id.replace("/", "_").replace("\\", "_")
        return self.storage_path / f"{safe_id}_longitudinal.json"

    def _get_alert_path(self, alert_id: str) -> Path:
        """Get file path for an alert."""
        return self.storage_path / "alerts" / f"{alert_id}.json"

    def _generate_id(self, prefix: str, content: str) -> str:
        """Generate unique ID using timestamp and hash."""
        timestamp = datetime.now().isoformat()
        hash_part = hashlib.md5(f"{content}:{timestamp}".encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp[:10].replace('-', '')}_{hash_part}"

    def _get_default_monitoring_rules(self) -> List[MonitoringRule]:
        """Get default palliative care monitoring rules."""
        return [
            MonitoringRule(
                rule_id="worsening_pain",
                name="Worsening Pain",
                description="Detect worsening pain trend over 7 days",
                category="symptom",
                entity_name_pattern="pain",
                min_observations=3,
                time_window_days=7,
                pattern_type="worsening_trend",
                alert_priority=AlertPriority.HIGH,
                alert_message_template=(
                    "आपके दर्द में पिछले हफ्ते से वृद्धि हुई है। "
                    "Your pain appears to be worsening over the past week. "
                    "Consider assessment for medication adjustment."
                ),
                suggested_actions=[
                    "Schedule follow-up call within 24 hours",
                    "Review current pain management plan",
                    "Assess for new complications"
                ],
                cooldown_hours=48
            ),
            MonitoringRule(
                rule_id="severe_breathlessness",
                name="Severe Breathlessness",
                description="Detect severe breathlessness episodes",
                category="symptom",
                entity_name_pattern="breathlessness",
                min_observations=1,
                time_window_days=1,
                pattern_type="severe_value",
                threshold_value=SeverityLevel.SEVERE,
                alert_priority=AlertPriority.URGENT,
                alert_message_template=(
                    "सांस फूलना एक गंभीर समस्या है। "
                    "Severe breathlessness reported. "
                    "Immediate clinical assessment recommended."
                ),
                suggested_actions=[
                    "Attempt immediate contact",
                    "Consider emergency referral if no response",
                    "Document for care team review"
                ],
                cooldown_hours=12
            ),
            MonitoringRule(
                rule_id="severe_pain",
                name="Severe Pain Episode",
                description="Detect severe pain reports",
                category="symptom",
                entity_name_pattern="pain",
                min_observations=1,
                time_window_days=1,
                pattern_type="severe_value",
                threshold_value=SeverityLevel.VERY_SEVERE,
                alert_priority=AlertPriority.HIGH,
                alert_message_template=(
                    "बहुत तीव्र दर्द की रिपोर्ट है। "
                    "Very severe pain reported. "
                    "Consider pain management review."
                ),
                suggested_actions=[
                    "Assess pain within 24 hours",
                    "Review current analgesic regimen",
                    "Check for breakthrough pain"
                ],
                cooldown_hours=24
            ),
            MonitoringRule(
                rule_id="new_severe_symptom",
                name="New Severe Symptom",
                description="Detect any new severe symptom",
                category="symptom",
                min_observations=1,
                time_window_days=1,
                pattern_type="new_severe",
                threshold_value=SeverityLevel.SEVERE,
                alert_priority=AlertPriority.HIGH,
                alert_message_template=(
                    "नया गंभीर लक्षण दिखाई दिया है। "
                    "New severe symptom reported. "
                    "Clinical assessment recommended."
                ),
                suggested_actions=[
                    "Document new symptom",
                    "Assess within 48 hours",
                    "Consider underlying cause"
                ],
                cooldown_hours=24
            ),
            MonitoringRule(
                rule_id="missed_checkin",
                name="Missed Check-in",
                description="Detect when patient hasn't been in touch",
                category="general",
                min_observations=0,
                time_window_days=14,
                pattern_type="gap_in_care",
                alert_priority=AlertPriority.LOW,
                alert_message_template=(
                    "लंबे समय से आपकी जानकारी नहीं मिली है। "
                    "Patient hasn't had any interaction in 14 days. "
                    "Proactive check-in recommended."
                ),
                suggested_actions=[
                    "Send caring check-in message",
                    "Offer callback time",
                    "Verify contact information"
                ],
                cooldown_hours=72
            ),
            MonitoringRule(
                rule_id="emotional_distress",
                name="Emotional Distress",
                description="Detect persistent emotional distress",
                category="emotional",
                min_observations=2,
                time_window_days=7,
                pattern_type="persistent",
                threshold_value=SeverityLevel.SEVERE,
                alert_priority=AlertPriority.MEDIUM,
                alert_message_template=(
                    "मरीज़ को भावनात्मक संकट हो सकता है। "
                    "Signs of emotional distress detected. "
                    "Consider psychosocial support."
                ),
                suggested_actions=[
                    "Reach out with empathetic call",
                    "Assess for depression/anxiety",
                    "Consider counseling referral"
                ],
                cooldown_hours=72
            )
        ]

    async def get_or_create_record(
        self,
        patient_id: str
    ) -> LongitudinalPatientRecord:
        """
        Get existing record or create new one.

        Args:
            patient_id: Patient identifier

        Returns:
            LongitudinalPatientRecord
        """
        # Check cache
        if patient_id in self._cache:
            return self._cache[patient_id]

        # Try to load from storage
        record = await self._load_record(patient_id)

        if not record:
            record = LongitudinalPatientRecord(patient_id=patient_id)
            # Initialize with default monitoring rules
            record.monitoring_rules = self.default_rules.copy()
            logger.info(f"Created longitudinal record for {patient_id}")

        self._cache[patient_id] = record
        return record

    async def _load_record(
        self,
        patient_id: str
    ) -> Optional[LongitudinalPatientRecord]:
        """Load record from storage."""
        file_path = self._get_patient_path(patient_id)

        if not file_path.exists():
            return None

        try:
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                if not content:
                    return None
                data = json.loads(content)
                return LongitudinalPatientRecord.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading longitudinal record for {patient_id}: {e}")
            return None

    async def save_record(
        self,
        record: LongitudinalPatientRecord
    ) -> None:
        """Save record to storage."""
        async with self._lock:
            file_path = self._get_patient_path(record.patient_id)
            record.updated_at = datetime.now()

            try:
                async with aiofiles.open(file_path, "w") as f:
                    await f.write(json.dumps(record.to_dict(), indent=2))
                self._cache[record.patient_id] = record
                logger.debug(f"Saved longitudinal record for {record.patient_id}")
            except Exception as e:
                logger.error(f"Error saving longitudinal record: {e}")

    async def add_observation(
        self,
        patient_id: str,
        observation: TimestampedObservation
    ) -> LongitudinalPatientRecord:
        """
        Add observation and optionally run monitoring checks.

        Args:
            patient_id: Patient identifier
            observation: Observation to add

        Returns:
            Updated LongitudinalPatientRecord
        """
        record = await self.get_or_create_record(patient_id)
        record.add_observation(observation)

        # Save record
        await self.save_record(record)

        return record

    async def get_longitudinal_summary(
        self,
        patient_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get longitudinal summary for context injection.

        Args:
            patient_id: Patient identifier
            days: Number of days to look back

        Returns:
            Dictionary with summaries, trends, and alerts
        """
        record = await self.get_or_create_record(patient_id)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Filter recent observations
        recent_observations = [
            o for o in record.observations
            if o.timestamp >= start_date
        ]

        # Group by category and entity
        by_category: Dict[str, Dict[str, List]] = {}
        for obs in recent_observations:
            if obs.category not in by_category:
                by_category[obs.category] = {}
            if obs.entity_name not in by_category[obs.category]:
                by_category[obs.category][obs.entity_name] = []
            by_category[obs.category][obs.entity_name].append(obs)

        # Generate summaries for each entity
        summaries = {}
        for category, entities in by_category.items():
            for entity, observations_list in entities.items():
                if len(observations_list) >= 2:  # Only summarize if we have multiple
                    summary = record.get_time_series_summary(entity, days)
                    if summary:
                        key = f"{category}:{entity}"
                        summaries[key] = summary.to_dict()

        # Active alerts
        active_alerts = [a.to_dict() for a in record.active_alerts if not a.resolved]

        # Current symptoms
        active_symptoms = record.get_active_symptoms()

        # Current medications
        current_medications = record.get_current_medications()

        return {
            "patient_id": patient_id,
            "period_days": days,
            "total_observations": len(recent_observations),
            "summaries": summaries,
            "active_alerts": active_alerts,
            "active_symptoms": [
                {"name": name, "severity": sev.value, "last_reported": ts.isoformat()}
                for name, sev, ts in active_symptoms
            ],
            "current_medications": current_medications,
            "care_team_size": len(record.care_team),
            "data_sources": [ds.value for ds in record.data_sources],
            "primary_condition": record.primary_condition
        }

    async def run_monitoring_checks(
        self,
        patient_id: str
    ) -> List[MonitoringAlert]:
        """
        Run all monitoring rules and generate alerts.

        Args:
            patient_id: Patient identifier

        Returns:
            List of new alerts generated
        """
        record = await self.get_or_create_record(patient_id)

        # Check if we need to run monitoring
        now = datetime.now()
        if record.last_alert_check:
            time_since_check = now - record.last_alert_check
            if time_since_check < timedelta(hours=12):  # Minimum 12 hours between checks
                return []

        record.last_alert_check = now
        new_alerts = []

        for rule in record.monitoring_rules:
            if not rule.enabled:
                continue

            # Check cooldown for this rule
            recent_alerts = [
                a for a in record.active_alerts
                if a.rule_id == rule.rule_id and not a.resolved
            ]
            for alert in recent_alerts:
                if (now - alert.created_at).total_seconds() < rule.cooldown_hours * 3600:
                    continue  # Skip if still in cooldown

            alert = await self._check_rule(record, rule)
            if alert:
                record.active_alerts.append(alert)
                new_alerts.append(alert)
                logger.info(
                    f"Alert generated for {record.patient_id}: "
                    f"{alert.title} ({alert.priority.value})"
                )

        if new_alerts:
            await self.save_record(record)

        return new_alerts

    async def _check_rule(
        self,
        record: LongitudinalPatientRecord,
        rule: MonitoringRule
    ) -> Optional[MonitoringAlert]:
        """Check if a monitoring rule triggers an alert."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=rule.time_window_days)

        # Filter observations by rule criteria
        observations = [
            o for o in record.observations
            if start_date <= o.timestamp <= end_date
        ]

        if rule.category:
            observations = [o for o in observations if o.category == rule.category]

        if rule.entity_name_pattern:
            pattern = rule.entity_name_pattern.replace("*", "").lower()
            observations = [
                o for o in observations
                if pattern in o.entity_name.lower()
            ]

        if rule.pattern_type == "gap_in_care":
            # Check for gap in interactions
            if record.observations:
                latest_obs = max(record.observations, key=lambda o: o.timestamp)
                days_since = (end_date - latest_obs.timestamp).days
                if days_since >= rule.time_window_days:
                    return self._create_alert(record, rule, [latest_obs])
            return None

        if len(observations) < rule.min_observations:
            return None

        # Check pattern type
        if rule.pattern_type == "worsening_trend":
            summary = record.get_time_series_summary(
                observations[0].entity_name,
                rule.time_window_days
            )
            if summary and summary.trend == TemporalTrend.WORSENING:
                return self._create_alert(record, rule, observations)

        elif rule.pattern_type == "severe_value":
            for obs in observations:
                severity = None
                if isinstance(obs.value, SeverityLevel):
                    severity = obs.value
                elif isinstance(obs, SymptomObservation):
                    severity = obs.severity

                if severity and severity >= rule.threshold_value:
                    return self._create_alert(record, rule, [obs])

        elif rule.pattern_type == "new_severe":
            # Check if any recent observation is severe
            for obs in observations:
                severity = None
                if isinstance(obs.value, SeverityLevel):
                    severity = obs.value
                elif isinstance(obs, SymptomObservation):
                    severity = obs.severity

                if severity and severity >= rule.threshold_value:
                    # Check if this is a "new" symptom (no observations in prior 30 days)
                    prior_start = start_date - timedelta(days=30)
                    prior_obs = [
                        o for o in record.observations
                        if prior_start <= o.timestamp < start_date
                        and o.entity_name.lower() == obs.entity_name.lower()
                    ]
                    if not prior_obs:
                        return self._create_alert(record, rule, [obs])

        elif rule.pattern_type == "persistent":
            # Check for persistent elevated values
            severe_count = 0
            for obs in observations:
                severity = None
                if isinstance(obs.value, SeverityLevel):
                    severity = obs.value
                elif isinstance(obs, SymptomObservation):
                    severity = obs.severity
                elif isinstance(obs, EmotionalObservation):
                    severity = obs.intensity

                if severity and severity >= rule.threshold_value:
                    severe_count += 1

            if severe_count >= rule.min_observations:
                return self._create_alert(record, rule, observations)

        return None

    def _create_alert(
        self,
        record: LongitudinalPatientRecord,
        rule: MonitoringRule,
        observations: List[TimestampedObservation]
    ) -> MonitoringAlert:
        """Create alert from triggered rule."""
        alert_id = self._generate_id("alert", f"{record.patient_id}:{rule.rule_id}")

        return MonitoringAlert(
            alert_id=alert_id,
            patient_id=record.patient_id,
            created_at=datetime.now(),
            priority=rule.alert_priority,
            category=rule.category or "general",
            title=rule.name,
            description=rule.alert_message_template,
            trigger_observation_ids=[o.observation_id for o in observations],
            pattern_description=rule.description,
            rule_id=rule.rule_id,
            suggested_actions=rule.suggested_actions.copy()
        )

    async def acknowledge_alert(
        self,
        patient_id: str,
        alert_id: str,
        acknowledged_by: str
    ) -> bool:
        """Acknowledge an alert."""
        record = await self.get_or_create_record(patient_id)

        for alert in record.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                await self.save_record(record)
                return True

        return False

    async def resolve_alert(
        self,
        patient_id: str,
        alert_id: str,
        resolution_notes: Optional[str] = None
    ) -> bool:
        """Resolve an alert."""
        record = await self.get_or_create_record(patient_id)

        for i, alert in enumerate(record.active_alerts):
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                alert.resolution_notes = resolution_notes
                # Move to history
                record.alert_history.append(alert)
                record.active_alerts.pop(i)
                await self.save_record(record)
                return True

        return False

    async def add_care_team_member(
        self,
        patient_id: str,
        member: CareTeamMember
    ) -> None:
        """Add a care team member."""
        record = await self.get_or_create_record(patient_id)
        record.add_care_team_member(member)
        await self.save_record(record)

    async def get_care_team(self, patient_id: str) -> List[CareTeamMember]:
        """Get care team for a patient."""
        record = await self.get_or_create_record(patient_id)
        return record.care_team.copy()

    async def get_statistics(self) -> Dict[str, Any]:
        """Get longitudinal memory statistics."""
        all_records = []
        for file_path in self.storage_path.glob("*_longitudinal.json"):
            try:
                async with aiofiles.open(file_path, "r") as f:
                    content = await f.read()
                    if content:
                        data = json.loads(content)
                        all_records.append(LongitudinalPatientRecord.from_dict(data))
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        if not all_records:
            return {
                "total_patients": 0,
                "total_observations": 0,
                "active_alerts": 0,
                "data_sources": []
            }

        total_observations = sum(r.total_observations for r in all_records)
        active_alerts = sum(len(r.active_alerts) for r in all_records)

        # Aggregate data sources
        all_sources = set()
        for record in all_records:
            all_sources.update(record.data_sources)

        return {
            "total_patients": len(all_records),
            "total_observations": total_observations,
            "active_alerts": active_alerts,
            "data_sources": [ds.value for ds in all_sources],
            "cached_records": len(self._cache)
        }

    async def cleanup_expired(self) -> int:
        """Clean up records older than retention period."""
        removed = 0
        cutoff = datetime.now() - timedelta(days=self.retention_years * 365)

        for file_path in self.storage_path.glob("*_longitudinal.json"):
            try:
                async with aiofiles.open(file_path, "r") as f:
                    content = await f.read()
                    if not content:
                        continue
                    data = json.loads(content)
                    updated = datetime.fromisoformat(data["updated_at"])

                    if updated < cutoff:
                        # Archive instead of delete
                        archive_path = self.storage_path / "archive" / file_path.name
                        archive_path.parent.mkdir(exist_ok=True)
                        file_path.rename(archive_path)
                        removed += 1
                        logger.info(f"Archived expired record: {file_path.name}")
            except Exception as e:
                logger.error(f"Error checking {file_path}: {e}")

        logger.info(f"Cleaned up {removed} expired records")
        return removed
