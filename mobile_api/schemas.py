from pydantic import BaseModel, Field
from typing import Any, Dict, Literal, Optional, List
from enum import Enum


class MobileQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    language: str = Field(default="en-IN")
    include_context: bool = Field(default=True)
    patient_id: Optional[str] = None


class MobileQueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    evidence_level: str
    emergency_level: str
    confidence: float
    validation_status: str
    disclaimer: Optional[str] = None


class VoiceQueryResponse(MobileQueryResponse):
    transcript: str
    audio_base64: Optional[str] = None


class DeviceRegistrationRequest(BaseModel):
    name: str
    phone_hash: str
    role: str
    language: str
    site_id: str
    pin: str = Field(..., min_length=4, max_length=6)
    abha_id: Optional[str] = None
    digital_literacy_score: Optional[int] = None


class LoginRequest(BaseModel):
    user_id: str
    pin: str


class RefreshRequest(BaseModel):
    """Body of /auth/refresh: the refresh token from the last sign-in or renewal."""
    refresh_token: str


class ReleaseStatusResponse(BaseModel):
    """What the app may know about the running study release (``client_view()``).

    Declared so the contract carries the release gate's fields and the Android
    client is generated against them rather than a hand-kept copy.
    """
    release_id: str
    release_name: Optional[str] = None
    status: str
    activated: bool
    approved: bool
    usable_with_participants: bool
    drifting: bool
    drift: List[dict] = Field(default_factory=list)


class AuthResponse(BaseModel):
    user_id: str
    token: str
    expires_at: float
    refresh_token: str


class SyncPushRequest(BaseModel):
    observations: Optional[List[dict]] = []
    medication_reminders: Optional[List[dict]] = []
    interaction_logs: Optional[List[dict]] = []
    vignette_responses: Optional[List[dict]] = []
    device_timestamp: float


class SyncAcknowledgement(BaseModel):
    """The server's answer for one pushed item.

    The app removes an item from its queue only when its acknowledgement says
    accepted. `index` is the item's position in its list of the push; `id` is
    the id the server stored it under."""
    kind: Literal["observation", "medication_reminder", "interaction_log", "vignette_response"]
    index: int
    id: Optional[str] = None
    status: Literal["accepted", "rejected"]
    reason: Optional[Literal["invalid", "not_owned", "save_failed", "unavailable", "over_limit"]] = None


class SyncPushResponse(BaseModel):
    accepted: int
    rejected: int
    acknowledgements: List[SyncAcknowledgement]
    conflicts: List[dict]


class SyncPullResponse(BaseModel):
    patients: List[dict]
    observations: List[dict]
    care_team_members: List[dict]
    medication_reminders: List[dict]
    query_cache_updates: List[dict]
    vignette_assignments: List[dict]
    server_timestamp: float


class CreateObservationRequest(BaseModel):
    patient_id: str
    category: str
    entity_name: str
    value: Optional[str] = None
    value_text: Optional[str] = None
    severity: Optional[int] = None
    location: Optional[str] = None
    duration: Optional[str] = None
    timestamp: Optional[float] = None


class ObservationResponse(BaseModel):
    observation_id: str
    status: str


class CreateReminderRequest(BaseModel):
    """
    A reminder call from the app. The dosage, if sent, is stored with the
    reminder and never spoken (ADR 0004, decision D10). A call needs a phone
    number; a request without one is refused rather than scheduled.
    """

    patient_id: str
    medication_name: str
    dosage: Optional[str] = None
    phone_number: Optional[str] = None
    scheduled_time: float
    language: str
    frequency: Optional[str] = None


class ReminderResponse(BaseModel):
    reminder_id: str
    status: str


class SusSubmissionRequest(BaseModel):
    scores: List[int] = Field(..., min_length=10, max_length=10)
    site_id: str
    language: str
    completed_at: float


class SusSubmissionResponse(BaseModel):
    submission_id: str
    sus_score: float
    status: str


class VignetteSubmissionRequest(BaseModel):
    vignette_id: str
    with_tool: bool
    response_text: Optional[str] = None
    started_at: float
    completed_at: float
    metadata: Optional[dict] = None


class VignetteSubmissionResponse(BaseModel):
    submission_id: str
    status: str


class InteractionLogBatch(BaseModel):
    logs: List[dict]
    device_id: str
    batch_timestamp: float


class InteractionLogResponse(BaseModel):
    accepted: int
    status: str


class CacheBundleResponse(BaseModel):
    version: str
    language: str
    generated_at: float
    queries: List[dict]
    treatments: List[dict]
    emergency_keywords: dict
    evidence_badge_metadata: dict


class PatientDetailResponse(BaseModel):
    patient: dict
    observations: List[dict]
    care_team: List[dict]
    medication_reminders: List[dict]
    last_updated: float


class AddCareTeamMemberRequest(BaseModel):
    name: str
    role: str
    organization: Optional[str] = None
    phone_number: Optional[str] = None
    primary_contact: bool = False


class FhirImportRequest(BaseModel):
    bundle: dict


class PatientInsightsResponse(BaseModel):
    patient_id: str
    insights: List[dict]
    total: int
    # The Dose Boundary's record for the insight text (None when there was none).
    safety: Optional[Dict[str, Any]] = None


class PatientMemoryQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    # The Supported Language of the question; the Deferral is given in it.
    language: str = Field(default="en")


class PatientMemoryQueryResponse(BaseModel):
    answer: str
    cited_memory_ids: List[str]
    cited_insight_ids: List[str]
    raw_memories_used: int
    insights_used: int
    # The Dose Boundary's record for the answer.
    safety: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    query_id: str
    positive: bool
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    evaluation_id: Optional[str] = None
