from pydantic import BaseModel, Field
from typing import Optional, List
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


class SyncPushResponse(BaseModel):
    accepted: int
    rejected: int
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
    patient_id: str
    medication_name: str
    dosage: str
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
    phone: Optional[str] = None
    relationship: Optional[str] = None


class FhirImportRequest(BaseModel):
    bundle: dict
