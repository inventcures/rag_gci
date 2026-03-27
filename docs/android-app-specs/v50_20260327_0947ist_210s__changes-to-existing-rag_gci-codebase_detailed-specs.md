# Changes to Existing rag_gci Codebase: Detailed Specification

**Version**: 0.1.0
**Date**: 27 March 2026
**Authors**: Ashish Makani, KCDH-A Team
**Status**: Draft — Pre-Implementation Spec
**Companion Document**: `v0_27march2026_0923st_210s_detailed-android-app-specs.md` (Android app specs)
**Reference**: [DeepWiki: rag_gci Architecture](https://deepwiki.com/inventcures/rag_gci)

---

## Table of Contents

1. [Overview and Design Principles](#1-overview-and-design-principles)
2. [New Module: Mobile API (`mobile_api/`)](#2-new-module-mobile-api)
3. [New Module: Authentication (`auth/`)](#3-new-module-authentication)
4. [New Module: Sync Engine (`sync/`)](#4-new-module-sync-engine)
5. [New Module: Evaluation (`evaluation/`)](#5-new-module-evaluation)
6. [New Module: Offline Cache (`offline/`)](#6-new-module-offline-cache)
7. [Changes to Existing Files](#7-changes-to-existing-files)
8. [Configuration Changes](#8-configuration-changes)
9. [New Dependencies](#9-new-dependencies)
10. [Testing Plan](#10-testing-plan)
11. [Migration and Backward Compatibility](#11-migration-and-backward-compatibility)
12. [File Manifest](#12-file-manifest)

---

## 1. Overview and Design Principles

### 1.1 Why These Changes

The existing Palli Sahayak backend (FastAPI, ~30,000 lines Python) serves telephony (Bolna), WhatsApp (Twilio), and web (Gradio) clients. The Android app introduces a fourth client surface with specific needs:

1. **Authentication**: The system currently has no auth. Mobile clients handling PHI need JWT-based auth.
2. **Sync**: Mobile clients work offline and need delta sync with conflict resolution.
3. **Mobile-optimized API**: Existing endpoints are designed for server-to-server or single-request patterns. Mobile needs aggregated responses, batch operations, and compressed payloads.
4. **Evaluation data**: The EVAH study requires SUS scores, vignette responses, and structured interaction logs that the current system doesn't collect.
5. **Offline cache**: Rural/remote sites need pre-computed response bundles downloadable over limited bandwidth.

### 1.2 Design Principles

| Principle | Implementation |
|-----------|----------------|
| **No business logic duplication** | Mobile endpoints are thin wrappers that delegate to existing services |
| **Backward compatible** | All existing endpoints remain unchanged and unauthenticated |
| **Modular** | Each new capability is a separate Python module with clean imports |
| **Maximize reuse** | New code calls existing functions from `simple_rag_server.py`, `safety_enhancements.py`, `personalization/`, etc. |
| **Clean interfaces** | Each module exposes a clear public API via `__init__.py` |
| **Opt-in auth** | Authentication is required only for `/api/mobile/v1/*` routes |
| **File-based storage** (consistent with existing patterns) | New data stored in `data/` subdirectories as JSON, matching the pattern used by `user_profiles/`, `longitudinal/`, etc. |

### 1.3 Architecture After Changes

```
simple_rag_server.py (existing — modified to mount mobile router)
    │
    ├── /api/query, /api/bolna/*, /api/sarvam/*, etc. (UNCHANGED)
    │
    └── /api/mobile/v1/* (NEW — mobile_api/router.py)
            │
            ├── auth/ ──────────────── JWT + PIN authentication
            │     └── calls: auth/jwt_handler.py, auth/pin_auth.py
            │
            ├── query ──────────────── Wraps SimpleRAGPipeline.query()
            │     └── calls: existing RAG pipeline, safety_enhancements.py
            │
            ├── sync/* ─────────────── Delta sync for offline-first
            │     └── calls: sync/delta_tracker.py, sync/conflict_resolver.py
            │     └── reads: personalization/longitudinal_memory.py (modified)
            │
            ├── medication/* ────────── Wraps MedicationVoiceReminderManager
            │     └── calls: existing medication_voice_reminders.py
            │
            ├── patient/* ──────────── Wraps LongitudinalMemoryManager
            │     └── calls: existing personalization/longitudinal_memory.py
            │
            ├── evaluation/* ────────── SUS, vignettes, interaction logs
            │     └── calls: evaluation/sus_collector.py, evaluation/vignette_manager.py
            │
            ├── cache/* ────────────── Offline cache bundles
            │     └── calls: offline/cache_builder.py
            │
            └── fhir/* ────────────── Wraps FHIRAdapter
                  └── calls: existing personalization/fhir_adapter.py
```

### 1.4 Directory Structure After Changes

```
rag_gci/
├── mobile_api/                    # NEW: Mobile-optimized API layer
│   ├── __init__.py
│   ├── router.py                  # FastAPI APIRouter, all /api/mobile/v1/* routes
│   ├── schemas.py                 # Pydantic request/response models
│   ├── dependencies.py            # FastAPI dependencies (auth, user context)
│   └── utils.py                   # Mobile-specific helpers
│
├── auth/                          # NEW: Authentication system
│   ├── __init__.py
│   ├── jwt_handler.py             # JWT creation, verification, refresh
│   ├── pin_auth.py                # PIN hashing (Argon2id) and verification
│   ├── models.py                  # User registration, device, session models
│   ├── middleware.py              # FastAPI dependency for JWT verification
│   └── storage.py                 # File-based credential storage
│
├── sync/                          # NEW: Sync engine
│   ├── __init__.py
│   ├── delta_tracker.py           # Track record modifications
│   ├── conflict_resolver.py       # Last-write-wins with server authority
│   └── batch_processor.py         # Efficient batch upload/download
│
├── evaluation/                    # NEW: EVAH evaluation data
│   ├── __init__.py
│   ├── sus_collector.py           # SUS score collection and analysis
│   ├── vignette_manager.py        # Vignette assignment and response collection
│   ├── interaction_logger.py      # Structured interaction event storage
│   ├── time_motion.py             # Time-motion data analysis
│   └── exporter.py                # CSV/JSON export for R/lme4
│
├── offline/                       # NEW: Offline cache generation
│   ├── __init__.py
│   ├── cache_builder.py           # Generate offline cache bundles
│   └── query_ranker.py            # Identify top queries by frequency
│
├── data/
│   ├── auth/                      # NEW: Auth credential storage
│   │   └── devices/               # Per-device registration files
│   ├── evaluation/                # NEW: Evaluation data storage
│   │   ├── sus_scores/            # SUS score files
│   │   ├── vignette_responses/    # Vignette response files
│   │   ├── interaction_logs/      # Interaction log files
│   │   └── vignettes/             # 20 standardized vignette definitions
│   └── cache/                     # NEW (extends existing): Offline cache bundles
│       └── mobile/                # Pre-computed mobile cache bundles
│
├── simple_rag_server.py           # MODIFIED: Mount mobile router (~20 lines added)
├── personalization/
│   ├── longitudinal_memory.py     # MODIFIED: Add updated_at tracking, delta query methods
│   ├── user_profile.py            # MODIFIED: Add updated_at tracking, delta query methods
│   └── fhir_adapter.py            # UNCHANGED (called via mobile_api)
├── medication_voice_reminders.py  # MODIFIED: Add updated_at tracking, delta query methods
├── safety_enhancements.py         # UNCHANGED (called via mobile_api)
├── analytics/
│   └── usage_analytics.py         # MODIFIED: Add get_top_queries() method
└── config.yaml                    # MODIFIED: Add mobile and evaluation sections
```

---

## 2. New Module: Mobile API (`mobile_api/`)

### 2.1 `mobile_api/__init__.py`

```python
"""
Mobile API Module for Palli Sahayak Android App.

Provides mobile-optimized REST endpoints under /api/mobile/v1/.
All endpoints require JWT authentication.
Delegates to existing services -- no business logic duplication.
"""

from mobile_api.router import mobile_router

__all__ = ["mobile_router"]
```

### 2.2 `mobile_api/router.py`

The central router that mounts all mobile endpoints:

```python
from fastapi import APIRouter, Depends, UploadFile, File, Query
from fastapi.responses import JSONResponse
from typing import Optional

from auth.middleware import require_auth, get_current_user
from mobile_api.schemas import (
    MobileQueryRequest, MobileQueryResponse,
    VoiceQueryResponse,
    SyncPushRequest, SyncPushResponse,
    SyncPullResponse,
    CreateObservationRequest, ObservationResponse,
    CreateReminderRequest, ReminderResponse,
    SusSubmissionRequest, SusSubmissionResponse,
    VignetteSubmissionRequest, VignetteSubmissionResponse,
    InteractionLogBatch, InteractionLogResponse,
    CacheBundleResponse,
    PatientDetailResponse,
)
from mobile_api.dependencies import get_rag_pipeline, get_safety_manager, get_memory_manager

mobile_router = APIRouter(prefix="/api/mobile/v1", tags=["mobile"])


# ── Health ──────────────────────────────────────────────────────────

@mobile_router.get("/health")
async def mobile_health():
    return {"status": "healthy", "api_version": "v1", "module": "mobile"}


# ── Authentication ──────────────────────────────────────────────────

@mobile_router.post("/auth/register")
async def register_device(request: DeviceRegistrationRequest):
    """Register a new device with PIN. Returns JWT token."""
    # Delegates to auth/pin_auth.py and auth/jwt_handler.py
    ...

@mobile_router.post("/auth/login")
async def login(request: LoginRequest):
    """Verify PIN and issue JWT token."""
    ...

@mobile_router.post("/auth/refresh")
async def refresh_token(user = Depends(get_current_user)):
    """Refresh an expiring JWT token."""
    ...


# ── Clinical Query ──────────────────────────────────────────────────

@mobile_router.post("/query", response_model=MobileQueryResponse)
async def mobile_query(
    request: MobileQueryRequest,
    user = Depends(require_auth),
    rag_pipeline = Depends(get_rag_pipeline),
    safety_manager = Depends(get_safety_manager),
):
    """
    Submit a text query. Wraps the existing SimpleRAGPipeline.query().
    Returns response with evidence badge, sources, and validation status.
    """
    result = await rag_pipeline.query(
        query_text=request.query,
        user_id=user.user_id,
        language=request.language,
    )

    safety_result = safety_manager.process_response(
        query=request.query,
        response=result.answer,
        sources=result.sources,
        language=request.language,
    )

    return MobileQueryResponse(
        answer=safety_result.response,
        sources=result.sources,
        evidence_level=safety_result.evidence_level,
        emergency_level=safety_result.emergency_level,
        confidence=safety_result.confidence,
        validation_status=safety_result.validation_status,
        disclaimer=safety_result.disclaimer,
    )


@mobile_router.post("/query/voice", response_model=VoiceQueryResponse)
async def voice_query(
    audio: UploadFile = File(...),
    language: str = Query(...),
    user = Depends(require_auth),
    rag_pipeline = Depends(get_rag_pipeline),
    safety_manager = Depends(get_safety_manager),
):
    """
    Submit a voice query. Processes: STT -> RAG -> TTS.
    Returns text response + base64 audio.
    """
    audio_bytes = await audio.read()

    # 1. STT via Sarvam (reuses existing sarvam_integration)
    transcript = await sarvam_stt(audio_bytes, language)

    # 2. RAG query (reuses existing pipeline)
    result = await rag_pipeline.query(
        query_text=transcript,
        user_id=user.user_id,
        language=language,
    )

    # 3. Safety processing (reuses existing safety_enhancements)
    safety_result = safety_manager.process_response(
        query=transcript,
        response=result.answer,
        sources=result.sources,
        language=language,
    )

    # 4. TTS via Sarvam (reuses existing sarvam_integration)
    audio_response = await sarvam_tts(safety_result.response, language)

    return VoiceQueryResponse(
        transcript=transcript,
        answer=safety_result.response,
        sources=result.sources,
        evidence_level=safety_result.evidence_level,
        emergency_level=safety_result.emergency_level,
        confidence=safety_result.confidence,
        audio_base64=audio_response,
    )


# ── Sync ────────────────────────────────────────────────────────────

@mobile_router.post("/sync/push", response_model=SyncPushResponse)
async def sync_push(
    request: SyncPushRequest,
    user = Depends(require_auth),
):
    """
    Batch upload local changes from the mobile device.
    Accepts observations, medication reminders, interaction logs.
    Returns conflict list (if any).
    """
    # Delegates to sync/batch_processor.py
    ...

@mobile_router.post("/sync/pull", response_model=SyncPullResponse)
async def sync_pull(
    last_sync_at: float = Query(..., description="Unix timestamp of last successful sync"),
    user = Depends(require_auth),
):
    """
    Delta pull: returns all records modified since last_sync_at.
    Includes patients, observations, care team, medication reminders.
    """
    # Delegates to sync/delta_tracker.py
    ...

@mobile_router.get("/sync/status")
async def sync_status(user = Depends(require_auth)):
    """Returns sync health: pending counts, last sync time, conflicts."""
    ...


# ── Patient ─────────────────────────────────────────────────────────

@mobile_router.get("/patient/{patient_id}", response_model=PatientDetailResponse)
async def get_patient(
    patient_id: str,
    user = Depends(require_auth),
    memory_manager = Depends(get_memory_manager),
):
    """
    Get patient with observations, care team, and medication reminders.
    Aggregated response to minimize round trips.
    """
    # Reuses personalization/longitudinal_memory.py
    record = await memory_manager.get_patient_record(patient_id)
    care_team = await memory_manager.get_care_team(patient_id)
    reminders = await get_medication_reminders(patient_id)

    return PatientDetailResponse(
        patient=record.to_dict(),
        observations=record.observations,
        care_team=care_team,
        medication_reminders=reminders,
    )

@mobile_router.get("/patients")
async def list_patients(
    user = Depends(require_auth),
    memory_manager = Depends(get_memory_manager),
):
    """List patients assigned to the authenticated user."""
    ...

@mobile_router.post("/observation", response_model=ObservationResponse)
async def create_observation(
    request: CreateObservationRequest,
    user = Depends(require_auth),
    memory_manager = Depends(get_memory_manager),
):
    """
    Create a new observation. Wraps LongitudinalMemoryManager.add_observation().
    """
    observation = await memory_manager.add_observation(
        patient_id=request.patient_id,
        category=request.category,
        entity_name=request.entity_name,
        value=request.value,
        value_text=request.value_text,
        severity=request.severity,
        source_type="app",
        reported_by=user.role,
    )
    return ObservationResponse(observation_id=observation.observation_id, status="created")


# ── Medication Reminders ────────────────────────────────────────────

@mobile_router.get("/medication/reminders")
async def get_reminders(
    user_id: Optional[str] = None,
    user = Depends(require_auth),
):
    """Get medication reminders for the user or a specific patient."""
    # Reuses medication_voice_reminders.py
    ...

@mobile_router.post("/medication/reminder", response_model=ReminderResponse)
async def create_reminder(
    request: CreateReminderRequest,
    user = Depends(require_auth),
):
    """Create a medication reminder. Wraps MedicationVoiceReminderManager."""
    ...

@mobile_router.get("/medication/adherence/{patient_id}")
async def get_adherence(
    patient_id: str,
    user = Depends(require_auth),
):
    """Get medication adherence stats. Wraps existing adherence endpoint."""
    ...


# ── Care Team ───────────────────────────────────────────────────────

@mobile_router.get("/careteam/{patient_id}")
async def get_care_team(
    patient_id: str,
    user = Depends(require_auth),
    memory_manager = Depends(get_memory_manager),
):
    """Get care team members. Wraps existing care team endpoint."""
    ...

@mobile_router.post("/careteam/{patient_id}/add")
async def add_care_team_member(
    patient_id: str,
    request: AddCareTeamMemberRequest,
    user = Depends(require_auth),
):
    """Add care team member. Wraps existing endpoint."""
    ...


# ── Evaluation ──────────────────────────────────────────────────────

@mobile_router.post("/evaluation/sus", response_model=SusSubmissionResponse)
async def submit_sus(
    request: SusSubmissionRequest,
    user = Depends(require_auth),
):
    """Submit SUS questionnaire scores."""
    # Delegates to evaluation/sus_collector.py
    ...

@mobile_router.get("/evaluation/vignettes")
async def get_assigned_vignettes(user = Depends(require_auth)):
    """Get vignettes assigned to this user (with/without tool assignment)."""
    # Delegates to evaluation/vignette_manager.py
    ...

@mobile_router.post("/evaluation/vignette", response_model=VignetteSubmissionResponse)
async def submit_vignette_response(
    request: VignetteSubmissionRequest,
    user = Depends(require_auth),
):
    """Submit response to a clinical vignette."""
    ...

@mobile_router.post("/evaluation/logs", response_model=InteractionLogResponse)
async def submit_interaction_logs(
    request: InteractionLogBatch,
    user = Depends(require_auth),
):
    """Batch upload interaction logs from the mobile device."""
    # Delegates to evaluation/interaction_logger.py
    ...

@mobile_router.get("/evaluation/export")
async def export_evaluation_data(
    format: str = Query("csv", description="Export format: csv or json"),
    user = Depends(require_auth),
):
    """Export evaluation data for analysis (R/lme4 compatible)."""
    # Delegates to evaluation/exporter.py
    ...


# ── Offline Cache ───────────────────────────────────────────────────

@mobile_router.get("/cache/bundle", response_model=CacheBundleResponse)
async def get_cache_bundle(
    language: str = Query(...),
    user = Depends(require_auth),
):
    """
    Download offline cache bundle for the specified language.
    Contains top 20 queries, top 50 treatments, emergency keywords.
    """
    # Delegates to offline/cache_builder.py
    ...


# ── FHIR ────────────────────────────────────────────────────────────

@mobile_router.get("/fhir/export/{patient_id}")
async def export_fhir(
    patient_id: str,
    user = Depends(require_auth),
):
    """Export patient data as FHIR R4 Bundle. Wraps existing FHIRAdapter."""
    # Reuses personalization/fhir_adapter.py
    ...

@mobile_router.post("/fhir/import")
async def import_fhir(
    request: FhirImportRequest,
    user = Depends(require_auth),
):
    """Import FHIR R4 Bundle. Wraps existing FHIRAdapter."""
    ...
```

### 2.3 `mobile_api/schemas.py`

Pydantic models for request/response validation:

```python
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
    evidence_level: str          # A, B, C, D, E
    emergency_level: str         # none, low, high, critical
    confidence: float            # 0.0 - 1.0
    validation_status: str       # passed, flagged, review_required
    disclaimer: Optional[str] = None


class VoiceQueryResponse(MobileQueryResponse):
    transcript: str
    audio_base64: Optional[str] = None  # TTS audio


class DeviceRegistrationRequest(BaseModel):
    name: str
    phone_hash: str              # SHA-256 hashed phone number
    role: str                    # asha_worker, caregiver, patient
    language: str
    site_id: str                 # cmc_vellore, kmc_manipal, etc.
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
    category: str                # symptom, medication, vital_sign, functional_status, emotional
    entity_name: str
    value: Optional[str] = None
    value_text: Optional[str] = None
    severity: Optional[int] = None  # 0-4
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
    scheduled_time: float        # Unix timestamp
    language: str
    frequency: Optional[str] = None


class ReminderResponse(BaseModel):
    reminder_id: str
    status: str


class SusSubmissionRequest(BaseModel):
    scores: List[int] = Field(..., min_length=10, max_length=10)  # 10 items, each 1-5
    site_id: str
    language: str
    completed_at: float


class SusSubmissionResponse(BaseModel):
    submission_id: str
    sus_score: float             # 0-100
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
    logs: List[dict]             # List of interaction events
    device_id: str
    batch_timestamp: float


class InteractionLogResponse(BaseModel):
    accepted: int
    status: str


class CacheBundleResponse(BaseModel):
    version: str
    language: str
    generated_at: float
    queries: List[dict]          # Top 20 pre-computed query-response pairs
    treatments: List[dict]       # Top 50 symptom-treatment pairs
    emergency_keywords: dict     # Language -> keyword list
    evidence_badge_metadata: dict


class PatientDetailResponse(BaseModel):
    patient: dict
    observations: List[dict]
    care_team: List[dict]
    medication_reminders: List[dict]
    last_updated: float
```

### 2.4 `mobile_api/dependencies.py`

FastAPI dependency injection for shared resources:

```python
from fastapi import Depends

_rag_pipeline = None
_safety_manager = None
_memory_manager = None
_medication_manager = None


def init_dependencies(rag_pipeline, safety_manager, memory_manager, medication_manager):
    global _rag_pipeline, _safety_manager, _memory_manager, _medication_manager
    _rag_pipeline = rag_pipeline
    _safety_manager = safety_manager
    _memory_manager = memory_manager
    _medication_manager = medication_manager


def get_rag_pipeline():
    return _rag_pipeline


def get_safety_manager():
    return _safety_manager


def get_memory_manager():
    return _memory_manager


def get_medication_manager():
    return _medication_manager
```

---

## 3. New Module: Authentication (`auth/`)

### 3.1 `auth/__init__.py`

```python
"""
Authentication module for Palli Sahayak mobile API.

Provides JWT-based authentication with PIN verification.
Auth is opt-in: only /api/mobile/v1/* endpoints require it.
Existing endpoints remain unauthenticated for backward compatibility.
"""

from auth.jwt_handler import JWTHandler
from auth.pin_auth import PinAuthenticator
from auth.middleware import require_auth, get_current_user
from auth.models import DeviceRegistration, AuthenticatedUser

__all__ = [
    "JWTHandler",
    "PinAuthenticator",
    "require_auth",
    "get_current_user",
    "DeviceRegistration",
    "AuthenticatedUser",
]
```

### 3.2 `auth/jwt_handler.py`

```python
import jwt
import time
import secrets
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenPair:
    access_token: str
    refresh_token: str
    expires_at: float


class JWTHandler:
    def __init__(self, secret: str, expiry_hours: int = 72):
        self.secret = secret
        self.expiry_seconds = expiry_hours * 3600
        self.algorithm = "HS256"

    def create_token_pair(self, user_id: str, role: str, site_id: str) -> TokenPair:
        now = time.time()
        expires_at = now + self.expiry_seconds

        access_payload = {
            "user_id": user_id,
            "role": role,
            "site_id": site_id,
            "iat": now,
            "exp": expires_at,
            "type": "access",
        }
        access_token = jwt.encode(access_payload, self.secret, algorithm=self.algorithm)

        refresh_payload = {
            "user_id": user_id,
            "iat": now,
            "exp": now + (self.expiry_seconds * 2),  # Refresh token lasts 2x
            "type": "refresh",
            "jti": secrets.token_hex(16),
        }
        refresh_token = jwt.encode(refresh_payload, self.secret, algorithm=self.algorithm)

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
        )

    def verify_token(self, token: str) -> Optional[dict]:
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None

    def refresh_access_token(self, refresh_token: str) -> Optional[TokenPair]:
        payload = self.verify_token(refresh_token)
        if payload and payload.get("type") == "refresh":
            return self.create_token_pair(
                user_id=payload["user_id"],
                role=payload.get("role", ""),
                site_id=payload.get("site_id", ""),
            )
        return None
```

### 3.3 `auth/pin_auth.py`

```python
import hashlib
import secrets
import json
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

DATA_DIR = Path("./data/auth/devices")


class PinAuthenticator:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def register_device(
        self,
        user_id: str,
        pin: str,
        name: str,
        phone_hash: str,
        role: str,
        language: str,
        site_id: str,
        abha_id: Optional[str] = None,
        digital_literacy_score: Optional[int] = None,
    ) -> dict:
        salt = secrets.token_hex(16)
        pin_hash = self._hash_pin(pin, salt)

        device_record = {
            "user_id": user_id,
            "name": name,
            "phone_hash": phone_hash,
            "role": role,
            "language": language,
            "site_id": site_id,
            "abha_id": abha_id,
            "digital_literacy_score": digital_literacy_score,
            "pin_hash": pin_hash,
            "pin_salt": salt,
            "failed_attempts": 0,
            "created_at": __import__("time").time(),
            "last_login_at": None,
        }

        filepath = DATA_DIR / f"{user_id}.json"
        with open(filepath, "w") as f:
            json.dump(device_record, f, indent=2)

        logger.info(f"Registered device for user {user_id} at site {site_id}")
        return {"user_id": user_id, "status": "registered"}

    def verify_pin(self, user_id: str, pin: str) -> bool:
        filepath = DATA_DIR / f"{user_id}.json"
        if not filepath.exists():
            return False

        with open(filepath) as f:
            record = json.load(f)

        if record.get("failed_attempts", 0) >= 5:
            logger.warning(f"User {user_id} locked out after 5 failed attempts")
            return False

        expected_hash = record["pin_hash"]
        salt = record["pin_salt"]
        actual_hash = self._hash_pin(pin, salt)

        if actual_hash == expected_hash:
            record["failed_attempts"] = 0
            record["last_login_at"] = __import__("time").time()
            with open(filepath, "w") as f:
                json.dump(record, f, indent=2)
            return True
        else:
            record["failed_attempts"] = record.get("failed_attempts", 0) + 1
            with open(filepath, "w") as f:
                json.dump(record, f, indent=2)
            return False

    def get_user_record(self, user_id: str) -> Optional[dict]:
        filepath = DATA_DIR / f"{user_id}.json"
        if not filepath.exists():
            return None
        with open(filepath) as f:
            record = json.load(f)
        record.pop("pin_hash", None)
        record.pop("pin_salt", None)
        return record

    def _hash_pin(self, pin: str, salt: str) -> str:
        return hashlib.pbkdf2_hmac(
            "sha256",
            pin.encode("utf-8"),
            salt.encode("utf-8"),
            iterations=100_000,
        ).hex()
```

### 3.4 `auth/middleware.py`

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dataclasses import dataclass
from typing import Optional

from auth.jwt_handler import JWTHandler

security = HTTPBearer()

_jwt_handler: Optional[JWTHandler] = None


def init_auth(jwt_secret: str, jwt_expiry_hours: int = 72):
    global _jwt_handler
    _jwt_handler = JWTHandler(secret=jwt_secret, expiry_hours=jwt_expiry_hours)


@dataclass
class AuthenticatedUser:
    user_id: str
    role: str
    site_id: str


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> AuthenticatedUser:
    if _jwt_handler is None:
        raise HTTPException(status_code=500, detail="Auth not initialized")

    payload = _jwt_handler.verify_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    return AuthenticatedUser(
        user_id=payload["user_id"],
        role=payload.get("role", ""),
        site_id=payload.get("site_id", ""),
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> AuthenticatedUser:
    return await require_auth(credentials)
```

### 3.5 `auth/models.py`

```python
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class DeviceRegistration:
    user_id: str
    name: str
    phone_hash: str
    role: str
    language: str
    site_id: str
    abha_id: Optional[str] = None
    digital_literacy_score: Optional[int] = None
    registered_at: float = field(default_factory=time.time)
```

---

## 4. New Module: Sync Engine (`sync/`)

### 4.1 `sync/__init__.py`

```python
"""
Sync engine for Palli Sahayak mobile clients.

Provides delta sync (push/pull) with conflict resolution.
Designed for offline-first mobile clients with intermittent connectivity.
"""

from sync.delta_tracker import DeltaTracker
from sync.conflict_resolver import ConflictResolver
from sync.batch_processor import BatchProcessor

__all__ = ["DeltaTracker", "ConflictResolver", "BatchProcessor"]
```

### 4.2 `sync/delta_tracker.py`

```python
import time
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class DeltaTracker:
    """Tracks record modifications and provides delta queries."""

    def __init__(self, memory_manager, medication_manager, user_profile_manager):
        self.memory_manager = memory_manager
        self.medication_manager = medication_manager
        self.user_profile_manager = user_profile_manager

    async def get_changes_since(self, user_id: str, since_timestamp: float) -> dict:
        """
        Returns all records modified after since_timestamp.
        This is the core of the delta pull sync.
        """
        patients = await self.memory_manager.get_patients_modified_since(
            user_id=user_id,
            since=since_timestamp,
        )

        observations = await self.memory_manager.get_observations_modified_since(
            user_id=user_id,
            since=since_timestamp,
        )

        care_team = await self.memory_manager.get_care_team_modified_since(
            user_id=user_id,
            since=since_timestamp,
        )

        reminders = await self.medication_manager.get_reminders_modified_since(
            user_id=user_id,
            since=since_timestamp,
        )

        return {
            "patients": [p.to_dict() for p in patients],
            "observations": [o.to_dict() for o in observations],
            "care_team_members": [c.to_dict() for c in care_team],
            "medication_reminders": [r.to_dict() for r in reminders],
            "server_timestamp": time.time(),
        }
```

### 4.3 `sync/conflict_resolver.py`

```python
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ConflictStrategy(Enum):
    SERVER_WINS = "server_wins"
    CLIENT_WINS = "client_wins"
    APPEND_ONLY = "append_only"


ENTITY_STRATEGIES = {
    "patient": ConflictStrategy.SERVER_WINS,
    "observation": ConflictStrategy.APPEND_ONLY,
    "care_team_member": ConflictStrategy.SERVER_WINS,
    "medication_reminder_schedule": ConflictStrategy.CLIENT_WINS,
    "medication_reminder_status": ConflictStrategy.SERVER_WINS,
    "interaction_log": ConflictStrategy.APPEND_ONLY,
    "vignette_response": ConflictStrategy.CLIENT_WINS,
}


class ConflictResolver:

    def resolve(self, entity_type: str, server_record: dict, client_record: dict) -> dict:
        strategy = ENTITY_STRATEGIES.get(entity_type, ConflictStrategy.SERVER_WINS)

        if strategy == ConflictStrategy.APPEND_ONLY:
            return client_record  # Always accept client appends

        if strategy == ConflictStrategy.CLIENT_WINS:
            return client_record

        if strategy == ConflictStrategy.SERVER_WINS:
            server_ts = server_record.get("updated_at", 0)
            client_ts = client_record.get("updated_at", 0)
            if client_ts > server_ts:
                logger.info(f"Conflict on {entity_type}: client newer but server wins per policy")
            return server_record

        return server_record
```

### 4.4 `sync/batch_processor.py`

```python
import time
from typing import List
import logging

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 50


class BatchProcessor:
    """Processes batch uploads from mobile devices efficiently."""

    def __init__(self, memory_manager, medication_manager, evaluation_logger):
        self.memory_manager = memory_manager
        self.medication_manager = medication_manager
        self.evaluation_logger = evaluation_logger

    async def process_push(self, push_data: dict, user_id: str) -> dict:
        accepted = 0
        rejected = 0
        conflicts = []

        for obs in (push_data.get("observations") or [])[:MAX_BATCH_SIZE]:
            try:
                await self.memory_manager.add_observation_from_sync(obs, source="mobile_sync")
                accepted += 1
            except Exception as e:
                logger.warning(f"Rejected observation {obs.get('observation_id')}: {e}")
                rejected += 1

        for reminder in (push_data.get("medication_reminders") or [])[:MAX_BATCH_SIZE]:
            try:
                await self.medication_manager.upsert_reminder_from_sync(reminder)
                accepted += 1
            except Exception as e:
                rejected += 1

        for log_entry in (push_data.get("interaction_logs") or [])[:MAX_BATCH_SIZE]:
            try:
                await self.evaluation_logger.ingest_log(log_entry, user_id=user_id)
                accepted += 1
            except Exception as e:
                rejected += 1

        for vignette in (push_data.get("vignette_responses") or [])[:MAX_BATCH_SIZE]:
            try:
                await self.evaluation_logger.ingest_vignette_response(vignette, user_id=user_id)
                accepted += 1
            except Exception as e:
                rejected += 1

        return {
            "accepted": accepted,
            "rejected": rejected,
            "conflicts": conflicts,
            "server_timestamp": time.time(),
        }
```

---

## 5. New Module: Evaluation (`evaluation/`)

### 5.1 `evaluation/__init__.py`

```python
"""
EVAH Evaluation Data Module for Palli Sahayak.

Collects and manages evaluation-specific data:
- SUS (System Usability Scale) scores
- Clinical vignette crossover responses
- Structured interaction logs
- Time-motion data
- CSV/JSON export for statistical analysis (R/lme4)
"""

from evaluation.sus_collector import SusCollector
from evaluation.vignette_manager import VignetteManager
from evaluation.interaction_logger import MobileInteractionLogger
from evaluation.exporter import EvaluationExporter

__all__ = ["SusCollector", "VignetteManager", "MobileInteractionLogger", "EvaluationExporter"]
```

### 5.2 `evaluation/sus_collector.py`

```python
import json
import time
import uuid
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

DATA_DIR = Path("./data/evaluation/sus_scores")


class SusCollector:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def submit_score(
        self,
        user_id: str,
        scores: List[int],
        site_id: str,
        language: str,
        completed_at: float,
    ) -> dict:
        if len(scores) != 10 or not all(1 <= s <= 5 for s in scores):
            raise ValueError("SUS requires exactly 10 scores, each 1-5")

        sus_score = self._calculate_sus(scores)
        submission_id = str(uuid.uuid4())

        record = {
            "submission_id": submission_id,
            "user_id": user_id,
            "scores": scores,
            "sus_score": sus_score,
            "site_id": site_id,
            "language": language,
            "completed_at": completed_at,
            "submitted_at": time.time(),
        }

        filepath = DATA_DIR / f"{submission_id}.json"
        with open(filepath, "w") as f:
            json.dump(record, f, indent=2)

        logger.info(f"SUS score submitted: {sus_score:.1f} by user {user_id} at {site_id}")
        return {"submission_id": submission_id, "sus_score": sus_score, "status": "accepted"}

    def _calculate_sus(self, scores: List[int]) -> float:
        """Standard SUS calculation: odd items (score-1), even items (5-score), sum * 2.5"""
        adjusted = []
        for i, score in enumerate(scores):
            if (i + 1) % 2 == 1:  # Odd items (1,3,5,7,9): positive
                adjusted.append(score - 1)
            else:                  # Even items (2,4,6,8,10): negative
                adjusted.append(5 - score)
        return sum(adjusted) * 2.5

    def get_scores_by_site(self, site_id: str) -> List[dict]:
        results = []
        for filepath in DATA_DIR.glob("*.json"):
            with open(filepath) as f:
                record = json.load(f)
            if record.get("site_id") == site_id:
                results.append(record)
        return sorted(results, key=lambda r: r["submitted_at"])
```

### 5.3 `evaluation/vignette_manager.py`

```python
import json
import time
import uuid
import random
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

VIGNETTES_DIR = Path("./data/evaluation/vignettes")
RESPONSES_DIR = Path("./data/evaluation/vignette_responses")
ASSIGNMENTS_DIR = Path("./data/evaluation/vignette_assignments")


class VignetteManager:
    """
    Manages the EVAH clinical vignette crossover assessment.

    Per EVAH section 6.4:
    - 20 standardized vignettes per participant
    - 10 with Palli Sahayak, 10 without
    - Randomized, counterbalanced assignment
    """

    def __init__(self):
        VIGNETTES_DIR.mkdir(parents=True, exist_ok=True)
        RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
        ASSIGNMENTS_DIR.mkdir(parents=True, exist_ok=True)

    def get_vignettes(self) -> List[dict]:
        vignettes = []
        for filepath in sorted(VIGNETTES_DIR.glob("*.json")):
            with open(filepath) as f:
                vignettes.append(json.load(f))
        return vignettes

    def assign_vignettes(self, user_id: str) -> List[dict]:
        """
        Assign 20 vignettes with randomized counterbalanced with/without-tool split.
        Returns list of {vignette_id, with_tool} assignments.
        """
        assignment_path = ASSIGNMENTS_DIR / f"{user_id}.json"
        if assignment_path.exists():
            with open(assignment_path) as f:
                return json.load(f)["assignments"]

        vignettes = self.get_vignettes()
        if len(vignettes) < 20:
            logger.warning(f"Only {len(vignettes)} vignettes available, expected 20")

        vignette_ids = [v["vignette_id"] for v in vignettes]
        random.shuffle(vignette_ids)

        assignments = []
        for i, vid in enumerate(vignette_ids):
            assignments.append({
                "vignette_id": vid,
                "with_tool": i < 10,  # First 10 with tool, last 10 without
                "order": i + 1,
            })

        random.shuffle(assignments)

        record = {
            "user_id": user_id,
            "assignments": assignments,
            "assigned_at": time.time(),
        }
        with open(assignment_path, "w") as f:
            json.dump(record, f, indent=2)

        return assignments

    def submit_response(
        self,
        user_id: str,
        vignette_id: str,
        with_tool: bool,
        response_text: Optional[str],
        started_at: float,
        completed_at: float,
        metadata: Optional[dict] = None,
    ) -> dict:
        response_id = str(uuid.uuid4())
        record = {
            "response_id": response_id,
            "user_id": user_id,
            "vignette_id": vignette_id,
            "with_tool": with_tool,
            "response_text": response_text,
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_ms": int((completed_at - started_at) * 1000),
            "metadata": metadata or {},
            "submitted_at": time.time(),
        }

        filepath = RESPONSES_DIR / f"{response_id}.json"
        with open(filepath, "w") as f:
            json.dump(record, f, indent=2)

        logger.info(f"Vignette response: {vignette_id} by {user_id} (with_tool={with_tool})")
        return {"submission_id": response_id, "status": "accepted"}
```

### 5.4 `evaluation/interaction_logger.py`

```python
import json
import time
import uuid
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

LOGS_DIR = Path("./data/evaluation/interaction_logs")


class MobileInteractionLogger:
    """
    Stores structured interaction events from mobile devices.
    Events are append-only and synced in batches.
    """

    def __init__(self):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    async def ingest_log(self, log_entry: dict, user_id: str) -> str:
        log_id = log_entry.get("log_id", str(uuid.uuid4()))
        record = {
            "log_id": log_id,
            "user_id": user_id,
            "session_id": log_entry.get("session_id", ""),
            "event_type": log_entry.get("event_type", ""),
            "event_data": log_entry.get("event_data"),
            "timestamp": log_entry.get("timestamp", time.time()),
            "duration_ms": log_entry.get("duration_ms"),
            "language": log_entry.get("language", ""),
            "site_id": log_entry.get("site_id", ""),
            "is_offline": log_entry.get("is_offline", False),
            "ingested_at": time.time(),
        }

        date_str = time.strftime("%Y-%m-%d", time.gmtime(record["timestamp"]))
        day_dir = LOGS_DIR / date_str
        day_dir.mkdir(exist_ok=True)

        filepath = day_dir / f"{log_id}.json"
        with open(filepath, "w") as f:
            json.dump(record, f)

        return log_id

    async def ingest_batch(self, logs: List[dict], user_id: str) -> int:
        count = 0
        for log_entry in logs:
            await self.ingest_log(log_entry, user_id)
            count += 1
        return count

    async def ingest_vignette_response(self, response: dict, user_id: str) -> str:
        response_id = response.get("response_id", str(uuid.uuid4()))
        filepath = Path("./data/evaluation/vignette_responses") / f"{response_id}.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        response["user_id"] = user_id
        response["ingested_at"] = time.time()
        with open(filepath, "w") as f:
            json.dump(response, f, indent=2)
        return response_id
```

### 5.5 `evaluation/exporter.py`

```python
import csv
import json
import io
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


class EvaluationExporter:
    """Exports evaluation data in formats compatible with R/lme4 analysis."""

    def export_sus_csv(self) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "submission_id", "user_id", "site_id", "language",
            "item_1", "item_2", "item_3", "item_4", "item_5",
            "item_6", "item_7", "item_8", "item_9", "item_10",
            "sus_score", "completed_at"
        ])

        sus_dir = Path("./data/evaluation/sus_scores")
        for filepath in sorted(sus_dir.glob("*.json")):
            with open(filepath) as f:
                record = json.load(f)
            scores = record["scores"]
            writer.writerow([
                record["submission_id"], record["user_id"],
                record["site_id"], record["language"],
                *scores,
                record["sus_score"], record["completed_at"],
            ])

        return output.getvalue()

    def export_vignettes_csv(self) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "response_id", "user_id", "vignette_id", "with_tool",
            "response_text", "started_at", "completed_at", "duration_ms",
            "submitted_at"
        ])

        responses_dir = Path("./data/evaluation/vignette_responses")
        for filepath in sorted(responses_dir.glob("*.json")):
            with open(filepath) as f:
                record = json.load(f)
            writer.writerow([
                record["response_id"], record["user_id"],
                record["vignette_id"], record["with_tool"],
                record.get("response_text", ""),
                record["started_at"], record["completed_at"],
                record.get("duration_ms", ""),
                record.get("submitted_at", ""),
            ])

        return output.getvalue()

    def export_interaction_logs_csv(self, start_date: str = None, end_date: str = None) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "log_id", "user_id", "session_id", "event_type",
            "timestamp", "duration_ms", "language", "site_id", "is_offline"
        ])

        logs_dir = Path("./data/evaluation/interaction_logs")
        for day_dir in sorted(logs_dir.iterdir()):
            if not day_dir.is_dir():
                continue
            if start_date and day_dir.name < start_date:
                continue
            if end_date and day_dir.name > end_date:
                continue
            for filepath in sorted(day_dir.glob("*.json")):
                with open(filepath) as f:
                    record = json.load(f)
                writer.writerow([
                    record["log_id"], record["user_id"],
                    record.get("session_id", ""),
                    record["event_type"],
                    record["timestamp"],
                    record.get("duration_ms", ""),
                    record.get("language", ""),
                    record.get("site_id", ""),
                    record.get("is_offline", False),
                ])

        return output.getvalue()
```

---

## 6. New Module: Offline Cache (`offline/`)

### 6.1 `offline/__init__.py`

```python
"""
Offline cache module for Palli Sahayak mobile clients.

Generates pre-computed cache bundles containing:
- Top 20 clinical queries with responses (in all supported languages)
- Top 50 symptom-treatment pairs from knowledge graph
- Emergency keywords in all supported languages
"""

from offline.cache_builder import CacheBundleBuilder

__all__ = ["CacheBundleBuilder"]
```

### 6.2 `offline/cache_builder.py`

```python
import json
import time
import hashlib
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

CACHE_DIR = Path("./data/cache/mobile")

SUPPORTED_LANGUAGES = ["ta-IN", "te-IN", "kn-IN", "ml-IN", "bn-IN", "as-IN", "hi-IN", "en-IN"]

EMERGENCY_KEYWORDS = {
    "en": ["bleeding", "unconscious", "not breathing", "chest pain", "seizure", "suicide", "severe pain", "choking"],
    "hi": ["खून बह रहा", "बेहोश", "सांस नहीं", "छाती में दर्द", "दौरा", "तेज दर्द"],
    "ta": ["இரத்தப்போக்கு", "மயக்கம்", "மூச்சு விடவில்லை", "நெஞ்சு வலி", "வலிப்பு"],
    "bn": ["রক্তপাত", "অচেতন", "শ্বাস নেই", "বুকে ব্যথা", "খিঁচুনি"],
    "kn": ["ರಕ್ತಸ್ರಾವ", "ಪ್ರಜ್ಞೆ ತಪ್ಪಿದ", "ಉಸಿರಾಟ ಇಲ್ಲ", "ಎದೆ ನೋವು"],
    "ml": ["രക്തസ്രാവം", "ബോധം ഇല്ല", "ശ്വാസം ഇല്ല", "നെഞ്ചുവേദന"],
    "te": ["రక్తస్రావం", "స్పృహ లేదు", "ఊపిరి ఆడటం లేదు", "ఛాతీ నొప్పి"],
    "as": ["ৰক্তক্ষৰণ", "অচেতন", "উশাহ নাই", "বুকুৰ বিষ"],
}

DEFAULT_TOP_QUERIES = [
    "How to manage pain at home",
    "What to do for nausea and vomiting",
    "How to manage breathlessness",
    "Morphine dosage and side effects",
    "How to manage constipation from opioids",
    "Signs of emergency that need hospital",
    "How to help with anxiety and fear",
    "What to do when patient cannot eat",
    "How to manage mouth sores",
    "How to help patient sleep better",
    "What to do for bedsores",
    "How to manage fever at home",
    "When to call the doctor",
    "How to give emotional support to patient",
    "What to tell family about prognosis",
    "How to manage swelling in legs",
    "Pain assessment for non-verbal patient",
    "How to manage secretions at end of life",
    "Caregiver self-care and burnout",
    "Medication schedule management",
]


class CacheBundleBuilder:
    def __init__(self, rag_pipeline=None, kg_client=None, usage_analytics=None):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.rag_pipeline = rag_pipeline
        self.kg_client = kg_client
        self.usage_analytics = usage_analytics

    async def build_bundle(self, language: str) -> dict:
        """Build a complete offline cache bundle for the given language."""
        bundle_version = time.strftime("%Y%m%d-%H%M%S")

        queries = await self._build_query_cache(language)
        treatments = await self._build_treatment_cache()

        bundle = {
            "version": bundle_version,
            "language": language,
            "generated_at": time.time(),
            "queries": queries,
            "treatments": treatments,
            "emergency_keywords": EMERGENCY_KEYWORDS,
            "evidence_badge_metadata": {
                "A": {"label": "Strong Evidence", "color": "#1B5E20"},
                "B": {"label": "Good Evidence", "color": "#2E7D32"},
                "C": {"label": "Moderate Evidence", "color": "#F57F17"},
                "D": {"label": "Limited Evidence", "color": "#E65100"},
                "E": {"label": "Insufficient Evidence", "color": "#C62828"},
            },
        }

        filepath = CACHE_DIR / f"bundle_{language}_{bundle_version}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False, indent=2)

        logger.info(f"Built cache bundle for {language}: {len(queries)} queries, {len(treatments)} treatments")
        return bundle

    async def _build_query_cache(self, language: str) -> List[dict]:
        """Pre-compute responses for top 20 queries."""
        queries = DEFAULT_TOP_QUERIES

        if self.usage_analytics:
            try:
                top = await self.usage_analytics.get_top_queries(n=20, language=language)
                if top:
                    queries = top
            except Exception:
                pass

        cached_queries = []
        for query_text in queries:
            try:
                if self.rag_pipeline:
                    result = await self.rag_pipeline.query(
                        query_text=query_text,
                        language=language,
                    )
                    cached_queries.append({
                        "query_hash": hashlib.sha256(query_text.lower().strip().encode()).hexdigest(),
                        "query_text": query_text,
                        "response_text": result.answer,
                        "evidence_level": result.evidence_level if hasattr(result, "evidence_level") else "C",
                        "sources": result.sources if hasattr(result, "sources") else [],
                    })
            except Exception as e:
                logger.warning(f"Failed to cache query '{query_text}': {e}")

        return cached_queries

    async def _build_treatment_cache(self) -> List[dict]:
        """Cache top 50 symptom-treatment pairs from knowledge graph."""
        common_symptoms = [
            "pain", "nausea", "vomiting", "breathlessness", "constipation",
            "anxiety", "depression", "insomnia", "fatigue", "appetite_loss",
            "mouth_sores", "fever", "cough", "diarrhea", "edema",
            "itching", "confusion", "delirium", "hiccups", "bleeding",
        ]

        treatments = []
        if self.kg_client:
            for symptom in common_symptoms:
                try:
                    result = await self.kg_client.get_treatments(symptom)
                    if result:
                        treatments.append({
                            "symptom": symptom,
                            "treatments": result,
                        })
                except Exception:
                    pass

        return treatments
```

### 6.3 `offline/query_ranker.py`

```python
from typing import List, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class QueryRanker:
    """Ranks queries by frequency to determine top-N for caching."""

    def __init__(self, usage_analytics):
        self.usage_analytics = usage_analytics

    async def get_top_queries(self, n: int = 20, language: Optional[str] = None) -> List[str]:
        """
        Returns the top N most frequent queries.
        Falls back to DEFAULT_TOP_QUERIES if analytics unavailable.
        """
        try:
            if hasattr(self.usage_analytics, "get_query_frequencies"):
                frequencies = await self.usage_analytics.get_query_frequencies(
                    days=30,
                    language=language,
                )
                if frequencies:
                    sorted_queries = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
                    return [q for q, _ in sorted_queries[:n]]
        except Exception as e:
            logger.warning(f"Could not get query frequencies: {e}")

        return None
```

---

## 7. Changes to Existing Files

### 7.1 `simple_rag_server.py` — Mount Mobile Router

**Location**: Near the top of the file where other routers are included (around lines 100-150).

**Changes** (additive only, ~25 lines):

```python
# ADD after existing router imports:
from mobile_api.router import mobile_router
from mobile_api.dependencies import init_dependencies as init_mobile_deps
from auth.middleware import init_auth

# ADD in the startup/lifespan function, after existing initialization:
mobile_config = config.get("mobile", {})
if mobile_config.get("enabled", False):
    jwt_secret = os.environ.get("MOBILE_JWT_SECRET", mobile_config.get("jwt_secret", ""))
    if jwt_secret:
        init_auth(jwt_secret=jwt_secret, jwt_expiry_hours=mobile_config.get("jwt_expiry_hours", 72))
        init_mobile_deps(
            rag_pipeline=rag_pipeline,
            safety_manager=safety_manager,
            memory_manager=memory_manager,
            medication_manager=medication_manager,
        )
        app.include_router(mobile_router)
        logger.info("Mobile API v1 enabled at /api/mobile/v1/")
    else:
        logger.warning("Mobile API disabled: MOBILE_JWT_SECRET not set")

# ADD CORS update for mobile app:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Impact**: ~25 lines added. No existing code modified. Mobile API only loads if `mobile.enabled: true` in config.

### 7.2 `personalization/longitudinal_memory.py` — Add Delta Sync Support

**Changes**: Add `updated_at` field to observations and delta query methods.

**Modification 1**: In the observation creation methods, add `updated_at` timestamp:

```python
# In add_observation() method (and similar methods that create observations):
# ADD to the observation dict before saving:
observation["updated_at"] = time.time()
```

**Modification 2**: Add new methods for delta sync:

```python
# ADD these methods to LongitudinalMemoryManager class:

async def get_observations_modified_since(self, user_id: str, since: float) -> list:
    """Return observations modified after the given timestamp."""
    patient_ids = await self._get_patient_ids_for_user(user_id)
    results = []
    for patient_id in patient_ids:
        record = await self.get_patient_record(patient_id)
        if record:
            for obs in record.observations:
                obs_dict = obs.to_dict() if hasattr(obs, "to_dict") else obs
                if obs_dict.get("updated_at", 0) > since:
                    results.append(obs_dict)
    return results

async def get_patients_modified_since(self, user_id: str, since: float) -> list:
    """Return patient records modified after the given timestamp."""
    patient_ids = await self._get_patient_ids_for_user(user_id)
    results = []
    for patient_id in patient_ids:
        record = await self.get_patient_record(patient_id)
        if record:
            record_dict = record.to_dict() if hasattr(record, "to_dict") else record
            if record_dict.get("updated_at", 0) > since:
                results.append(record_dict)
    return results

async def get_care_team_modified_since(self, user_id: str, since: float) -> list:
    """Return care team members modified after the given timestamp."""
    patient_ids = await self._get_patient_ids_for_user(user_id)
    results = []
    for patient_id in patient_ids:
        team = await self.get_care_team(patient_id)
        for member in team:
            member_dict = member.to_dict() if hasattr(member, "to_dict") else member
            if member_dict.get("updated_at", 0) > since:
                results.append(member_dict)
    return results

async def add_observation_from_sync(self, observation_data: dict, source: str = "mobile_sync"):
    """Ingest an observation from mobile sync. Deduplicates by observation_id."""
    obs_id = observation_data.get("observation_id")
    if not obs_id:
        return
    existing = await self._get_observation_by_id(obs_id)
    if existing:
        return  # Already exists, skip (append-only, no updates)
    observation_data["source_type"] = source
    observation_data["updated_at"] = time.time()
    await self._save_observation(observation_data)
```

**Impact**: ~60 lines added to `longitudinal_memory.py`. No existing methods modified.

### 7.3 `personalization/user_profile.py` — Add Delta Sync Support

**Changes**: Add `updated_at` tracking and delta query method.

```python
# ADD to UserProfileManager class:

async def get_profile_changes_since(self, user_id: str, since: float) -> Optional[dict]:
    """Return user profile if modified after the given timestamp."""
    profile = await self.get_profile(user_id)
    if profile:
        profile_dict = profile.to_dict() if hasattr(profile, "to_dict") else profile
        if profile_dict.get("updated_at", 0) > since:
            return profile_dict
    return None
```

**Impact**: ~10 lines added.

### 7.4 `medication_voice_reminders.py` — Add Delta Sync Support

**Changes**: Add `updated_at` tracking and delta query methods.

```python
# ADD to MedicationVoiceReminderManager class:

async def get_reminders_modified_since(self, user_id: str, since: float) -> list:
    """Return reminders modified after the given timestamp."""
    all_reminders = await self.get_reminders(user_id)
    return [r for r in all_reminders if r.get("updated_at", 0) > since]

async def upsert_reminder_from_sync(self, reminder_data: dict):
    """Upsert a reminder from mobile sync."""
    reminder_data["updated_at"] = time.time()
    reminder_id = reminder_data.get("reminder_id")
    if reminder_id:
        existing = await self._get_reminder_by_id(reminder_id)
        if existing:
            existing.update(reminder_data)
            await self._save_reminder(existing)
        else:
            await self._save_reminder(reminder_data)
```

**Impact**: ~20 lines added.

### 7.5 `analytics/usage_analytics.py` — Add Top Queries Method

```python
# ADD to UsageAnalytics class:

async def get_top_queries(self, n: int = 20, language: Optional[str] = None) -> Optional[list]:
    """Return the top N most frequent queries from the last 30 days."""
    try:
        query_counts = {}
        # Aggregate from interaction logs or existing analytics data
        # ... implementation depends on existing analytics storage format
        sorted_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
        return [q for q, _ in sorted_queries[:n]]
    except Exception:
        return None
```

**Impact**: ~15 lines added.

---

## 8. Configuration Changes

### 8.1 `config.yaml` — Add Mobile and Evaluation Sections

```yaml
# ADD to config.yaml:

mobile:
  enabled: true
  jwt_expiry_hours: 72
  pin_min_length: 4
  cache_bundle_refresh_hours: 24
  sync_batch_max_size: 50
  max_offline_days: 30
  cors_origins:
    - "*"

evaluation:
  enabled: true
  sus_collection_months: [4, 8]
  vignette_count: 20
  vignettes_with_tool: 10
  vignettes_without_tool: 10
  interaction_log_retention_days: 365
  export_formats: ["csv", "json"]
```

### 8.2 `.env` — Add Mobile JWT Secret

```bash
# ADD to .env:
MOBILE_JWT_SECRET=<generate-a-strong-random-secret>
```

### 8.3 `requirements.txt` — Add New Dependencies

```
# ADD to requirements.txt:
PyJWT>=2.8.0
```

---

## 9. New Dependencies

| Package | Version | Purpose | Impact |
|---------|---------|---------|--------|
| PyJWT | >=2.8.0 | JWT token creation/verification | Lightweight, no transitive deps |

All other functionality uses Python stdlib (`hashlib`, `json`, `csv`, `uuid`, `time`, `secrets`, `pathlib`). No heavy new dependencies.

---

## 10. Testing Plan

### 10.1 Unit Tests

| Module | Test File | Key Tests |
|--------|-----------|-----------|
| `auth/jwt_handler.py` | `tests/test_auth_jwt.py` | Token creation, verification, expiry, refresh, invalid token |
| `auth/pin_auth.py` | `tests/test_auth_pin.py` | PIN registration, verification, lockout after 5 failures |
| `sync/delta_tracker.py` | `tests/test_sync_delta.py` | Delta queries return only records after timestamp |
| `sync/conflict_resolver.py` | `tests/test_sync_conflict.py` | Server-wins, client-wins, append-only strategies |
| `sync/batch_processor.py` | `tests/test_sync_batch.py` | Batch push with mixed success/failure |
| `evaluation/sus_collector.py` | `tests/test_evaluation_sus.py` | SUS score calculation (known values), validation |
| `evaluation/vignette_manager.py` | `tests/test_evaluation_vignette.py` | Assignment randomization, response storage |
| `evaluation/exporter.py` | `tests/test_evaluation_export.py` | CSV format validation for R compatibility |
| `offline/cache_builder.py` | `tests/test_offline_cache.py` | Bundle generation, emergency keywords |

### 10.2 Integration Tests

```python
# tests/test_mobile_api_integration.py

async def test_full_mobile_flow():
    """End-to-end test of the mobile API."""
    # 1. Register device
    response = await client.post("/api/mobile/v1/auth/register", json={
        "name": "Test ASHA",
        "phone_hash": "abc123hash",
        "role": "asha_worker",
        "language": "hi-IN",
        "site_id": "cmc_vellore",
        "pin": "1234",
    })
    assert response.status_code == 200
    token = response.json()["token"]

    # 2. Submit text query
    headers = {"Authorization": f"Bearer {token}"}
    response = await client.post("/api/mobile/v1/query", json={
        "query": "How to manage pain at home",
        "language": "hi-IN",
    }, headers=headers)
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "evidence_level" in response.json()

    # 3. Create observation
    response = await client.post("/api/mobile/v1/observation", json={
        "patient_id": "patient_001",
        "category": "symptom",
        "entity_name": "pain",
        "severity": 2,
        "value_text": "Moderate pain in lower back",
    }, headers=headers)
    assert response.status_code == 200

    # 4. Sync pull
    response = await client.post("/api/mobile/v1/sync/pull?last_sync_at=0", headers=headers)
    assert response.status_code == 200
    assert "observations" in response.json()

    # 5. Get cache bundle
    response = await client.get("/api/mobile/v1/cache/bundle?language=hi-IN", headers=headers)
    assert response.status_code == 200
    assert len(response.json()["queries"]) > 0

    # 6. Submit SUS
    response = await client.post("/api/mobile/v1/evaluation/sus", json={
        "scores": [4, 2, 5, 1, 4, 2, 5, 1, 4, 2],
        "site_id": "cmc_vellore",
        "language": "hi-IN",
        "completed_at": time.time(),
    }, headers=headers)
    assert response.status_code == 200
    assert response.json()["sus_score"] == 80.0
```

### 10.3 Regression Tests

After all changes, run the existing test suite to verify no regressions:

```bash
python -c "from simple_rag_server import *"
python tests/test_v2_modules.py
python tests/test_graphrag_config.py
curl http://localhost:8000/api/query -X POST -d '{"query": "test"}' -H "Content-Type: application/json"
curl http://localhost:8000/health
```

All existing endpoints must work identically before and after changes.

---

## 11. Migration and Backward Compatibility

### 11.1 Zero-Downtime Deployment

The mobile API is **additive only**:
- New routes are mounted under `/api/mobile/v1/` -- no existing route changes
- New modules are imported conditionally (`if mobile_config.get("enabled", False)`)
- New data directories are created lazily (on first write)
- If `MOBILE_JWT_SECRET` is not set, mobile API simply doesn't load

### 11.2 Existing Client Impact

| Client | Impact |
|--------|--------|
| Telephony (Bolna) | None -- uses `/api/bolna/*` (unchanged) |
| WhatsApp (Twilio) | None -- uses `/webhook` (unchanged) |
| Web (Gradio) | None -- uses `/admin` (unchanged) |
| Sarvam AI | None -- uses `/api/sarvam/*` (unchanged) |
| Retell AI | None -- uses `/api/retell/*` (unchanged) |

### 11.3 Data Migration

No data migration needed. New data directories are created empty:
- `data/auth/devices/` -- populated when devices register
- `data/evaluation/` -- populated when evaluation data is submitted
- `data/cache/mobile/` -- populated when cache bundles are built

Existing data in `data/user_profiles/`, `data/longitudinal/`, `data/medication_voice_reminders/` is accessed via existing managers (read-only access pattern from mobile API).

---

## 12. File Manifest

### New Files (26 files)

```
mobile_api/__init__.py
mobile_api/router.py
mobile_api/schemas.py
mobile_api/dependencies.py
mobile_api/utils.py

auth/__init__.py
auth/jwt_handler.py
auth/pin_auth.py
auth/models.py
auth/middleware.py

sync/__init__.py
sync/delta_tracker.py
sync/conflict_resolver.py
sync/batch_processor.py

evaluation/__init__.py
evaluation/sus_collector.py
evaluation/vignette_manager.py
evaluation/interaction_logger.py
evaluation/time_motion.py
evaluation/exporter.py

offline/__init__.py
offline/cache_builder.py
offline/query_ranker.py

data/evaluation/vignettes/          (directory + 20 vignette JSON files)
data/auth/devices/                  (empty directory)
data/cache/mobile/                  (empty directory)
```

### Modified Files (5 files, additive changes only)

```
simple_rag_server.py                (~25 lines added: mobile router mount)
personalization/longitudinal_memory.py   (~60 lines added: delta sync methods)
personalization/user_profile.py     (~10 lines added: delta sync method)
medication_voice_reminders.py       (~20 lines added: delta sync methods)
analytics/usage_analytics.py        (~15 lines added: top queries method)
```

### Modified Configuration (2 files)

```
config.yaml                         (mobile + evaluation sections added)
.env                                (MOBILE_JWT_SECRET added)
```

### New Dependencies (1 package)

```
requirements.txt                    (PyJWT>=2.8.0 added)
```

### New Test Files (10 files)

```
tests/test_auth_jwt.py
tests/test_auth_pin.py
tests/test_sync_delta.py
tests/test_sync_conflict.py
tests/test_sync_batch.py
tests/test_evaluation_sus.py
tests/test_evaluation_vignette.py
tests/test_evaluation_export.py
tests/test_offline_cache.py
tests/test_mobile_api_integration.py
```

---

## Summary of Changes

| Category | Count | Lines Added | Lines Modified |
|----------|-------|-------------|----------------|
| New Python modules | 5 modules, 23 files | ~1,800 | 0 |
| Modified existing files | 5 files | ~130 | 0 |
| New test files | 10 files | ~500 | 0 |
| Configuration | 2 files | ~25 | 0 |
| New data directories | 3 directories | 0 | 0 |
| New dependencies | 1 package | 1 | 0 |
| **Total** | **41 new files** | **~2,455** | **0** |

Every change is additive. Zero existing lines are modified or deleted. The mobile API is a clean extension that maximizes reuse of existing services through thin wrapper endpoints.

---

**End of Document**

*Version 0.1.0 | 27 March 2026 | Changes to Existing rag_gci Codebase — Detailed Specification*
