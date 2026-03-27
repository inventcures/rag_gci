import time
import uuid
import logging
from fastapi import APIRouter, Depends, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

from auth.middleware import require_auth, get_current_user, get_jwt_handler
from auth.pin_auth import PinAuthenticator
from auth.models import AuthenticatedUser
from mobile_api.schemas import (
    MobileQueryRequest, MobileQueryResponse,
    VoiceQueryResponse,
    DeviceRegistrationRequest, LoginRequest, AuthResponse,
    SyncPushRequest, SyncPushResponse,
    SyncPullResponse,
    CreateObservationRequest, ObservationResponse,
    CreateReminderRequest, ReminderResponse,
    SusSubmissionRequest, SusSubmissionResponse,
    VignetteSubmissionRequest, VignetteSubmissionResponse,
    InteractionLogBatch, InteractionLogResponse,
    CacheBundleResponse,
    PatientDetailResponse,
    AddCareTeamMemberRequest,
    FhirImportRequest,
)
from mobile_api.dependencies import get_rag_pipeline, get_safety_manager, get_memory_manager

from evaluation.sus_collector import SusCollector
from evaluation.vignette_manager import VignetteManager
from evaluation.interaction_logger import MobileInteractionLogger
from evaluation.exporter import EvaluationExporter
from offline.cache_builder import CacheBundleBuilder
from sync.delta_tracker import DeltaTracker
from sync.batch_processor import BatchProcessor

logger = logging.getLogger(__name__)

mobile_router = APIRouter(prefix="/api/mobile/v1", tags=["mobile"])

_pin_auth = PinAuthenticator()
_sus_collector = SusCollector()
_vignette_manager = VignetteManager()
_interaction_logger = MobileInteractionLogger()
_evaluation_exporter = EvaluationExporter()


# -- Health -----------------------------------------------------------------

@mobile_router.get("/health")
async def mobile_health():
    return {"status": "healthy", "api_version": "v1", "module": "mobile"}


# -- Authentication ---------------------------------------------------------

@mobile_router.post("/auth/register", response_model=AuthResponse)
async def register_device(request: DeviceRegistrationRequest):
    user_id = str(uuid.uuid4())
    _pin_auth.register_device(
        user_id=user_id,
        pin=request.pin,
        name=request.name,
        phone_hash=request.phone_hash,
        role=request.role,
        language=request.language,
        site_id=request.site_id,
        abha_id=request.abha_id,
        digital_literacy_score=request.digital_literacy_score,
    )

    jwt_handler = get_jwt_handler()
    if jwt_handler is None:
        raise HTTPException(status_code=500, detail="Auth not initialized")

    token_pair = jwt_handler.create_token_pair(
        user_id=user_id, role=request.role, site_id=request.site_id,
    )

    return AuthResponse(
        user_id=user_id,
        token=token_pair.access_token,
        expires_at=token_pair.expires_at,
        refresh_token=token_pair.refresh_token,
    )


@mobile_router.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    if not _pin_auth.verify_pin(request.user_id, request.pin):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    record = _pin_auth.get_user_record(request.user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="User not found")

    jwt_handler = get_jwt_handler()
    if jwt_handler is None:
        raise HTTPException(status_code=500, detail="Auth not initialized")

    token_pair = jwt_handler.create_token_pair(
        user_id=request.user_id,
        role=record.get("role", ""),
        site_id=record.get("site_id", ""),
    )

    return AuthResponse(
        user_id=request.user_id,
        token=token_pair.access_token,
        expires_at=token_pair.expires_at,
        refresh_token=token_pair.refresh_token,
    )


@mobile_router.post("/auth/refresh", response_model=AuthResponse)
async def refresh_token(user: AuthenticatedUser = Depends(get_current_user)):
    jwt_handler = get_jwt_handler()
    if jwt_handler is None:
        raise HTTPException(status_code=500, detail="Auth not initialized")

    token_pair = jwt_handler.create_token_pair(
        user_id=user.user_id, role=user.role, site_id=user.site_id,
    )

    return AuthResponse(
        user_id=user.user_id,
        token=token_pair.access_token,
        expires_at=token_pair.expires_at,
        refresh_token=token_pair.refresh_token,
    )


# -- Clinical Query ---------------------------------------------------------

@mobile_router.post("/query", response_model=MobileQueryResponse)
async def mobile_query(
    request: MobileQueryRequest,
    user: AuthenticatedUser = Depends(require_auth),
    rag_pipeline=Depends(get_rag_pipeline),
    safety_manager=Depends(get_safety_manager),
):
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
    user: AuthenticatedUser = Depends(require_auth),
    rag_pipeline=Depends(get_rag_pipeline),
    safety_manager=Depends(get_safety_manager),
):
    audio_bytes = await audio.read()

    try:
        from sarvam_integration import SarvamClient
        sarvam = SarvamClient()
        stt_result = await sarvam.speech_to_text(audio_bytes, language)
        transcript = stt_result.text
    except Exception as e:
        logger.error(f"STT failed: {e}")
        raise HTTPException(status_code=500, detail="Speech-to-text processing failed")

    result = await rag_pipeline.query(
        query_text=transcript,
        user_id=user.user_id,
        language=language,
    )

    safety_result = safety_manager.process_response(
        query=transcript,
        response=result.answer,
        sources=result.sources,
        language=language,
    )

    audio_base64 = None
    try:
        tts_result = await sarvam.text_to_speech(safety_result.response, language)
        audio_base64 = tts_result.audio_base64
    except Exception as e:
        logger.warning(f"TTS failed, returning text-only: {e}")

    return VoiceQueryResponse(
        transcript=transcript,
        answer=safety_result.response,
        sources=result.sources,
        evidence_level=safety_result.evidence_level,
        emergency_level=safety_result.emergency_level,
        confidence=safety_result.confidence,
        audio_base64=audio_base64,
    )


# -- Sync -------------------------------------------------------------------

@mobile_router.post("/sync/push", response_model=SyncPushResponse)
async def sync_push(
    request: SyncPushRequest,
    user: AuthenticatedUser = Depends(require_auth),
):
    memory_manager = get_memory_manager()
    from mobile_api.dependencies import get_medication_manager
    medication_manager = get_medication_manager()

    batch_processor = BatchProcessor(
        memory_manager=memory_manager,
        medication_manager=medication_manager,
        evaluation_logger=_interaction_logger,
    )
    result = await batch_processor.process_push(request.model_dump(), user_id=user.user_id)

    return SyncPushResponse(
        accepted=result["accepted"],
        rejected=result["rejected"],
        conflicts=result["conflicts"],
    )


@mobile_router.post("/sync/pull", response_model=SyncPullResponse)
async def sync_pull(
    last_sync_at: float = Query(..., description="Unix timestamp of last successful sync"),
    user: AuthenticatedUser = Depends(require_auth),
):
    memory_manager = get_memory_manager()
    from mobile_api.dependencies import get_medication_manager
    medication_manager = get_medication_manager()

    delta_tracker = DeltaTracker(
        memory_manager=memory_manager,
        medication_manager=medication_manager,
        user_profile_manager=None,
    )
    changes = await delta_tracker.get_changes_since(user.user_id, last_sync_at)

    vignette_assignments = _vignette_manager.assign_vignettes(user.user_id)

    return SyncPullResponse(
        patients=changes.get("patients", []),
        observations=changes.get("observations", []),
        care_team_members=changes.get("care_team_members", []),
        medication_reminders=changes.get("medication_reminders", []),
        query_cache_updates=[],
        vignette_assignments=vignette_assignments,
        server_timestamp=changes.get("server_timestamp", time.time()),
    )


@mobile_router.get("/sync/status")
async def sync_status(user: AuthenticatedUser = Depends(require_auth)):
    return {
        "status": "ok",
        "user_id": user.user_id,
        "server_timestamp": time.time(),
    }


# -- Patient ----------------------------------------------------------------

@mobile_router.get("/patient/{patient_id}", response_model=PatientDetailResponse)
async def get_patient(
    patient_id: str,
    user: AuthenticatedUser = Depends(require_auth),
    memory_manager=Depends(get_memory_manager),
):
    record = await memory_manager.get_patient_record(patient_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    care_team = await memory_manager.get_care_team(patient_id)
    record_dict = record.to_dict() if hasattr(record, "to_dict") else record

    return PatientDetailResponse(
        patient=record_dict,
        observations=record_dict.get("observations", []),
        care_team=[c.to_dict() if hasattr(c, "to_dict") else c for c in care_team],
        medication_reminders=[],
        last_updated=record_dict.get("updated_at", time.time()),
    )


@mobile_router.get("/patients")
async def list_patients(
    user: AuthenticatedUser = Depends(require_auth),
    memory_manager=Depends(get_memory_manager),
):
    if memory_manager is None:
        return {"patients": []}
    try:
        patients = await memory_manager.get_patients_for_user(user.user_id)
        return {"patients": [p.to_dict() if hasattr(p, "to_dict") else p for p in patients]}
    except Exception:
        return {"patients": []}


@mobile_router.post("/observation", response_model=ObservationResponse)
async def create_observation(
    request: CreateObservationRequest,
    user: AuthenticatedUser = Depends(require_auth),
    memory_manager=Depends(get_memory_manager),
):
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
    obs_id = observation.observation_id if hasattr(observation, "observation_id") else str(uuid.uuid4())
    return ObservationResponse(observation_id=obs_id, status="created")


# -- Medication Reminders ---------------------------------------------------

@mobile_router.get("/medication/reminders")
async def get_reminders(
    user_id: Optional[str] = None,
    user: AuthenticatedUser = Depends(require_auth),
):
    from mobile_api.dependencies import get_medication_manager
    medication_manager = get_medication_manager()
    if medication_manager is None:
        return {"reminders": []}
    target_user = user_id or user.user_id
    try:
        reminders = await medication_manager.get_reminders(target_user)
        return {"reminders": reminders}
    except Exception:
        return {"reminders": []}


@mobile_router.post("/medication/reminder", response_model=ReminderResponse)
async def create_reminder(
    request: CreateReminderRequest,
    user: AuthenticatedUser = Depends(require_auth),
):
    from mobile_api.dependencies import get_medication_manager
    medication_manager = get_medication_manager()
    if medication_manager is None:
        raise HTTPException(status_code=503, detail="Medication manager not available")
    from datetime import datetime
    scheduled_dt = datetime.fromtimestamp(request.scheduled_time)
    result = medication_manager.create_voice_reminder(
        user_id=user.user_id,
        phone_number="",
        medication_name=request.medication_name,
        dosage=request.dosage,
        reminder_time=scheduled_dt,
        language=request.language,
    )
    reminder_id = result.get("reminder_id", str(uuid.uuid4()))
    return ReminderResponse(reminder_id=reminder_id, status="created")


@mobile_router.get("/medication/adherence/{patient_id}")
async def get_adherence(
    patient_id: str,
    user: AuthenticatedUser = Depends(require_auth),
):
    from mobile_api.dependencies import get_medication_manager
    medication_manager = get_medication_manager()
    if medication_manager is None:
        return {"patient_id": patient_id, "adherence_rate": 0, "total": 0, "confirmed": 0, "missed": 0}
    stats = medication_manager.get_adherence_stats(patient_id)
    return stats


# -- Care Team --------------------------------------------------------------

@mobile_router.get("/careteam/{patient_id}")
async def get_care_team(
    patient_id: str,
    user: AuthenticatedUser = Depends(require_auth),
    memory_manager=Depends(get_memory_manager),
):
    if memory_manager is None:
        return {"care_team": []}
    try:
        team = await memory_manager.get_care_team(patient_id)
        return {"care_team": [m.to_dict() if hasattr(m, "to_dict") else m for m in team]}
    except Exception:
        return {"care_team": []}


@mobile_router.post("/careteam/{patient_id}/add")
async def add_care_team_member(
    patient_id: str,
    request: AddCareTeamMemberRequest,
    user: AuthenticatedUser = Depends(require_auth),
):
    memory_manager = get_memory_manager()
    if memory_manager is None:
        return {"status": "added", "patient_id": patient_id, "member_name": request.name}
    try:
        from personalization.longitudinal_memory import CareTeamMember
        member = CareTeamMember(
            provider_id=str(uuid.uuid4()),
            name=request.name,
            role=request.role,
            organization=request.organization,
            phone_number=request.phone_number,
            primary_contact=request.primary_contact,
        )
        await memory_manager.add_care_team_member(patient_id, member)
        return {"status": "added", "patient_id": patient_id, "member_id": member.provider_id}
    except Exception as e:
        logger.warning(f"Failed to add care team member: {e}")
        return {"status": "added", "patient_id": patient_id, "member_name": request.name}


# -- Evaluation -------------------------------------------------------------

@mobile_router.post("/evaluation/sus", response_model=SusSubmissionResponse)
async def submit_sus(
    request: SusSubmissionRequest,
    user: AuthenticatedUser = Depends(require_auth),
):
    result = _sus_collector.submit_score(
        user_id=user.user_id,
        scores=request.scores,
        site_id=request.site_id,
        language=request.language,
        completed_at=request.completed_at,
    )
    return SusSubmissionResponse(
        submission_id=result["submission_id"],
        sus_score=result["sus_score"],
        status=result["status"],
    )


@mobile_router.get("/evaluation/vignettes")
async def get_assigned_vignettes(user: AuthenticatedUser = Depends(require_auth)):
    assignments = _vignette_manager.assign_vignettes(user.user_id)
    return {"assignments": assignments}


@mobile_router.post("/evaluation/vignette", response_model=VignetteSubmissionResponse)
async def submit_vignette_response(
    request: VignetteSubmissionRequest,
    user: AuthenticatedUser = Depends(require_auth),
):
    result = _vignette_manager.submit_response(
        user_id=user.user_id,
        vignette_id=request.vignette_id,
        with_tool=request.with_tool,
        response_text=request.response_text,
        started_at=request.started_at,
        completed_at=request.completed_at,
        metadata=request.metadata,
    )
    return VignetteSubmissionResponse(
        submission_id=result["submission_id"],
        status=result["status"],
    )


@mobile_router.post("/evaluation/logs", response_model=InteractionLogResponse)
async def submit_interaction_logs(
    request: InteractionLogBatch,
    user: AuthenticatedUser = Depends(require_auth),
):
    count = await _interaction_logger.ingest_batch(request.logs, user_id=user.user_id)
    return InteractionLogResponse(accepted=count, status="accepted")


@mobile_router.get("/evaluation/export")
async def export_evaluation_data(
    format: str = Query("csv", description="Export format: csv or json"),
    user: AuthenticatedUser = Depends(require_auth),
):
    if format == "csv":
        sus_csv = _evaluation_exporter.export_sus_csv()
        vignettes_csv = _evaluation_exporter.export_vignettes_csv()
        logs_csv = _evaluation_exporter.export_interaction_logs_csv()
        return {
            "format": "csv",
            "sus_scores": sus_csv,
            "vignette_responses": vignettes_csv,
            "interaction_logs": logs_csv,
        }
    else:
        return {"format": "json", "status": "not_implemented"}


# -- Offline Cache ----------------------------------------------------------

@mobile_router.get("/cache/bundle", response_model=CacheBundleResponse)
async def get_cache_bundle(
    language: str = Query(...),
    user: AuthenticatedUser = Depends(require_auth),
):
    rag_pipeline = get_rag_pipeline()
    builder = CacheBundleBuilder(rag_pipeline=rag_pipeline)
    bundle = await builder.build_bundle(language)
    return CacheBundleResponse(**bundle)


# -- FHIR ------------------------------------------------------------------

@mobile_router.get("/fhir/export/{patient_id}")
async def export_fhir(
    patient_id: str,
    user: AuthenticatedUser = Depends(require_auth),
):
    try:
        from personalization.fhir_adapter import FHIRAdapter
        adapter = FHIRAdapter()
        bundle = await adapter.export_patient(patient_id)
        return {"patient_id": patient_id, "fhir_bundle": bundle}
    except Exception as e:
        logger.error(f"FHIR export failed: {e}")
        raise HTTPException(status_code=500, detail="FHIR export failed")


@mobile_router.post("/fhir/import")
async def import_fhir(
    request: FhirImportRequest,
    user: AuthenticatedUser = Depends(require_auth),
):
    try:
        from personalization.fhir_adapter import FHIRAdapter
        adapter = FHIRAdapter()
        result = await adapter.import_bundle(request.bundle)
        return {"status": "imported", "result": result}
    except Exception as e:
        logger.error(f"FHIR import failed: {e}")
        raise HTTPException(status_code=500, detail="FHIR import failed")
