"""
Which server routes return an answer, so every one of them passes the Dose Boundary.

An answer route returns model- or graph-derived text: a generated answer, graph
or index content, a summary of remembered conversations. Each one passes the
answer gate (the Dose Boundary's ``apply``) before anything reaches the caller,
through ``guard_query_result`` / ``process_response`` for a single answer string,
or ``guard_answer_payload`` for structured graph and index data.

Every route of the app is classified here, as ``METHOD path`` (``WS`` for a
socket), exactly like ``auth/policy.py`` declares access: either in
``ANSWER_ROUTES`` (with where its gate runs) or in ``NO_ANSWER_TEXT`` (with why
it returns no answer text). ``tests/test_every_answer_route_passes_the_dose_boundary.py`` fails
when a route is in neither, and drives every answer route with a stub that emits
a dose. So a new answer-returning route cannot ship without the gate: it has to
be classified, and once classified as an answer route it needs a dose test.
"""

from typing import Any, Dict, Iterable, List, Mapping

from fastapi.routing import APIRoute, APIWebSocketRoute

from auth.policy import iter_route_paths

# Routes that return model- or graph-derived text, and where the gate runs.
ANSWER_ROUTES: Dict[str, str] = {
    "POST /api/query": "answer_web_query -> guard_query_result",
    "POST /api/bolna/query": "guard_query_result",
    "POST /webhook": "WhatsApp: guard_query_result before any send",
    "WS /ws/retell/llm/{call_id}": "guard_query_result / AnswerStream",
    "WS /ws/voice": "web Live voice; the transcript check is ticket 12",
    "POST /api/mobile/v1/query": "process_response",
    "POST /api/mobile/v1/query/voice": "VoiceTurnHandler -> process_response",
    "WS /api/mobile/v1/ws/voice": "VoiceTurnHandler -> process_response",
    "GET /api/mobile/v1/cache/bundle": "CacheBundleBuilder -> process_response",
    "POST /api/mobile/v1/patient/{patient_id}/query-memory": "guard_answer_payload (answer)",
    "GET /api/mobile/v1/patient/{patient_id}/insights": "guard_answer_payload",
    "POST /api/kg/query": "gate_answer_payload",
    "POST /api/kg/extract": "gate_answer_payload",
    "GET /api/kg/entity/{entity_name}": "gate_answer_payload",
    "GET /api/kg/treatments/{symptom}": "gate_answer_payload",
    "GET /api/kg/side-effects/{medication}": "gate_answer_payload",
    "GET /api/kg/search": "gate_answer_payload",
    "GET /api/kg/visualization/{entity_name}": "gate_answer_payload before the page is drawn",
    "POST /api/graphrag/query": "gate_answer_payload (response); the admin tab gates it again",
    "GET /api/graphrag/entities": "gate_answer_payload",
    "GET /api/graphrag/entity/{entity_name}/relationships": "gate_answer_payload",
    "GET /api/graphrag/communities": "gate_answer_payload",
    "POST /api/pageindex/query": "gate_answer_payload",
    "GET /api/pageindex/tree/{doc_id}": "gate_answer_payload (model-written node summaries)",
}

_HEALTH = "health or status only"
_COUNTS = "counts and index statistics"
_RECORD = "stored records, not generated text"
_CALLER = "returns what the caller sent, transformed; not an answer"
_STATIC = "static page or schema"

# Routes that return no answer text, and why.
NO_ANSWER_TEXT: Dict[str, str] = {
    "GET /": _STATIC,
    "GET /health": _HEALTH,
    "GET /voice": _STATIC,
    "GET /docs": _STATIC,
    "GET /docs/oauth2-redirect": _STATIC,
    "GET /redoc": _STATIC,
    "GET /openapi.json": _STATIC,
    "GET /api/voice/models": "voice model list",
    "GET /api/voice/languages": "language list",
    "GET /api/debug": "vector store diagnostics for staff",
    "POST /api/upload": "queues a document upload",
    "GET /api/upload/{job_id}": "indexing job status",
    "GET /media/{filename}": "audio file of an already-guarded answer",
    # PageIndex
    "GET /api/pageindex/health": _HEALTH,
    "GET /api/pageindex/stats": _COUNTS,
    "POST /api/pageindex/index": "starts indexing",
    "POST /api/pageindex/index/batch": "starts indexing",
    "GET /api/pageindex/index/status": "tree metadata: names, status, counts",
    "GET /api/pageindex/trees": "tree metadata: names, status, counts",
    # Phone providers
    "POST /api/bolna/webhook": "call lifecycle events",
    "GET /api/bolna/stats": _COUNTS,
    "GET /api/bolna/calls": "call records",
    "POST /api/retell/webhook": "call lifecycle events",
    "GET /api/retell/health": _HEALTH,
    "GET /api/retell/stats": _COUNTS,
    "GET /api/retell/calls": "call records",
    "GET /api/retell/vobiz": "telephony configuration",
    "GET /api/sarvam/health": _HEALTH,
    "GET /api/sarvam/languages": "language list",
    "POST /api/sarvam/stt": _CALLER,
    "POST /api/sarvam/tts": _CALLER,
    "POST /api/sarvam/translate": _CALLER,
    "POST /api/sarvam/webhook": "call lifecycle events",
    "GET /api/sarvam/stats": _COUNTS,
    # Medication reminders (ticket 06 keeps doses out of what they say)
    "POST /api/medication/voice-reminder": "schedules a reminder call",
    "GET /api/medication/voice-reminders/{user_id}": _RECORD,
    "POST /api/medication/voice-reminder/callback": "provider callback",
    "GET /api/medication/adherence/{user_id}": _RECORD,
    "GET /api/medication/pending-calls": _RECORD,
    # Knowledge graph and GraphRAG administration
    "GET /api/kg/health": _HEALTH,
    "GET /api/kg/stats": _COUNTS,
    "POST /api/kg/import-base": "imports the base graph; returns counts",
    "GET /api/graphrag/health": _HEALTH,
    "GET /api/graphrag/stats": _COUNTS,
    "POST /api/graphrag/index": "starts indexing",
    "GET /api/graphrag/index/status": "indexing status",
    "POST /api/graphrag/verify": "index integrity checks and counts",
    # Temporal reasoning: rule-based summaries of the patient record, whose
    # wording never advises a dose change (tests/test_temporal_reasoner.py)
    "GET /api/temporal/health": _HEALTH,
    "POST /api/temporal/symptom-progression": "rule-based record summary; no dose advice",
    "POST /api/temporal/medication-effectiveness": "rule-based record summary; no dose advice",
    "POST /api/temporal/correlations": "rule-based record summary; no dose advice",
    "GET /api/temporal/patient/{patient_id}/summary": "rule-based record summary; no dose advice",
    # Care team and FHIR
    "GET /api/careteam/{patient_id}": _RECORD,
    "POST /api/careteam/{patient_id}/add": _RECORD,
    "GET /api/careteam/{patient_id}/primary": _RECORD,
    "POST /api/careteam/{patient_id}/attribute": _RECORD,
    "POST /api/careteam/{patient_id}/notify": "sends a care-team notification",
    "POST /api/fhir/export/{patient_id}": _RECORD,
    "POST /api/fhir/import": _RECORD,
    "POST /api/fhir/validate": "validation result",
    "GET /api/fhir/snomed-codes": "code list",
    # Study dashboard
    "GET /admin/study/": "study dashboard page of logged, already-guarded answers",
    "GET /admin/study/api": "study summary",
    "GET /admin/study/api/interactions": "logged, already-guarded answers",
    "GET /admin/study/health": _HEALTH,
    "GET /admin/study/api/voice/provider": "voice provider setting",
    "POST /admin/study/api/voice/provider": "voice provider setting",
    # Mobile
    "GET /api/mobile/v1/health": _HEALTH,
    "POST /api/mobile/v1/auth/register": "sign-in",
    "POST /api/mobile/v1/auth/login": "sign-in",
    "POST /api/mobile/v1/auth/refresh": "sign-in",
    "GET /api/mobile/v1/release/status": "release gate",
    "POST /api/mobile/v1/sync/push": _RECORD,
    "POST /api/mobile/v1/sync/pull": _RECORD,
    "GET /api/mobile/v1/sync/status": _HEALTH,
    "GET /api/mobile/v1/patient/{patient_id}": _RECORD,
    "GET /api/mobile/v1/patients": _RECORD,
    "POST /api/mobile/v1/observation": _RECORD,
    "GET /api/mobile/v1/medication/reminders": _RECORD,
    "POST /api/mobile/v1/medication/reminder": _RECORD,
    "GET /api/mobile/v1/medication/adherence/{patient_id}": _RECORD,
    "GET /api/mobile/v1/careteam/{patient_id}": _RECORD,
    "POST /api/mobile/v1/careteam/{patient_id}/add": _RECORD,
    "POST /api/mobile/v1/evaluation/sus": "study questionnaire",
    "GET /api/mobile/v1/evaluation/vignettes": "fixed study vignettes",
    "POST /api/mobile/v1/evaluation/vignette": "study questionnaire",
    "POST /api/mobile/v1/evaluation/logs": "study log upload",
    "GET /api/mobile/v1/evaluation/export": "study export of logged, already-guarded answers",
    "GET /api/mobile/v1/fhir/export/{patient_id}": _RECORD,
    "POST /api/mobile/v1/fhir/import": _RECORD,
    "POST /api/mobile/v1/feedback": "feedback receipt",
}


def route_keys(routes: Iterable[Any]) -> List[str]:
    """``METHOD path`` for every HTTP route and ``WS path`` for every socket.

    Mounts (the Gradio admin panel) are not listed: the panel's GraphRAG tab
    reads ``POST /api/graphrag/query`` and gates what it shows."""
    keys: List[str] = []
    for path, route in iter_route_paths(routes):
        if isinstance(route, APIRoute):
            keys.extend(f"{method} {path}" for method in sorted(route.methods))
        elif isinstance(route, APIWebSocketRoute):
            keys.append(f"WS {path}")
    return keys


def unclassified_routes(
    app: Any,
    answers: Mapping[str, str] = ANSWER_ROUTES,
    others: Mapping[str, str] = NO_ANSWER_TEXT,
) -> List[str]:
    """Routes of `app` declared neither an answer route nor one without answer text."""
    return sorted({key for key in route_keys(app.routes)
                   if key not in answers and key not in others})
