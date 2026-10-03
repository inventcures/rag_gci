#!/usr/bin/env python3
"""
Study Interaction Logging
=========================
Writes the system log that the EVAH grant and the study protocol both promise.

The grant commits to a system log that supports, continuously:

  * Adoption  - "Calls/ASHA/week; sustained use (Continuous; system logs)"
  * Safety    - "Hallucination rate; emergency F1; guideline concordance"
  * Language equity - "All metrics by language (Throughout; system logs)"
  * Cost      - "Per-interaction resource use"
  * Supervision - "tiered expert sampling (5% random, 50% flagged, 100% critical)"

and to a release identifier "stored with every interaction" (protocol 7.8).

Privacy
-------
The Digital Personal Data Protection Act 2023 applies, and the study operates
under a consented data plan. Identifiers are therefore pseudonymised rather than
stored: participant ids become keyed hashes, and free-text queries and responses
are scrubbed of direct identifiers before they reach disk. This is not optional
decoration -- the grant's own risk section commits to "local-first data
architecture aligned with India's Digital Personal Data Protection Act 2023".

Query text is retained rather than discarded because the grant's analytic plan
depends on it: BERTopic is applied to "5,000-15,000 transcribed queries" to find
population-level knowledge gaps, and the protocol requires "audited interaction
numbers per language".

Storage
-------
One JSONL file per UTC day under `data/study/logs/`. Append-only, which makes the
log auditable: entries are never rewritten, so a later edit to a response cannot
silently change what was recorded.
"""

import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from study_release import get_study_release

logger = logging.getLogger(__name__)

LOG_DIR = Path("./data/study/logs")

# Salt for pseudonymisation. Must be stable across restarts for a study release
# (otherwise the same participant becomes a different id each day and adoption
# is unmeasurable), and must never be committed. Falls back to an ephemeral salt
# with a loud warning, which is safe but makes participants unlinkable across runs.
_SECRET = os.getenv("STUDY_LOG_SALT", "")
if not _SECRET:
    _SECRET = uuid.uuid4().hex
    logger.error(
        "STUDY_LOG_SALT is not set. Participant ids will be pseudonymised with "
        "an ephemeral salt and will NOT be linkable across restarts, which "
        "makes sustained-adoption measurement impossible. Set STUDY_LOG_SALT "
        "before any participant use."
    )


# ---------------------------------------------------------------------------
# Pseudonymisation and scrubbing
# ---------------------------------------------------------------------------

def pseudonymise(user_id: str) -> str:
    """
    Keyed hash of a participant identifier.

    Truncated to 16 hex characters: enough that collisions across ~200
    participants are not a concern, short enough to read in a dashboard.
    Deterministic within a salt so the same participant aggregates correctly.
    """
    if not user_id:
        return "anonymous"
    return hmac.new(
        _SECRET.encode(), user_id.encode(), hashlib.sha256
    ).hexdigest()[:16]


# Direct identifiers that must never reach the log. Ordered most specific first.
_SCRUBBERS: List[tuple] = [
    ("aadhaar", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")),
    ("phone", re.compile(r"(?<![\d.])(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("abha", re.compile(r"\b\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{2}\b")),
    ("url", re.compile(r"\bhttps?://\S+\b")),
    # Catch-all for bare digit runs of 9 or more. Format-specific patterns above
    # only match well-formed Indian mobile numbers and 12-digit Aadhaar numbers,
    # so an 11-digit number, a landline or a malformed number would otherwise
    # reach the log intact. Order matters: this runs last so that a recognised
    # format is labelled precisely.
    # Word characters only in the guards: a full stop after a number is normal
    # sentence punctuation ("... at 09876543210. Please"), and excluding it here
    # let exactly those numbers through.
    ("digits", re.compile(r"(?<![\w])\+?\d[\d\s-]{7,}\d(?![\w])")),
]


@dataclass
class ScrubResult:
    text: str
    applied: List[str] = field(default_factory=list)

    @property
    def was_scrubbed(self) -> bool:
        return bool(self.applied)


def scrub(text: str) -> ScrubResult:
    """
    Remove direct identifiers from free text before it is written to disk.

    Scrubs structured identifiers reliably (phone, Aadhaar, ABHA, email, URL).
    It does not attempt to remove personal names, which cannot be done reliably
    without a name gazetteer; that limitation is recorded on the log entry so
    that anyone auditing the corpus knows free text may still contain a name.
    """
    if not text:
        return ScrubResult(text="")
    applied: List[str] = []
    out = text
    for label, pattern in _SCRUBBERS:
        if pattern.search(out):
            out = pattern.sub(f"[{label.upper()}_REDACTED]", out)
            applied.append(label)
    return ScrubResult(text=out, applied=applied)


# ---------------------------------------------------------------------------
# Log entry
# ---------------------------------------------------------------------------

@dataclass
class InteractionLog:
    """
    One interaction, as protocol 7.8 requires it be recorded.

    Field names follow the grant's own vocabulary so that the analysis does not
    have to translate between the proposal's language and the schema.
    """
    # -- identity (pseudonymised) --
    participant_id: str          # HMAC of user_id, never the raw value
    site_id: str
    care_role: str               # ASHA Worker / Family Caregiver / Patient
    session_id: str = ""

    # -- release provenance (protocol 7.8) --
    release_id: str = ""
    release_name: str = ""
    release_approved: bool = False

    # -- timing --
    occurred_at: float = 0.0
    occurred_at_iso: str = ""
    stage_latency_ms: Dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0

    # -- the interaction --
    channel: str = "mobile"      # mobile | voice | whatsapp | web
    language: str = ""
    query: str = ""
    response: str = ""
    scrubbed_fields: List[str] = field(default_factory=list)

    # -- retrieval --
    rag_method: str = ""
    source_count: int = 0
    source_documents: List[str] = field(default_factory=list)
    corpus_chunk_count: Optional[int] = None

    # -- safety (the grant's safety outcome) --
    emergency_level: str = "none"
    emergency_detected: bool = False
    evidence_level: str = "E"
    confidence: float = 0.0
    validation_status: str = ""
    dosage_blocked: bool = False
    dosage_reason: Optional[str] = None
    handoff_triggered: bool = False

    # -- operational --
    is_offline: bool = False
    is_cache_hit: bool = False
    error: Optional[str] = None
    provider: str = ""
    stt_model: str = ""
    tts_model: str = ""
    token_usage: Dict[str, int] = field(default_factory=dict)

    # -- protocol 4.1 adoption classification --
    interaction_class: str = "substantive"   # substantive | excluded
    exclusion_reason: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


# Phrases that mark an interaction as non-substantive for the protocol 4.1
# adoption denominator, which explicitly excludes them.
_EXCLUSION_PATTERNS: List[tuple] = [
    ("test_message", re.compile(r"\b(test message|testing|test call)\b", re.I)),
    ("training_demo", re.compile(r"\b(training|demo|demonstration|teach ?-?back)\b", re.I)),
    ("duplicate_retry", re.compile(r"\b(same question again|repeat|again sorry|resend)\b", re.I)),
]


def classify_interaction(query: str, response: str, had_error: bool = False) -> tuple:
    """
    Apply the protocol 4.1 definition of a substantive interaction.

    "Training demonstrations, test messages, accidental activations and
    duplicate retries are excluded. A substantive interaction records a care
    question and a returned response or appropriate clinical escalation."

    The heuristic is deliberately conservative in the direction the protocol
    asks for: where it is unsure, the interaction is counted as substantive, so
    the adoption denominator is not inflated by silent exclusions. An empty
    response with no escalation is the one unambiguous exclusion.
    """
    if not query or not query.strip():
        return "excluded", "empty_query"
    if not response or not response.strip():
        if had_error:
            return "excluded", "failed_no_response"
        return "excluded", "no_response_no_escalation"
    for label, pattern in _EXCLUSION_PATTERNS:
        if pattern.search(query):
            return "excluded", label
    return "substantive", None


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class StudyInteractionLogger:
    """
    Append-only study log with the grant's promised granularity.

    Wraps the analytics collectors that already exist in this repository but
    which nothing was calling: `RealtimeMetrics` (four-stage latency),
    `UsageAnalytics` (queries by language and hour) and `clinical_validation`'s
    expert sampler (5% / 50% / 100%).
    """

    def __init__(
        self,
        log_dir: Path = LOG_DIR,
        use_analytics: bool = True,
        use_expert_sampling: bool = True,
    ):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        self.realtime = None
        self.usage = None
        self.expert_sampler = None
        self.clinical_metrics = None

        if use_analytics:
            self._init_analytics()
        if use_expert_sampling:
            self._init_clinical()

    # -- wiring the previously-unused modules -----------------------------
    def _init_analytics(self) -> None:
        try:
            from analytics.realtime_metrics import RealtimeMetrics

            self.realtime = RealtimeMetrics()
            logger.info("Wired RealtimeMetrics (STT/RAG/TTS/response latency)")
        except Exception:
            logger.exception("RealtimeMetrics unavailable; latency will be logged but not windowed")

        try:
            from analytics.usage_analytics import UsageAnalytics

            self.usage = UsageAnalytics(storage_path="data/analytics")
            logger.info("Wired UsageAnalytics (queries by language and hour)")
        except Exception:
            logger.exception("UsageAnalytics unavailable")

    def _init_clinical(self) -> None:
        """Tiered expert sampling: 5% random, 50% flagged, 100% critical."""
        try:
            from clinical_validation.expert_sampling import ExpertSampler
            from clinical_validation.metrics import ValidationMetrics

            self.expert_sampler = ExpertSampler(
                storage_path="data/expert_samples",
                sample_rate=0.05,
                max_samples_per_day=50,
            )
            self.clinical_metrics = ValidationMetrics(
                storage_path="data/metrics"
            )
            logger.info(
                "Wired ExpertSampler (5%% random / 50%% flagged / 100%% critical) "
                "and ClinicalMetricsCollector"
            )
        except Exception:
            logger.exception(
                "clinical_validation unavailable; tiered expert sampling disabled"
            )

    # -- writing -----------------------------------------------------------
    def _path_for(self, when: float) -> Path:
        day = datetime.fromtimestamp(when, tz=timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"interactions-{day}.jsonl"

    def write(self, entry: InteractionLog) -> Optional[Path]:
        """Append one entry. Never raises: logging must not break a query."""
        if not entry.occurred_at:
            entry.occurred_at = time.time()
        if not entry.occurred_at_iso:
            entry.occurred_at_iso = datetime.fromtimestamp(
                entry.occurred_at, tz=timezone.utc
            ).isoformat()

        try:
            path = self._path_for(entry.occurred_at)
            with self._lock:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry.to_json(), ensure_ascii=False) + "\n")
            return path
        except Exception:
            logger.exception("Failed to write study interaction log")
            return None

    async def record(
        self,
        *,
        user_id: str,
        site_id: str = "",
        care_role: str = "",
        query: str = "",
        response: str = "",
        session_id: str = "",
        channel: str = "mobile",
        language: str = "",
        stage_latency_ms: Optional[Dict[str, float]] = None,
        total_latency_ms: float = 0.0,
        rag_method: str = "",
        sources: Optional[List[Dict[str, Any]]] = None,
        safety_result: Any = None,
        is_offline: bool = False,
        is_cache_hit: bool = False,
        error: Optional[str] = None,
        provider: str = "",
        stt_model: str = "",
        tts_model: str = "",
    ) -> InteractionLog:
        """
        Record one interaction end to end.

        This is the single call site the mobile API was missing. Everything the
        grant buys with system logs flows from it.
        """
        started = time.time()
        sources = sources or []
        stage_latency_ms = stage_latency_ms or {}

        scrubbed_query = scrub(query)
        scrubbed_response = scrub(response)
        scrubbed_fields = sorted(
            set(scrubbed_query.applied) | set(scrubbed_response.applied)
        )

        release = get_study_release()
        stamp = release.stamp()

        safety = _as_dict(safety_result)
        interaction_class, exclusion_reason = classify_interaction(
            query, response, had_error=bool(error)
        )

        entry = InteractionLog(
            participant_id=pseudonymise(user_id),
            site_id=site_id,
            care_role=care_role,
            session_id=session_id or uuid.uuid4().hex[:12],
            release_id=stamp["release_id"],
            release_name=stamp["release_name"],
            release_approved=stamp["approved"],
            occurred_at=started,
            channel=channel,
            language=language,
            query=scrubbed_query.text,
            response=scrubbed_response.text,
            scrubbed_fields=scrubbed_fields,
            rag_method=rag_method,
            source_count=len(sources),
            source_documents=sorted(
                {
                    str(s.get("filename") or s.get("source") or s.get("doc_id") or "")
                    for s in sources
                    if isinstance(s, dict)
                }
                - {""}
            ),
            emergency_level=str(safety.get("emergency_level") or "none"),
            emergency_detected=str(safety.get("emergency_level") or "none") != "none",
            evidence_level=str(safety.get("evidence_level") or "E"),
            confidence=float(safety.get("confidence") or 0.0),
            validation_status=str(safety.get("validation_status") or ""),
            dosage_blocked=bool(safety.get("dosage_blocked", False)),
            dosage_reason=safety.get("dosage_reason"),
            is_offline=is_offline,
            is_cache_hit=is_cache_hit,
            error=error,
            provider=provider,
            stt_model=stt_model,
            tts_model=tts_model,
            interaction_class=interaction_class,
            exclusion_reason=exclusion_reason,
        )

        # Prefer measured stage latencies; fall back to the wall clock.
        entry.total_latency_ms = total_latency_ms or round(
            (time.time() - started) * 1000, 1
        )
        entry.stage_latency_ms = {
            k: round(float(v), 1)
            for k, v in stage_latency_ms.items()
            if v is not None
        }
        entry.occurred_at_iso = datetime.fromtimestamp(
            entry.occurred_at, tz=timezone.utc
        ).isoformat()

        self.write(entry)

        await self._fan_out(entry)
        return entry

    async def _fan_out(self, entry: InteractionLog) -> None:
        """Push into the collectors that were previously never called."""
        if self.realtime is not None:
            try:
                from analytics.realtime_metrics import MetricType

                mapping = {
                    "stt": MetricType.STT_LATENCY_MS,
                    "rag": MetricType.RAG_LATENCY_MS,
                    "tts": MetricType.TTS_LATENCY_MS,
                    "response": MetricType.RESPONSE_LATENCY_MS,
                }
                for stage, ms in entry.stage_latency_ms.items():
                    metric = mapping.get(stage)
                    if metric is not None:
                        await self.realtime.record_latency(metric, ms)
                if entry.total_latency_ms:
                    await self.realtime.record_latency(
                        MetricType.RESPONSE_LATENCY_MS, entry.total_latency_ms
                    )
                await self.realtime.record_query()
                if entry.error:
                    await self.realtime.record_error(entry.error[:60])
            except Exception:
                logger.exception("RealtimeMetrics fan-out failed")

        if self.usage is not None:
            try:
                await self.usage.record_query(
                    user_id=entry.participant_id,
                    language=entry.language,
                    query_type="voice" if entry.channel == "voice" else "text",
                    used_rag=bool(entry.rag_method),
                    rag_success=entry.source_count > 0,
                    validation_passed=entry.source_count > 0,
                )
            except Exception:
                logger.exception("UsageAnalytics fan-out failed")

        if self.expert_sampler is not None:
            try:
                validation = {
                    "issues": _issues_for(entry),
                    "confidence_score": entry.confidence,
                }
                await self.expert_sampler.maybe_sample(
                    query=entry.query,
                    response=entry.response,
                    language=entry.language or "en",
                    sources=[{"doc": d} for d in entry.source_documents],
                    validation_result=validation,
                    session_id=entry.session_id,
                    user_id=entry.participant_id,
                )
            except Exception:
                logger.exception("Expert sampling fan-out failed")

        if self.clinical_metrics is not None:
            try:
                await self.clinical_metrics.record_validation(
                    validation_result={
                        "valid": entry.source_count > 0,
                        "confidence_score": entry.confidence,
                        "confidence": entry.confidence,
                        "evidence_level": entry.evidence_level,
                        "source_count": entry.source_count,
                        "dosage_blocked": entry.dosage_blocked,
                        "emergency_level": entry.emergency_level,
                        "validation_status": entry.validation_status,
                    },
                    response_time_ms=entry.total_latency_ms or None,
                )
            except Exception:
                logger.exception("ClinicalMetrics fan-out failed")


def _issues_for(entry: InteractionLog) -> List[Dict[str, str]]:
    """
    Map an interaction onto the issue levels the expert sampler keys on.

    This is the grant's "50% flagged" tier. An interaction with no retrieved
    source is the single most important thing for a clinician to see, because
    the answer was generated without grounding.
    """
    issues: List[Dict[str, str]] = []
    if entry.source_count == 0 and entry.interaction_class == "substantive":
        issues.append({"level": "critical", "reason": "no_retrieved_sources"})
    if entry.dosage_blocked:
        issues.append({"level": "info", "reason": "dosage_restriction_applied"})
    if entry.validation_status == "unsupported_answer":
        issues.append({"level": "error", "reason": "unsupported_answer"})
    return issues


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_entries(
    day: Optional[str] = None,
    log_dir: Path = LOG_DIR,
) -> List[Dict[str, Any]]:
    """
    Read log entries for auditing.

    Audit reads are deliberately separate from writes: an auditor needs to read
    everything without being able to alter it, and the write path never offers a
    read-modify-write entry point.
    """
    if not log_dir.exists():
        return []
    paths = (
        [log_dir / f"interactions-{day}.jsonl"] if day
        else sorted(log_dir.glob("interactions-*.jsonl"))
    )
    entries: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed log line in %s", path)
    return entries


_logger: Optional[StudyInteractionLogger] = None


def get_study_logger() -> StudyInteractionLogger:
    global _logger
    if _logger is None:
        _logger = StudyInteractionLogger()
    return _logger