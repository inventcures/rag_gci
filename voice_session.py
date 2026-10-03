#!/usr/bin/env python3
"""
Real-time voice session
=======================
Turn orchestration for the WebSocket voice path.

The important property is that this is **not a second safety path**. Whatever
route produced the answer, the text goes through the same
`SafetyEnhancementsManager.process_response` the text API uses, and the
interaction is written to the same study log with the voice path recorded. A
voice route that bypassed the filter would be exactly the hole ADR 0004 exists to
close, and the symptom would be an ASHA worker hearing a dose read aloud.

Frame protocol, JSON both ways:

    client -> server   {"type": "start", "language": "hi-IN", "session_id": "..."}
                        {"type": "audio", "data": "<base64 pcm>"}
                        {"type": "stop"}          # end of the user's turn
                        {"type": "interrupt"}     # user cut in, discard playback
                        {"type": "ping"}

    server -> client   {"type": "ready", "session_id": "...", "voice_path": "live"}
                        {"type": "transcript", "text": "..."}
                        {"type": "response", "text": "...", "audio_base64": "...",
                         "answer_kind": "...", "evidence_level": "B",
                         "emergency_level": "none", "source_count": 3,
                         "voice_path": "live", "release_id": "..."}
                        {"type": "pong"}
                        {"type": "error", "message": "..."}
"""

import base64
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Voice paths, matching the Android constants so the log reads the same on both
# sides. ADR 0007.
VOICE_PATH_LIVE = "live"
VOICE_PATH_FALLBACK = "fallback"
VOICE_PATH_CACHE = "cache"

# ADR 0007: fallback triggers on health, not reachability. A Live session that
# connects and then goes quiet mid-answer is worse than one that never connects,
# because the user is left mid-sentence.
FIRST_TOKEN_TIMEOUT_S = 4.0
PER_TURN_STALL_TIMEOUT_S = 8.0


@dataclass
class VoiceTurn:
    """One user turn and the answer it produced."""
    transcript: str
    answer: str
    answer_kind: str
    evidence_level: str
    emergency_level: str
    source_count: int
    voice_path: str
    audio_base64: Optional[str] = None
    latency_ms: int = 0
    fallback_reason: Optional[str] = None


@dataclass
class VoiceSession:
    """
    Per-connection state.

    Held in memory because a session is a live conversation on one connection.
    Nothing here is the study's record; the study log is written separately and is
    append-only.
    """
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    language: str = "en-IN"
    release_id: str = ""
    participant_id: str = ""
    care_role: str = ""
    site_id: str = ""
    release_approved: bool = False
    history: List[Dict[str, str]] = field(default_factory=list)
    interrupted: bool = False

    def record_turn(self, question: str, answer: str) -> None:
        self.history.append({"question": question, "answer": answer})
        # Bounded so a long session cannot grow without limit on a device that
        # has been sitting in a patient's home all day.
        if len(self.history) > 20:
            self.history = self.history[-20:]


# The server-side SafetyResult has no answer_kind field; it reports whether the
# dose restriction fired. The client distinguishes a partial redaction from a
# full deferral by whether clinical content survived, so the wire format stays
# identical to the Android client.
ANSWER = "ANSWER"
REDACTED_ANSWER = "REDACTED_ANSWER"
DEFERRAL = "DEFERRAL"


def _answer_kind(safety: Any) -> str:
    if not getattr(safety, "dosage_blocked", False):
        return ANSWER
    # A deferral is the whole refusal text; a redaction still carries guidance and
    # is followed by an explanation note.
    text = getattr(safety, "response", "") or ""
    return DEFERRAL if text.startswith("I understand this is what you need") else REDACTED_ANSWER


def classify_fallback(
    first_token_latency_s: Optional[float],
    turn_latency_s: Optional[float],
) -> Optional[str]:
    """
    Decide whether a Live turn should fall back to Sarvam.

    Returns None when the turn is healthy, otherwise a short reason recorded with
    the interaction so the fallback rate is measurable. That rate is the evidence
    that the primary path works, and a high rate says it does not.

    Kept separate and pure so the thresholds can be tested without a socket, a
    provider, or a clock.
    """
    if first_token_latency_s is None:
        return "no_first_token"
    if first_token_latency_s > FIRST_TOKEN_TIMEOUT_S:
        return "first_token_timeout"
    if turn_latency_s is not None and turn_latency_s > PER_TURN_STALL_TIMEOUT_S:
        return "turn_stalled"
    return None


class VoiceTurnHandler:
    """
    Runs one turn: transcript in, grounded and safety-filtered answer out.

    Dependencies are injected so the orchestration is testable without a
    microphone, a socket or a provider. Nothing here reaches for a global.
    """

    def __init__(
        self,
        rag_pipeline: Any,
        safety_manager: Any,
        study_logger: Any = None,
        synthesise: Optional[Callable[[str, str], Awaitable[Optional[str]]]] = None,
    ):
        self.rag_pipeline = rag_pipeline
        self.safety_manager = safety_manager
        self.study_logger = study_logger
        self.synthesise = synthesise

    async def handle_audio(
        self,
        session: VoiceSession,
        audio: bytes,
        voice_path: str = VOICE_PATH_LIVE,
    ) -> Optional[VoiceTurn]:
        """
        Transcribe a recording, then answer it.

        Transcribing here rather than on the client is deliberate. The client cannot
        know what the user said, so making it fetch the transcript first costs a
        round trip on every turn for no benefit. One frame in, one frame out, and
        the transcript comes back alongside the answer.
        """
        transcript = await self._transcribe(audio, session.language)
        if not transcript:
            return None
        return await self.handle(session, transcript, voice_path=voice_path)

    async def _transcribe(self, audio: bytes, language: str) -> str:
        try:
            from sarvam_integration import SarvamClient

            result = await SarvamClient().speech_to_text(audio, language)
            return (result.transcript or "").strip()
        except Exception:
            # No transcript means no question. Returning an empty string here would
            # be recorded as an empty substantive interaction, which protocol 4.1
            # would then have to exclude.
            logger.warning("Transcription unavailable for %s", language)
            return ""

    async def handle(
        self,
        session: VoiceSession,
        transcript: str,
        voice_path: str = VOICE_PATH_LIVE,
        fallback_reason: Optional[str] = None,
    ) -> VoiceTurn:
        """
        Answer one spoken question.

        The safety boundary is applied to the generated text regardless of where it
        came from, which is the whole reason this lives in one place rather than
        being duplicated per voice provider.
        """
        started = time.time()

        result = await self.rag_pipeline.query(
            question=transcript,
            user_id=f"voice__{session.session_id}",
            language=session.language,
        )
        raw_answer = result.get("answer", "") or ""
        sources = result.get("sources", []) or []

        safety = self.safety_manager.process_response(
            query=transcript,
            response=raw_answer,
            sources=sources,
            language=session.language,
        )

        answer_kind = _answer_kind(safety)

        audio_base64 = None
        if self.synthesise is not None:
            audio_base64 = await self.synthesise(safety.response, session.language)

        session.record_turn(transcript, safety.response)

        turn = VoiceTurn(
            transcript=transcript,
            answer=safety.response,
            answer_kind=answer_kind,
            evidence_level=str(getattr(safety, "evidence_level", "E")),
            emergency_level=str(getattr(safety, "emergency_level", "none")),
            source_count=len(sources),
            voice_path=voice_path,
            audio_base64=audio_base64,
            latency_ms=int((time.time() - started) * 1000),
            fallback_reason=fallback_reason,
        )

        await self._record(session, turn)
        return turn

    async def _record(self, session: VoiceSession, turn: VoiceTurn) -> None:
        """
        Write the interaction to the study log.

        Best effort by design. A logging failure must not take down a live
        conversation in someone's home, but it must be loud, because a voice
        interaction that goes unrecorded is a gap in the study's evidence.
        """
        if self.study_logger is None:
            return
        try:
            await self.study_logger.record(
                user_id=session.participant_id or f"voice__{session.session_id}",
                site_id=session.site_id,
                care_role=session.care_role,
                session_id=session.session_id,
                query=turn.transcript,
                response=turn.answer,
                language=session.language,
                channel="voice",
                stage_latency_ms={"response": float(turn.latency_ms)},
                total_latency_ms=float(turn.latency_ms),
                rag_method="vector",
                sources=[{"filename": ""} for _ in range(turn.source_count)],
                safety_result={
                    "answer_kind": turn.answer_kind,
                    "evidence_level": turn.evidence_level,
                    "emergency_level": turn.emergency_level,
                    "validation_status": "voice_turn",
                },
                provider=turn.voice_path,
                stt_model="saaras-v3",
            )
        except Exception:
            logger.exception("Voice interaction was not recorded; evidence gap")


def turn_frame(turn: VoiceTurn, release_id: str = "") -> str:
    """Serialise a turn for the wire."""
    return json.dumps(
        {
            "type": "response",
            "text": turn.answer,
            "audio_base64": turn.audio_base64,
            "answer_kind": turn.answer_kind,
            "evidence_level": turn.evidence_level,
            "emergency_level": turn.emergency_level,
            "source_count": turn.source_count,
            "voice_path": turn.voice_path,
            "fallback_reason": turn.fallback_reason,
            "latency_ms": turn.latency_ms,
            "release_id": release_id,
        },
        ensure_ascii=False,
    )


def error_frame(message: str) -> str:
    return json.dumps({"type": "error", "message": message}, ensure_ascii=False)


def decode_audio(frame: Dict[str, Any]) -> bytes:
    """
    Decode base64 PCM from a frame.

    Raises on malformed input rather than returning empty bytes: a silent empty
    decode would send an empty question to the RAG and record it as a substantive
    interaction, which protocol 4.1 would then have to exclude.
    """
    data = frame.get("data")
    if not isinstance(data, str) or not data:
        raise ValueError("audio frame has no data")
    return base64.b64decode(data, validate=True)
