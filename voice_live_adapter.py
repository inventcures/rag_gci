"""
Gemini Live turn adapter
========================

`voice_router.LiveVoiceProvider` calls `service.stream_audio(audio, language)`.
`gemini_live/service.py` had no such method, so the primary voice path could not
serve a turn and every request fell through to Sarvam. That is the same shape of
silent failure as the English-only retrieval index: nothing errored, the system
simply quietly stopped using the provider it was supposed to prefer.

Live is used here as a transcription upgrade, not as a second answer generator.
Grounding and the dose boundary stay in `VoiceTurnHandler`, which is the only place
they are applied. Letting Live answer in its own words would mean a second safety
path, and a second safety path is a second set of rules that can drift from the
first. The trade-off is real and worth stating: Live's latency advantage is mostly
given up, because the round trip is transcribe, then generate, rather than duplex
streaming. Taking that trade keeps one boundary, and one boundary is worth more
than a faster wrong one.

True duplex with barge-in needs a protocol change on `/ws/voice`, not an adapter.
That belongs in a later ticket, and this module is deliberately shaped so it can be
replaced when that happens.

The `google-genai` SDK is imported inside `gemini_live.service`, never here, so this
module loads and its tests run in an environment without the SDK installed. An
absent SDK is a normal condition on the study server, and the router already has a
fallback for it.
"""

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)

# How long we wait for a Live transcription before treating the turn as failed.
# Long enough for a rural 2G uplink on a short recording, short enough that a user
# notices a stall before it becomes a support call.
TRANSCRIPTION_TIMEOUT_S = 8.0

VOICE_PATH_LIVE = "live"


class LiveSessionUnavailable(RuntimeError):
    """
    The Live path could not produce a transcript.

    Distinct from silence. Silence yields an empty string and no answer. This
    raises, because the router must record a reason for falling back, and "the
    provider produced nothing" is a different event from "the user said nothing".
    """


async def transcribe_with_live(
    service: Any,
    session: Any,
    audio: bytes,
    timeout: float = TRANSCRIPTION_TIMEOUT_S,
    language: Optional[str] = None,
) -> str:
    """
    Transcribe one recording through Gemini Live.

    Raises `LiveSessionUnavailable` when the provider cannot serve the turn. Returns
    an empty string only when the audio genuinely held no speech.

    `language` wins over the session's own language when both are present. The
    session records what the connection opened as, while this argument is what the
    client asked for on this turn. Letting the session win meant a client that
    switched language mid-connection was still transcribed in the opening language.
    """
    available = getattr(service, "is_available", None)
    if callable(available) and not available():
        raise LiveSessionUnavailable("gemini_live_reports_unavailable")

    effective_language = language or getattr(session, "language", "en-IN")
    try:
        live_session = await service.create_session(language=effective_language)
    except Exception as exc:  # noqa: BLE001
        raise LiveSessionUnavailable(f"session_open_failed:{type(exc).__name__}") from exc

    try:
        await live_session.connect()
        await live_session.send_audio(audio, effective_language)
        transcript = await live_session.wait_for_transcription(timeout)
    except asyncio.TimeoutError as exc:
        raise LiveSessionUnavailable("transcription_stalled") from exc
    except LiveSessionUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LiveSessionUnavailable(f"live_error:{type(exc).__name__}") from exc
    finally:
        try:
            await live_session.disconnect()
        except Exception:  # noqa: BLE001
            # A session that will not close must not mask the turn's real outcome.
            logger.warning("Live session did not close cleanly")

    return (transcript or "").strip()


class GeminiLiveTurnAdapter:
    """
    Adapts Gemini Live to the turn shape the router expects.

    `LiveVoiceProvider` consumes an async iterator and takes the last event marked
    `final`. That is the whole contract.
    """

    def __init__(self, service: Any, handler: Any, session: Any):
        self.service = service
        self.handler = handler
        self.session = session

    async def stream_audio(
        self, audio: bytes, language: str
    ) -> AsyncIterator[Dict[str, Any]]:
        transcript = await transcribe_with_live(
            self.service, self.session, audio, language=language
        )

        if not transcript:
            # Silence is not a substantive turn. Recording it as one would put an
            # empty interaction into the adoption denominator.
            yield {"final": True, "turn": None, "transcript": ""}
            return

        turn: Optional[Any] = await self.handler.handle(
            self.session, transcript, voice_path=VOICE_PATH_LIVE, fallback_reason=None
        )
        yield {"final": True, "turn": turn, "transcript": transcript}


def build_live_service_factory(handler: Any) -> Any:
    """
    Build the factory `VoiceRouter` calls to obtain a Live-backed turn adapter.

    `VoiceRouter` takes `live_service_factory(session)` and treats its return value
    as anything exposing `stream_audio`. That is this adapter, so the router needs
    no knowledge of Google.

    The service is constructed per turn rather than held, because the Google client
    is cheap to create and a long-lived one would outlive the credentials that made
    it work.
    """

    def factory(session: Any) -> GeminiLiveTurnAdapter:
        from gemini_live.service import GeminiLiveService

        # The safety manager is passed in so the tool-call path filters grounding
        # text before the model can speak it. Built from the handler rather than
        # imported from a global, because the router must not depend on the app
        # having initialised anything.
        service = GeminiLiveService(
            rag_pipeline=getattr(handler, "rag_pipeline", None),
            safety_manager=getattr(handler, "safety_manager", None),
        )
        return GeminiLiveTurnAdapter(service, handler, session)

    return factory


async def collect_live_events(
    service: Any, handler: Any, session: Any, audio: bytes, language: str
) -> List[Dict[str, Any]]:
    """Convenience for callers that want the event list rather than a stream."""
    adapter = GeminiLiveTurnAdapter(service, handler, session)
    return [event async for event in adapter.stream_audio(audio, language)]