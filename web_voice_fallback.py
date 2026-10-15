#!/usr/bin/env python3
"""
The Sarvam path for web voice (/ws/voice)
=========================================
ADR 0007: real-time voice is Gemini Live, with the India-resident Sarvam path
as its fallback. The administrator can force that path for the whole
deployment, and a Live session that stalls falls back to it automatically.
Web voice used to have neither: it opened Live whatever the override said, and
a Live session that went quiet left the caller in silence.

This module is what the web socket uses for both, and for speaking the text
the Dose Boundary releases in place of a withheld Live turn:

  * `LiveTurnClock` watches a Live turn against the release's deadlines
    (`voice_fallback.trigger`: "first-token timeout or per-turn stall",
    the thresholds in voice_session) and names the reason when Live misses
    them.
  * `WebSarvamTurns` answers the caller's recording on the Sarvam path:
    Saaras transcribes it, the answer goes through the same VoiceTurnHandler
    (and so the same Dose Boundary) as the mobile voice routes, and Bulbul
    speaks it back in the session's language.
  * `speak_for_web` is that Bulbul call, returning the PCM the web client
    plays (24 kHz, 16-bit, mono: Live's output format).

Frames follow ticket 12's per-turn order: the audio, the assistant
transcription, the caller's transcription, then turn_complete. The Sarvam
path's frames also carry `voice_path` and `fallback_reason`.
"""

import asyncio
import io
import logging
import wave
from typing import Any, Optional

from voice_router import record_voice_path
from voice_session import (
    FIRST_TOKEN_TIMEOUT_S,
    PER_TURN_STALL_TIMEOUT_S,
    VOICE_PATH_FALLBACK,
    VoiceSession,
    VoiceTurn,
    VoiceTurnHandler,
)

logger = logging.getLogger(__name__)

CHANNEL = "web_voice"
OPERATOR_FORCED = "operator_forced_fallback"

WEB_INPUT_RATE = 16000   # what the web client records (Int16 PCM, mono)
WEB_OUTPUT_RATE = 24000  # what it plays (Live's output format)
# One recording, bounded: two minutes of caller audio.
MAX_TURN_PCM_BYTES = WEB_INPUT_RATE * 2 * 120


def pcm_to_wav(pcm: bytes, rate: int = WEB_INPUT_RATE) -> bytes:
    """Wrap 16-bit mono PCM in a WAV header (Saaras takes a file, not raw PCM)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)
    return buffer.getvalue()


def wav_to_pcm(data: bytes) -> bytes:
    """The PCM frames of a 16-bit mono WAV. Raises wave.Error on anything else."""
    with wave.open(io.BytesIO(data)) as source:
        if source.getsampwidth() != 2 or source.getnchannels() != 1:
            raise wave.Error("expected 16-bit mono audio")
        if source.getframerate() != WEB_OUTPUT_RATE:
            logger.warning("Bulbul returned %s Hz audio; the web client plays %s Hz",
                           source.getframerate(), WEB_OUTPUT_RATE)
        return source.readframes(source.getnframes())


async def speak_for_web(text: str, language: str) -> Optional[bytes]:
    """
    Speak `text` in the Supported Language's Sarvam voice, as web-client PCM.

    None when there is nothing to say, the language has no Sarvam voice (it is
    never spoken in Hindi instead), or synthesis is unavailable: the text frame
    still carries the words.
    """
    if not text or not text.strip():
        return None
    from sarvam_integration.config import sarvam_voice

    voice = sarvam_voice(language)
    if voice is None:
        logger.warning("No Sarvam voice for %s; the reply is text only", language)
        return None
    try:
        from sarvam_integration import SarvamClient

        result = await SarvamClient().text_to_speech(
            text, voice.language, voice=voice.name, sample_rate=WEB_OUTPUT_RATE)
        if not result.success:
            logger.warning("Sarvam TTS failed for %s: %s", language, result.error)
            return None
        return wav_to_pcm(result.get_audio_bytes()) or None
    except (wave.Error, EOFError, ValueError, OSError) as exc:
        logger.warning("Sarvam audio unusable for %s: %s", language, exc)
        return None


class LiveTurnClock:
    """
    Whether Live answers the caller's recording within the release's deadlines.

    The socket's response relay reports Live's output (`live_output`) and turn
    completions (`turn_completed`); the socket reports the caller starting
    (`caller_started`). When the caller stops, `wait_for_reply` returns None
    once Live completes its turn, or the reason it missed a deadline:
    "first_token_timeout" when Live said nothing within FIRST_TOKEN_TIMEOUT_S
    of the caller stopping, "turn_stalled" when its turn was not complete
    within PER_TURN_STALL_TIMEOUT_S.
    """

    def __init__(self) -> None:
        self._progress = asyncio.Event()
        self._complete = asyncio.Event()
        self.mid_turn = False   # Live has started a turn it has not finished
        self.answered = False   # Live completed a turn since the caller started

    def caller_started(self) -> None:
        """The caller began a new recording."""
        self.answered = False
        if not self.mid_turn:
            self._progress.clear()
        self._complete.clear()

    def live_output(self) -> None:
        """Live produced audio for a turn."""
        self.mid_turn = True
        self._progress.set()

    def interrupted(self) -> None:
        """The caller barged in; Live's turn was dropped."""
        self.mid_turn = False

    def turn_completed(self) -> None:
        """Live completed a turn."""
        self.mid_turn = False
        self.answered = True
        self._progress.set()
        self._complete.set()

    async def wait_for_reply(self) -> Optional[str]:
        """None when Live answers in time, otherwise the fallback reason."""
        if self.answered and not self.mid_turn:
            return None
        loop = asyncio.get_running_loop()
        started = loop.time()
        if self.mid_turn:
            self._complete.clear()
        else:
            try:
                await asyncio.wait_for(self._progress.wait(), timeout=FIRST_TOKEN_TIMEOUT_S)
            except asyncio.TimeoutError:
                return "first_token_timeout"
        remaining = PER_TURN_STALL_TIMEOUT_S - (loop.time() - started)
        try:
            await asyncio.wait_for(self._complete.wait(), timeout=max(remaining, 0.0))
        except asyncio.TimeoutError:
            return "turn_stalled"
        return None


class WebSarvamTurns:
    """
    Answers a web voice caller's recordings on the Sarvam path.

    One per socket, so the turn history stays with the connection. The answer
    is the VoiceTurnHandler's: the same grounding, the same Dose Boundary and
    the same voice length as the mobile voice routes. Each answered turn
    writes one study row (channel "web_voice", voice path "fallback"), keyed
    like the Live path's rows on the socket's session key: a signed-in
    participant's id, or an ephemeral id, which the log records as
    "anonymous".
    """

    def __init__(self, rag_pipeline: Any, safety_manager: Any, session_key: str,
                 participant: Any = None) -> None:
        # The row is written here, as "web_voice"; the handler writes none.
        self.handler = VoiceTurnHandler(rag_pipeline=rag_pipeline, safety_manager=safety_manager,
                                        study_logger=None, synthesise=None)
        self.session_key = session_key
        self.participant = participant
        # The answer path keys patient context on the participant; without
        # one its id is ephemeral.
        self.session = VoiceSession(
            participant_id=getattr(participant, "user_id", "") if participant is not None else "",
            care_role=getattr(participant, "role", "") or "",
            site_id=getattr(participant, "site_id", "") or "",
        )

    async def answer(self, websocket: Any, pcm: bytes, language: str, reason: str,
                     session_id: str = "") -> Optional[VoiceTurn]:
        """
        Answer one recording and send its frames.

        `reason` is why Sarvam answers: "operator_forced_fallback", or the
        deadline Live missed. It is recorded on the turn, on its frames, in its
        study row and in the deployment's voice report (forced turns apart
        from Live failures). `session_id` groups the row with the Live
        session it replaced, when there was one.
        """
        self.session.language = language
        turn = None
        if pcm:
            turn = await self.handler.handle_audio(self.session, pcm_to_wav(pcm),
                                                   voice_path=VOICE_PATH_FALLBACK)
        if turn is None:
            logger.info("Web voice recording had no question (%s)", reason)
            await websocket.send_json({"type": "turn_complete", "voice_path": VOICE_PATH_FALLBACK,
                                       "fallback_reason": reason, "heard": False})
            return None

        turn.voice_path = VOICE_PATH_FALLBACK
        turn.fallback_reason = reason
        record_voice_path(CHANNEL, reason)
        await self._record(turn, language, session_id or self.session.session_id)

        audio = await speak_for_web(turn.answer, language)
        if audio:
            await websocket.send_bytes(audio)
        await websocket.send_json({
            "type": "transcription", "role": "assistant", "text": turn.answer,
            "answer_kind": turn.answer_kind, "emergency_level": turn.emergency_level,
            "voice_path": VOICE_PATH_FALLBACK, "fallback_reason": reason,
        })
        await websocket.send_json({"type": "transcription", "role": "user", "text": turn.transcript})
        await websocket.send_json({"type": "turn_complete", "voice_path": VOICE_PATH_FALLBACK,
                                   "fallback_reason": reason})
        return turn

    async def _record(self, turn: VoiceTurn, language: str, session_id: str) -> None:
        """The turn's one study row, written before any of it is sent."""
        from study_logging import record_answer

        await record_answer(
            channel=CHANNEL, voice_path=VOICE_PATH_FALLBACK, user_id=self.session_key,
            session_id=session_id, site_id=self.session.site_id, care_role=self.session.care_role,
            query=turn.transcript, response=turn.answer, language=language,
            result={"sources": [s for s in turn.sources if isinstance(s, dict)],
                    "retrieval": turn.retrieval, "kg_check": turn.kg_check},
            safety={**turn.safety, "answer_kind": turn.answer_kind, "fallback_reason": turn.fallback_reason},
            provider="sarvam", stt_model="saaras-v3", tts_model="bulbul-v3",
        )
