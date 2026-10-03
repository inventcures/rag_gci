#!/usr/bin/env python3
"""
Tests for the Gemini Live turn adapter.

ADR 0007 makes Gemini Live the primary voice path with Sarvam as the automatic
fallback. T5 left one gap: LiveVoiceProvider calls `service.stream_audio(...)` and
`gemini_live/service.py` had no such method, so the primary path could not serve a
turn and every request fell through to Sarvam. That is the same class of silent
failure as the English-only index, so it is tested here rather than assumed fixed.

The seam under test is the boundary where this project's real risk sits: whether a
Live turn is grounded and safety-filtered through the same path as every other voice
turn, or whether it becomes a second answer generator with its own, weaker rules. So
the fakes replace only Google's SDK and nothing in this repository.

The real `google-genai` package is not installed in this environment, and these
tests must still run. That is deliberate: it forces the adapter to declare its
dependency rather than hide it behind a live import.
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_live_adapter import (  # noqa: E402
    GeminiLiveTurnAdapter,
    LiveSessionUnavailable,
    transcribe_with_live,
)
from voice_session import VoiceSession, VoiceTurn  # noqa: E402


class FakeSafety:
    """Mirrors what SafetyEnhancementsManager returns for a clean answer."""

    def __init__(self, response="Pain is managed with opioids."):
        self.response = response
        self.evidence_level = "B"
        self.emergency_level = "none"


class RecordingHandler:
    """
    Stands in for VoiceTurnHandler.

    Real grounding and real safety are not what is under test here, so they are not
    faked into appearing to pass. This records what it was asked to answer, which is
    the adapter's actual responsibility.
    """

    def __init__(self, answer="Pain is managed with opioids.", fail=False):
        self.calls = []
        self._answer = answer
        self._fail = fail

    async def handle(self, session, transcript, voice_path="", fallback_reason=None):
        self.calls.append(
            {
                "transcript": transcript,
                "voice_path": voice_path,
                "fallback_reason": fallback_reason,
                "session_id": session.session_id,
            }
        )
        if self._fail:
            raise RuntimeError("handler exploded")
        return VoiceTurn(
            transcript=transcript,
            answer=self._answer,
            answer_kind="ANSWER",
            evidence_level="B",
            emergency_level="none",
            source_count=2,
            voice_path=voice_path,
        )


class FakeLiveSession:
    """Duck-types GeminiLiveSession for the calls the adapter makes."""

    def __init__(self, transcript="what is morphine for?", stall=False):
        self.transcript = transcript
        self.stall = stall
        self.connected = False
        self.disconnected = False
        self.sent_audio: bytes = b""
        self.sent_language = None

    async def connect(self):
        self.connected = True

    async def send_audio(self, audio: bytes, language=None):
        self.sent_audio = audio
        self.sent_language = language

    async def wait_for_transcription(self, timeout):
        if self.stall:
            raise asyncio.TimeoutError("no transcription")
        return self.transcript

    def get_transcription(self, clear=True):
        return self.transcript

    async def disconnect(self):
        self.disconnected = True


class FakeLiveService:
    def __init__(self, session=None, connect_error=None, available=True):
        self._session = session
        self._connect_error = connect_error
        self._available = available
        self.sessions_created = 0

    def is_available(self):
        return self._available

    async def create_session(self, language="en-IN", **kwargs):
        if self._connect_error:
            raise self._connect_error
        self.sessions_created += 1
        return self._session


def make_session():
    return VoiceSession(session_id="vs_test", language="hi-IN")


async def drain(agen):
    return [event async for event in agen]


class TestTranscribeWithLive(unittest.IsolatedAsyncioTestCase):
    """The transcription primitive every other guarantee rests on."""

    async def test_returns_the_transcript(self):
        svc = FakeLiveService(FakeLiveSession(transcript="morphine kya hai?"))
        out = await transcribe_with_live(svc, make_session(), b"\x00\x01")
        self.assertEqual(out, "morphine kya hai?")

    async def test_audio_reaches_the_live_session(self):
        """Otherwise the transcript is fiction."""
        fake = FakeLiveSession()
        await transcribe_with_live(FakeLiveService(fake), make_session(), b"BYTES")
        self.assertEqual(fake.sent_audio, b"BYTES")

    async def test_absent_sdk_raises_rather_than_returning_silence(self):
        """The router must record a real reason, not fall back for no reason."""
        with self.assertRaises(LiveSessionUnavailable):
            await transcribe_with_live(
                FakeLiveService(available=False), make_session(), b""
            )

    async def test_session_that_will_not_open_raises(self):
        with self.assertRaises(LiveSessionUnavailable):
            await transcribe_with_live(
                FakeLiveService(connect_error=ConnectionError("dial failed")),
                make_session(),
                b"",
            )

    async def test_silence_is_not_a_failure(self):
        """
        A quiet room must not look like a provider outage.

        It yields no transcript, which the caller handles, and it must not raise.
        """
        svc = FakeLiveService(FakeLiveSession(transcript=""))
        self.assertEqual(await transcribe_with_live(svc, make_session(), b""), "")

    async def test_session_is_released_even_when_the_turn_goes_wrong(self):
        fake = FakeLiveSession(stall=True)
        with self.assertRaises(LiveSessionUnavailable):
            await transcribe_with_live(FakeLiveService(fake), make_session(), b"")
        self.assertTrue(fake.disconnected, "Live session leaked on failure")

    async def test_a_closing_failure_does_not_mask_the_turn(self):
        class StubbornSession(FakeLiveSession):
            async def disconnect(self):
                raise RuntimeError("socket already gone")

        # The transcript still comes back. A close failure is not the user's problem.
        svc = FakeLiveService(StubbornSession(transcript="hello"))
        self.assertEqual(await transcribe_with_live(svc, make_session(), b""), "hello")


class TestGeminiLiveTurnAdapter(unittest.IsolatedAsyncioTestCase):
    async def test_healthy_turn_reaches_the_shared_pipeline(self):
        live = FakeLiveService(FakeLiveSession(transcript="what is morphine for?"))
        handler = RecordingHandler()
        adapter = GeminiLiveTurnAdapter(live, handler, make_session())
        events = await drain(adapter.stream_audio(b"\x00", "hi-IN"))
        final = events[-1]
        self.assertTrue(final.get("final"), "adapter must emit a final event")
        self.assertEqual(final["turn"].transcript, "what is morphine for?")
        self.assertEqual(final["turn"].voice_path, "live")
        self.assertEqual(len(handler.calls), 1)

    async def test_answer_comes_from_the_shared_handler_not_from_live(self):
        """
        The most important assertion in this file.

        If Live generated the answer itself, the dose boundary would have a second
        and weaker implementation.
        """
        live = FakeLiveService(FakeLiveSession())
        handler = RecordingHandler(answer="Pain is managed with opioids.")
        adapter = GeminiLiveTurnAdapter(live, handler, make_session())
        events = await drain(adapter.stream_audio(b"", "hi-IN"))
        self.assertEqual(events[-1]["turn"].answer, "Pain is managed with opioids.")

    async def test_empty_transcript_produces_no_turn(self):
        """
        Nothing substantive is recorded, so protocol 4.1's denominator stays honest.
        """
        handler = RecordingHandler()
        adapter = GeminiLiveTurnAdapter(
            FakeLiveService(FakeLiveSession(transcript="")), handler, make_session()
        )
        events = await drain(adapter.stream_audio(b"", "hi-IN"))
        self.assertTrue(events[-1].get("final"))
        self.assertIsNone(events[-1]["turn"])
        self.assertEqual(handler.calls, [], "empty transcript reached the pipeline")

    async def test_absent_sdk_surfaces_for_the_router(self):
        adapter = GeminiLiveTurnAdapter(
            FakeLiveService(available=False), RecordingHandler(), make_session()
        )
        with self.assertRaises(LiveSessionUnavailable):
            await drain(adapter.stream_audio(b"", "hi-IN"))

    async def test_client_language_reaches_the_provider(self):
        fake = FakeLiveSession()
        adapter = GeminiLiveTurnAdapter(
            FakeLiveService(fake), RecordingHandler(), make_session()
        )
        await drain(adapter.stream_audio(b"", "mr-IN"))
        self.assertEqual(fake.sent_language, "mr-IN")

    async def test_session_id_is_carried_for_the_study_log(self):
        handler = RecordingHandler()
        session = make_session()
        adapter = GeminiLiveTurnAdapter(FakeLiveService(FakeLiveSession()), handler, session)
        await drain(adapter.stream_audio(b"", "hi-IN"))
        self.assertEqual(handler.calls[0]["session_id"], session.session_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)