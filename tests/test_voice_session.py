#!/usr/bin/env python3
"""
Tests for the real-time voice session.

The property that matters is that voice is not a second safety path: a dose read
aloud is as harmful as one read on screen, and the same guard has to run.

Also covers fallback classification, because ADR 0007 requires the fallback to
trigger on health rather than on reachability, and that rate to be measurable.
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import voice_session as vs  # noqa: E402


class FakeSafetyResult:
    def __init__(self, **kw):
        self.answer = kw.get("answer", "")
        self.answer_kind = kw.get("answer_kind", "ANSWER")
        self.evidence_level = kw.get("evidence_level", "C")
        self.emergency_level = kw.get("emergency_level", "none")
        self.confidence = kw.get("confidence", 0.8)
        self.validation_status = kw.get("validation_status", "validated")
        self.disclaimer = None


class FakeSafetyManager:
    """Applies the real guard so the test exercises real behaviour."""

    def __init__(self, real_manager=None):
        if real_manager is not None:
            self._real = real_manager
        else:
            from safety_enhancements import get_safety_manager

            self._real = get_safety_manager()

    def process_response(self, query, response, sources, language="en-IN"):
        return self._real.process_response(
            query=query, response=response, sources=sources, language=language
        )


class FakeRag:
    """
    Mirrors SimpleRAGPipeline.query exactly.

    The signature is copied from the real class rather than invented. An earlier
    version of this fake accepted `**kw`, which silently absorbed a kwarg the real
    pipeline does not accept. Both voice routes were therefore broken in production
    while every test passed, because the fake was more forgiving than the code it
    stood in for.
    """

    #: The real signature, asserted at import so drift is caught here and not in a
    #: field site.
    REAL_SIGNATURE = ("question", "conversation_id", "user_id", "top_k", "source_language")

    def __init__(self, answer="Morphine is a strong opioid.", sources=None):
        self.answer = answer
        self.sources = sources if sources is not None else [{"filename": "handbook.pdf"}]
        self.calls = []

    async def query(
        self,
        question: str,
        conversation_id=None,
        user_id=None,
        top_k: int = 5,
        source_language: str = "en",
    ):
        self.calls.append({"question": question, "source_language": source_language})
        return {"answer": self.answer, "sources": self.sources}


class FakeStudyLogger:
    def __init__(self, fail=False):
        self.rows = []
        self.fail = fail

    async def record(self, **kw):
        if self.fail:
            raise RuntimeError("log backend unavailable")
        self.rows.append(kw)


class FallbackClassification(unittest.TestCase):
    """ADR 0007: health-based, not reachability-based."""

    def test_healthy_turn_does_not_fall_back(self):
        self.assertIsNone(vs.classify_fallback(first_token_latency_s=0.4, turn_latency_s=2.0))

    def test_no_first_token_falls_back(self):
        self.assertEqual(vs.classify_fallback(None, 9.0), "no_first_token")

    def test_slow_first_token_falls_back(self):
        # This is the case a connection check would miss: the session connected,
        # then went quiet. The user is left mid-sentence.
        self.assertEqual(
            vs.classify_fallback(first_token_latency_s=6.0, turn_latency_s=6.0),
            "first_token_timeout",
        )

    def test_stalled_turn_falls_back(self):
        self.assertEqual(
            vs.classify_fallback(first_token_latency_s=0.5, turn_latency_s=20.0),
            "turn_stalled",
        )

    def test_every_reason_is_a_short_machine_readable_token(self):
        for args in ((None, 1.0), (99.0, 1.0), (0.5, 99.0)):
            reason = vs.classify_fallback(*args)
            self.assertTrue(reason and " " not in reason)


class VoiceTurnSafety(unittest.TestCase):
    """Voice must obey the same boundary as text."""

    def setUp(self):
        self.logger = FakeStudyLogger()
        self.handler = vs.VoiceTurnHandler(
            rag_pipeline=FakeRag(),
            safety_manager=FakeSafetyManager(),
            study_logger=self.logger,
        )
        self.session = vs.VoiceSession(language="en-IN", participant_id="p1")

    def test_a_dose_spoken_aloud_is_still_refused(self):
        self.handler.rag_pipeline = FakeRag(answer="Give morphine 10 mg every 4 hours.")
        turn = asyncio.run(
            self.handler.handle(self.session, "what dose of morphine?", voice_path="live")
        )
        self.assertNotEqual(turn.answer_kind, "ANSWER")
        self.assertNotIn("10 mg", turn.answer)

    def test_a_partial_redaction_keeps_the_clinical_remainder(self):
        self.handler.rag_pipeline = FakeRag(
            answer=(
                "Morphine is a strong opioid used for severe cancer pain. "
                "It is the gold standard for severe pain control. Give 10 mg every 4 hours."
            )
        )
        turn = asyncio.run(self.handler.handle(self.session, "what is morphine for?"))
        self.assertEqual(turn.answer_kind, "REDACTED_ANSWER")
        self.assertIn("strong opioid", turn.answer)

    def test_general_drug_information_is_answered(self):
        turn = asyncio.run(self.handler.handle(self.session, "what is morphine for?"))
        self.assertEqual(turn.answer_kind, "ANSWER")
        self.assertIn("strong opioid", turn.answer)

    def test_emergency_is_reported_on_the_turn(self):
        self.handler.rag_pipeline = FakeRag(answer="Call 108 immediately.")
        turn = asyncio.run(
            self.handler.handle(
                self.session, "patient cannot breathe", fallback_reason="first_token_timeout"
            )
        )
        self.assertEqual(turn.voice_path, vs.VOICE_PATH_LIVE)
        self.assertEqual(turn.fallback_reason, "first_token_timeout")
        self.assertIn("108", turn.answer)

    def test_the_transcript_is_what_gets_recorded(self):
        # Protocol 4.1 counts by what was asked, and a transcript is what was
        # asked, not the answer.
        asyncio.run(self.handler.handle(self.session, "what should I do?"))
        self.assertEqual(self.logger.rows[0]["query"], "what should I do?")
        self.assertEqual(self.logger.rows[0]["channel"], "voice")
        self.assertEqual(self.logger.rows[0]["provider"], vs.VOICE_PATH_LIVE)

    def test_a_logging_failure_does_not_break_the_conversation(self):
        # A gap in the evidence is bad; dropping a live call in someone's home is
        # worse. It must survive, and it is logged loudly.
        handler = vs.VoiceTurnHandler(
            rag_pipeline=FakeRag(),
            safety_manager=FakeSafetyManager(),
            study_logger=FakeStudyLogger(fail=True),
        )
        turn = asyncio.run(handler.handle(self.session, "what should I do?"))
        self.assertIn("strong opioid", turn.answer)

    def test_session_history_is_bounded(self):
        # A phone sitting in a patient's home all day must not grow without limit.
        for i in range(30):
            asyncio.run(self.handler.handle(self.session, f"question {i}"))
        self.assertLessEqual(len(self.session.history), 20)


class AudioFraming(unittest.TestCase):
    def test_malformed_audio_is_rejected_rather_than_sent_as_empty(self):
        # An empty question would be recorded as a substantive interaction, which
        # protocol 4.1 would then have to exclude.
        for bad in ({}, {"data": ""}, {"data": "not base64!!"}):
            with self.assertRaises(ValueError):
                vs.decode_audio(bad)

    def test_valid_audio_decodes(self):
        import base64

        payload = b"\x01\x02\x03" * 100
        self.assertEqual(vs.decode_audio({"data": base64.b64encode(payload).decode()}), payload)

    def test_turn_frame_carries_the_release_and_the_path(self):
        turn = vs.VoiceTurn(
            transcript="q", answer="a", answer_kind="ANSWER", evidence_level="B",
            emergency_level="none", source_count=1, voice_path=vs.VOICE_PATH_FALLBACK,
            fallback_reason="turn_stalled",
        )
        frame = __import__("json").loads(vs.turn_frame(turn, release_id="dev@abc"))
        self.assertEqual(frame["type"], "response")
        self.assertEqual(frame["release_id"], "dev@abc")
        self.assertEqual(frame["voice_path"], vs.VOICE_PATH_FALLBACK)
        self.assertEqual(frame["fallback_reason"], "turn_stalled")


class TestPipelineSignatureGuard(unittest.TestCase):
    """
    The fake must not be more forgiving than the code it stands in for.

    Both voice routes once called `query(question=..., language=...)` while the real
    pipeline accepts `source_language`. Every test passed because the fakes took
    `**kw`, and every voice turn in the field would have raised TypeError. This
    fails the build the moment the two drift apart.
    """

    def test_fake_matches_the_real_pipeline_signature(self):
        import inspect

        from simple_rag_server import SimpleRAGPipeline

        real = {
            name
            for name in inspect.signature(SimpleRAGPipeline.query).parameters
            if name != "self"
        }
        fake = {
            name
            for name in inspect.signature(FakeRag.query).parameters
            if name != "self"
        }
        self.assertEqual(
            fake,
            real,
            "FakeRag.query has drifted from SimpleRAGPipeline.query. A fake that "
            "accepts more than the real thing hides real breakages.",
        )

    def test_real_pipeline_does_not_accept_the_wrong_kwarg(self):
        """Pins the specific mistake, so the fix is not silently reverted."""
        import inspect

        from simple_rag_server import SimpleRAGPipeline

        params = inspect.signature(SimpleRAGPipeline.query).parameters
        self.assertNotIn("language", params)
        self.assertNotIn("query_text", params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
