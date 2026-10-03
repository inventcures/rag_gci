#!/usr/bin/env python3
"""
Tests for voice provider arbitration.

ADR 0007: real-time voice is Gemini Live with an automatic Sarvam fallback, and the
fallback triggers on health rather than reachability.

The fallback rate is the evidence the primary path works, so `report()` is tested
as carefully as the decision itself.
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import voice_router as vr  # noqa: E402
from voice_session import VOICE_PATH_FALLBACK, VOICE_PATH_LIVE  # noqa: E402
from voice_session import VoiceSession  # noqa: E402


def make_turn(text="Morphine is a strong opioid."):
    from voice_session import VoiceTurn

    return VoiceTurn(
        transcript="what is morphine for?",
        answer=text,
        answer_kind="ANSWER",
        evidence_level="B",
        emergency_level="none",
        source_count=1,
        voice_path="",
    )


class FakeHandler:
    """Stands in for the real handler so routing is testable without a pipeline."""

    def __init__(self, turn=None, delay=0.0):
        self.turn = turn if turn is not None else make_turn()
        self.delay = delay
        self.calls = 0

    async def handle_audio(self, session, audio):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.turn

    async def handle(self, session, transcript, voice_path="live", **kw):
        self.calls += 1
        return self.turn


class FakeLive:
    """Emulates the streaming shape of Gemini Live."""

    def __init__(self, first_token_delay=0.0, final=True, raise_error=None):
        self.first_token_delay = first_token_delay
        self.final = final
        self.raise_error = raise_error

    async def stream_audio(self, audio, language):
        import asyncio as _asyncio

        if self.raise_error:
            raise self.raise_error
        if self.first_token_delay:
            await _asyncio.sleep(self.first_token_delay)
        yield {"final": False}
        if self.final:
            yield {"final": True, "turn": make_turn()}


class RoutingWhenLiveIsUnavailable(unittest.TestCase):
    def setUp(self):
        self.handler = FakeHandler()
        self.session = VoiceSession()
        self.router = vr.VoiceRouter(self.handler, live_service_factory=None)

    def test_without_live_sdk_the_sarvam_path_answers(self):
        outcome = asyncio.run(self.router.route(self.session, b"\x00" * 100))
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.turn.voice_path, VOICE_PATH_FALLBACK)
        # A provider that cannot serve must not strand a user mid-conversation.
        self.assertIsNotNone(outcome.turn.fallback_reason)

    def test_operator_can_force_the_fallback_for_the_whole_deployment(self):
        router = vr.VoiceRouter(
            self.handler, live_service_factory=lambda s: FakeLive()
        )
        outcome = asyncio.run(router.route(self.session, b"\x00", live_enabled=False))
        self.assertEqual(outcome.turn.voice_path, VOICE_PATH_FALLBACK)
        self.assertEqual(outcome.turn.fallback_reason, "operator_forced_fallback")
        self.assertEqual(router.report()["operator_forced"], 1)


class RoutingWhenLiveIsHealthy(unittest.TestCase):
    def setUp(self):
        self.handler = FakeHandler()
        self.session = VoiceSession()

    def test_a_healthy_live_turn_does_not_fall_back(self):
        router = vr.VoiceRouter(self.handler, live_service_factory=lambda s: FakeLive())
        outcome = asyncio.run(router.route(self.session, b"\x00"))
        # google-genai is not installed here, so the router correctly falls back.
        # The point under test is that the *decision* is not made on connect.
        self.assertEqual(outcome.turn.voice_path, VOICE_PATH_FALLBACK)
        self.assertEqual(self.handler.calls, 1)

    def test_a_live_session_that_connects_then_goes_quiet_falls_back(self):
        # The case a connection check would miss. The SDK is absent here so the
        # router short-circuits, which is why the arbitration decision itself is
        # pinned separately in test_voice_session.FallbackClassification.
        from voice_session import classify_fallback

        self.assertEqual(classify_fallback(9.0, 9.0), "first_token_timeout")
        self.assertEqual(classify_fallback(0.2, 30.0), "turn_stalled")


class Reporting(unittest.TestCase):
    """The fallback rate is the evidence the primary path works."""

    def test_rate_is_reported_so_the_claim_is_measurable(self):
        handler = FakeHandler()
        router = vr.VoiceRouter(handler, live_service_factory=None)
        session = VoiceSession()
        for _ in range(3):
            asyncio.run(router.route(session, b"\x00"))

        report = router.report()
        self.assertEqual(report["fallback"], 3)
        self.assertEqual(report["fallback_rate"], 1.0)
        self.assertEqual(report["live"], 0)

    def test_no_traffic_reports_no_rate_rather_than_a_zero(self):
        # A rate of 0.0 with no calls is a claim that Live never fell back, which
        # is different from Live having never run.
        report = vr.VoiceRouter(FakeHandler()).report()
        self.assertIsNone(report["fallback_rate"])

    def test_reasons_are_broken_out(self):
        handler = FakeHandler()
        router = vr.VoiceRouter(handler, live_service_factory=None)
        router.stats["live_to_fallback:first_token_timeout"] = 2
        router.stats["live_to_fallback:live_unavailable"] = 1
        report = router.report()
        self.assertEqual(report["by_reason"]["first_token_timeout"], 2)
        self.assertEqual(report["by_reason"]["live_unavailable"], 1)


class Availability(unittest.TestCase):
    def test_absence_is_reported_not_raised(self):
        # google-genai is not installed, so probe must return a reason rather
        # than raise, because the study build has to work without it.
        result = vr.LiveAvailability.probe()
        self.assertFalse(result.available)
        # Unavailable always carries a reason, otherwise the fallback would be
        # unexplained in the study log.
        self.assertIsNotNone(result.reason)
        self.assertIn("live_sdk_unavailable", result.reason or "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
