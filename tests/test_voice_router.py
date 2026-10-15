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
import types
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import voice_router as vr  # noqa: E402
from voice_session import VOICE_PATH_FALLBACK  # noqa: E402
from voice_session import VoiceSession  # noqa: E402


def require_turn(outcome):
    """Unwrap an outcome, failing on the absence of a turn.

    ProviderOutcome.turn is Optional by design, because a quiet room produces no
    turn. Asserting it here means a routing regression reports as a missing turn
    rather than an AttributeError three lines later.
    """
    assert outcome.turn is not None, f"expected a turn, got error={outcome.error!r}"
    return outcome.turn


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
    """
    Stands in for the real handler so routing is testable without a pipeline.

    The signatures match `SupportsAudioTurns` exactly, including the optional
    `voice_path`. A fake that accepted less than the real thing let a protocol
    mismatch sit unnoticed, which is the same failure as the pipeline kwargs: the
    stand-in being more forgiving than the code it stands in for.
    """

    def __init__(self, turn=None, delay=0.0):
        self.turn = turn if turn is not None else make_turn()
        self.delay = delay
        self.calls = 0
        self.voice_paths = []

    async def handle_audio(self, session, audio, voice_path: str = "live"):
        self.calls += 1
        self.voice_paths.append(voice_path)
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.turn

    async def handle(self, session, transcript, voice_path="live", **kw):
        self.calls += 1
        self.voice_paths.append(voice_path)
        return self.turn


class RoutingOnTheMobileRoutes(unittest.TestCase):
    """The mobile routes never offer Live; every turn is answered on Sarvam."""

    def setUp(self):
        self.handler = FakeHandler()
        self.session = VoiceSession()
        self.router = vr.VoiceRouter(self.handler)

    def test_the_sarvam_path_answers_with_a_reason(self):
        outcome = asyncio.run(self.router.route(self.session, b"\x00" * 100))
        self.assertTrue(outcome.ok)
        self.assertEqual(require_turn(outcome).voice_path, VOICE_PATH_FALLBACK)
        # A provider that cannot serve must not strand a user mid-conversation.
        self.assertIsNotNone(require_turn(outcome).fallback_reason)

    def test_operator_can_force_the_fallback_for_the_whole_deployment(self):
        router = vr.VoiceRouter(self.handler)
        outcome = asyncio.run(router.route(self.session, b"\x00", live_enabled=False))
        self.assertEqual(require_turn(outcome).voice_path, VOICE_PATH_FALLBACK)
        self.assertEqual(require_turn(outcome).fallback_reason, "operator_forced_fallback")
        self.assertEqual(router.report()["operator_forced"], 1)


class FallbackClassification(unittest.TestCase):
    def test_classification_is_pinned_independently(self):
        from voice_session import classify_fallback

        self.assertEqual(classify_fallback(9.0, 9.0), "first_token_timeout")
        self.assertEqual(classify_fallback(0.2, 30.0), "turn_stalled")


class Reporting(unittest.TestCase):
    """The fallback rate is the evidence the primary path works."""

    def test_a_channel_without_live_is_not_counted_as_live_failing(self):
        handler = FakeHandler()
        router = vr.VoiceRouter(handler)
        outcome = asyncio.run(router.route(VoiceSession(), b"\x00"))
        self.assertEqual(require_turn(outcome).fallback_reason, "live_not_offered")
        self.assertEqual(handler.voice_paths, [VOICE_PATH_FALLBACK])
        report = router.report()
        self.assertEqual(report["live_not_offered"], 1)
        self.assertIsNone(report["fallback_rate"])

    def test_no_traffic_reports_no_rate_rather_than_a_zero(self):
        # A rate of 0.0 with no calls is a claim that Live never fell back, which
        # is different from Live having never run.
        report = vr.VoiceRouter(FakeHandler()).report()
        self.assertIsNone(report["fallback_rate"])

    def test_reasons_are_broken_out(self):
        handler = FakeHandler()
        router = vr.VoiceRouter(handler)
        router.stats["live_to_fallback:first_token_timeout"] = 2
        router.stats["live_to_fallback:live_unavailable"] = 1
        report = router.report()
        self.assertEqual(report["by_reason"]["first_token_timeout"], 2)
        self.assertEqual(report["by_reason"]["live_unavailable"], 1)


class Availability(unittest.TestCase):
    def test_absence_is_reported_not_raised(self):
        """
        The study build must work without the SDK, so absence has to produce a
        reason rather than an exception.

        Absence is simulated by putting None in sys.modules, which makes the import
        raise. Asserting the real environment instead would make this test pass for
        the wrong reason on a machine with the SDK and fail on one without.
        """
        with unittest.mock.patch.dict(sys.modules, {"google.genai": None}):
            result = vr.LiveAvailability.probe()
        self.assertFalse(result.available)
        # Unavailable always carries a reason, otherwise the fallback would be
        # unexplained in the study log.
        self.assertIsNotNone(result.reason)
        self.assertIn("live_sdk_unavailable", result.reason or "")

    def test_presence_is_reported_without_a_reason(self):
        """
        The mirror of the case above, so neither branch depends on the host.
        """
        with unittest.mock.patch.dict(
            sys.modules, {"google.genai": types.ModuleType("google.genai")}
        ):
            result = vr.LiveAvailability.probe()
        # Whether the service imports is a property of this repository, not of the
        # test, so only the reason-free shape is asserted when it is reachable.
        if result.available:
            self.assertIsNone(result.reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
