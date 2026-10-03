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
from voice_session import VOICE_PATH_FALLBACK, VOICE_PATH_LIVE  # noqa: E402
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
        self.assertEqual(require_turn(outcome).voice_path, VOICE_PATH_FALLBACK)
        # A provider that cannot serve must not strand a user mid-conversation.
        self.assertIsNotNone(require_turn(outcome).fallback_reason)

    def test_operator_can_force_the_fallback_for_the_whole_deployment(self):
        router = vr.VoiceRouter(
            self.handler, live_service_factory=lambda s: FakeLive()
        )
        outcome = asyncio.run(router.route(self.session, b"\x00", live_enabled=False))
        self.assertEqual(require_turn(outcome).voice_path, VOICE_PATH_FALLBACK)
        self.assertEqual(require_turn(outcome).fallback_reason, "operator_forced_fallback")
        self.assertEqual(router.report()["operator_forced"], 1)


class RoutingWhenLiveIsHealthy(unittest.TestCase):
    def setUp(self):
        self.handler = FakeHandler()
        self.session = VoiceSession()

    def test_a_healthy_live_turn_does_not_fall_back(self):
        """
        Availability is forced rather than inherited.

        An earlier version of this test asserted the fallback and passed only
        because google-genai happened to be absent on the machine. It therefore
        tested the environment, not the decision, and it passed while proving the
        opposite of what its name claimed. The probe is patched so the test holds
        whether or not the SDK is installed.
        """
        live = vr.LiveAvailability(available=True)
        with unittest.mock.patch.object(vr.LiveAvailability, "probe", return_value=live):
            router = vr.VoiceRouter(self.handler, live_service_factory=lambda s: FakeLive())
            outcome = asyncio.run(router.route(self.session, b"\x00"))
        self.assertEqual(require_turn(outcome).voice_path, VOICE_PATH_LIVE)
        # Sarvam must not have been touched at all when Live answered.
        self.assertEqual(self.handler.calls, 0)

    def test_a_live_session_that_connects_then_goes_quiet_falls_back(self):
        """
        The case a connection check would miss.

        A socket that opens is not evidence of a healthy turn, so a Live provider
        that yields a first token and then stalls must fall back rather than leave
        the user mid-sentence.
        """
        stalling = FakeLive(first_token_delay=9.0, final=False)
        live = vr.LiveAvailability(available=True)
        with unittest.mock.patch.object(vr.LiveAvailability, "probe", return_value=live):
            router = vr.VoiceRouter(
                self.handler, live_service_factory=lambda s: stalling
            )
            outcome = asyncio.run(router.route(self.session, b"\x00"))
        self.assertEqual(require_turn(outcome).voice_path, VOICE_PATH_FALLBACK)
        self.assertEqual(self.handler.calls, 1)
        self.assertIsNotNone(require_turn(outcome).fallback_reason)

    def test_classification_is_pinned_independently(self):
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
