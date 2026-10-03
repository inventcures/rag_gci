#!/usr/bin/env python3
"""
Tests for voice provider selection.

The dangerous failure here is not a crash. It is an operator override that quietly
becomes the permanent configuration, after which nobody is maintaining the Live
path and nobody remembers who turned it off. So the tests care about defaults,
auditability, and whether a change needs to be explainable.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_provider import (  # noqa: E402
    PROVIDER_LIVE,
    PROVIDER_SARVAM,
    VoiceProviderControl,
    VoiceProviderSetting,
)


class TempControl(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="ps-voice-"))
        self.control = VoiceProviderControl(
            path=self.dir / "setting.json",
            audit_path=self.dir / "audit.jsonl",
        )


class DefaultIsLive(TempControl):
    def test_no_setting_means_live(self):
        # ADR 0007 makes Live the primary path, so silence means Live.
        self.assertTrue(self.control.current().live_enabled)
        self.assertEqual(self.control.current().provider, PROVIDER_LIVE)

    def test_an_unreadable_setting_falls_back_to_live_not_to_sarvam(self):
        # An unreadable file must not strand the deployment on the fallback path
        # with nobody aware that it happened.
        self.control.path.write_text("{ this is not json", encoding="utf-8")
        fresh = VoiceProviderControl(path=self.control.path, audit_path=self.control.audit_path)
        self.assertTrue(fresh.current().live_enabled)

    def test_an_unknown_provider_is_treated_as_live(self):
        self.control.path.write_text(json.dumps({"provider": "whisper"}), encoding="utf-8")
        fresh = VoiceProviderControl(path=self.control.path, audit_path=self.control.audit_path)
        self.assertEqual(fresh.current().provider, PROVIDER_LIVE)


class ChangingProvider(TempControl):
    def test_an_admin_can_force_sarvam(self):
        setting = self.control.set_provider(PROVIDER_SARVAM, "ashish", "Live quota exhausted")
        self.assertEqual(setting.provider, PROVIDER_SARVAM)
        self.assertFalse(setting.live_enabled)
        self.assertEqual(setting.previous_provider, PROVIDER_LIVE)

    def test_the_change_persists(self):
        self.control.set_provider(PROVIDER_SARVAM, "ashish", "incident 412")
        reloaded = VoiceProviderControl(
            path=self.control.path, audit_path=self.control.audit_path
        )
        self.assertFalse(reloaded.current().live_enabled)

    def test_a_change_back_to_live_is_allowed_and_recorded(self):
        self.control.set_provider(PROVIDER_SARVAM, "ashish", "incident 412")
        setting = self.control.set_provider(PROVIDER_LIVE, "ashish", "quota restored")
        self.assertTrue(setting.live_enabled)
        self.assertEqual(setting.previous_provider, PROVIDER_SARVAM)

    def test_a_reason_is_required(self):
        # An override nobody can explain later is how a deliberate incident
        # response turns into the permanent configuration.
        for reason in ("", "   "):
            with self.assertRaises(ValueError):
                self.control.set_provider(PROVIDER_SARVAM, "ashish", reason)

    def test_an_unknown_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            self.control.set_provider("whisper", "ashish", "why not")

    def test_the_setting_is_read_every_turn(self):
        # The router reads this per turn, so an override applies without a restart.
        self.assertTrue(self.control.current().live_enabled)
        self.control.set_provider(PROVIDER_SARVAM, "ashish", "incident")
        self.assertFalse(self.control.current().live_enabled)


class AuditTrail(TempControl):
    def test_every_change_is_audited(self):
        self.control.set_provider(PROVIDER_SARVAM, "ashish", "incident 412")
        self.control.set_provider(PROVIDER_LIVE, "ashish", "restored")

        history = self.control.history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["provider"], PROVIDER_LIVE)
        self.assertEqual(history[0]["previous_provider"], PROVIDER_SARVAM)
        self.assertEqual(history[1]["reason"], "incident 412")

    def test_a_no_op_change_is_not_audited(self):
        # Otherwise the trail fills with entries that say nothing happened.
        self.control.set_provider(PROVIDER_LIVE, "ashish", "already live")
        self.assertEqual(self.control.history(), [])

    def test_an_audit_failure_does_not_lose_the_change(self):
        self.control.audit_path = Path("/proc/definitely/not/writable/audit.jsonl")
        setting = self.control.set_provider(PROVIDER_SARVAM, "ashish", "incident")
        # The operator's intent must land even if the audit write fails; the
        # failure itself is logged loudly in the implementation.
        self.assertEqual(setting.provider, PROVIDER_SARVAM)
        self.assertFalse(self.control.current().live_enabled)


class RoutingRespectsTheSelection(TempControl):
    def test_forcing_sarvam_marks_turns_as_operator_forced(self):
        import asyncio

        from voice_router import VoiceRouter
        from voice_session import VOICE_PATH_FALLBACK, VoiceSession

        session = VoiceSession()

        class Handler:
            async def handle_audio(self, session, audio, voice_path="live"):
                from voice_session import VoiceTurn

                return VoiceTurn(
                    transcript="q", answer="a", answer_kind="ANSWER",
                    evidence_level="B", emergency_level="none",
                    source_count=0, voice_path="",
                )

        self.control.set_provider(PROVIDER_SARVAM, "ashish", "incident 412")
        router = VoiceRouter(Handler(), live_service_factory=lambda s: None,
                             control=self.control)

        outcome = asyncio.run(router.route(object(), b"\x00"))
        self.assertEqual(outcome.turn.voice_path, VOICE_PATH_FALLBACK)
        self.assertEqual(outcome.turn.fallback_reason, "operator_forced_fallback")
        self.assertEqual(router.report()["operator_forced"], 1)

    def test_the_report_distinguishes_forced_from_automatic(self):
        # A deliberate choice and a provider failure are different events and
        # must not be added into one number.
        import asyncio

        from voice_router import VoiceRouter
        from voice_session import VoiceSession

        session = VoiceSession()

        class Handler:
            async def handle_audio(self, session, audio, voice_path="live"):
                from voice_session import VoiceTurn

                return VoiceTurn(
                    transcript="q", answer="a", answer_kind="ANSWER",
                    evidence_level="B", emergency_level="none",
                    source_count=0, voice_path="",
                )

        router = VoiceRouter(Handler(), live_service_factory=lambda s: None,
                             control=self.control)
        asyncio.run(router.route(session, b"\x00"))

        report = router.report()
        self.assertEqual(report["selected_provider"], PROVIDER_LIVE)
        self.assertEqual(report["operator_forced"], 0)
        self.assertEqual(report["fallback"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
