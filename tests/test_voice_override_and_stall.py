#!/usr/bin/env python3
"""
The voice override governs every voice channel, and a stalled Live session
falls back (review 3, ticket 16).

Before this:

  * the administrator's "force Sarvam" setting was read only by the mobile
    router, which never offers Live; web voice (/ws/voice) and WhatsApp voice
    notes opened Live regardless, so forcing Sarvam changed nothing;
  * a web Live session that went quiet was never cut short and the caller
    waited in silence;
  * operator-forced turns were counted into the Live failure rate;
  * on web voice a turn the Dose Boundary withheld was shown as text only:
    the caller, listening, heard silence.
"""

import asyncio
import base64
import io
import json
import os
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path
from typing import Dict, List
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import voice_router as vr  # noqa: E402
from voice_provider import PROVIDER_SARVAM, VoiceProviderControl  # noqa: E402
from voice_session import VOICE_PATH_FALLBACK, VOICE_PATH_LIVE, VoiceSession, VoiceTurn  # noqa: E402



def a_turn() -> VoiceTurn:
    return VoiceTurn(transcript="q", answer="a", answer_kind="ANSWER", evidence_level="B",
                     emergency_level="none", source_count=0, voice_path="")


class SarvamHandler:
    """The Sarvam path's handler, as the router sees it."""

    def __init__(self):
        self.calls = 0

    async def handle_audio(self, session, audio, voice_path: str = "live"):
        self.calls += 1
        return a_turn()


def forced_sarvam_control(test: unittest.TestCase) -> VoiceProviderControl:
    """A control in a temp dir with Sarvam forced, installed as the deployment's."""
    tmp = Path(tempfile.mkdtemp())
    control = VoiceProviderControl(path=tmp / "voice_provider.json", audit_path=tmp / "audit.jsonl")
    control.set_provider(PROVIDER_SARVAM, "dr-ashish", "incident 16")
    patcher = mock.patch("voice_provider._control", control)
    patcher.start()
    test.addCleanup(patcher.stop)
    return control


def fresh_deployment_tally(test: unittest.TestCase) -> None:
    """Every channel's turns count into one deployment-wide tally; start it empty."""
    patcher = mock.patch.object(vr, "_DEPLOYMENT_STATS", {}, create=True)
    patcher.start()
    test.addCleanup(patcher.stop)


class TheLiveFailureRateExcludesOperatorForcedTurns(unittest.TestCase):
    def setUp(self):
        fresh_deployment_tally(self)

    def test_forced_turns_are_not_live_failures(self):
        vr.record_voice_path("web_voice", VOICE_PATH_LIVE)
        router = vr.VoiceRouter(SarvamHandler())
        asyncio.run(router.route(VoiceSession(), b"\x00", live_enabled=False))
        asyncio.run(router.route(VoiceSession(), b"\x00", live_enabled=False))
        report = vr.voice_path_report()
        self.assertEqual(report["live"], 1)
        self.assertEqual(report["operator_forced"], 2)
        self.assertEqual(report["fallback"], 0)
        self.assertEqual(report["fallback_rate"], 0.0)
        self.assertEqual(report["sarvam_turns"], 2)

    def test_the_deployment_report_counts_every_channel(self):
        vr.record_voice_path("web_voice", "operator_forced_fallback")
        vr.record_voice_path("whatsapp", VOICE_PATH_LIVE)
        vr.record_voice_path("web_voice", "turn_stalled")
        report = vr.voice_path_report()
        self.assertEqual(report["live"], 1)
        self.assertEqual(report["operator_forced"], 1)
        self.assertEqual(report["fallback"], 1)
        self.assertEqual(report["fallback_rate"], 0.5)
        self.assertEqual(report["by_reason"], {"turn_stalled": 1})
        self.assertEqual(report["by_channel"]["web_voice"], {"operator_forced_fallback": 1, "turn_stalled": 1})


class TheMobileRoutesReadTheDeploymentOverride(unittest.TestCase):
    def test_a_router_built_without_a_control_reads_the_deployments(self):
        forced_sarvam_control(self)
        fresh_deployment_tally(self)
        router = vr.VoiceRouter(SarvamHandler())  # as mobile_api/router.py builds it
        outcome = asyncio.run(router.route(VoiceSession(), b"\x00"))
        self.assertEqual(outcome.turn.fallback_reason, "operator_forced_fallback")


# --------------------------------------------------------------------------
# Web voice (/ws/voice), over the real app's socket.
# --------------------------------------------------------------------------

from tests.test_web_live_voice import (  # noqa: E402
    DOSE_REPLY, QUESTION, ScriptedLiveService, ScriptedLiveSession, WebVoiceSocket,
    setUpModule as _web_setup, tearDownModule as _web_teardown,
)
from tests.test_route_policy import _FakePipeline  # noqa: E402

SARVAM_PCM = b"\x11\x22" * 240  # what the stubbed Bulbul says, as 24 kHz PCM
SARVAM_ANSWER = "Keep him comfortable, sitting up, and call the palliative care nurse."
HEARD = "He is breathless, what can I do?"


def setUpModule():
    _web_setup()


def tearDownModule():
    _web_teardown()


def wav_of(pcm: bytes, rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)
    return buffer.getvalue()


class SarvamStub:
    """Sarvam's network: Saaras hears HEARD, Bulbul returns SARVAM_PCM as WAV."""

    def __init__(self):
        self.stt_audio: List[bytes] = []
        self.tts: List[Dict] = []

    async def speech_to_text(self, client, audio, language="hi-IN", **kwargs):
        from sarvam_integration.client import SarvamSTTResult

        self.stt_audio.append(audio)
        return SarvamSTTResult(success=True, transcript=HEARD, language_code=language)

    async def text_to_speech(self, client, text, language="hi-IN", voice="priya", **kwargs):
        from sarvam_integration.client import SarvamTTSResult

        self.tts.append({"text": text, "language": language, "voice": voice, **kwargs})
        return SarvamTTSResult(success=True, audio_base64=base64.b64encode(wav_of(SARVAM_PCM)).decode(),
                               sample_rate=kwargs.get("sample_rate", 22050))


class WebVoiceOnSarvam(WebVoiceSocket):
    """The web socket with Sarvam's network and the answer pipeline stubbed."""

    def setUp(self):
        super().setUp()
        from sarvam_integration.client import SarvamClient

        fresh_deployment_tally(self)
        self.sarvam = SarvamStub()

        async def query(pipeline, question, user_id=None, source_language=None, **kwargs):
            return {"status": "success", "answer": SARVAM_ANSWER, "sources": []}

        for patcher in (
            mock.patch.object(SarvamClient, "is_available", lambda client: True),
            mock.patch.object(SarvamClient, "speech_to_text",
                              lambda client, *a, **k: self.sarvam.speech_to_text(client, *a, **k)),
            mock.patch.object(SarvamClient, "text_to_speech",
                              lambda client, *a, **k: self.sarvam.text_to_speech(client, *a, **k)),
            mock.patch.object(_FakePipeline, "query", query, create=True),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def audio_of(frames: List[Dict]) -> List[bytes]:
        return [f["bytes"] for f in frames if f["type"] == "audio"]

    @staticmethod
    def assistant(frames: List[Dict]) -> List[Dict]:
        return [f for f in frames if f["type"] == "transcription" and f.get("role") == "assistant"]


class WithTheOverrideSetWebVoiceOpensNoLiveSession(WebVoiceOnSarvam):
    def test_forced_sarvam_answers_on_sarvam_and_opens_no_live_session(self):
        forced_sarvam_control(self)
        with self.client.websocket_connect("/ws/voice") as ws:
            ws.send_json({"type": "config", "language": "en-IN"})
            ack = self.frames_until(ws, "config_ack")[-1]
            ws.send_json({"type": "start_audio"})
            ws.send_bytes(b"\x00\x01" * 160)
            frames = self.exchange(ws, {"type": "stop_audio"})
        self.assertEqual(ScriptedLiveService.instances[-1].sessions, [], "a Live session was opened")
        self.assertEqual(ack.get("voice_path"), VOICE_PATH_FALLBACK)
        self.assertEqual(self.audio_of(frames), [SARVAM_PCM])
        spoken = self.assistant(frames)
        self.assertEqual(len(spoken), 1, frames)
        self.assertIn("comfortable", spoken[0]["text"])
        self.assertEqual(spoken[0]["fallback_reason"], "operator_forced_fallback")
        done = [f for f in frames if f["type"] == "turn_complete"]
        self.assertEqual(done[0]["voice_path"], VOICE_PATH_FALLBACK)
        # The caller's recording reached Saaras as a WAV of what they sent.
        with wave.open(io.BytesIO(self.sarvam.stt_audio[0])) as heard:
            self.assertEqual(heard.getframerate(), 16000)
            self.assertEqual(heard.readframes(heard.getnframes()), b"\x00\x01" * 160)
        report = vr.voice_path_report()
        self.assertEqual(report["operator_forced"], 1)
        self.assertIsNone(report["fallback_rate"], "a forced turn counted as a Live failure")


class QuietLiveSession(ScriptedLiveSession):
    """Connects, takes the caller's audio, and never answers."""

    async def send_audio(self, audio_chunk: bytes) -> None:
        return None


class AStalledWebLiveSessionFallsBack(WebVoiceOnSarvam):
    def test_a_live_session_that_goes_quiet_past_the_deadline_is_answered_by_sarvam(self):
        import web_voice_fallback

        with mock.patch.object(web_voice_fallback, "FIRST_TOKEN_TIMEOUT_S", 0.1), \
                mock.patch.object(web_voice_fallback, "PER_TURN_STALL_TIMEOUT_S", 0.3), \
                mock.patch("tests.test_web_live_voice.ScriptedLiveSession", QuietLiveSession):
            with self.client.websocket_connect("/ws/voice") as ws:
                self.start(ws, {})
                live = self.service.sessions[-1]
                ws.send_bytes(b"\x00\x01" * 160)
                ws.send_json({"type": "stop_audio"})
                frames = self.frames_until(ws, "turn_complete")
        self.assertIsInstance(live, QuietLiveSession)
        self.assertEqual(self.audio_of(frames), [SARVAM_PCM])
        spoken = self.assistant(frames)
        self.assertEqual(spoken[0]["fallback_reason"], "first_token_timeout")
        self.assertIn("comfortable", spoken[0]["text"])
        self.assertEqual(vr.voice_path_report()["by_reason"], {"first_token_timeout": 1})


class AWithheldWebTurnIsSpoken(WebVoiceOnSarvam):
    """The Deferral that replaces a withheld Live turn is heard, in the session's language."""

    def test_the_deferral_is_spoken_in_the_sessions_language(self):
        from dosage_guard import deferral_for
        from voice_safety_wrapper import fit_for_voice

        ScriptedLiveService.script = [([b"DOSE-AUDIO"], "मॉर्फिन 10 mg हर 4 घंटे में दें।",
                                        "दर्द के लिए कितनी दवा दूं?")]
        with self.client.websocket_connect("/ws/voice") as ws:
            ws.send_json({"type": "config", "language": "hi-IN"})
            self.frames_until(ws, "config_ack")
            ws.send_json({"type": "start_audio"})
            self.frames_until(ws, "session_created")
            frames = self.turn(ws)
        self.assertEqual(self.audio_of(frames), [SARVAM_PCM], "the Deferral was not spoken")
        self.assertEqual(len(self.sarvam.tts), 1)
        self.assertEqual(self.sarvam.tts[0]["text"], fit_for_voice(deferral_for("hi-IN")))
        self.assertEqual(self.sarvam.tts[0]["language"], "hi-IN")
        self.assertEqual(self.sarvam.tts[0]["sample_rate"], 24000)
        self.assertEqual(self.assistant(frames)[0]["text"], deferral_for("hi-IN"))

    def test_a_language_without_a_sarvam_voice_is_not_spoken_in_hindi(self):
        import web_voice_fallback

        with mock.patch("sarvam_integration.config.sarvam_voice", return_value=None):
            pcm = asyncio.run(web_voice_fallback.speak_for_web("Please ask your doctor.", "as-IN"))
        self.assertIsNone(pcm)
        self.assertEqual(self.sarvam.tts, [])


class AWebSarvamTurnWritesOneStudyRow(WebVoiceOnSarvam):
    """A Sarvam-answered web turn writes exactly one row: web_voice, fallback."""

    def setUp(self):
        super().setUp()
        import study_logging
        from study_logging import StudyInteractionLogger

        self.log_dir = Path(tempfile.mkdtemp(prefix="t16-rows-"))
        log = StudyInteractionLogger(log_dir=self.log_dir, use_analytics=False, use_expert_sampling=False)
        for patcher in (mock.patch.object(study_logging, "_logger", log),
                        mock.patch.dict(os.environ, {"RETRIEVAL_AUDIT_DIR": tempfile.mkdtemp()})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def rows(self) -> List[Dict]:
        from study_logging import read_entries

        return read_entries(log_dir=self.log_dir)

    def forced_turn(self, headers=None) -> List[Dict]:
        forced_sarvam_control(self)
        with self.client.websocket_connect("/ws/voice", headers=headers or {}) as ws:
            ws.send_json({"type": "config", "language": "en-IN", "user_id": "patient-of-someone-else"})
            self.frames_until(ws, "config_ack")
            ws.send_json({"type": "start_audio"})
            ws.send_bytes(b"\x00\x01" * 160)
            return self.exchange(ws, {"type": "stop_audio"})

    def assert_one_fallback_row(self) -> Dict:
        rows = self.rows()
        self.assertEqual(len(rows), 1, rows)
        self.assertEqual(rows[0]["channel"], "web_voice")
        self.assertEqual(rows[0]["voice_path"], VOICE_PATH_FALLBACK)
        self.assertEqual(rows[0]["query"], HEARD)
        self.assertIn("comfortable", rows[0]["response"])
        return rows[0]

    def test_a_signed_in_participants_forced_turn_is_keyed_on_their_token(self):
        from study_logging import pseudonymise

        self.forced_turn(headers=self.signed_in)
        row = self.assert_one_fallback_row()
        self.assertEqual(row["participant_id"], pseudonymise("asha-1"))

    def test_an_anonymous_forced_turn_is_recorded_as_anonymous(self):
        self.forced_turn()
        row = self.assert_one_fallback_row()
        self.assertEqual(row["participant_id"], "anonymous")

    def test_a_stalled_live_turn_writes_only_the_sarvam_row(self):
        import web_voice_fallback

        with mock.patch.object(web_voice_fallback, "FIRST_TOKEN_TIMEOUT_S", 0.1), \
                mock.patch.object(web_voice_fallback, "PER_TURN_STALL_TIMEOUT_S", 0.3), \
                mock.patch("tests.test_web_live_voice.ScriptedLiveSession", QuietLiveSession):
            with self.client.websocket_connect("/ws/voice", headers=self.signed_in) as ws:
                _, live = self.start(ws, {})
                ws.send_bytes(b"\x00\x01" * 160)
                ws.send_json({"type": "stop_audio"})
                self.frames_until(ws, "turn_complete")
        row = self.assert_one_fallback_row()
        self.assertEqual(row["session_id"], live.session_id)


# --------------------------------------------------------------------------
# WhatsApp voice notes.
# --------------------------------------------------------------------------

class WithTheOverrideSetAWhatsAppVoiceNoteOpensNoLiveSession(unittest.TestCase):
    def test_forced_sarvam_takes_the_traditional_pipeline(self):
        from tests.test_whatsapp_fast import NUMBER, SAFE_ANSWER, live_bot

        forced_sarvam_control(self)
        fresh_deployment_tally(self)
        bot, session = live_bot(SAFE_ANSWER, spoken="Sit upright and use a fan.")
        opened = []
        manager = bot.gemini_session_manager
        original = manager.get_or_create_session

        async def spy(**kwargs):
            opened.append(kwargs)
            return await original(**kwargs)

        manager.get_or_create_session = spy
        # The traditional pipeline runs on the fakes; only whether Live was tried matters.
        with mock.patch.object(type(bot), "_handle_twilio_media_message_gemini") as live_path:
            asyncio.run(bot._handle_twilio_media_message(NUMBER, "https://x/m", "audio/ogg"))
        self.assertEqual(opened, [])
        live_path.assert_not_called()
        self.assertEqual(vr.voice_path_report()["operator_forced"], 1)


if __name__ == "__main__":
    unittest.main()
