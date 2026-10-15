#!/usr/bin/env python3
"""
Every channel writes a study row (review 3, ticket 13).

Before this only the mobile router wrote study rows. Web, WhatsApp, Bolna,
Retell and the web voice socket answered without recording anything, so the
ADR 0004 refusal log missed every dose refusal outside the app, and the mobile
voice rows carried no Dose Boundary outcome at all.

What these pin down, each channel driven at its own seam (the HTTP app, the
WhatsApp bot, the Retell socket handler, the web and mobile voice sockets):

  * one interaction writes exactly one study row, naming the channel and the
    voice path it was answered on
  * a dose refusal on that channel is in the row's safety record
  * an anonymous web voice session leaves no longitudinal record: its row is
    not keyed on a participant
"""

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

import test_route_policy as harness  # noqa: E402
from study_logging import StudyInteractionLogger, pseudonymise, read_entries  # noqa: E402
from tests.test_text_path_dose_guard import DOSE_ANSWER, RealisticRag, make_bot  # noqa: E402

QUESTION = "How much morphine should I give for the pain?"
CALLER = "+919812345678"


def setUpModule():
    harness.setUpModule()


def tearDownModule():
    harness.tearDownModule()


class StudyLogHarness(unittest.TestCase):
    """Points the one study logger every channel uses at a fresh directory."""

    def setUp(self):
        import mobile_api.router as router_module
        import study_logging

        self.log_dir = Path(tempfile.mkdtemp(prefix="study-rows-"))
        study_log = StudyInteractionLogger(
            log_dir=self.log_dir, use_analytics=False, use_expert_sampling=False)
        for patcher in (
            mock.patch.object(study_logging, "_logger", study_log),
            mock.patch.object(router_module, "_study_logger", study_log),
            mock.patch.dict(os.environ, {"RETRIEVAL_AUDIT_DIR": tempfile.mkdtemp()}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def rows(self) -> List[Dict[str, Any]]:
        return read_entries(log_dir=self.log_dir)

    def assert_one_refusal_row(self, channel: str, voice_path: str) -> Dict[str, Any]:
        """Exactly one row, on `channel` and `voice_path`, recording the dose refusal."""
        rows = self.rows()
        self.assertEqual(len(rows), 1, rows)
        row = rows[0]
        self.assertEqual(row["channel"], channel)
        self.assertEqual(row["voice_path"], voice_path)
        self.assertTrue(row["dosage_blocked"], f"{channel}: the refusal is not in the row")
        self.assertTrue(row["release_id"])
        self.assertNotIn("10 mg", row["response"])
        return row


# -- the HTTP app: web and Bolna -------------------------------------------------


async def _dose_query(self, question, conversation_id=None, user_id=None, top_k=5,
                      source_language="en", **kwargs):
    return {"status": "success", "answer": DOSE_ANSWER,
            "sources": [{"filename": "handbook.pdf"}],
            "retrieval": {"rag_method": "vector"}}


class HttpChannels(StudyLogHarness):
    app = None

    @classmethod
    def setUpClass(cls):
        if HttpChannels.app is None:
            HttpChannels.app = harness.build_server_app()

    def setUp(self):
        super().setUp()
        from auth.middleware import get_jwt_handler, init_auth
        from webhook_auth import require_bolna

        patcher = mock.patch.object(harness._FakePipeline, "query", _dose_query, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app.dependency_overrides[require_bolna] = lambda: None
        self.addCleanup(self.app.dependency_overrides.pop, require_bolna, None)
        self.client = TestClient(self.app, raise_server_exceptions=False)
        init_auth(jwt_secret=harness.MOBILE_SECRET)
        token = get_jwt_handler().create_token_pair("asha-1", "asha_worker", "site-1").access_token
        self.participant = {"Authorization": f"Bearer {token}"}

    def test_a_web_question_writes_one_row_with_its_refusal(self):
        response = self.client.post("/api/query", data={"query": QUESTION, "language": "hi"},
                                    headers=self.participant)
        self.assertEqual(response.status_code, 200, response.text)
        row = self.assert_one_refusal_row("web", "text")
        self.assertEqual(row["participant_id"], pseudonymise("asha-1"))
        self.assertEqual(row["care_role"], "asha_worker")
        self.assertEqual(row["language"], "hi")
        self.assertEqual(row["source_count"], 1)

    def test_a_bolna_turn_writes_one_row_with_its_refusal(self):
        # Bolna's agent LLM is this server (ticket 17); released and enabled here,
        # tests/test_phone_providers_residency.py covers the refusal.
        import phone_residency
        from webhook_auth import require_bolna_key

        for patcher in (
            mock.patch.object(phone_residency, "released_services",
                              lambda: set(phone_residency.services("bolna"))),
            mock.patch.object(phone_residency, "bolna_enabled", lambda: True),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.app.dependency_overrides[require_bolna_key] = lambda: None
        self.addCleanup(self.app.dependency_overrides.pop, require_bolna_key, None)
        response = self.client.post("/api/bolna/llm/en/v1/chat/completions", json={
            "model": "palli-sahayak", "messages": [
                {"role": "system", "content": f"Caller number: {CALLER}; call id: e1"},
                {"role": "user", "content": QUESTION}]})
        self.assertEqual(response.status_code, 200, response.text)
        row = self.assert_one_refusal_row("bolna", "phone")
        self.assertEqual(row["provider"], "bolna")
        self.assertEqual(row["participant_id"], pseudonymise(f"bolna_{CALLER}"))
        self.assertNotIn(CALLER, str(row))


# -- WhatsApp ----------------------------------------------------------------------


class WhatsApp(StudyLogHarness):
    NUMBER = f"whatsapp:{CALLER}"

    def test_a_text_message_writes_one_row_with_its_refusal(self):
        bot = make_bot(RealisticRag())
        asyncio.run(bot._handle_twilio_text_message(self.NUMBER, QUESTION))
        row = self.assert_one_refusal_row("whatsapp", "text")
        self.assertEqual(row["participant_id"], pseudonymise(self.NUMBER))

    def test_a_voice_note_on_the_sarvam_path_writes_one_row(self):
        bot = make_bot(RealisticRag())
        bot.twilio_api.client = object()

        async def download(url):
            return os.path.join(tempfile.mkdtemp(), "note.ogg")

        bot.twilio_api.download_media = download
        asyncio.run(bot._handle_twilio_media_message(self.NUMBER, "https://x/m", "audio/ogg"))
        self.assert_one_refusal_row("whatsapp", "fallback")

    def test_a_live_voice_note_whose_reply_names_a_dose_writes_one_row(self):
        from tests.test_whatsapp_fast import SAFE_ANSWER, live_bot

        bot, _ = live_bot(SAFE_ANSWER, spoken=DOSE_ANSWER)
        asyncio.run(bot._handle_twilio_media_message(self.NUMBER, "https://x/m", "audio/ogg"))
        # Live's reply was refused and the guarded text spoken by Sarvam instead.
        self.assert_one_refusal_row("whatsapp", "fallback")

    def test_a_live_voice_note_heard_from_live_writes_one_live_row(self):
        from tests.test_whatsapp_fast import SAFE_ANSWER, live_bot

        bot, _ = live_bot(SAFE_ANSWER, spoken="Sit upright and use a fan.")
        asyncio.run(bot._handle_twilio_media_message(self.NUMBER, "https://x/m", "audio/ogg"))
        rows = self.rows()
        self.assertEqual(len(rows), 1, rows)
        self.assertEqual((rows[0]["channel"], rows[0]["voice_path"]), ("whatsapp", "live"))
        self.assertFalse(rows[0]["dosage_blocked"])
        self.assertIn("fan", rows[0]["response"])


# -- Retell ------------------------------------------------------------------------


class Retell(StudyLogHarness):
    def test_a_turn_writes_one_row_with_its_refusal(self):
        from retell_integration.custom_llm_server import RetellCustomLLMHandler
        from tests.test_retell_turns import FakeSocket, turn, wait_for

        async def go():
            handler = RetellCustomLLMHandler(rag_pipeline=RealisticRag(answer=DOSE_ANSWER))
            ws = FakeSocket()
            call = asyncio.create_task(handler.handle_websocket(ws, "call-1"))
            await ws.inbox.put({"interaction_type": "call_details",
                                "call": {"from_number": CALLER, "metadata": {"language": "en-IN"}}})
            await ws.inbox.put(turn(1, QUESTION))
            await wait_for(lambda: any(r.get("content_complete") for r in ws.responses()))
            await wait_for(lambda: self.rows())
            await ws.inbox.put(None)
            await call

        asyncio.run(go())
        row = self.assert_one_refusal_row("retell", "phone")
        self.assertEqual(row["provider"], "retell")


# -- the mobile voice socket ---------------------------------------------------------


class MobileVoiceSocket(StudyLogHarness):
    def test_a_turn_writes_one_row_with_its_refusal(self):
        import voice_router as vr
        from tests.test_voice_endpoint_integration import FakeStt, PCM, build_app, signed_in
        from tests.test_voice_endpoint_integration import RealisticRag as VoiceRag
        import base64

        stt = FakeStt(QUESTION)
        for patcher in (
            mock.patch("sarvam_integration.SarvamClient", return_value=stt),
            mock.patch.object(vr.LiveAvailability, "probe", return_value=vr.LiveAvailability(
                available=False, reason="live_sdk_unavailable: test")),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        client = TestClient(build_app(VoiceRag(answer=DOSE_ANSWER)))
        with client.websocket_connect("/api/mobile/v1/ws/voice?language=en-IN",
                                      headers=signed_in()) as ws:
            ws.receive_json()
            ws.send_json({"type": "audio", "language": "en-IN",
                          "audio_base64": base64.b64encode(PCM).decode()})
            reply = ws.receive_json()
        self.assertEqual(reply["type"], "response", reply)
        row = self.assert_one_refusal_row("voice", "fallback")
        self.assertEqual(row["participant_id"], pseudonymise("participant-1"))


# -- the web voice socket --------------------------------------------------------------


class WebVoiceSocket(StudyLogHarness):
    """The real /ws/voice, Live scripted (tests/test_web_live_voice.py)."""

    def setUp(self):
        super().setUp()
        import test_web_live_voice as live

        self.live = live
        self.socket = live.WebVoiceSocket(methodName="setUp")
        self.socket.app = live.WebVoiceSocket.app or live.build_server_app()
        live.WebVoiceSocket.app = self.socket.app
        self.socket.setUp()
        self.addCleanup(self.socket.doCleanups)

    def speak(self, headers=None, spoken=None):
        live = self.live
        live.ScriptedLiveService.script = [([b"AUDIO-1"], spoken or live.DOSE_REPLY, live.QUESTION)]
        with self.socket.client.websocket_connect("/ws/voice", headers=headers or {}) as ws:
            _, session = self.socket.start(ws, {"user_id": live.OTHER_USER})
            frames = self.socket.turn(ws)
        return frames, session

    def test_a_turn_writes_one_row_with_its_refusal(self):
        frames, session = self.speak(headers=self.socket.signed_in)
        row = self.assert_one_refusal_row("web_voice", "live")
        self.assertEqual(row["participant_id"], pseudonymise("asha-1"))
        self.assertEqual(row["session_id"], session.session_id)
        self.assertEqual(row["query"], self.live.QUESTION)
        types_seen = [f["type"] for f in frames]
        self.assertLess(types_seen.index("transcription"), types_seen.index("turn_complete"))

    def test_an_anonymous_session_writes_a_row_keyed_on_no_participant(self):
        _, session = self.speak()
        rows = self.rows()
        self.assertEqual(len(rows), 1, rows)
        self.assertEqual(rows[0]["participant_id"], "anonymous")
        self.assertNotEqual(rows[0]["participant_id"], pseudonymise(self.live.OTHER_USER))
        self.assertEqual(rows[0]["session_id"], session.session_id)

    def test_a_released_turn_records_what_live_said(self):
        self.speak(headers=self.socket.signed_in, spoken=self.live.SAFE_REPLY)
        rows = self.rows()
        self.assertEqual(len(rows), 1, rows)
        self.assertFalse(rows[0]["dosage_blocked"])
        self.assertEqual(rows[0]["response"], self.live.SAFE_REPLY)


if __name__ == "__main__":
    unittest.main()
