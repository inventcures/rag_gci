#!/usr/bin/env python3
"""
Errors and logs leak nothing, and document metadata can't be wiped.

What these pin down:

  * every error response from the server has one shape,
    {"status": "error", "error": ...}, and no body carries a traceback; a
    route that raises gets a generic message, not the exception text
  * the pipeline's own error dicts (query, add, remove) carry no traceback,
    so a route that returns them can't leak one either
  * a torn document-metadata file raises a clear error at load and is left on
    disk untouched; the save is atomic, so a failed write leaves the old file
  * at INFO, a full Bolna call (start, function call, extraction, end, the
    RAG query and the fallback webhook) and a WhatsApp voice answer log no
    phone number, question, summary, concern or answer text; Query
    Translation logs neither the question nor its translation
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PHONE = "+919812345678"
QUESTION = "my father coughs blood every night"
SUMMARY = "caller described haemoptysis in her father"
CONCERN = "bleeding from the mouth"
ANSWER = "Keep him propped up and call the palliative nurse today"


def run(coro):
    return asyncio.run(coro)


class CapturedLogs(logging.Handler):
    """Every record at INFO or above, from every logger."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())

    def __enter__(self):
        root = logging.getLogger()
        self._old_level = root.level
        root.setLevel(logging.INFO)
        root.addHandler(self)
        return self

    def __exit__(self, *exc):
        root = logging.getLogger()
        root.removeHandler(self)
        root.setLevel(self._old_level)
        return False

    def assert_clean(self, test, *secrets):
        test.assertTrue(self.lines, "nothing was logged at INFO; the capture is broken")
        text = "\n".join(self.lines).lower()
        for secret in secrets:
            test.assertNotIn(secret.lower(), text, f"{secret!r} logged at INFO")


# ---------------------------------------------------------------------------
# One error shape, no traceback
# ---------------------------------------------------------------------------


class ErrorResponsesHaveOneShape(unittest.TestCase):
    """The server's app, with a route that raises, a 404 and a bad body."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        cls.app = build_server_app()

        @cls.app.get("/__test__/boom")
        async def boom():
            raise RuntimeError("secret internal detail /srv/app/data.db")

        cls.client = TestClient(cls.app, raise_server_exceptions=False)

    def assert_one_shape(self, response, status):
        self.assertEqual(response.status_code, status, response.text)
        body = response.json()
        self.assertEqual(body.get("status"), "error", body)
        self.assertIsInstance(body.get("error"), str, body)
        self.assertNotIn("traceback", response.text.lower())
        self.assertNotIn('File "', response.text)

    def test_a_route_that_raises_returns_the_error_shape_without_a_traceback(self):
        response = self.client.get("/__test__/boom")
        self.assert_one_shape(response, 500)
        self.assertNotIn("secret internal detail", response.text)

    def test_an_unknown_route_returns_the_error_shape(self):
        self.assert_one_shape(self.client.get("/__test__/no-such-route"), 404)

    def test_an_invalid_body_returns_the_error_shape(self):
        self.assert_one_shape(self.client.post("/api/query", json={"no_query": 1}), 422)

    def test_no_route_in_the_route_table_answers_a_forced_error_with_a_traceback(self):
        """Every JSON route, with the pipeline made to raise, answers without a traceback."""
        from fastapi.routing import APIRoute

        broken = mock.MagicMock(side_effect=RuntimeError("forced"))
        routes = [r for r in self.app.routes if isinstance(r, APIRoute)
                  and "{" not in r.path and not r.path.startswith("/__test__")]
        self.assertGreater(len(routes), 20, "the route table did not build")
        with mock.patch.object(_FakePipeline, "__getattr__", lambda self, name: broken):
            for route in routes:
                for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
                    response = self.client.request(method, route.path, json={})
                    self.assertNotIn("traceback", response.text.lower(), f"{method} {route.path}")
                    self.assertNotIn('Traceback (most recent call last)', response.text,
                                     f"{method} {route.path}")


class _FakePipeline:
    """Stands in for SimpleRAGPipeline inside create_app: no models, no disk."""

    document_metadata: dict = {}
    conversations: dict = {}

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return mock.MagicMock(name=name)


def build_server_app():
    """create_app() with the heavy pieces stubbed (pipeline, admin UI, WhatsApp bot)."""
    import simple_rag_server as srv

    patches = [
        mock.patch.object(srv, "SimpleRAGPipeline", _FakePipeline),
        mock.patch.object(srv.SimpleAdminUI, "__init__", lambda self, *a, **k: None),
        mock.patch.object(srv.SimpleAdminUI, "create_gradio_interface", lambda self: None),
        mock.patch.object(srv.gr, "mount_gradio_app", lambda app, *a, **k: app),
        mock.patch.dict(os.environ, {"TWILIO_ACCOUNT_SID": "", "TWILIO_AUTH_TOKEN": ""}),
    ]
    for patch in patches:
        patch.start()
    try:
        return srv.create_app(srv._parse_args([]))
    finally:
        for patch in patches:
            patch.stop()


class PipelineErrorsCarryNoTraceback(unittest.TestCase):
    """query, add and remove return error dicts that routes pass straight on."""

    def pipeline(self):
        from simple_rag_server import SimpleRAGPipeline

        pipeline = SimpleRAGPipeline.__new__(SimpleRAGPipeline)
        pipeline.document_metadata = {"doc-1": {"chunk_count": 2, "filename": "a.pdf"}}
        pipeline.vector_db = mock.MagicMock()
        pipeline.vector_db.delete.side_effect = RuntimeError("forced")
        return pipeline

    def test_remove_document_error_has_no_traceback(self):
        result = run(self.pipeline().remove_document("doc-1"))
        self.assertEqual(result["status"], "error")
        self.assertNotIn("traceback", result)

    def test_add_documents_error_has_no_traceback(self):
        pipeline = self.pipeline()
        with tempfile.NamedTemporaryFile(suffix=".txt") as f, \
                mock.patch.object(type(pipeline), "_copy_to_permanent_storage",
                                  side_effect=RuntimeError("forced"), create=True):
            result = run(pipeline.add_documents([f.name]))
        self.assertEqual(result["status"], "error")
        self.assertNotIn("traceback", result)

    def test_query_error_has_no_traceback(self):
        import simple_rag_server as srv

        pipeline = srv.SimpleRAGPipeline.__new__(srv.SimpleRAGPipeline)
        with mock.patch.object(srv, "SAFETY_ENHANCEMENTS_AVAILABLE", True), \
                mock.patch.object(srv, "get_safety_manager", create=True), \
                mock.patch.object(srv.SimpleRAGPipeline, "_catch_up_conversations",
                                  side_effect=RuntimeError("forced")):
            result = run(pipeline._query_impl(QUESTION, conversation_id="c1", user_id="u1",
                                              _trace={"stage_latency_ms": {}}))
        self.assertEqual(result["status"], "error")
        self.assertNotIn("traceback", result)


# ---------------------------------------------------------------------------
# Document metadata
# ---------------------------------------------------------------------------


class MetadataCannotBeWiped(unittest.TestCase):
    def pipeline(self, metadata_file):
        from simple_rag_server import SimpleRAGPipeline

        pipeline = SimpleRAGPipeline.__new__(SimpleRAGPipeline)
        pipeline.metadata_file = Path(metadata_file)
        return pipeline

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="metadata-"))
        self.file = self.dir / "document_metadata.json"

    def test_a_truncated_metadata_file_raises_a_clear_error(self):
        from simple_rag_server import MetadataLoadError

        torn = '{"doc-1": {"filename": "guideline.pdf", "chunk_c'
        self.file.write_text(torn)
        with self.assertRaises(MetadataLoadError) as raised:
            self.pipeline(self.file)._load_metadata()
        self.assertIn(str(self.file), str(raised.exception))
        self.assertEqual(self.file.read_text(), torn, "the torn file was changed")

    def test_metadata_that_is_not_an_object_raises(self):
        from simple_rag_server import MetadataLoadError

        self.file.write_text("[1, 2]")
        with self.assertRaises(MetadataLoadError):
            self.pipeline(self.file)._load_metadata()

    def test_a_missing_metadata_file_is_an_empty_index(self):
        self.assertEqual(self.pipeline(self.file)._load_metadata(), {})

    def test_a_failed_save_leaves_the_old_file_whole(self):
        old = {"doc-1": {"filename": "guideline.pdf", "chunk_count": 4}}
        self.file.write_text(json.dumps(old))
        pipeline = self.pipeline(self.file)
        pipeline.document_metadata = {"doc-2": {"filename": "new.pdf", "chunk_count": 1}}

        real_dump = json.dump

        def dies_halfway(obj, fp, **kw):
            fp.write('{"doc-2": {"filen')
            raise OSError("disk full")

        with mock.patch("simple_rag_server.json.dump", side_effect=dies_halfway):
            with self.assertRaises(OSError):
                pipeline._save_metadata()
        self.assertEqual(json.loads(self.file.read_text()), old)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), [self.file.name],
                         "a temporary file was left behind")
        self.assertIs(json.dump, real_dump)

    def test_a_save_round_trips(self):
        pipeline = self.pipeline(self.file)
        pipeline.document_metadata = {"doc-2": {"filename": "new.pdf", "chunk_count": 1}}
        pipeline._save_metadata()
        self.assertEqual(pipeline._load_metadata(), pipeline.document_metadata)


# ---------------------------------------------------------------------------
# Identifiable fields stay out of INFO logs
# ---------------------------------------------------------------------------


class BolnaCallLogsNothingIdentifiable(unittest.TestCase):
    CALL = "call-123"

    def test_a_full_bolna_call_through_the_webhook_handler(self):
        from bolna_integration.webhooks import BolnaWebhookHandler

        handler = BolnaWebhookHandler()
        events = [
            {"event": "call_started", "call_id": self.CALL, "phone_number": PHONE},
            {"event": "transcription", "call_id": self.CALL, "text": QUESTION, "role": "user"},
            {"event": "function_call", "call_id": self.CALL, "function_name": "query_rag_knowledge_base",
             "parameters": {"query": QUESTION, "caller": PHONE}},
            {"event": "call_ended", "call_id": self.CALL, "duration_seconds": 42,
             "summary": SUMMARY, "transcript": QUESTION},
            {"event": "extraction_completed", "call_id": self.CALL,
             "extracted_data": {"user_concern": CONCERN, "urgency_level": "high",
                                "follow_up_needed": True, "caller_phone": PHONE}},
        ]
        with CapturedLogs() as logs:
            for event in events:
                run(handler.handle_event(event))
        logs.assert_clean(self, PHONE, QUESTION, SUMMARY, CONCERN)
        self.assertEqual(handler.get_completed_call(self.CALL).summary, SUMMARY)


class BolnaRoutesLogNothingIdentifiable(unittest.TestCase):
    """The server's own Bolna routes: the RAG query and the fallback webhook."""

    EVENTS = [
        {"event": "call_ended", "call_id": "c1", "duration": 30, "summary": SUMMARY},
        {"event": "extraction_completed", "call_id": "c1",
         "extracted_data": {"user_concern": CONCERN, "urgency_level": "emergency",
                            "caller_phone": PHONE}},
    ]

    @staticmethod
    def verified(app):
        """Skip webhook verification: these tests are about logs, and
        tests/test_webhook_verification.py covers the verifier itself."""
        from webhook_auth import require_bolna

        app.dependency_overrides[require_bolna] = lambda: None
        return app

    def post_events(self, client):
        with CapturedLogs() as logs:
            for event in self.EVENTS:
                self.assertEqual(client.post("/api/bolna/webhook", json=event).status_code, 200)
        logs.assert_clean(self, PHONE, SUMMARY, CONCERN)

    def test_the_webhook_route_logs_no_summary_or_concern(self):
        from fastapi.testclient import TestClient

        self.post_events(TestClient(self.verified(build_server_app())))

    def test_the_fallback_webhook_without_the_handler_logs_no_summary_or_concern(self):
        from fastapi.testclient import TestClient

        import simple_rag_server as srv

        with mock.patch.object(srv, "BOLNA_AVAILABLE", False):
            app = build_server_app()
        self.post_events(TestClient(self.verified(app)))

    def test_the_rag_query_route_logs_no_question_caller_or_answer(self):
        from fastapi.testclient import TestClient

        async def query(self, question, conversation_id=None, user_id=None,
                        top_k=5, source_language="en"):
            return {"status": "success", "answer": ANSWER,
                    "sources": [{"filename": "handbook.pdf"}]}

        client = TestClient(self.verified(build_server_app()))
        with CapturedLogs() as logs, mock.patch.object(_FakePipeline, "query", query, create=True):
            response = client.post("/api/bolna/query", json={
                "query": QUESTION, "language": "en", "from_number": PHONE, "execution_id": "e1"})
        self.assertEqual(response.json()["status"], "success", response.text)
        logs.assert_clean(self, PHONE, QUESTION, ANSWER[:30])


class WhatsAppVoiceAnswerLogsNothingIdentifiable(unittest.TestCase):
    NUMBER = f"whatsapp:{PHONE}"

    def test_a_twilio_voice_note_and_its_spoken_answer(self):
        import whatsapp_bot
        from whatsapp_bot import EnhancedTTSService, EnhancedWhatsAppBot, TwilioWhatsAppAPI

        class FakeStt:
            supported_languages = {"hi": "Hindi", "en": "English"}

            async def transcribe_audio(self, path):
                return {"status": "success", "text": QUESTION,
                        "detected_language": "en", "language_name": "English"}

        class FakeRag:
            async def query(self, question, conversation_id=None, user_id=None,
                            top_k=5, source_language="en"):
                return {"status": "success", "answer": ANSWER, "sources": []}

        class FakeCommunicate:
            def __init__(self, text, voice):
                pass

            async def save(self, path):
                Path(path).write_bytes(b"ID3fake")

        sent = mock.MagicMock()
        sent.sid, sent.status = "SM1", "queued"
        twilio = TwilioWhatsAppAPI.__new__(TwilioWhatsAppAPI)
        twilio.account_sid, twilio.auth_token = "AC1", "token"
        twilio.from_number = "whatsapp:+14155238886"
        twilio.client = mock.MagicMock()
        twilio.client.messages.create.return_value = sent

        async def download(url):
            path = Path(tempfile.mkdtemp()) / "note.ogg"
            path.write_bytes(b"OggS")
            return str(path)

        twilio.download_media = download

        tts = EnhancedTTSService.__new__(EnhancedTTSService)
        tts.supported_voices = {"en": "en-IN-NeerjaNeural", "hi": "hi-IN-SwaraNeural"}
        tts.output_dir = Path(tempfile.mkdtemp())

        bot = EnhancedWhatsAppBot.__new__(EnhancedWhatsAppBot)
        bot.rag_pipeline = FakeRag()
        bot.stt_service = FakeStt()
        bot.tts_service = tts
        bot.twilio_api = twilio
        bot.whatsapp_api = twilio
        bot.user_preferences = {}
        bot.media_files = {}
        bot.whatsapp_char_limit = 1550
        bot.clarification_system = None
        bot.conversation_history = {}
        bot.gemini_live_enabled = False

        with CapturedLogs() as logs, \
                mock.patch.object(whatsapp_bot.edge_tts, "Communicate", FakeCommunicate), \
                mock.patch.dict(os.environ, {"PUBLIC_BASE_URL": "https://example.test"}):
            run(bot._handle_twilio_media_message(PHONE, "https://api.twilio.com/m/1", "audio/ogg"))
        self.assertTrue(twilio.client.messages.create.called, "nothing was sent")
        logs.assert_clean(self, PHONE.lstrip("+"), QUESTION, ANSWER[:30])


class TranslationLogsNothingIdentifiable(unittest.TestCase):
    def test_query_translation_logs_neither_question_nor_translation(self):
        from simple_rag_server import SimpleRAGPipeline

        pipeline = SimpleRAGPipeline.__new__(SimpleRAGPipeline)
        hindi = "मेरे पिता हर रात खून की खांसी करते हैं"

        async def llm(prompt, **kw):
            return QUESTION, "gemini-test"

        pipeline._call_llm = llm
        with CapturedLogs() as logs:
            logging.getLogger("simple_rag_server").info("capture check")
            result = run(pipeline.translate_query_to_english(hindi, "hi"))
        self.assertEqual(result["translated_query"], QUESTION)
        logs.assert_clean(self, hindi, QUESTION)

    def test_answer_translation_logs_no_answer_text(self):
        from simple_rag_server import SimpleRAGPipeline

        pipeline = SimpleRAGPipeline.__new__(SimpleRAGPipeline)
        hindi_answer = "उन्हें सहारा देकर बिठाएं और आज ही नर्स को बुलाएं"

        async def llm(prompt, **kw):
            return hindi_answer, "gemini-test"

        pipeline._call_llm = llm
        with CapturedLogs() as logs:
            run(pipeline.translate_text(ANSWER, "hi"))
        logs.assert_clean(self, hindi_answer, ANSWER[:30])


if __name__ == "__main__":
    unittest.main()
