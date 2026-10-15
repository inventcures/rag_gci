#!/usr/bin/env python3
"""
The dose boundary on the text channels.

Voice and the mobile app already passed every answer through
`SafetyEnhancementsManager.process_response` (which applies the dosage guard).
The web API, the admin query tab and WhatsApp handed `SimpleRAGPipeline.query()`
drafts straight to the user, and the drafting prompt asked the model for
"specific drug names, doses, routes, frequencies". So the boundary held on the
channels that were tested and not on the ones that were not.

What these tests pin down:

  * a dose-bearing draft from query() reaches neither the web response, the
    admin tab, a WhatsApp message nor the text WhatsApp sends to TTS
  * the drafting prompts no longer ask for doses, and tell the model to refer
    the dose to the doctor or nurse
  * the mobile /query route calls query() with its real signature, reads the
    dict it really returns, and returns (and logs) the guarded answer
  * the offline cache bundle calls query() correctly and stores guarded text

The fake pipeline copies SimpleRAGPipeline.query's signature verbatim and does
not accept **kw: a forgiving fake is what hid the mobile route's wrong kwargs.
"""

import asyncio
import inspect
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOSE_ANSWER = (
    "Morphine is the usual strong opioid for cancer pain. "
    "Start morphine at 10 mg every 4 hours and titrate by 5 mg until pain is "
    "controlled. Also give paracetamol 1 g twice daily."
)
LEAKS = ("10 mg", "5 mg", "1 g", "every 4 hours", "twice daily", "titrate by")


def run(coro):
    return asyncio.run(coro)


def assert_no_dose(test, text, where):
    lowered = (text or "").lower()
    for leaked in LEAKS:
        test.assertNotIn(leaked, lowered, f"dose leaked through {where}: {leaked!r}")


class RealisticRag:
    """Mirrors SimpleRAGPipeline.query exactly. Deliberately no **kw."""

    def __init__(self, answer=DOSE_ANSWER, status="success"):
        self.answer = answer
        self.status = status
        self.calls = []

    async def query(
        self,
        question: str,
        conversation_id=None,
        user_id=None,
        top_k: int = 5,
        source_language: str = "en",
    ):
        self.calls.append(
            {"question": question, "user_id": user_id, "source_language": source_language}
        )
        return {
            "status": self.status,
            "answer": self.answer,
            "sources": [{"filename": "handbook.pdf", "chunk_index": 0, "total_chunks": 3}],
            "model_used": "fake-llm",
            "retrieval": {"rag_method": "vector", "retrievers_active": ["vector"]},
            "kg_check": {"mode": "off", "status": "skipped"},
        }

    async def translate_text(self, text, target_language):
        # Identity "translation", so whatever is sent is exactly what was translated.
        return {"status": "success", "translated_text": f"[{target_language}] {text}"}


class FakeSignatureMatchesTheRealPipeline(unittest.TestCase):
    def test_fake_has_the_real_signature(self):
        from simple_rag_server import SimpleRAGPipeline

        real = [n for n in inspect.signature(SimpleRAGPipeline.query).parameters if n != "self"]
        fake = [n for n in inspect.signature(RealisticRag.query).parameters if n != "self"]
        self.assertEqual(fake, real, "RealisticRag.query has drifted from SimpleRAGPipeline.query")


# ---------------------------------------------------------------------------
# The shared helper
# ---------------------------------------------------------------------------


class GuardQueryResult(unittest.TestCase):
    def test_replaces_the_draft_and_records_the_outcome(self):
        from safety_enhancements import guard_query_result

        result = run(RealisticRag().query("what dose of morphine?"))
        safety = guard_query_result(result, query="what dose of morphine?", language="en")

        self.assertTrue(safety.dosage_blocked)
        self.assertEqual(result["answer"], safety.response)
        assert_no_dose(self, result["answer"], "guard_query_result")
        self.assertTrue(result["safety"]["dosage_blocked"])
        self.assertNotIn(DOSE_ANSWER, repr(result), "the draft must not ride along in the result")

    def test_error_results_pass_untouched(self):
        from safety_enhancements import guard_query_result

        result = {"status": "error", "error": "boom"}
        self.assertIsNone(guard_query_result(result, query="q"))
        self.assertEqual(result, {"status": "error", "error": "boom"})

    def test_a_clean_answer_is_not_blocked(self):
        from safety_enhancements import guard_query_result

        clean = "Morphine is used for severe cancer pain. Confirm the dose with your doctor or nurse."
        result = {"status": "success", "answer": clean, "sources": []}
        safety = guard_query_result(result, query="what helps cancer pain?")
        self.assertFalse(safety.dosage_blocked)
        self.assertIn("Morphine is used for severe cancer pain.", result["answer"])


# ---------------------------------------------------------------------------
# Web
# ---------------------------------------------------------------------------


class WebAnswersAreGuarded(unittest.TestCase):
    def setUp(self):
        self.audit_dir = tempfile.mkdtemp(prefix="retrieval-audit-")
        self.env = mock.patch.dict(os.environ, {"RETRIEVAL_AUDIT_DIR": self.audit_dir})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_api_query_returns_the_guarded_answer(self):
        from simple_rag_server import answer_web_query

        rag = RealisticRag()
        body = run(answer_web_query(rag, "how much morphine should I give?"))
        self.assertEqual(rag.calls[0]["question"], "how much morphine should I give?")
        assert_no_dose(self, body["answer"], "POST /api/query")
        self.assertTrue(body["safety"]["dosage_blocked"])

    def test_api_query_through_the_real_query_method(self):
        """The real query() drafts a dose; the web body must not carry it."""
        from test_answer_path_retrieval import make_pipeline

        from retrieval.settings import RetrievalSettings, set_active_settings
        import simple_rag_server

        set_active_settings(RetrievalSettings())
        simple_rag_server._rag_method = "vector"
        pipeline = make_pipeline(answer=DOSE_ANSWER)
        body = run(simple_rag_server.answer_web_query(pipeline, "morphine for cancer pain?"))
        self.assertEqual(body["status"], "success")
        assert_no_dose(self, body["answer"], "POST /api/query (real query())")

    def test_the_route_uses_the_guarded_helper(self):
        """/api/query is built inside create_app(), so pin that it delegates to the helper."""
        import simple_rag_server

        source = inspect.getsource(simple_rag_server.create_app)
        route = source[source.index('@app.post("/api/query")'):]
        route = route[: route.index("@app.", 10)]
        self.assertIn("answer_web_query(rag_pipeline, query, language=language,", route)
        self.assertNotIn("rag_pipeline.query(", route)

    def test_bolna_answers_are_guarded_before_voice_optimisation(self):
        import simple_rag_server

        source = inspect.getsource(simple_rag_server.create_app)
        route = source[source.index('@app.post("/api/bolna/query"'):]
        route = route[: route.index("@app.", 10)]
        guard_at = route.index("guard_query_result(rag_result")
        self.assertLess(guard_at, route.index("optimize_for_voice"))

    def test_admin_query_tab_shows_the_guarded_answer(self):
        from simple_rag_server import SimpleAdminUI

        ui = SimpleAdminUI.__new__(SimpleAdminUI)
        ui.rag_pipeline = RealisticRag()
        answer, info = ui._handle_query("morphine dose?")
        assert_no_dose(self, answer, "the admin query tab")
        self.assertEqual(info["sources"][0]["filename"], "handbook.pdf")


# ---------------------------------------------------------------------------
# WhatsApp
# ---------------------------------------------------------------------------


class FakeSender:
    def __init__(self):
        self.texts = []
        self.audio = []

    async def send_text_message(self, to, text):
        self.texts.append(text)
        return {"status": "success"}

    async def send_audio_message(self, to, *args):
        self.audio.append(args)
        return {"status": "success"}


class FakeTts:
    def __init__(self):
        self.spoken = []

    async def synthesize_speech(self, text, language):
        self.spoken.append(text)
        return {"audio_available": False}


class FakeStt:
    supported_languages = {"hi": "Hindi", "en": "English"}

    def __init__(self, text="morphine kitna dena hai?", language="hi"):
        self.text = text
        self.language = language

    async def transcribe_audio(self, path):
        return {
            "status": "success", "text": self.text,
            "detected_language": self.language, "language_name": "Hindi",
        }


def make_bot(rag):
    from whatsapp_bot import EnhancedWhatsAppBot

    bot = EnhancedWhatsAppBot.__new__(EnhancedWhatsAppBot)
    bot.rag_pipeline = rag
    bot.stt_service = FakeStt()
    bot.tts_service = FakeTts()
    bot.twilio_api = FakeSender()
    bot.whatsapp_api = FakeSender()
    bot.user_preferences = {}
    bot.media_files = {}
    bot.whatsapp_char_limit = 1550
    bot.clarification_system = None
    bot.conversation_history = {}
    bot.gemini_live_enabled = False
    return bot


class WhatsAppAnswersAreGuarded(unittest.TestCase):
    NUMBER = "whatsapp:+910000000000"

    def everything_sent(self, bot):
        return (
            bot.twilio_api.texts + bot.whatsapp_api.texts + bot.tts_service.spoken
            + list(bot.conversation_history.get(self.NUMBER, []))
        )

    def assert_bot_clean(self, bot, where):
        sent = self.everything_sent(bot)
        self.assertTrue(sent, f"{where}: nothing was sent")
        for text in sent:
            assert_no_dose(self, text, where)

    def test_twilio_text_message_and_its_translation(self):
        bot = make_bot(RealisticRag())
        bot.user_preferences[self.NUMBER] = {"language": "hi"}
        run(bot._handle_twilio_text_message(self.NUMBER, "how much morphine?"))
        self.assertEqual(len(bot.twilio_api.texts), 2, bot.twilio_api.texts)  # English + Hindi
        self.assert_bot_clean(bot, "WhatsApp (Twilio text)")

    def test_query_after_clarification(self):
        bot = make_bot(RealisticRag())
        run(bot._process_query_directly(self.NUMBER, "how much morphine?"))
        self.assert_bot_clean(bot, "WhatsApp (after clarification)")

    def test_twilio_voice_note_text_and_tts(self):
        bot = make_bot(RealisticRag())
        bot.twilio_api.client = object()

        async def download(url):
            return os.path.join(tempfile.mkdtemp(), "note.ogg")

        bot.twilio_api.download_media = download
        run(bot._handle_twilio_media_message(self.NUMBER, "https://x/m", "audio/ogg"))
        self.assertTrue(bot.tts_service.spoken, "no TTS text was produced")
        self.assert_bot_clean(bot, "WhatsApp (Twilio voice note, text and TTS)")

    def test_cloud_api_text_message_and_tts(self):
        bot = make_bot(RealisticRag())
        message = {"text": {"body": "how much morphine?"}}
        run(bot._handle_text_message(message, self.NUMBER))
        self.assertTrue(bot.whatsapp_api.texts and bot.tts_service.spoken)
        self.assert_bot_clean(bot, "WhatsApp (Cloud API text, text and TTS)")

    def test_cloud_api_voice_note_text_and_tts(self):
        bot = make_bot(RealisticRag())

        async def download(media_id):
            return os.path.join(tempfile.mkdtemp(), "note.ogg")

        bot.whatsapp_api.download_media = download
        run(bot._handle_audio_message({"audio": {"id": "m1"}}, self.NUMBER))
        self.assertTrue(bot.tts_service.spoken, "no TTS text was produced")
        self.assert_bot_clean(bot, "WhatsApp (Cloud API voice note, text and TTS)")

    def test_every_query_call_goes_through_the_guard(self):
        """A new handler that calls rag_pipeline.query directly would bypass the guard."""
        import whatsapp_bot

        source = inspect.getsource(whatsapp_bot.EnhancedWhatsAppBot)
        self.assertEqual(
            source.count("self.rag_pipeline.query("), 1,
            "only _guarded_query may call the pipeline directly",
        )

    def test_a_failing_guard_sends_the_error_not_the_draft(self):
        bot = make_bot(RealisticRag())
        with mock.patch(
            "safety_enhancements.SafetyEnhancementsManager.process_response",
            side_effect=RuntimeError("guard down"),
        ):
            run(bot._process_query_directly(self.NUMBER, "how much morphine?"))
        self.assertEqual(len(bot.twilio_api.texts), 1)
        self.assertIn("technical difficulties", bot.twilio_api.texts[0])
        self.assert_bot_clean(bot, "WhatsApp (guard failure)")


# ---------------------------------------------------------------------------
# The drafting prompt
# ---------------------------------------------------------------------------


class PromptNoLongerAsksForDoses(unittest.TestCase):
    def capture_prompts(self):
        from simple_rag_server import SimpleRAGPipeline

        p = SimpleRAGPipeline.__new__(SimpleRAGPipeline)
        p.gemini_api_key = "x"
        p.groq_api_key = None
        p.response_model = "gemini"
        prompts = []

        async def fake_llm(prompt, max_tokens=0, temperature=0.0):
            prompts.append(prompt)
            return "Error: captured", "error"

        p._call_llm = fake_llm
        meta = [{"filename": "handbook.pdf", "chunk_index": 0, "total_chunks": 2}]
        run(p._generate_answer_with_citations("q?", "ctx", meta, should_fuse=False))
        run(p._generate_answer_with_citations("q?", "ctx", meta, should_fuse=True))
        run(p._generate_answer("q?", "ctx"))
        self.assertEqual(len(prompts), 3)
        return prompts

    def test_prompts_do_not_request_doses(self):
        for prompt in self.capture_prompts():
            self.assertNotIn("doses, routes, frequencies", prompt)
            self.assertIn("Never give a dose", prompt)
            self.assertIn("confirm the dose with the patient's doctor or nurse", prompt)
            # The rest of the prompt is intact.
            self.assertIn("Paragraph 2: Practical action plan", prompt)
            self.assertIn("QUALITY REQUIREMENTS", prompt)
            self.assertIn("QUESTION: q?", prompt)

    def test_medgemma_path_is_gone(self):
        # Its prompt carried a dosed worked example; every answer now goes
        # through the shared LLM client and the prompt above.
        from simple_rag_server import SimpleRAGPipeline

        self.assertFalse(hasattr(SimpleRAGPipeline, "_call_medgemma_endpoint"))


# ---------------------------------------------------------------------------
# Mobile
# ---------------------------------------------------------------------------


class MobileQueryUsesTheRealPipeline(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        import mobile_api.router as router_module
        from auth.middleware import require_auth
        from auth.models import AuthenticatedUser
        from mobile_api.dependencies import init_dependencies
        from safety_enhancements import SafetyEnhancementsManager

        self.rag = RealisticRag()
        self.recorded = []

        async def fake_record(**kwargs):
            self.recorded.append(kwargs)

        patcher = mock.patch.object(router_module._study_logger, "record", new=fake_record)
        patcher.start()
        self.addCleanup(patcher.stop)

        app = FastAPI()
        app.include_router(router_module.mobile_router)
        app.dependency_overrides[require_auth] = lambda: AuthenticatedUser(
            user_id="asha-1", role="ASHA Worker", site_id="site-1"
        )
        init_dependencies(self.rag, SafetyEnhancementsManager(), None, None)
        self.client = TestClient(app)

    def test_query_called_with_the_real_signature_and_answer_guarded(self):
        resp = self.client.post(
            "/api/mobile/v1/query",
            json={"query": "how much morphine?", "language": "hi-IN"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(
            self.rag.calls,
            [{"question": "how much morphine?", "user_id": "asha-1", "source_language": "hi-IN"}],
        )
        body = resp.json()
        assert_no_dose(self, body["answer"], "mobile /query")
        self.assertEqual(body["validation_status"], "dosage_restricted")
        self.assertEqual(body["sources"][0]["filename"], "handbook.pdf")

        # The study log holds what the user received, with the real retrieval record.
        self.assertEqual(len(self.recorded), 1)
        row = self.recorded[0]
        self.assertEqual(row["response"], body["answer"])
        self.assertEqual(row["rag_method"], "vector")
        self.assertEqual(row["retrieval"]["retrievers_active"], ["vector"])
        self.assertTrue(row["safety_result"].dosage_blocked)


class OfflineBundleIsGuarded(unittest.TestCase):
    def test_bundle_queries_use_the_real_signature_and_store_guarded_text(self):
        from offline import cache_builder
        from safety_enhancements import SafetyEnhancementsManager

        rag = RealisticRag()
        with mock.patch.object(cache_builder, "CACHE_DIR", cache_builder.Path(tempfile.mkdtemp())):
            builder = cache_builder.CacheBundleBuilder(
                rag_pipeline=rag, safety_manager=SafetyEnhancementsManager()
            )
            entries = run(builder._build_query_cache("en-IN"))

        self.assertTrue(entries, "no answers were cached")
        self.assertEqual(len(entries), len(rag.calls))
        self.assertEqual(rag.calls[0]["source_language"], "en-IN")
        for entry in entries:
            assert_no_dose(self, entry["response_text"], "the offline bundle")
            self.assertEqual(entry["sources"][0]["filename"], "handbook.pdf")


if __name__ == "__main__":
    unittest.main(verbosity=2)
