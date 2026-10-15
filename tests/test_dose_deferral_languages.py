#!/usr/bin/env python3
"""
The Deferral and the Redacted Answer note in every Supported Language.

The Dose Boundary carried its copy in English, Hindi, Bengali and Tamil only, so a
Marathi, Kannada, Odia or Punjabi participant whose answer held a dose was
refused in English. And the web query took no language at all, so a Hindi web
question got the English Deferral too.

What these pin down, at the answer gate (`DosageGuard.apply`) and the HTTP app:

  * for each of the eleven Supported Languages, a dose-bearing answer becomes a
    Deferral (or a Redacted Answer with its note) written in that language's
    script, with no fallback recorded
  * every spelling the table accepts ("or", "od", "or-IN", "hi_IN") reaches the
    same copy
  * a language outside the eleven gets the English Deferral, and the fallback is
    recorded in the safety record, the result's safety dict and the study log
  * the copy itself carries no dose, so the gate never redacts its own words
  * POST /api/query with language=hi returns the Hindi Deferral
"""

import asyncio
import tempfile
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supported_languages import SUPPORTED_LANGUAGES, get_language, scripts_in  # noqa: E402

# Too little survives redaction, so the gate answers with the Deferral.
DOSE_ONLY_ANSWER = "Give morphine 10 mg every 4 hours."

# Enough clinical content survives that the gate keeps it, adding the note.
PARTLY_DOSE_ANSWER = (
    "Morphine is the usual strong opioid for cancer pain and is safe when a doctor "
    "supervises it. Keep a written record of each time pain breaks through. "
    "Give morphine 10 mg every 4 hours."
)


def own_script_only(test: unittest.TestCase, text: str, code: str) -> None:
    """`text` is written in `code`'s script; Latin is allowed only for "OPD"."""
    lang = get_language(code)
    letters = {s for s in scripts_in(text.replace("OPD", ""))}
    test.assertEqual(letters, {lang.script}, f"{code}: copy is not in {lang.script} only: {text!r}")


class DeferralInEverySupportedLanguage(unittest.TestCase):
    def setUp(self):
        from dosage_guard import DosageGuard

        self.guard = DosageGuard()

    def test_each_language_gets_its_own_deferral(self):
        for lang in SUPPORTED_LANGUAGES:
            with self.subTest(language=lang.code):
                result = self.guard.apply(DOSE_ONLY_ANSWER, language=lang.code)
                self.assertTrue(result.blocked)
                self.assertFalse(result.redacted)
                own_script_only(self, result.response, lang.code)
                self.assertIsNone(result.language_fallback)
                self.assertIsNone(result.to_dict()["language_fallback"])

    def test_each_language_gets_its_own_redacted_answer_note(self):
        from dosage_guard import _redaction_note

        for lang in SUPPORTED_LANGUAGES:
            with self.subTest(language=lang.code):
                result = self.guard.apply(PARTLY_DOSE_ANSWER, language=lang.code)
                self.assertTrue(result.redacted)
                note = _redaction_note(lang.code)
                self.assertTrue(result.response.endswith(note))
                own_script_only(self, note, lang.code)
                self.assertIsNone(result.language_fallback)

    def test_every_table_spelling_reaches_the_same_copy(self):
        from dosage_guard import _redaction_note, _refusal_for

        for spelling, code in (("or-IN", "or"), ("od", "or"), ("od-IN", "or"),
                               ("hi_IN", "hi"), ("PA-in", "pa"), ("mr-IN", "mr")):
            with self.subTest(spelling=spelling):
                self.assertEqual(_refusal_for(spelling), _refusal_for(code))
                self.assertEqual(_redaction_note(spelling), _redaction_note(code))
                own_script_only(self, _refusal_for(spelling), code)

    def test_no_two_languages_share_a_deferral(self):
        from dosage_guard import _redaction_note, _refusal_for

        codes = [lang.code for lang in SUPPORTED_LANGUAGES]
        self.assertEqual(len({_refusal_for(c) for c in codes}), len(codes))
        self.assertEqual(len({_redaction_note(c) for c in codes}), len(codes))

    def test_the_copy_carries_no_dose(self):
        from dosage_guard import _redaction_note, _refusal_for, detect_doses

        for lang in SUPPORTED_LANGUAGES:
            with self.subTest(language=lang.code):
                self.assertEqual(detect_doses(_refusal_for(lang.code)), [])
                self.assertEqual(detect_doses(_redaction_note(lang.code)), [])


class EnglishFallbackIsRecorded(unittest.TestCase):
    def setUp(self):
        from dosage_guard import DosageGuard

        self.guard = DosageGuard()

    def test_language_outside_the_eleven_gets_the_english_deferral_with_the_fallback_recorded(self):
        from dosage_guard import _refusal_for

        result = self.guard.apply(DOSE_ONLY_ANSWER, language="as")
        self.assertTrue(result.blocked)
        self.assertEqual(result.response, _refusal_for("en"))
        self.assertEqual(result.language_fallback, "as")
        self.assertEqual(result.to_dict()["language_fallback"], "as")

    def test_redacted_answer_note_fallback_is_recorded(self):
        from dosage_guard import _redaction_note

        result = self.guard.apply(PARTLY_DOSE_ANSWER, language="xx-YY")
        self.assertTrue(result.redacted)
        self.assertTrue(result.response.endswith(_redaction_note("en")))
        self.assertEqual(result.language_fallback, "xx-YY")

    def test_no_fallback_is_recorded_when_nothing_was_withheld(self):
        result = self.guard.apply("Morphine is used for severe cancer pain.", language="as")
        self.assertFalse(result.blocked)
        self.assertIsNone(result.language_fallback)

    def test_the_fallback_reaches_the_answers_safety_record_and_the_study_log(self):
        from safety_enhancements import guard_query_result
        import study_logging

        result = {"status": "success", "answer": DOSE_ONLY_ANSWER, "sources": []}
        safety = guard_query_result(result, query="how much morphine?", language="as")
        self.assertEqual(safety.dosage_language_fallback, "as")
        self.assertEqual(result["safety"]["dosage_language_fallback"], "as")

        logs = Path(tempfile.mkdtemp(prefix="deferral-study-"))
        study_logger = study_logging.StudyInteractionLogger(
            log_dir=logs, use_analytics=False, use_expert_sampling=False,
        )
        asyncio.run(study_logger.record(
            user_id="p1", query="how much morphine?", response=result["answer"],
            channel="web", language="as", safety_result=result["safety"],
        ))
        rows = study_logging.read_entries(log_dir=logs)
        self.assertEqual(rows[-1]["dosage_language_fallback"], "as")


class WebQueryCarriesTheLanguage(unittest.TestCase):
    """POST /api/query, through the server's app and its route policy."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from auth.policy import issue_admin_token
        from test_errors_and_logs_leak_nothing import build_server_app
        import test_errors_and_logs_leak_nothing as harness

        cls.env = mock.patch.dict(os.environ, {
            "ADMIN_JWT_SECRET": "deferral-test-admin-secret-0123456789abcdef",
            "RETRIEVAL_AUDIT_DIR": "",
        })
        cls.env.start()
        cls.calls = []
        calls = cls.calls

        async def query(self, question, conversation_id=None, user_id=None, top_k=5,
                        source_language="en"):
            calls.append(source_language)
            return {"status": "success", "answer": DOSE_ONLY_ANSWER, "sources": []}

        cls.query_patch = mock.patch.object(harness._FakePipeline, "query", query, create=True)
        cls.query_patch.start()
        cls.client = TestClient(build_server_app(), raise_server_exceptions=False)
        cls.client.headers["Authorization"] = f"Bearer {issue_admin_token('deferral-test')}"

    @classmethod
    def tearDownClass(cls):
        cls.query_patch.stop()
        cls.env.stop()

    def test_hindi_web_question_gets_the_hindi_deferral(self):
        from dosage_guard import _refusal_for

        response = self.client.post("/api/query", data={"query": "मॉर्फिन कितना दें?", "language": "hi"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["answer"], _refusal_for("hi"))
        self.assertTrue(body["safety"]["dosage_blocked"])
        self.assertIsNone(body["safety"]["dosage_language_fallback"])
        self.assertEqual(self.calls[-1], "hi")

    def test_web_language_tag_reaches_the_pipeline_as_the_table_code(self):
        from dosage_guard import _refusal_for

        response = self.client.post("/api/query", data={"query": "q", "language": "or-IN"})
        self.assertEqual(response.json()["answer"], _refusal_for("or"))
        self.assertEqual(self.calls[-1], "or")

    def test_web_language_outside_the_eleven_is_answered_in_english_with_the_fallback_recorded(self):
        from dosage_guard import _refusal_for

        response = self.client.post("/api/query", data={"query": "q", "language": "as"})
        body = response.json()
        self.assertEqual(body["answer"], _refusal_for("en"))
        self.assertEqual(body["safety"]["dosage_language_fallback"], "as")
        self.assertEqual(self.calls[-1], "en")

    def test_web_question_without_a_language_is_still_english(self):
        from dosage_guard import _refusal_for

        response = self.client.post("/api/query", data={"query": "how much morphine?"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["answer"], _refusal_for("en"))
        self.assertEqual(self.calls[-1], "en")


if __name__ == "__main__":
    unittest.main()
