#!/usr/bin/env python3
"""
The Emergency Override comes only from the question, and it speaks the guidance.

Before this (review B1, B2):

  * the Dose Boundary also stood aside whenever the *answer* said "call 108",
    so "Give 10mg morphine. If breathing stops call 108 immediately." went out
    whole, and so did the guard's own Deferral text;
  * when a CRITICAL question did invoke the override, the dose went out and no
    emergency guidance was added on any channel;
  * the guard re-ran the emergency detector itself with a hard-coded "en".

What these pin down, at the answer gate and on each channel:

  * only a CRITICAL alert computed on the question invokes the override;
  * the override puts the 108 guidance, in the participant's Supported
    Language, as the first sentence of the answer;
  * the alert the answer path already computed is the one the guard uses, and
    the safety record carries its level.
"""

import asyncio
import os
import re
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dosage_guard  # noqa: E402
from dosage_guard import DosageGuard, detect_doses  # noqa: E402
from safety_enhancements import (  # noqa: E402
    EmergencyDetectionSystem,
    EmergencyLevel,
    SafetyEnhancementsManager,
    guard_query_result,
)
from tests.test_text_path_dose_guard import (  # noqa: E402
    DOSE_ANSWER,
    RealisticRag,
    make_bot,
)

ELEVEN = ("en", "hi", "bn", "ta", "gu", "mr", "te", "kn", "ml", "pa", "or")
CRITICAL_QUESTION = "He is not breathing properly, how much morphine?"
ROUTINE_QUESTION = "What helps cancer pain at home?"
ANSWER_SAYING_108 = "Give 10mg morphine. If breathing stops call 108 immediately."


def run(coro):
    return asyncio.run(coro)


def first_sentence(text: str) -> str:
    """The text up to the first sentence end (. ! ? or the danda)."""
    return re.split(r"(?<=[.!?।])\s+", (text or "").strip(), maxsplit=1)[0]


def alert(level: EmergencyLevel, language: str = "en"):
    """A real alert at `level`, built the way the detector builds one."""
    return EmergencyDetectionSystem()._create_alert(level, ["test"], language)


class OnlyTheQuestionInvokesTheOverride(unittest.TestCase):
    def test_an_answer_saying_call_108_is_still_redacted(self):
        guard = DosageGuard(emergency_system=EmergencyDetectionSystem())
        result = guard.apply(ANSWER_SAYING_108, language="en", query=ROUTINE_QUESTION)
        self.assertFalse(result.emergency_overrode)
        self.assertTrue(result.blocked)
        self.assertNotIn("10mg", result.response)

    def test_an_answer_saying_call_108_is_redacted_on_the_answer_path(self):
        result = SafetyEnhancementsManager().process_response(
            query=ROUTINE_QUESTION, response=ANSWER_SAYING_108, language="en-IN"
        )
        self.assertFalse(result.emergency_overrode_dosage)
        self.assertTrue(result.dosage_blocked)
        self.assertNotIn("10mg", result.response)

    def test_the_deferral_text_never_triggers_the_override(self):
        guard = DosageGuard(emergency_system=EmergencyDetectionSystem())
        for language in ELEVEN:
            with self.subTest(language=language):
                deferral = dosage_guard._refusal_for(language)
                result = guard.apply(deferral, language=language, query=ROUTINE_QUESTION)
                self.assertFalse(result.emergency_overrode)

    def test_a_high_alert_does_not_invoke_the_override(self):
        result = DosageGuard().apply(
            DOSE_ANSWER, language="en", query="severe pain, what dose?",
            alert=alert(EmergencyLevel.HIGH),
        )
        self.assertFalse(result.emergency_overrode)
        self.assertTrue(result.blocked)


class TheOverrideSpeaksTheGuidance(unittest.TestCase):
    def test_every_supported_language_has_108_guidance_free_of_doses(self):
        for language in ELEVEN:
            with self.subTest(language=language):
                guidance = dosage_guard.emergency_guidance_for(language)
                self.assertIn("108", first_sentence(guidance))
                self.assertEqual(detect_doses(guidance), [])
        self.assertNotEqual(
            dosage_guard.emergency_guidance_for("ta"), dosage_guard.emergency_guidance_for("en"))

    def test_a_critical_question_gets_the_guidance_first_in_its_language(self):
        manager = SafetyEnhancementsManager()
        for language in ELEVEN:
            with self.subTest(language=language):
                result = manager.process_response(
                    query=CRITICAL_QUESTION, response=DOSE_ANSWER, language=f"{language}-IN"
                )
                guidance = dosage_guard.emergency_guidance_for(language)
                self.assertTrue(result.emergency_overrode_dosage)
                self.assertTrue(result.response.startswith(guidance), result.response[:80])
                self.assertIn("108", first_sentence(result.response))

    def test_the_guidance_is_not_added_twice(self):
        guard = DosageGuard()
        once = guard.apply(DOSE_ANSWER, alert=alert(EmergencyLevel.CRITICAL)).response
        twice = guard.apply(once, alert=alert(EmergencyLevel.CRITICAL)).response
        self.assertEqual(once, twice)


class TheAlertIsPassedIn(unittest.TestCase):
    def test_the_safety_record_carries_the_alert_level_passed_in(self):
        guard = DosageGuard()
        for level, expected in ((EmergencyLevel.CRITICAL, "critical"),
                                (EmergencyLevel.HIGH, "high")):
            with self.subTest(level=level):
                record = guard.apply(DOSE_ANSWER, alert=alert(level)).to_dict()
                self.assertEqual(record["emergency_level"], expected)
        self.assertEqual(guard.apply(DOSE_ANSWER).to_dict()["emergency_level"], "none")

    def test_the_answer_path_detects_once_in_the_participants_language(self):
        # Patched on the class: the guard is a singleton and may hold the
        # detector of an earlier manager.
        manager = SafetyEnhancementsManager()
        original = EmergencyDetectionSystem.detect_emergency
        with mock.patch.object(
            EmergencyDetectionSystem, "detect_emergency", autospec=True, side_effect=original,
        ) as detect:
            result = manager.process_response(
                query=CRITICAL_QUESTION, response=DOSE_ANSWER, language="ta-IN"
            )
        languages = [c.args[3] if len(c.args) > 3 else c.kwargs.get("language")
                     for c in detect.call_args_list]
        self.assertEqual(languages, ["ta-IN"])
        self.assertTrue(result.emergency_overrode_dosage)


class EveryChannelSpeaksTheGuidance(unittest.TestCase):
    """A CRITICAL question on each channel gets the 108 guidance first."""

    def test_web(self):
        from simple_rag_server import answer_web_query

        body = run(answer_web_query(RealisticRag(), CRITICAL_QUESTION))
        self.assertIn("108", first_sentence(body["answer"]))
        self.assertTrue(body["safety"]["emergency_overrode_dosage"])

    def test_whatsapp(self):
        bot = make_bot(RealisticRag())
        run(bot._process_query_directly("whatsapp:+910000000000", CRITICAL_QUESTION))
        self.assertTrue(bot.twilio_api.texts)
        self.assertIn("108", first_sentence(bot.twilio_api.texts[0]))

    def test_bolna_and_retell_escalation_reply(self):
        """Bolna and Retell answer a CRITICAL question with the voice wrapper's reply."""
        from voice_safety_wrapper import VoiceSafetyWrapper

        wrapper = VoiceSafetyWrapper()
        for language in ELEVEN:
            with self.subTest(language=language):
                checked = run(wrapper.check_voice_query(
                    user_id="caller", transcript=CRITICAL_QUESTION, language=language))
                self.assertTrue(checked.should_escalate)
                self.assertTrue(checked.safety_message.startswith(
                    dosage_guard.emergency_guidance_for(language)), checked.safety_message[:80])

    def test_bolna_and_retell_answer_path(self):
        """When the voice pre-check is unavailable both routes call guard_query_result."""
        for language in ("en", "hi", "mr"):
            with self.subTest(language=language):
                result = run(RealisticRag().query(CRITICAL_QUESTION))
                guard_query_result(result, query=CRITICAL_QUESTION, language=language)
                self.assertTrue(result["answer"].startswith(
                    dosage_guard.emergency_guidance_for(language)))

    def test_streaming_path(self):
        from answer_stream import AnswerStream

        async def deltas():
            for word in "Turn him onto his side. Keep his airway clear.".split(" "):
                yield word + " "

        for language in ("en", "kn"):
            with self.subTest(language=language):
                sent = []

                async def send(sentence):
                    sent.append(sentence)

                stream = AnswerStream(
                    send, query=CRITICAL_QUESTION, language=language,
                    guard=SafetyEnhancementsManager().dosage_guard,
                )
                run(stream.consume(deltas()))
                self.assertEqual(sent[0], dosage_guard.emergency_guidance_for(language))
                self.assertEqual(sent[1:], ["Turn him onto his side.", "Keep his airway clear."])

    def test_retell_streamed_tail_does_not_repeat_the_guidance(self):
        from answer_stream import AnswerStream
        from retell_integration.custom_llm_server import RetellCustomLLMHandler

        async def send(sentence):
            pass

        async def deltas():
            yield "Turn him onto his side. "

        stream = AnswerStream(send, query=CRITICAL_QUESTION, language="en",
                              guard=SafetyEnhancementsManager().dosage_guard)
        run(stream.consume(deltas()))
        handler = RetellCustomLLMHandler(rag_pipeline=RealisticRag())
        session = mock.Mock(language="en")
        result = {"status": "success", "answer": stream.text + " Keep his airway clear."}
        tail = handler._streamed_tail(session, stream, result, CRITICAL_QUESTION)
        self.assertNotIn("108", tail)
        self.assertIn("Keep his airway clear.", tail)

    def test_streaming_routine_question_gets_no_guidance(self):
        from answer_stream import AnswerStream

        async def deltas():
            yield "Keep the mouth moist with a damp cloth. "

        sent = []

        async def send(sentence):
            sent.append(sentence)

        stream = AnswerStream(send, query=ROUTINE_QUESTION, language="en",
                              guard=SafetyEnhancementsManager().dosage_guard)
        run(stream.consume(deltas()))
        self.assertEqual(sent, ["Keep the mouth moist with a damp cloth."])


if __name__ == "__main__":
    unittest.main(verbosity=2)
