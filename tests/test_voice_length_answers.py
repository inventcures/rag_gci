"""
Voice callers hear an answer written for listening.

Before this, every channel got the 200-400 word text answer with a formatted
evidence badge appended; voice channels then cut it to 65 words, and the cut
landed on the citations and the evidence note. The redaction note after a
removed dose sentence was in parentheses, which the voice cleanup deleted.

What these pin down:

  * inside a voice turn the answer prompt is the voice prompt (60-80 words,
    dose referral kept, no sources line) with a small token cap; text turns
    keep the text prompt
  * a voice answer carries no formatted badge; the badge's fields are in the
    result for the channel's spoken note
  * the redaction note survives the voice cleanup; a voice-length answer is
    not cut by the safety net
  * mobile voice sends voice-length text to TTS and records exactly that
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import client as llm_client  # noqa: E402
from voice_safety_wrapper import VOICE_SAFETY_NET_S, clean_for_voice, fit_for_voice  # noqa: E402

EIGHTY_WORDS = (
    "Noisy breathing near the end of life is common and usually does not distress "
    "the patient. Turn her gently onto her side and keep her mouth moist with a damp "
    "cloth. Glycopyrronium under the skin can reduce the secretions; confirm the dose "
    "with her doctor or nurse. Call the palliative care team at once if she seems "
    "to struggle for breath, turns blue, or becomes very restless, because she may "
    "need a medicine change today."
)


def voice(coro_factory):
    async def go():
        token = llm_client._voice_turn.set(True)
        try:
            return await coro_factory()
        finally:
            llm_client._voice_turn.reset(token)

    return asyncio.run(go())


class Prompt(unittest.TestCase):
    def pipeline(self, reply):
        from simple_rag_server import SimpleRAGPipeline

        p = SimpleRAGPipeline.__new__(SimpleRAGPipeline)
        p.calls = []

        async def call_llm(prompt, max_tokens=2048, temperature=0.3, purpose="answer"):
            p.calls.append({"prompt": prompt, "max_tokens": max_tokens})
            return reply, "fake"

        p._call_llm = call_llm
        return p

    def test_voice_turn_uses_the_voice_prompt(self):
        from simple_rag_server import VOICE_ANSWER_MAX_TOKENS

        p = self.pipeline(EIGHTY_WORDS + " [ Sources: handbook_pg4 ]")
        meta = [{"filename": "handbook.pdf", "chunk_index": 3, "total_chunks": 10}]
        answer, _ = voice(lambda: p._generate_answer_with_citations("q?", "context", meta))
        prompt = p.calls[0]["prompt"]
        self.assertIn("spoken aloud", prompt)
        self.assertIn("60 to 80 words", prompt)
        self.assertIn("Never give a dose", prompt)
        self.assertIn("doctor or nurse", prompt)
        self.assertNotIn("CITATION FORMAT", prompt)
        self.assertEqual(p.calls[0]["max_tokens"], VOICE_ANSWER_MAX_TOKENS)
        self.assertNotIn("Sources", answer)

    def test_text_turn_keeps_the_text_prompt(self):
        p = self.pipeline("A long answer.")
        meta = [{"filename": "handbook.pdf", "chunk_index": 3, "total_chunks": 10}]
        asyncio.run(p._generate_answer_with_citations("q?", "context", meta))
        self.assertIn("2 to 3 substantive paragraphs", p.calls[0]["prompt"])
        self.assertEqual(p.calls[0]["max_tokens"], 2048)


class Badge(unittest.TestCase):
    def run_query(self, in_voice):
        from tests.test_answer_path_retrieval import make_pipeline

        p = make_pipeline(answer=EIGHTY_WORDS)
        if in_voice:
            return voice(lambda: p.query("noisy breathing at the end of life"))
        return asyncio.run(p.query("noisy breathing at the end of life"))

    def test_voice_answer_has_no_formatted_badge(self):
        import simple_rag_server

        if not simple_rag_server.SAFETY_ENHANCEMENTS_AVAILABLE:
            self.skipTest("safety enhancements unavailable")
        spoken = self.run_query(in_voice=True)
        written = self.run_query(in_voice=False)
        self.assertNotIn("─", spoken["answer"])
        self.assertIn("─", written["answer"])
        for result in (spoken, written):
            badge = result["safety_enhancements"]["evidence_badge"]
            self.assertIsInstance(badge, dict)
            self.assertIn("level", badge)
            self.assertIn("consult_physician", badge)


class SafetyNet(unittest.TestCase):
    def test_redaction_note_is_spoken(self):
        import dosage_guard

        for lang in ("en", "hi"):
            note = dosage_guard._redaction_note(lang)
            spoken = clean_for_voice("Keep her comfortable (handbook_pg4)." + note)
            inner = note.strip().strip("_").strip("()")
            self.assertIn(inner[:30], spoken)
            self.assertNotIn("handbook_pg4", spoken)

    def test_voice_length_answer_is_not_cut(self):
        self.assertEqual(fit_for_voice(EIGHTY_WORDS), clean_for_voice(EIGHTY_WORDS))

    def test_long_answer_is_capped(self):
        long = " ".join(["Keep the patient comfortable and call the nurse."] * 60)
        spoken = fit_for_voice(long)
        self.assertLessEqual(len(spoken.split()), VOICE_SAFETY_NET_S / 60 * 130)
        self.assertLessEqual(len(spoken), VOICE_SAFETY_NET_S * 12 + 1)


class MobileVoice(unittest.TestCase):
    def test_tts_gets_voice_length_text_and_the_record_matches(self):
        import voice_session as vs
        from tests.test_voice_session import FakeRag, FakeSafetyManager

        spoken_texts = []

        async def synthesise(text, language):
            spoken_texts.append(text)
            return "audio"

        long = " ".join(["Keep the patient comfortable and call the nurse."] * 60)
        handler = vs.VoiceTurnHandler(rag_pipeline=FakeRag(answer=long),
                                      safety_manager=FakeSafetyManager(), synthesise=synthesise)
        session = vs.VoiceSession()
        turn = asyncio.run(handler.handle(session, "breathless at night"))
        self.assertEqual(spoken_texts, [turn.answer])
        self.assertLessEqual(len(turn.answer.split()), VOICE_SAFETY_NET_S / 60 * 130)


if __name__ == "__main__":
    unittest.main()
