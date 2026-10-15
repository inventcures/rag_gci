"""
A long WhatsApp answer reaches the participant within Twilio's message limit.

Moved from the root `test_whatsapp_length_limit.py` (round-3 ticket 20), which
printed lengths and asserted nothing. Twilio rejects a WhatsApp body over 1600
characters, so an answer that is not cut is an answer the participant never sees.

What these pin down, through the Twilio text path a participant's message takes:

  * a short answer arrives whole
  * every message sent for a long answer, English and translated, fits the limit
  * a long English answer is cut at a sentence end and says it was truncated
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_text_path_dose_guard import RealisticRag, make_bot  # noqa: E402

NUMBER = "whatsapp:+910000000000"
TWILIO_LIMIT = 1600
SHORT_ANSWER = "Sit upright, use a fan near the face, and call your nurse if it worsens."
LONG_ENGLISH = "Change the patient's position often to protect the skin. " * 60
LONG_HINDI = (
    "पैलियेटिव केयर एक विशेषज्ञ चिकित्सा देखभाल है जो गंभीर बीमारी से पीड़ित रोगियों के लिए "
    "दर्द और लक्षणों से राहत प्रदान करती है। "
) * 30


def answer(text: str, language: str = "en"):
    """Send one participant message through the Twilio text path; return what was sent."""
    bot = make_bot(RealisticRag(answer=text))
    bot.user_preferences[NUMBER] = {"language": language}
    asyncio.run(bot._handle_twilio_text_message(NUMBER, "how do I care for pressure sores?"))
    return bot, bot.twilio_api.texts


class LongWhatsAppAnswersFitTheLimit(unittest.TestCase):
    def test_a_short_answer_arrives_whole(self):
        _, sent = answer(SHORT_ANSWER)
        self.assertEqual(len(sent), 1, sent)
        self.assertIn(SHORT_ANSWER, sent[0])
        self.assertNotIn("truncated", sent[0])

    def test_every_message_for_a_long_english_answer_fits(self):
        bot, sent = answer(LONG_ENGLISH)
        self.assertTrue(sent)
        for text in sent:
            self.assertLessEqual(len(text), bot.whatsapp_char_limit)
            self.assertLessEqual(len(text), TWILIO_LIMIT)

    def test_a_long_english_answer_ends_at_a_sentence_and_says_it_was_cut(self):
        _, sent = answer(LONG_ENGLISH)
        english = sent[0]
        self.assertIn("(message truncated due to length limit)", english)
        body = english.split("... (message truncated")[0]
        self.assertTrue(body.rstrip().endswith("."), body[-80:])

    def test_a_long_answer_and_its_translation_both_fit(self):
        bot, sent = answer(LONG_HINDI, language="hi")
        self.assertEqual(len(sent), 2, "expected the English text and the Hindi translation")
        for text in sent:
            self.assertLessEqual(len(text), bot.whatsapp_char_limit)


if __name__ == "__main__":
    unittest.main()
