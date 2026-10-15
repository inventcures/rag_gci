"""
One Supported Language table, and WhatsApp answering in the right language.

Before this, WhatsApp kept its own language tables, each covering a different
subset of the eleven Supported Languages (ADR 0002):

  * TTS had voices for five languages and spoke the other six in Hindi;
  * Bengali and Gujarati voice notes were sent to Live as English;
  * a language the bot could not place defaulted to Hindi;
  * the Tamil "/lang ta" confirmation contained Bengali letters.

What these pin down, for each of the eleven Supported Languages:

  * the table holds exactly the eleven, each with an India-hosted voice;
  * a WhatsApp voice note is spoken back in its own language's voice, or the
    participant is told plainly, in their language, that speech is unavailable;
  * a WhatsApp text message is translated into the language it was written in;
  * never Hindi for a non-Hindi input;
  * Bengali and Gujarati voice notes go to Live in their own language;
  * every message the table carries is written only in its language's script.
"""

import asyncio
import base64
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_text_path_dose_guard import FakeSender, RealisticRag, make_bot  # noqa: E402

NUMBER = "whatsapp:+919800000002"
SAFE_ANSWER = "Sit upright, use a fan near the face, and call your nurse if it worsens."

# ADR 0002: the eleven Supported Languages, with the code Sarvam's Bulbul
# synthesises each one in (Sarvam spells Odia "od-IN").
SARVAM_TAG = {
    "hi": "hi-IN", "bn": "bn-IN", "kn": "kn-IN", "ml": "ml-IN", "mr": "mr-IN",
    "or": "od-IN", "pa": "pa-IN", "ta": "ta-IN", "te": "te-IN", "gu": "gu-IN",
    "en": "en-IN",
}

# The script each Supported Language is written in (ISO 15924).
SCRIPT = {
    "hi": "Deva", "mr": "Deva", "bn": "Beng", "kn": "Knda", "ml": "Mlym",
    "or": "Orya", "pa": "Guru", "ta": "Taml", "te": "Telu", "gu": "Gujr",
    "en": "Latn",
}

# "Breathless at night, what should I do?", as a participant might type it.
QUESTION = {
    "hi": "रात में सांस फूलती है, क्या करें?",
    "bn": "রাতে শ্বাসকষ্ট হয়, কী করব?",
    "kn": "ರಾತ್ರಿ ಉಸಿರಾಟದ ತೊಂದರೆ ಆಗುತ್ತದೆ, ಏನು ಮಾಡಬೇಕು?",
    "ml": "രാത്രിയിൽ ശ്വാസം മുട്ടുന്നു, എന്ത് ചെയ്യണം?",
    "mr": "रात्री श्वास घ्यायला त्रास होतो, मी काय करू?",
    "or": "ରାତିରେ ନିଶ୍ୱାସ ନେବାରେ କଷ୍ଟ ହୁଏ, କଣ କରିବି?",
    "pa": "ਰਾਤ ਨੂੰ ਸਾਹ ਚੜ੍ਹਦਾ ਹੈ, ਕੀ ਕਰੀਏ?",
    "ta": "இரவில் மூச்சுத் திணறல் ஏற்படுகிறது, என்ன செய்வது?",
    "te": "రాత్రి ఊపిరి ఆడటం లేదు, ఏం చేయాలి?",
    "gu": "રાત્રે શ્વાસ ચડે છે, શું કરવું?",
    "en": "Breathless at night, what should I do?",
}

ELEVEN = sorted(SARVAM_TAG)


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fakes at the speech providers' boundary (Sarvam's client, edge-tts)
# ---------------------------------------------------------------------------


class FakeSarvamClient:
    """Stands in for sarvam_integration.SarvamClient; records each synthesis."""

    calls = []
    fail = False

    def __init__(self, *args, **kwargs):
        pass

    def is_available(self):
        return True

    async def text_to_speech(self, text, language="hi-IN", voice="priya", **kwargs):
        FakeSarvamClient.calls.append({"text": text, "language": language, "voice": voice})
        if FakeSarvamClient.fail:
            return SimpleNamespace(success=False, audio_base64="", error="sarvam down")
        return SimpleNamespace(success=True, audio_base64=base64.b64encode(b"RIFFfake").decode(), error=None)


class FakeEdgeCommunicate:
    """Stands in for edge_tts.Communicate; records the voice asked for."""

    voices = []

    def __init__(self, text, voice, *args, **kwargs):
        FakeEdgeCommunicate.voices.append(voice)

    async def save(self, path):
        Path(path).write_bytes(b"ID3fake")


class SpeechProviders(unittest.TestCase):
    """Real EnhancedTTSService, with fakes where it calls out to the providers."""

    def setUp(self):
        FakeSarvamClient.calls = []
        FakeSarvamClient.fail = False
        FakeEdgeCommunicate.voices = []
        cwd = os.getcwd()
        os.chdir(tempfile.mkdtemp(prefix="tts-"))
        self.addCleanup(os.chdir, cwd)
        for target, fake in (("sarvam_integration.SarvamClient", FakeSarvamClient),
                             ("sarvam_integration.client.SarvamClient", FakeSarvamClient),
                             ("edge_tts.Communicate", FakeEdgeCommunicate)):
            patcher = mock.patch(target, fake)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tts(self):
        from whatsapp_bot import EnhancedTTSService

        return EnhancedTTSService()

    def spoken_languages(self):
        return [c["language"] for c in FakeSarvamClient.calls]


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


class TheTable(unittest.TestCase):
    def test_holds_exactly_the_eleven_supported_languages(self):
        from supported_languages import SUPPORTED_LANGUAGES

        self.assertEqual(sorted(lang.code for lang in SUPPORTED_LANGUAGES), ELEVEN)

    def test_every_supported_language_has_an_india_hosted_voice(self):
        from sarvam_integration.config import SARVAM_TTS_LANGUAGES
        from supported_languages import get_language

        for code in ELEVEN:
            voice = get_language(code).voice("sarvam")
            self.assertIsNotNone(voice, code)
            self.assertEqual(voice.language, SARVAM_TAG[code])
            self.assertIn(voice.language, SARVAM_TTS_LANGUAGES)

    def test_each_language_is_found_by_its_code_tag_or_provider_spelling(self):
        from supported_languages import get_language

        for spelling in ("or", "or-IN", "od", "od-IN", "OR_in"):
            self.assertEqual(get_language(spelling).code, "or", spelling)
        self.assertEqual(get_language("ta-IN").code, "ta")

    def test_a_language_outside_the_eleven_is_not_supported(self):
        from supported_languages import get_language

        for code in ("as", "as-IN", "ne", "tu-IN", "", None):
            self.assertIsNone(get_language(code), code)

    def test_live_supports_bengali_and_gujarati_but_not_odia_or_punjabi(self):
        from supported_languages import get_language

        self.assertEqual(get_language("bn").voice("gemini_live").language, "bn-IN")
        self.assertEqual(get_language("gu").voice("gemini_live").language, "gu-IN")
        self.assertIsNone(get_language("or").voice("gemini_live"))
        self.assertIsNone(get_language("pa").voice("gemini_live"))

    def test_the_script_check_sees_the_bengali_letters_in_the_old_tamil_string(self):
        from supported_languages import scripts_in

        self.assertEqual(scripts_in("இப்போது நான் உங்களுக்கு தமிழில் பதিলளிப்পேন்।"), {"Taml", "Beng"})

    def test_every_message_in_the_table_is_written_only_in_its_own_script(self):
        from supported_languages import get_language, scripts_in

        for code in ELEVEN:
            lang = get_language(code)
            self.assertEqual(lang.script, SCRIPT[code], code)
            for message in (lang.voice_unavailable, lang.language_set, lang.native_name):
                self.assertEqual(scripts_in(message), {SCRIPT[code]}, f"{code}: {message}")

    def test_a_question_is_placed_in_the_language_it_was_written_in(self):
        from supported_languages import detect_language

        for code in ELEVEN:
            self.assertEqual(detect_language(QUESTION[code]).code, code, QUESTION[code])


# ---------------------------------------------------------------------------
# WhatsApp speech
# ---------------------------------------------------------------------------


class WhatsAppSpeech(SpeechProviders):
    def test_each_supported_language_is_spoken_in_its_own_india_hosted_voice(self):
        for code in ELEVEN:
            result = run(self.tts().synthesize_speech("text", code))
            self.assertTrue(result["audio_available"], code)
        self.assertEqual(self.spoken_languages(), [SARVAM_TAG[c] for c in ELEVEN])
        self.assertEqual(FakeEdgeCommunicate.voices, [], "the India-hosted voice was not used first")

    def test_never_hindi_when_the_india_hosted_voice_fails(self):
        FakeSarvamClient.fail = True
        for code in ELEVEN:
            result = run(self.tts().synthesize_speech("text", code))
            if code != "hi":
                self.assertNotIn("hi-IN", "".join(FakeEdgeCommunicate.voices), code)
            if not result["audio_available"]:
                self.assertEqual(result["status"], "unavailable", code)
            FakeEdgeCommunicate.voices = []

    def test_a_language_outside_the_eleven_is_not_spoken_in_hindi(self):
        result = run(self.tts().synthesize_speech("text", "as"))
        self.assertFalse(result["audio_available"])
        self.assertEqual(result["status"], "unavailable")
        self.assertNotIn("hi-IN", self.spoken_languages() + FakeEdgeCommunicate.voices)


class WhatsAppVoiceNotes(SpeechProviders):
    """A voice note in each language, through the traditional STT, RAG, TTS path."""

    def bot(self, heard):
        from whatsapp_bot import EnhancedWhatsAppBot

        bot = make_bot(RealisticRag(answer=SAFE_ANSWER))
        bot.tts_service = self.tts()
        bot.twilio_api.client = object()

        async def download(url):
            path = Path(tempfile.mkdtemp()) / "note.ogg"
            path.write_bytes(b"ogg")
            return str(path)

        async def transcribe(path, detected_language=None):
            return {"status": "success", "text": QUESTION.get(heard, "question"),
                    "detected_language": heard, "language_name": "?"}

        bot.twilio_api.download_media = download
        bot.stt_service = SimpleNamespace(supported_languages={}, transcribe_audio=transcribe)
        self.assertIsInstance(bot, EnhancedWhatsAppBot)
        return bot

    def test_each_voice_note_is_answered_in_its_own_language_and_voice(self):
        for code in ELEVEN:
            FakeSarvamClient.calls = []
            bot = self.bot(code)
            run(bot._handle_twilio_media_message(NUMBER, "https://x/m", "audio/ogg"))
            self.assertEqual(self.spoken_languages(), [SARVAM_TAG[code]], code)
            self.assertEqual(len(bot.twilio_api.audio), 1, code)
            if code != "en":
                self.assertTrue(any(t.startswith("🇮🇳") and f"[{code}]" in t for t in bot.twilio_api.texts),
                                f"{code}: no translation sent: {bot.twilio_api.texts}")

    def test_unavailable_speech_is_said_plainly_in_the_participants_language(self):
        from supported_languages import get_language

        FakeSarvamClient.fail = True
        bot = self.bot("pa")  # Punjabi: no voice other than Sarvam's
        run(bot._handle_twilio_media_message(NUMBER, "https://x/m", "audio/ogg"))
        self.assertEqual(bot.twilio_api.audio, [])
        self.assertIn(get_language("pa").voice_unavailable, bot.twilio_api.texts)
        self.assertNotIn("hi-IN", FakeEdgeCommunicate.voices)

    def test_a_voice_note_outside_the_eleven_is_told_so_in_english_not_hindi(self):
        from supported_languages import UNSUPPORTED_LANGUAGE_MESSAGE

        bot = self.bot("as")  # Assamese: Sarvam hears it but cannot speak it
        run(bot._handle_twilio_media_message(NUMBER, "https://x/m", "audio/ogg"))
        self.assertIn(UNSUPPORTED_LANGUAGE_MESSAGE, bot.twilio_api.texts)
        self.assertNotIn("hi-IN", self.spoken_languages() + FakeEdgeCommunicate.voices)
        self.assertFalse(any(t.startswith("🇮🇳") for t in bot.twilio_api.texts), "translated into a guess")


class WhatsAppText(unittest.TestCase):
    def test_a_text_message_is_translated_into_the_language_it_was_written_in(self):
        for code in ELEVEN:
            bot = make_bot(RealisticRag(answer=SAFE_ANSWER))
            run(bot._handle_twilio_text_message(NUMBER, QUESTION[code]))
            translations = [t for t in bot.twilio_api.texts if t.startswith("🇮🇳")]
            if code == "en":
                self.assertEqual(translations, [], "an English question was translated")
            else:
                self.assertEqual(len(translations), 1, f"{code}: {bot.twilio_api.texts}")
                self.assertIn(f"[{code}]", translations[0])
            self.assertNotIn("[hi]", "".join(bot.twilio_api.texts) if code != "hi" else "")

    def test_the_language_command_confirms_in_that_language_only(self):
        from supported_languages import scripts_in

        for code in ELEVEN:
            bot = make_bot(RealisticRag(answer=SAFE_ANSWER))
            run(bot._handle_language_command(f"/lang {code}", NUMBER))
            self.assertEqual(bot.user_preferences[NUMBER]["language"], code)
            confirmation = bot.twilio_api.texts[-1]
            self.assertEqual(scripts_in(confirmation), {SCRIPT[code]}, f"{code}: {confirmation}")

    def test_the_language_help_lists_all_eleven_without_stray_scripts(self):
        bot = make_bot(RealisticRag(answer=SAFE_ANSWER))
        run(bot._handle_language_command("/lang", NUMBER))
        help_text = bot.twilio_api.texts[-1]
        for code in ELEVEN:
            self.assertIn(f"/lang {code} ", help_text)
        self.assertNotIn("්", help_text, "a Sinhala sign in the Tamil name")


# ---------------------------------------------------------------------------
# WhatsApp Live
# ---------------------------------------------------------------------------


class WhatsAppLive(unittest.TestCase):
    def live_bot(self, heard):
        from tests.test_whatsapp_fast import live_bot

        bot, session = live_bot(SAFE_ANSWER, spoken="Sit upright and use a fan.")
        opened = []

        class Manager:
            async def get_or_create_session(self, user_id, language, voice):
                opened.append(language)
                return session

        async def transcribe(path, detected_language=None):
            return {"status": "success", "text": QUESTION[heard],
                    "detected_language": heard, "language_name": "?"}

        bot.gemini_session_manager = Manager()
        bot.stt_service = SimpleNamespace(supported_languages={}, transcribe_audio=transcribe)
        return bot, opened

    def test_bengali_and_gujarati_voice_notes_go_to_live_in_their_own_language(self):
        for code, tag in (("bn", "bn-IN"), ("gu", "gu-IN"), ("ta", "ta-IN"), ("hi", "hi-IN")):
            bot, opened = self.live_bot(code)
            run(bot._handle_twilio_media_message(NUMBER, "https://x/m", "audio/ogg"))
            self.assertEqual(opened, [tag], code)

    def test_a_language_live_cannot_speak_is_answered_by_the_sarvam_path_instead(self):
        for code in ("or", "pa"):
            bot, opened = self.live_bot(code)
            spoken = []

            async def synthesize(text, language):
                spoken.append(language)
                return {"audio_available": False}

            bot.tts_service = SimpleNamespace(synthesize_speech=synthesize)
            run(bot._handle_twilio_media_message(NUMBER, "https://x/m", "audio/ogg"))
            self.assertEqual(opened, [], f"{code} was sent to Live, which cannot speak it")
            self.assertEqual(spoken, [code])


if __name__ == "__main__":
    unittest.main()
