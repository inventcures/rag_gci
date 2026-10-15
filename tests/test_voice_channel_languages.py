"""
Every voice channel reads the Supported Language table, and the old tables go.

Before this, each channel kept its own language table, each covering a
different subset of the eleven Supported Languages (ADR 0002):

  * Gemini Live and the web client's picker offered Assamese, with a Hindi
    voice, and left out Odia and Punjabi;
  * Sarvam kept an unused table sending eleven STT-only languages to Hindi TTS;
  * Bolna answered Punjabi and Malayalam in a Hindi voice, and fell back to a
    Hindi agent for any language it did not list;
  * Retell knew four languages, fell back to Hindi, and turned "od-IN" into a
    second Odia code, "od";
  * the offline bundle listed Assamese and left out Marathi, Odia, Punjabi
    and Gujarati, and keyed Odia "od-IN" while the table says "or".

What these pin down, for each channel and each of the eleven languages: the
channel resolves to that language, or to an explicit "unavailable" answer, and
never to Hindi for a non-Hindi input. And no module outside the table defines a
language map of its own.
"""

import ast
import asyncio
import os
import re
import sys
import tempfile
import unittest
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supported_languages import (  # noqa: E402
    ENGLISH_CODE,
    PROVIDER_BOLNA,
    PROVIDER_RETELL,
    PROVIDER_SARVAM,
    SUPPORTED_LANGUAGES,
    UNSUPPORTED_LANGUAGE_MESSAGE,
    get_language,
    scripts_in,
)

ELEVEN = sorted(lang.code for lang in SUPPORTED_LANGUAGES)
# Languages outside the eleven that the old tables carried (Assamese) or a
# provider can report (Nepali): each must be refused plainly, never answered in Hindi.
OUTSIDE = ("as", "as-IN", "ne-IN")


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# The table carries what the channels need
# ---------------------------------------------------------------------------


class TheTableGreets(unittest.TestCase):
    def test_every_greeting_is_written_only_in_its_own_script(self):
        for lang in SUPPORTED_LANGUAGES:
            self.assertEqual(scripts_in(lang.greeting), {lang.script}, f"{lang.code}: {lang.greeting}")

    def test_no_message_holds_an_unassigned_character(self):
        # The old Sarvam Odia greeting ended in U+0B64, which is not a character.
        for lang in SUPPORTED_LANGUAGES:
            for message in (lang.greeting, lang.voice_unavailable, lang.language_set):
                bad = [hex(ord(c)) for c in message if unicodedata.category(c) == "Cn"]
                self.assertEqual(bad, [], f"{lang.code}: {message}")


# ---------------------------------------------------------------------------
# No module outside the table defines a language map
# ---------------------------------------------------------------------------

# Every spelling of a language a provider or an old table used.
_CODES = {"hi", "bn", "kn", "ml", "mr", "or", "od", "pa", "ta", "te", "gu", "en", "as", "ur",
          "ne", "sa", "sd", "ks", "kok", "gom", "mai", "mni", "brx", "doi"}
_TAG = re.compile(r"^[a-z]{2,3}[-_]IN$")
_NAMES = {"Hindi", "Bengali", "Kannada", "Malayalam", "Marathi", "Odia", "Oriya", "Punjabi",
          "Tamil", "Telugu", "Gujarati", "English", "Assamese", "English (India)"}
# Keys of a per-language settings dict (a voice, a code, a name), as opposed to
# a dict of localised messages.
_SETTING_KEYS = {"voice", "voice_id", "voice_name", "tts_fallback", "transcriber_language",
                 "language_code", "retell_language", "name", "native_name", "tts_supported",
                 "stt_supported", "female", "male", "default_voice", "flag"}
_NOT_SCANNED = {"kotaemon-main", "tests", "docs", "android-app", "node_modules", "cache", "out",
                "venv", ".venv", ".git", ".claude", "data"}


def _is_language(node) -> bool:
    return (isinstance(node, ast.Constant) and isinstance(node.value, str)
            and (node.value in _CODES or bool(_TAG.match(node.value))))


def _is_setting(node) -> bool:
    """A value that says how to handle a language: another language code, a
    voice, a language name, or a dict of such settings."""
    if isinstance(node, ast.Dict):
        keys = {k.value for k in node.keys if isinstance(k, ast.Constant)}
        return bool(keys & _SETTING_KEYS)
    if isinstance(node, (ast.Name, ast.Subscript, ast.Attribute)):
        return True
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        text = node.value
        return bool(_TAG.match(text) or text in _CODES or text in _NAMES
                    or re.match(r"^[a-z]{2}-[A-Z]{2}-\w+Neural$", text))
    return False


def language_maps(source: str):
    """(line, what) for every language map in `source`: a dict keyed by three or
    more language codes whose values are settings (codes, voices, names), or a
    list of language codes that enumerates the set (eight or more) or names a
    language outside the eleven."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Dict):
            pairs = [(k, v) for k, v in zip(node.keys, node.values) if k is not None and _is_language(k)]
            if len(pairs) >= 3 and all(_is_setting(v) for _, v in pairs):
                found.append((node.lineno, "dict " + ",".join(k.value for k, _ in pairs)))
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            codes = [e.value for e in node.elts if _is_language(e)]
            if len(codes) >= 3 and len(codes) == len(node.elts) and (
                    len(codes) >= 8 or any(get_language(c) is None for c in codes)):
                found.append((node.lineno, "list " + ",".join(codes)))
    return found


class NoOtherLanguageMap(unittest.TestCase):
    def test_the_scan_finds_the_shapes_the_old_tables_had(self):
        old = '''
GEMINI = {"en-IN": {"name": "English", "tts_fallback": "en-IN-NeerjaNeural"},
          "hi-IN": {"name": "Hindi"}, "as-IN": {"name": "Assamese", "tts_fallback": "hi-IN-SwaraNeural"}}
CODES = {"en": "en-IN", "hi": "hi-IN", "as": "as-IN"}
FALLBACK = {"as-IN": "hi-IN", "ne-IN": "hi-IN", "sa-IN": "hi-IN"}
NAMES = {"en-IN": "English", "hi-IN": "Hindi", "ta-IN": "Tamil"}
OFFLINE = ["ta-IN", "te-IN", "as-IN", "hi-IN"]
ALL = ["en-IN", "hi-IN", "bn-IN", "kn-IN", "ml-IN", "mr-IN", "or-IN", "pa-IN"]
'''
        self.assertEqual(len(language_maps(old)), 6)

    def test_localised_messages_and_word_lists_are_not_language_maps(self):
        fine = '''
MESSAGES = {"hi": "कृपया दोबारा बताएं।", "en": "Please repeat.", "ta": "மீண்டும் சொல்லுங்கள்."}
TEMPLATES = {"en-IN": {"welcome_back": ["Welcome back."]}, "hi-IN": {"welcome_back": ["x"]}, "ta-IN": {}}
UNITS = {"hi": ("मिलीग्राम",), "bn": ("মিলিগ্রাম",), "or": ("ମିଲିଗ୍ରାମ",)}
SAMPLE = ("hi", "bn", "ta", "gu")
'''
        self.assertEqual(language_maps(fine), [])

    def test_no_module_outside_the_table_defines_a_language_map(self):
        found = []
        for path in sorted(ROOT.rglob("*.py")):
            rel = path.relative_to(ROOT)
            if rel.parts[0] in _NOT_SCANNED or rel.as_posix() == "supported_languages.py":
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            try:
                maps = language_maps(source)
            except SyntaxError:
                continue
            found.extend(f"{rel}:{line} {what}" for line, what in maps)
        self.assertEqual(found, [], "language maps outside supported_languages.py:\n" + "\n".join(found))

    def test_the_web_client_hard_codes_no_language_list(self):
        for path in sorted((ROOT / "web_client").glob("*.*")):
            text = path.read_text(encoding="utf-8")
            tags = set(re.findall(r"['\"]([a-z]{2,3}-IN)['\"]", text))
            # One default ('en-IN') before the server's list arrives, no more.
            self.assertLessEqual(tags, {"en-IN"}, f"{path.name} lists languages: {sorted(tags)}")
            self.assertNotIn("data-lang=", text, f"{path.name} hard-codes language buttons")
            self.assertNotIn("Assamese", text, path.name)


# ---------------------------------------------------------------------------
# Gemini Live
# ---------------------------------------------------------------------------


class GeminiLive(unittest.TestCase):
    def service(self):
        from gemini_live.service import GeminiLiveService

        service = GeminiLiveService()
        service.client = object()  # no network: create_session only builds the config
        return service

    def test_each_language_opens_live_in_that_language_or_is_refused_in_it(self):
        from gemini_live.service import GeminiLiveError

        for code in ELEVEN:
            lang = get_language(code)
            for spelling in (code, lang.tag):
                with self.subTest(language=spelling):
                    if lang.live_supported:
                        session = run(self.service().create_session(f"s-{spelling}", language=spelling))
                        self.assertEqual(session.language, lang.tag)
                    else:
                        with self.assertRaises(GeminiLiveError) as caught:
                            run(self.service().create_session(f"s-{spelling}", language=spelling))
                        self.assertIn(lang.voice_unavailable, str(caught.exception))

    def test_a_language_outside_the_eleven_is_refused_not_spoken_in_hindi_or_english(self):
        from gemini_live.service import GeminiLiveError

        for spelling in OUTSIDE:
            with self.subTest(language=spelling), self.assertRaises(GeminiLiveError) as caught:
                run(self.service().create_session("s-out", language=spelling))
            self.assertIn(UNSUPPORTED_LANGUAGE_MESSAGE, str(caught.exception))

    def test_live_languages_are_the_tables(self):
        from gemini_live.config import GeminiLiveConfig

        config = GeminiLiveConfig.from_yaml(str(ROOT / "config.yaml"))
        expected = [lang.tag for lang in SUPPORTED_LANGUAGES if lang.live_supported]
        self.assertEqual(config.supported_languages, expected)
        for code in ELEVEN:
            self.assertEqual(config.is_language_supported(code), get_language(code).live_supported, code)
        self.assertFalse(config.is_language_supported("as-IN"))

    def test_the_system_instruction_names_the_live_languages_only(self):
        instruction = self.service()._build_system_instruction("hi-IN")
        self.assertNotIn("Assamese", instruction)
        for lang in SUPPORTED_LANGUAGES:
            if lang.live_supported and lang.code != ENGLISH_CODE:
                self.assertIn(lang.name, instruction)


# ---------------------------------------------------------------------------
# The web client's picker
# ---------------------------------------------------------------------------


class WebPicker(unittest.TestCase):
    def test_the_picker_lists_the_live_languages_from_the_table(self):
        from fastapi.testclient import TestClient
        from tests.test_route_policy import build_server_app

        response = TestClient(build_server_app()).get("/api/voice/languages")
        self.assertEqual(response.status_code, 200, response.text)
        listed = response.json()["languages"]
        live = [lang for lang in SUPPORTED_LANGUAGES if lang.live_supported]
        self.assertEqual([item["tag"] for item in listed], [lang.tag for lang in live])
        for item, lang in zip(listed, live):
            self.assertEqual(item["native_name"], lang.native_name)
            self.assertEqual(item["name"], lang.name)
            self.assertEqual(item["greeting"], lang.greeting)
        self.assertNotIn("as-IN", [item["tag"] for item in listed])

    def test_the_page_builds_its_picker_from_the_server(self):
        page = (ROOT / "web_client" / "index.html").read_text(encoding="utf-8")
        self.assertIn("/api/voice/languages", page)
        self.assertNotIn("welcomeMessages", page)


# ---------------------------------------------------------------------------
# Sarvam (the mobile voice path)
# ---------------------------------------------------------------------------


class FakeSarvam:
    """Stands in for sarvam_integration.SarvamClient; records each call."""

    tts = []
    stt = []

    def __init__(self, *args, **kwargs):
        pass

    async def text_to_speech(self, text, language="hi-IN", voice="priya", **kwargs):
        FakeSarvam.tts.append((language, voice))
        return SimpleNamespace(success=True, audio_base64="UklGRg==", error=None)

    async def speech_to_text(self, audio, language="hi-IN", **kwargs):
        FakeSarvam.stt.append(language)
        return SimpleNamespace(success=True, transcript="question", language_code=language)


class SarvamVoice(unittest.TestCase):
    def setUp(self):
        FakeSarvam.tts, FakeSarvam.stt = [], []
        patcher = mock.patch("sarvam_integration.SarvamClient", FakeSarvam)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_each_language_is_spoken_in_its_own_sarvam_voice(self):
        from mobile_api.router import _synthesise_voice

        for code in ELEVEN:
            voice = get_language(code).voice(PROVIDER_SARVAM)
            for spelling in {code, get_language(code).tag, voice.language}:
                with self.subTest(language=spelling):
                    FakeSarvam.tts = []
                    self.assertIsNotNone(run(_synthesise_voice("text", spelling)))
                    self.assertEqual(FakeSarvam.tts, [(voice.language, voice.name)])

    def test_a_language_outside_the_eleven_is_not_spoken_in_hindi(self):
        from mobile_api.router import _synthesise_voice

        for spelling in OUTSIDE:
            self.assertIsNone(run(_synthesise_voice("text", spelling)), spelling)
        self.assertEqual(FakeSarvam.tts, [])

    def test_each_language_is_heard_in_its_own_sarvam_code_and_others_are_detected(self):
        from voice_session import VoiceTurnHandler

        handler = VoiceTurnHandler(rag_pipeline=None, safety_manager=None)
        for code in ELEVEN:
            run(handler._transcribe(b"audio", get_language(code).tag))
        for spelling in OUTSIDE:
            run(handler._transcribe(b"audio", spelling))
        self.assertEqual(FakeSarvam.stt, [get_language(c).voice(PROVIDER_SARVAM).language for c in ELEVEN]
                         + ["unknown"] * len(OUTSIDE))

    def test_the_hindi_fallback_table_is_gone(self):
        import sarvam_integration.config as config

        for name in ("_TTS_FALLBACK", "get_tts_language", "SARVAM_VOICE_MAP", "SARVAM_LANGUAGE_CONFIGS",
                     "_SHORT_TO_BCP47", "SARVAM_STT_LANGUAGES", "SARVAM_TTS_LANGUAGES"):
            self.assertFalse(hasattr(config, name), name)


# ---------------------------------------------------------------------------
# Bolna and Retell (the phone line)
# ---------------------------------------------------------------------------


class Bolna(unittest.TestCase):
    def config(self, language):
        from bolna_integration.config import get_palli_sahayak_agent_config

        return get_palli_sahayak_agent_config(server_url="https://example.org", language=language)

    def test_each_language_gets_its_own_agent_or_is_refused(self):
        from supported_languages import CARTESIA_VOICES

        for code in ELEVEN:
            lang = get_language(code)
            voice = lang.voice(PROVIDER_BOLNA)
            with self.subTest(language=code):
                if voice is None:
                    with self.assertRaises(ValueError) as caught:
                        self.config(code)
                    self.assertIn(lang.name, str(caught.exception))
                    continue
                config = self.config(code)
                tools = config["tasks"][0]["tools_config"]
                self.assertEqual(tools["transcriber"]["language"], voice.language)
                self.assertEqual(tools["synthesizer"]["provider_config"]["voice_id"], CARTESIA_VOICES[voice.name])
                self.assertEqual(config["agent_welcome_message"], lang.greeting)
                self.assertEqual(scripts_in(config["agent_welcome_message"]), {lang.script})
                if code != "hi":
                    self.assertNotEqual(voice.language, "hi")

    def test_punjabi_and_malayalam_are_no_longer_answered_in_hindi(self):
        for code in ("pa", "ml"):
            with self.assertRaises(ValueError):
                self.config(code)

    def test_a_language_outside_the_eleven_gets_no_hindi_agent(self):
        for spelling in OUTSIDE + ("hinglish",):
            with self.subTest(language=spelling), self.assertRaises(ValueError):
                self.config(spelling)

    def test_the_caller_can_report_any_of_the_eleven(self):
        from bolna_integration.config import EXTRACTION_SCHEMA, RAG_QUERY_FUNCTION

        enum = RAG_QUERY_FUNCTION["parameters"]["properties"]["user_language"]["enum"]
        self.assertEqual(sorted(enum), ELEVEN)
        self.assertEqual(sorted(EXTRACTION_SCHEMA["language_used"]["enum"]), ELEVEN)


class Retell(unittest.TestCase):
    def config(self, language):
        from retell_integration.config import get_palli_sahayak_retell_config

        return get_palli_sahayak_retell_config(llm_websocket_url="wss://example.org/ws", language=language)

    def test_each_language_gets_its_own_agent_or_is_refused(self):
        from supported_languages import CARTESIA_VOICES

        for code in ELEVEN:
            lang = get_language(code)
            voice = lang.voice(PROVIDER_RETELL)
            with self.subTest(language=code):
                if voice is None:
                    with self.assertRaises(ValueError) as caught:
                        self.config(code)
                    self.assertIn(lang.name, str(caught.exception))
                    continue
                config = self.config(code)
                self.assertEqual(config.language, voice.language)
                self.assertEqual(config.voice_id, CARTESIA_VOICES[voice.name])
                self.assertEqual(config.welcome_message, lang.greeting)
                if code != "hi":
                    self.assertNotEqual(config.language, "hi-IN")

    def test_a_language_outside_the_eleven_gets_no_hindi_agent(self):
        for spelling in OUTSIDE:
            with self.subTest(language=spelling), self.assertRaises(ValueError):
                self.config(spelling)

    def details(self, metadata):
        from retell_integration.custom_llm_server import RetellCustomLLMHandler, RetellSession

        session = RetellSession(call_id="c1", websocket=None)
        run(RetellCustomLLMHandler()._handle_call_details(session, {"call": {"metadata": metadata}}))
        return session

    def test_the_call_language_resolves_through_the_table(self):
        for code in ELEVEN:
            lang = get_language(code)
            for spelling in {code, lang.tag, *lang.aliases}:
                with self.subTest(language=spelling):
                    self.assertEqual(self.details({"language": spelling}).language, code)

    def test_a_call_language_outside_the_eleven_is_answered_in_english_not_hindi(self):
        for spelling in OUTSIDE:
            self.assertEqual(self.details({"language": spelling}).language, ENGLISH_CODE, spelling)

    def test_with_no_call_language_the_turn_is_placed_by_its_script_not_hindi(self):
        from retell_integration.custom_llm_server import RetellCustomLLMHandler
        from tests.test_supported_languages import QUESTION

        handler = RetellCustomLLMHandler()
        for code in ELEVEN:
            session = self.details({})
            self.assertEqual(handler._turn_language(session, QUESTION[code]), code, code)


class OdiaHasOneCode(unittest.TestCase):
    """Sarvam and the Android app spell Odia "od-IN"; the table's code is "or".
    Every spelling must reach the same Odia text."""

    def test_the_index_refusal_is_in_odia_under_either_spelling(self):
        from retrieval_index import INDEX_UNAVAILABLE_MESSAGE, index_unavailable_message

        for spelling in ("or", "or-IN", "od", "od-IN"):
            self.assertEqual(index_unavailable_message(spelling), INDEX_UNAVAILABLE_MESSAGE["or"], spelling)

    def test_dose_words_key_odia_by_the_tables_code(self):
        from dose_detector import _INDIC_FREQUENCY_WORDS, UNIT_WORDS

        for words in (UNIT_WORDS, _INDIC_FREQUENCY_WORDS):
            self.assertIn("or", words)
            self.assertNotIn("od", words)


# ---------------------------------------------------------------------------
# The offline bundle
# ---------------------------------------------------------------------------


class OfflineBundle(unittest.TestCase):
    def setUp(self):
        from offline import cache_builder

        self.cache_builder = cache_builder
        for target, value in ((cache_builder, "CACHE_DIR"),):
            patcher = mock.patch.object(target, value, Path(tempfile.mkdtemp(prefix="bundles-")))
            patcher.start()
            self.addCleanup(patcher.stop)
        cache_builder._BUNDLES.clear()
        self.addCleanup(cache_builder._BUNDLES.clear)

    def test_the_bundle_language_list_is_the_eleven(self):
        from offline.questions import SUPPORTED_LANGUAGES as BUNDLE_LANGUAGES, has_translation

        self.assertEqual(BUNDLE_LANGUAGES, [lang.tag for lang in SUPPORTED_LANGUAGES])
        for tag in BUNDLE_LANGUAGES:
            self.assertTrue(has_translation(tag), tag)

    def test_odia_has_one_bundle_under_either_spelling(self):
        builder = self.cache_builder.CacheBundleBuilder()
        bundles = [run(builder.get_bundle(spelling)) for spelling in ("od-IN", "or-IN", "or")]
        self.assertEqual({key for _, key in bundles}, {bundles[0][1]})
        bundle = bundles[0][0]
        self.assertEqual(bundle["language"], "or-IN")
        self.assertTrue(bundle["questions_translated"])

    def test_each_language_bundle_is_in_that_language(self):
        from offline.questions import questions_for

        for lang in SUPPORTED_LANGUAGES:
            for spelling in {lang.code, lang.tag, *lang.aliases}:
                with self.subTest(language=spelling):
                    questions = questions_for(spelling)
                    self.assertTrue(all(lang.script in scripts_in(q) for q in questions), spelling)

    def test_a_language_outside_the_eleven_gets_no_bundle(self):
        builder = self.cache_builder.CacheBundleBuilder()
        for spelling in OUTSIDE:
            with self.subTest(language=spelling), self.assertRaises(ValueError):
                run(builder.get_bundle(spelling))


if __name__ == "__main__":
    unittest.main(verbosity=2)
