"""
The Supported Language table: the one list of the eleven languages the app both
listens in and speaks (ADR 0002), with what each channel needs to know about them.

Each row carries the language's short code, its BCP-47 tag, its script, the voice
each speech provider uses for it (or none, when that provider cannot speak it),
and the two short messages a channel sends in that language. Live support is the
presence of a ``gemini_live`` voice.

Rules the table exists to enforce:

  * A language outside the eleven is not supported. ``get_language`` returns
    None, and the channel says so plainly (``UNSUPPORTED_LANGUAGE_MESSAGE``).
  * Nothing here falls back to Hindi. A provider without a voice for a language
    has no entry, and the channel tells the participant speech is unavailable
    (``SupportedLanguage.voice_unavailable``) rather than speaking another
    language.
  * Every message is written only in its language's script (pinned by
    tests/test_supported_languages.py).

This is the expand step of the expand-migrate-contract in the round-3 spec:
WhatsApp reads this table; the other channels' tables are migrated and deleted
by ticket 15.

The voice_unavailable and language_set messages are machine-drafted and await
clinical-language review before a study release.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Set, Tuple

# Speech providers a row can name a voice for.
PROVIDER_SARVAM = "sarvam"            # Sarvam Bulbul: India-hosted
PROVIDER_EDGE = "edge"                # Microsoft Edge neural voices: not India-hosted
PROVIDER_GEMINI_LIVE = "gemini_live"  # Gemini Live native audio

# Bulbul v3 speakers are language-agnostic (sarvam_integration/config.py).
_SARVAM_SPEAKER = "priya"
# The Live voice the WhatsApp channel opens sessions with.
_LIVE_VOICE = "Aoede"


@dataclass(frozen=True)
class Voice:
    """How one provider speaks one language: its language code and voice name."""

    provider: str
    language: str
    name: str


@dataclass(frozen=True)
class SupportedLanguage:
    """One row of the Supported Language table."""

    code: str            # the short code channels pass around, e.g. "hi", "or"
    tag: str             # BCP-47, e.g. "hi-IN", "or-IN"
    name: str            # English name
    native_name: str     # the language's own name, in its own script
    script: str          # ISO 15924 code, e.g. "Deva"
    voice_unavailable: str  # "spoken reply unavailable, read the text", in this language
    language_set: str       # "I will now answer you in <this language>", in this language
    voices: Mapping[str, Voice] = field(default_factory=dict)
    aliases: Tuple[str, ...] = ()  # other spellings providers use, e.g. Sarvam's "od-IN"

    def voice(self, provider: str) -> Optional[Voice]:
        """The voice ``provider`` speaks this language in, or None if it cannot."""
        return self.voices.get(provider)

    @property
    def live_supported(self) -> bool:
        """Whether Gemini Live can hold a conversation in this language."""
        return PROVIDER_GEMINI_LIVE in self.voices


def _voices(tag: str, *, sarvam: str, edge: Optional[str], live: bool) -> Dict[str, Voice]:
    voices = {PROVIDER_SARVAM: Voice(PROVIDER_SARVAM, sarvam, _SARVAM_SPEAKER)}
    if edge:
        voices[PROVIDER_EDGE] = Voice(PROVIDER_EDGE, tag, edge)
    if live:
        voices[PROVIDER_GEMINI_LIVE] = Voice(PROVIDER_GEMINI_LIVE, tag, _LIVE_VOICE)
    return voices


# Live support follows gemini_live/config.py, which defines no Odia or Punjabi.
# Edge has no Odia or Punjabi voice in this app.
SUPPORTED_LANGUAGES: Tuple[SupportedLanguage, ...] = (
    SupportedLanguage(
        "hi", "hi-IN", "Hindi", "हिन्दी", "Deva",
        voice_unavailable="क्षमा करें, अभी हिंदी में आवाज़ वाला जवाब उपलब्ध नहीं है। कृपया लिखा हुआ जवाब पढ़ें।",
        language_set="अब मैं आपको हिंदी में जवाब दूंगा।",
        voices=_voices("hi-IN", sarvam="hi-IN", edge="hi-IN-SwaraNeural", live=True),
    ),
    SupportedLanguage(
        "bn", "bn-IN", "Bengali", "বাংলা", "Beng",
        voice_unavailable="দুঃখিত, এখন বাংলায় ভয়েস উত্তর পাওয়া যাচ্ছে না। অনুগ্রহ করে লেখা উত্তরটি পড়ুন।",
        language_set="এখন আমি আপনাকে বাংলায় উত্তর দেব।",
        voices=_voices("bn-IN", sarvam="bn-IN", edge="bn-IN-TanishaaNeural", live=True),
    ),
    SupportedLanguage(
        "kn", "kn-IN", "Kannada", "ಕನ್ನಡ", "Knda",
        voice_unavailable="ಕ್ಷಮಿಸಿ, ಈಗ ಕನ್ನಡದಲ್ಲಿ ಧ್ವನಿ ಉತ್ತರ ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ಬರೆದ ಉತ್ತರವನ್ನು ಓದಿ.",
        language_set="ಈಗ ನಾನು ನಿಮಗೆ ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸುತ್ತೇನೆ.",
        voices=_voices("kn-IN", sarvam="kn-IN", edge="kn-IN-SapnaNeural", live=True),
    ),
    SupportedLanguage(
        "ml", "ml-IN", "Malayalam", "മലയാളം", "Mlym",
        voice_unavailable="ക്ഷമിക്കണം, ഇപ്പോൾ മലയാളത്തിൽ ശബ്ദ മറുപടി ലഭ്യമല്ല. ദയവായി എഴുതിയ മറുപടി വായിക്കുക.",
        language_set="ഇനി ഞാൻ നിങ്ങൾക്ക് മലയാളത്തിൽ മറുപടി നൽകും.",
        voices=_voices("ml-IN", sarvam="ml-IN", edge="ml-IN-SobhanaNeural", live=True),
    ),
    SupportedLanguage(
        "mr", "mr-IN", "Marathi", "मराठी", "Deva",
        voice_unavailable="क्षमस्व, सध्या मराठीत आवाजातील उत्तर उपलब्ध नाही. कृपया लिहिलेले उत्तर वाचा.",
        language_set="आता मी तुम्हाला मराठीत उत्तर देईन.",
        voices=_voices("mr-IN", sarvam="mr-IN", edge="mr-IN-AarohiNeural", live=True),
    ),
    SupportedLanguage(
        "or", "or-IN", "Odia", "ଓଡ଼ିଆ", "Orya",
        voice_unavailable="କ୍ଷମା କରନ୍ତୁ, ବର୍ତ୍ତମାନ ଓଡ଼ିଆରେ ସ୍ୱର ଉତ୍ତର ଉପଲବ୍ଧ ନାହିଁ। ଦୟାକରି ଲିଖିତ ଉତ୍ତର ପଢ଼ନ୍ତୁ।",
        language_set="ବର୍ତ୍ତମାନ ମୁଁ ଆପଣଙ୍କୁ ଓଡ଼ିଆରେ ଉତ୍ତର ଦେବି।",
        voices=_voices("or-IN", sarvam="od-IN", edge=None, live=False),
        aliases=("od", "od-IN"),
    ),
    SupportedLanguage(
        "pa", "pa-IN", "Punjabi", "ਪੰਜਾਬੀ", "Guru",
        voice_unavailable="ਮਾਫ਼ ਕਰਨਾ, ਇਸ ਵੇਲੇ ਪੰਜਾਬੀ ਵਿੱਚ ਆਵਾਜ਼ ਵਾਲਾ ਜਵਾਬ ਉਪਲਬਧ ਨਹੀਂ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਲਿਖਿਆ ਜਵਾਬ ਪੜ੍ਹੋ।",
        language_set="ਹੁਣ ਮੈਂ ਤੁਹਾਨੂੰ ਪੰਜਾਬੀ ਵਿੱਚ ਜਵਾਬ ਦੇਵਾਂਗਾ।",
        voices=_voices("pa-IN", sarvam="pa-IN", edge=None, live=False),
    ),
    SupportedLanguage(
        "ta", "ta-IN", "Tamil", "தமிழ்", "Taml",
        voice_unavailable="மன்னிக்கவும், இப்போது தமிழில் குரல் பதில் கிடைக்கவில்லை. தயவுசெய்து எழுத்துப் பதிலைப் படிக்கவும்.",
        language_set="இப்போது நான் உங்களுக்கு தமிழில் பதிலளிப்பேன்.",
        voices=_voices("ta-IN", sarvam="ta-IN", edge="ta-IN-PallaviNeural", live=True),
    ),
    SupportedLanguage(
        "te", "te-IN", "Telugu", "తెలుగు", "Telu",
        voice_unavailable="క్షమించండి, ప్రస్తుతం తెలుగులో వాయిస్ సమాధానం అందుబాటులో లేదు. దయచేసి రాసిన సమాధానాన్ని చదవండి.",
        language_set="ఇప్పుడు నేను మీకు తెలుగులో సమాధానం ఇస్తాను.",
        voices=_voices("te-IN", sarvam="te-IN", edge="te-IN-ShrutiNeural", live=True),
    ),
    SupportedLanguage(
        "gu", "gu-IN", "Gujarati", "ગુજરાતી", "Gujr",
        voice_unavailable="માફ કરશો, અત્યારે ગુજરાતીમાં અવાજવાળો જવાબ ઉપલબ્ધ નથી. કૃપા કરીને લખેલો જવાબ વાંચો.",
        language_set="હવે હું તમને ગુજરાતીમાં જવાબ આપીશ.",
        voices=_voices("gu-IN", sarvam="gu-IN", edge="gu-IN-DhwaniNeural", live=True),
    ),
    SupportedLanguage(
        "en", "en-IN", "English", "English", "Latn",
        voice_unavailable="Sorry, a spoken reply in English is not available right now. Please read the text reply.",
        language_set="I will now respond to you in English.",
        voices=_voices("en-IN", sarvam="en-IN", edge="en-IN-NeerjaNeural", live=True),
    ),
)

ENGLISH_CODE = "en"

# Sent, in English, when a participant uses a language outside the eleven: the
# app cannot answer in it, and will not guess another Indian language for them.
UNSUPPORTED_LANGUAGE_MESSAGE = (
    "Sorry, I cannot answer in this language yet, so I will reply in English. "
    "I can answer in: " + ", ".join(lang.native_name for lang in SUPPORTED_LANGUAGES) + "."
)


def _key(code: str) -> str:
    return code.strip().replace("_", "-").lower()


_BY_KEY: Dict[str, SupportedLanguage] = {}
for _lang in SUPPORTED_LANGUAGES:
    for _spelling in (_lang.code, _lang.tag, *_lang.aliases, *(v.language for v in _lang.voices.values())):
        _BY_KEY[_key(_spelling)] = _lang


def get_language(code: Optional[str]) -> Optional[SupportedLanguage]:
    """The Supported Language for a short code, BCP-47 tag or provider spelling
    (case and ``_``/``-`` ignored), or None when it is not one of the eleven."""
    if not code:
        return None
    return _BY_KEY.get(_key(code))


def is_supported(code: Optional[str]) -> bool:
    """Whether ``code`` names one of the eleven Supported Languages."""
    return get_language(code) is not None


# ---------------------------------------------------------------------------
# Scripts
# ---------------------------------------------------------------------------

# Unicode blocks of the scripts the eleven languages are written in.
_SCRIPT_BLOCKS: Tuple[Tuple[int, int, str], ...] = (
    (0x0900, 0x097F, "Deva"),
    (0x0980, 0x09FF, "Beng"),
    (0x0A00, 0x0A7F, "Guru"),
    (0x0A80, 0x0AFF, "Gujr"),
    (0x0B00, 0x0B7F, "Orya"),
    (0x0B80, 0x0BFF, "Taml"),
    (0x0C00, 0x0C7F, "Telu"),
    (0x0C80, 0x0CFF, "Knda"),
    (0x0D00, 0x0D7F, "Mlym"),
    (0x0D80, 0x0DFF, "Sinh"),
)
# The danda and double danda sit in the Devanagari block but end sentences in
# Bengali, Odia, Punjabi and others too, so they belong to no one script.
_SHARED_MARKS = {"।", "॥"}


def _script_of(char: str) -> Optional[str]:
    if char in _SHARED_MARKS:
        return None
    point = ord(char)
    for low, high, script in _SCRIPT_BLOCKS:
        if low <= point <= high:
            return script
    if char.isascii() and char.isalpha():
        return "Latn"
    if unicodedata.category(char).startswith("L"):
        return "Other"
    return None


def scripts_in(text: str) -> Set[str]:
    """The scripts (ISO 15924) of the letters in ``text``; punctuation, digits,
    spaces and the shared danda are ignored."""
    return {s for s in (_script_of(c) for c in text or "") if s}


# Common Marathi words that Hindi does not use, to tell the two Devanagari
# languages apart.
_MARATHI_MARKERS: Tuple[str, ...] = ("आहे", "नाही", "मला", "आणि", "काय", "तुम्ही", "आम्ही", "माझ")
_LANGUAGE_FOR_SCRIPT = {"Beng": "bn", "Guru": "pa", "Gujr": "gu", "Orya": "or", "Taml": "ta",
                        "Telu": "te", "Knda": "kn", "Mlym": "ml", "Latn": "en"}


def detect_language(text: str) -> Optional[SupportedLanguage]:
    """The Supported Language ``text`` is written in, judged by its dominant script.

    Devanagari is Marathi when a common Marathi word appears and Hindi otherwise;
    Latin script is English. Returns None for text with no letters, or whose
    dominant script is not one the eleven are written in.
    """
    counts: Dict[str, int] = {}
    for char in text or "":
        script = _script_of(char)
        if script:
            counts[script] = counts.get(script, 0) + 1
    if not counts:
        return None
    script = max(counts, key=counts.get)
    if script == "Deva":
        return get_language("mr" if any(m in text for m in _MARATHI_MARKERS) else "hi")
    code = _LANGUAGE_FOR_SCRIPT.get(script)
    return get_language(code) if code else None
