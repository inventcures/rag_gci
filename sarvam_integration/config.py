"""
Sarvam AI Configuration for Palli Sahayak Voice AI Helpline

Language configs (22 ASR / 11 TTS), voice mapping, and system prompt
for the Sarvam voice provider integration.

Documentation: https://docs.sarvam.ai/api-reference-docs/introduction
"""

import os
from typing import Dict, Any, Optional

SARVAM_STT_LANGUAGES = [
    "hi-IN", "bn-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN",
    "ta-IN", "te-IN", "en-IN", "gu-IN", "as-IN", "brx-IN", "doi-IN",
    "gom-IN", "ks-IN", "kok-IN", "mai-IN", "mni-IN", "ne-IN", "sa-IN", "sd-IN",
]

SARVAM_TTS_LANGUAGES = [
    "hi-IN", "bn-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN",
    "ta-IN", "te-IN", "en-IN", "gu-IN",
]

SARVAM_VOICE_MAP: Dict[str, Dict[str, str]] = {
    "hi-IN": {"female": "meera", "male": "amol"},
    "bn-IN": {"female": "diya", "male": "arnab"},
    "kn-IN": {"female": "pavithra", "male": "raghav"},
    "ml-IN": {"female": "ammu", "male": "arjun"},
    "mr-IN": {"female": "ananya", "male": "advait"},
    "od-IN": {"female": "shruti", "male": "manash"},
    "pa-IN": {"female": "divya", "male": "gurpreet"},
    "ta-IN": {"female": "thara", "male": "aravind"},
    "te-IN": {"female": "lakshmi", "male": "karthik"},
    "en-IN": {"female": "sita", "male": "neel"},
    "gu-IN": {"female": "riddhi", "male": "chirag"},
}

# Short code <-> BCP-47 mapping
_SHORT_TO_BCP47 = {
    "hi": "hi-IN", "en": "en-IN", "mr": "mr-IN", "ta": "ta-IN",
    "bn": "bn-IN", "kn": "kn-IN", "ml": "ml-IN", "te": "te-IN",
    "gu": "gu-IN", "pa": "pa-IN", "od": "od-IN", "as": "as-IN",
    "brx": "brx-IN", "doi": "doi-IN", "gom": "gom-IN", "ks": "ks-IN",
    "kok": "kok-IN", "mai": "mai-IN", "mni": "mni-IN", "ne": "ne-IN",
    "sa": "sa-IN", "sd": "sd-IN",
}

# TTS fallback for STT-only languages (fall back to Hindi TTS)
_TTS_FALLBACK = {
    "as-IN": "hi-IN", "brx-IN": "hi-IN", "doi-IN": "hi-IN",
    "gom-IN": "hi-IN", "ks-IN": "hi-IN", "kok-IN": "hi-IN",
    "mai-IN": "hi-IN", "mni-IN": "hi-IN", "ne-IN": "hi-IN",
    "sa-IN": "hi-IN", "sd-IN": "hi-IN",
}

SARVAM_LANGUAGE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "hi-IN": {
        "name": "Hindi",
        "stt_supported": True,
        "tts_supported": True,
        "default_voice": "meera",
        "welcome_message": "\u0928\u092e\u0938\u094d\u0924\u0947! \u092e\u0948\u0902 \u092a\u0932\u094d\u0932\u0940 \u0938\u0939\u093e\u092f\u0915 \u0939\u0942\u0902\u0964 \u0906\u091c \u092e\u0948\u0902 \u0906\u092a\u0915\u0940 \u0915\u0948\u0938\u0947 \u092e\u0926\u0926 \u0915\u0930 \u0938\u0915\u0924\u0940 \u0939\u0942\u0902?",
        "tts_fallback": None,
    },
    "en-IN": {
        "name": "English (India)",
        "stt_supported": True,
        "tts_supported": True,
        "default_voice": "sita",
        "welcome_message": "Hello! I am Palli Sahayak, your palliative care assistant. How can I help you today?",
        "tts_fallback": None,
    },
    "bn-IN": {
        "name": "Bengali",
        "stt_supported": True,
        "tts_supported": True,
        "default_voice": "diya",
        "welcome_message": "\u09a8\u09ae\u09b8\u09cd\u0995\u09be\u09b0! \u0986\u09ae\u09bf \u09aa\u09b2\u09cd\u09b2\u09c0 \u09b8\u09b9\u09be\u09df\u0995\u0964 \u0986\u099c \u0986\u09ae\u09bf \u0986\u09aa\u09a8\u09be\u0995\u09c7 \u0995\u09c0\u09ad\u09be\u09ac\u09c7 \u09b8\u09be\u09b9\u09be\u09af\u09cd\u09af \u0995\u09b0\u09a4\u09c7 \u09aa\u09be\u09b0\u09bf?",
        "tts_fallback": None,
    },
    "kn-IN": {
        "name": "Kannada",
        "stt_supported": True,
        "tts_supported": True,
        "default_voice": "pavithra",
        "welcome_message": "\u0ca8\u0cae\u0cb8\u0ccd\u0c95\u0cbe\u0cb0! \u0ca8\u0cbe\u0ca8\u0cc1 \u0caa\u0cb2\u0ccd\u0cb2\u0cbf \u0cb8\u0cb9\u0cbe\u0caf\u0c95\u0ccd. \u0c88 \u0cb9\u0cc7\u0c97\u0cc6 \u0ca8\u0cbf\u0cae\u0c97\u0cc6 \u0cb8\u0cb9\u0cbe\u0caf \u0cae\u0cbe\u0ca1\u0cac\u0cb9\u0cc1\u0ca6\u0cc1?",
        "tts_fallback": None,
    },
    "ml-IN": {
        "name": "Malayalam",
        "stt_supported": True,
        "tts_supported": True,
        "default_voice": "ammu",
        "welcome_message": "\u0d28\u0d2e\u0d38\u0d4d\u0d15\u0d3e\u0d30\u0d02! \u0d1e\u0d3e\u0d28\u0d4d\u200d \u0d2a\u0d32\u0d4d\u0d32\u0d3f \u0d38\u0d39\u0d3e\u0d2f\u0d15\u0d4d \u0d06\u0d23\u0d4d. \u0d07\u0d28\u0d4d\u0d28\u0d4d \u0d1e\u0d3e\u0d28\u0d4d\u200d \u0d28\u0d3f\u0d19\u0d4d\u0d19\u0d33\u0d46 \u0d0e\u0d19\u0d4d\u0d19\u0d28\u0d46 \u0d38\u0d39\u0d3e\u0d2f\u0d3f\u0d15\u0d4d\u0d15\u0d3e\u0d02?",
        "tts_fallback": None,
    },
    "mr-IN": {
        "name": "Marathi",
        "stt_supported": True,
        "tts_supported": True,
        "default_voice": "ananya",
        "welcome_message": "\u0928\u092e\u0938\u094d\u0915\u093e\u0930! \u092e\u0940 \u092a\u0932\u094d\u0932\u0940 \u0938\u0939\u093e\u092f\u0915 \u0906\u0939\u0947. \u0906\u091c \u092e\u0940 \u0924\u0941\u092e\u094d\u0939\u093e\u0932\u093e \u0915\u0936\u0940 \u092e\u0926\u0924 \u0915\u0930\u0942 \u0936\u0915\u0924\u0947?",
        "tts_fallback": None,
    },
    "od-IN": {
        "name": "Odia",
        "stt_supported": True,
        "tts_supported": True,
        "default_voice": "shruti",
        "welcome_message": "\u0b28\u0b2e\u0b38\u0b4d\u0b15\u0b3e\u0b30! \u0b2e\u0b41\u0b01 \u0b2a\u0b32\u0b4d\u0b32\u0b40 \u0b38\u0b39\u0b3e\u0b5f\u0b15\u0b4d\u0b64 \u0b06\u0b1c\u0b3f \u0b2e\u0b41\u0b01 \u0b06\u0b2a\u0b23\u0b19\u0b4d\u0b15\u0b41 \u0b15\u0b3f\u0b2a\u0b30\u0b3f \u0b38\u0b3e\u0b39\u0b3e\u0b2f\u0b4d\u0b5f \u0b15\u0b30\u0b3f\u0b2a\u0b3e\u0b30\u0b3f\u0b2c\u0b3f?",
        "tts_fallback": None,
    },
    "pa-IN": {
        "name": "Punjabi",
        "stt_supported": True,
        "tts_supported": True,
        "default_voice": "divya",
        "welcome_message": "\u0a38\u0a24 \u0a38\u0a4d\u0a30\u0a40 \u0a05\u0a15\u0a3e\u0a32! \u0a2e\u0a48\u0a02 \u0a2a\u0a71\u0a32\u0a40 \u0a38\u0a39\u0a3e\u0a07\u0a15 \u0a39\u0a3e\u0a02\u0964 \u0a05\u0a71\u0a1c \u0a2e\u0a48\u0a02 \u0a24\u0a41\u0a39\u0a3e\u0a21\u0a40 \u0a15\u0a3f\u0a35\u0a47\u0a02 \u0a2e\u0a26\u0a26 \u0a15\u0a30 \u0a38\u0a15\u0a26\u0a40 \u0a39\u0a3e\u0a02?",
        "tts_fallback": None,
    },
    "ta-IN": {
        "name": "Tamil",
        "stt_supported": True,
        "tts_supported": True,
        "default_voice": "thara",
        "welcome_message": "\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd! \u0ba8\u0bbe\u0ba9\u0bcd \u0baa\u0bb2\u0bcd\u0bb2\u0bbf \u0b9a\u0b95\u0bbe\u0baf\u0b95\u0bcd. \u0b87\u0ba9\u0bcd\u0bb1\u0bc1 \u0ba8\u0bbe\u0ba9\u0bcd \u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bc1\u0b95\u0bcd\u0b95\u0bc1 \u0b8e\u0baa\u0bcd\u0baa\u0b9f\u0bbf \u0b89\u0ba4\u0bb5 \u0bae\u0bc1\u0b9f\u0bbf\u0baf\u0bc1\u0bae\u0bcd?",
        "tts_fallback": None,
    },
    "te-IN": {
        "name": "Telugu",
        "stt_supported": True,
        "tts_supported": True,
        "default_voice": "lakshmi",
        "welcome_message": "\u0c28\u0c2e\u0c38\u0c4d\u0c15\u0c3e\u0c30\u0c02! \u0c28\u0c47\u0c28\u0c41 \u0c2a\u0c32\u0c4d\u0c32\u0c3f \u0c38\u0c39\u0c3e\u0c2f\u0c15\u0c4d. \u0c08 \u0c30\u0c4b\u0c1c\u0c41 \u0c28\u0c47\u0c28\u0c41 \u0c2e\u0c40\u0c15\u0c41 \u0c0e\u0c32\u0c3e \u0c38\u0c39\u0c3e\u0c2f\u0c02 \u0c1a\u0c47\u0c2f\u0c17\u0c32\u0c28\u0c41?",
        "tts_fallback": None,
    },
    "gu-IN": {
        "name": "Gujarati",
        "stt_supported": True,
        "tts_supported": True,
        "default_voice": "riddhi",
        "welcome_message": "\u0aa8\u0aae\u0ab8\u0acd\u0a95\u0abe\u0ab0! \u0ab9\u0ac1\u0a82 \u0aaa\u0ab2\u0acd\u0ab2\u0ac0 \u0ab8\u0ab9\u0abe\u0aaf\u0a95 \u0a9b\u0ac1\u0a82. \u0a86\u0a9c\u0ac7 \u0ab9\u0ac1\u0a82 \u0aa4\u0aae\u0aa8\u0ac7 \u0a95\u0ac7\u0ab5\u0ac0 \u0ab0\u0ac0\u0aa4\u0ac7 \u0aae\u0aa6\u0aa6 \u0a95\u0ab0\u0ac0 \u0ab6\u0a95\u0ac1\u0a82?",
        "tts_fallback": None,
    },
    "as-IN": {
        "name": "Assamese",
        "stt_supported": True,
        "tts_supported": False,
        "default_voice": "meera",
        "welcome_message": "\u09a8\u09ae\u09b8\u09cd\u0995\u09be\u09f0! \u09ae\u0987 \u09aa\u09b2\u09cd\u09b2\u09c0 \u09b8\u09b9\u09be\u09af\u09bc\u0995\u0964",
        "tts_fallback": "hi-IN",
    },
    "brx-IN": {
        "name": "Bodo",
        "stt_supported": True,
        "tts_supported": False,
        "default_voice": "meera",
        "welcome_message": "Namaskar! Ang Palli Sahayak.",
        "tts_fallback": "hi-IN",
    },
    "doi-IN": {
        "name": "Dogri",
        "stt_supported": True,
        "tts_supported": False,
        "default_voice": "meera",
        "welcome_message": "\u0928\u092e\u0938\u094d\u0915\u093e\u0930! \u092e\u0948\u0902 \u092a\u0932\u094d\u0932\u0940 \u0938\u0939\u093e\u092f\u0915 \u0939\u093e\u0902\u0964",
        "tts_fallback": "hi-IN",
    },
    "gom-IN": {
        "name": "Goan Konkani",
        "stt_supported": True,
        "tts_supported": False,
        "default_voice": "meera",
        "welcome_message": "Namaskar! Hanv Palli Sahayak.",
        "tts_fallback": "hi-IN",
    },
    "ks-IN": {
        "name": "Kashmiri",
        "stt_supported": True,
        "tts_supported": False,
        "default_voice": "meera",
        "welcome_message": "\u0627\u064e\u0633\u0644\u0627\u0645 \u0639\u0644\u06cc\u06a9\u0645! \u0628\u06c1 \u067e\u0644\u0651\u06cc \u0633\u06c1\u0627\u06cc\u06a9 \u0686\u06c1\u06c1.",
        "tts_fallback": "hi-IN",
    },
    "kok-IN": {
        "name": "Konkani",
        "stt_supported": True,
        "tts_supported": False,
        "default_voice": "meera",
        "welcome_message": "\u0928\u092e\u0938\u094d\u0915\u093e\u0930! \u0939\u093e\u0902\u0935 \u092a\u0932\u094d\u0932\u0940 \u0938\u0939\u093e\u092f\u0915.",
        "tts_fallback": "hi-IN",
    },
    "mai-IN": {
        "name": "Maithili",
        "stt_supported": True,
        "tts_supported": False,
        "default_voice": "meera",
        "welcome_message": "\u092a\u094d\u0930\u0923\u093e\u092e! \u0939\u092e \u092a\u0932\u094d\u0932\u0940 \u0938\u0939\u093e\u092f\u0915 \u091b\u0940\u0964",
        "tts_fallback": "hi-IN",
    },
    "mni-IN": {
        "name": "Manipuri",
        "stt_supported": True,
        "tts_supported": False,
        "default_voice": "meera",
        "welcome_message": "Namaskar! Ei Palli Sahayak ni.",
        "tts_fallback": "hi-IN",
    },
    "ne-IN": {
        "name": "Nepali",
        "stt_supported": True,
        "tts_supported": False,
        "default_voice": "meera",
        "welcome_message": "\u0928\u092e\u0938\u094d\u0924\u0947! \u092e \u092a\u0932\u094d\u0932\u0940 \u0938\u0939\u093e\u092f\u0915 \u0939\u0941\u0901\u0964",
        "tts_fallback": "hi-IN",
    },
    "sa-IN": {
        "name": "Sanskrit",
        "stt_supported": True,
        "tts_supported": False,
        "default_voice": "meera",
        "welcome_message": "\u0928\u092e\u0938\u094d\u0915\u093e\u0930\u0903! \u0905\u0939\u0902 \u092a\u0932\u094d\u0932\u0940 \u0938\u0939\u093e\u092f\u0915\u0903\u0964",
        "tts_fallback": "hi-IN",
    },
    "sd-IN": {
        "name": "Sindhi",
        "stt_supported": True,
        "tts_supported": False,
        "default_voice": "meera",
        "welcome_message": "\u0646\u0645\u0633\u062a\u06d2! \u0645\u0627\u0646 \u067e\u0644\u0651\u064a \u0633\u0647\u0627\u064a\u06a9 \u0622\u0647\u064a\u0627\u0646.",
        "tts_fallback": "hi-IN",
    },
}

SARVAM_SYSTEM_PROMPT = """You are Palli Sahayak, a compassionate palliative care voice assistant for the Palli Sahayak Voice AI Agent Helpline, powered by Sarvam AI.

## YOUR ROLE
- Provide empathetic support for patients and caregivers dealing with serious illness
- Answer questions about pain management, symptom control, and comfort care
- Offer emotional support and guidance during difficult times
- Help navigate palliative care services and resources

## LANGUAGE GUIDELINES
- Respond in the SAME LANGUAGE the user speaks
- Sarvam supports 22 Indian languages for understanding and 11 for speaking
- Use simple, clear language - avoid complex medical jargon
- Be warm, patient, gentle, and understanding at all times
- Acknowledge emotions before providing information

## CRITICAL RULES
1. ALWAYS query the knowledge base FIRST when user asks any health question
2. NEVER provide specific medication dosages - always recommend consulting their doctor
3. For emergencies (severe pain, breathing difficulty, unconsciousness), immediately advise calling emergency services (108/112)
4. Keep voice responses concise - 2-3 sentences maximum for easy listening
5. Express empathy and validate feelings before giving medical information
"""


def sarvam_language_code(short_code: str) -> str:
    """Convert short language code (e.g. 'hi') to Sarvam BCP-47 format (e.g. 'hi-IN')."""
    return _SHORT_TO_BCP47.get(short_code, f"{short_code}-IN")


def short_language_code(sarvam_code: str) -> str:
    """Convert Sarvam BCP-47 code (e.g. 'hi-IN') to short format (e.g. 'hi')."""
    return sarvam_code.split("-")[0]


def get_tts_language(language: str) -> str:
    """Get TTS language, falling back to Hindi for unsupported languages."""
    if language in SARVAM_TTS_LANGUAGES:
        return language
    bcp47 = sarvam_language_code(language) if "-" not in language else language
    return _TTS_FALLBACK.get(bcp47, "hi-IN")


def get_default_voice(language: str, gender: str = "female") -> str:
    """Get default voice name for a language and gender."""
    bcp47 = sarvam_language_code(language) if "-" not in language else language
    tts_lang = get_tts_language(bcp47)
    voices = SARVAM_VOICE_MAP.get(tts_lang, SARVAM_VOICE_MAP["hi-IN"])
    return voices.get(gender, voices["female"])


def get_sarvam_config_from_env() -> Dict[str, Any]:
    """
    Load Sarvam configuration from environment variables.

    Environment variables:
    - SARVAM_API_KEY: API key (required)
    - SARVAM_BASE_URL: Base URL (default: https://api.sarvam.ai)
    - SARVAM_DEFAULT_LANGUAGE: Default language (default: hi-IN)
    - SARVAM_TTS_VOICE: Default TTS voice (default: meera)
    - SARVAM_TTS_PACE: TTS pace 0.5-2.0 (default: 1.0)
    - SARVAM_STT_MODEL: STT model (default: saaras:v3)
    """
    return {
        "api_key": os.getenv("SARVAM_API_KEY", ""),
        "base_url": os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai"),
        "default_language": os.getenv("SARVAM_DEFAULT_LANGUAGE", "hi-IN"),
        "tts_voice": os.getenv("SARVAM_TTS_VOICE", "meera"),
        "tts_pace": float(os.getenv("SARVAM_TTS_PACE", "1.0")),
        "stt_model": os.getenv("SARVAM_STT_MODEL", "saaras:v3"),
    }
