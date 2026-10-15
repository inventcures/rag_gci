"""
Sarvam AI Configuration for Palli Sahayak Voice AI Helpline

Models, the table lookups and system prompt for the Sarvam voice provider
integration. The languages are the Supported Language table's.

Documentation: https://docs.sarvam.ai/api-reference-docs/introduction
"""

import os
from typing import Dict, Any, Optional

from supported_languages import PROVIDER_SARVAM, Voice, get_language

# Model registry. Saaras v4 / Bulbul v4 were announced at Sarvam Epoch
# (July 30, 2026). As of mid-Aug 2026 the live API accepts saaras:v4 but
# NOT yet bulbul:v4 (only bulbul:v2 / v3-beta / v3) — the client therefore
# falls back to bulbul:v3 automatically until Sarvam enables the v4 tag.
SARVAM_STT_MODELS: Dict[str, str] = {
    "saaras:v3": "Stable ASR, 22 Indic languages + English (default)",
    "saaras:v4": "Latest ASR (Epoch 2026), 22 Indic languages + English",
    "saaras:flash": "Low-latency ASR",
    "saarika:v2.5": "Legacy ASR (deprecated, 11 languages)",
}

SARVAM_TTS_MODELS: Dict[str, str] = {
    "bulbul:v3": "Stable multilingual TTS, 11 languages (default)",
    "bulbul:v4": "Latest TTS (Epoch 2026): richer emotion/expression; "
                 "auto-falls back to v3 until the API accepts the tag",
    "bulbul:v3-beta": "v3 beta channel",
    "bulbul:v2": "Legacy TTS (supports pitch/loudness controls)",
}

DEFAULT_STT_MODEL = "saaras:v3"
DEFAULT_TTS_MODEL = "bulbul:v3"
TTS_FALLBACK_MODEL = "bulbul:v3"


def get_stt_model() -> str:
    """STT model to use (override with SARVAM_STT_MODEL, e.g. saaras:v4)."""
    return os.getenv("SARVAM_STT_MODEL", DEFAULT_STT_MODEL)


def get_tts_model() -> str:
    """TTS model to use (override with SARVAM_TTS_MODEL, e.g. bulbul:v4)."""
    return os.getenv("SARVAM_TTS_MODEL", DEFAULT_TTS_MODEL)


# Languages, codes and speakers come from the Supported Language table
# (supported_languages.py): each row's "sarvam" voice is the code Bulbul speaks
# it in (Odia is "od-IN") and its speaker. Bulbul v3 speakers are
# language-agnostic, so every row uses the same one. A language outside the
# eleven has no Sarvam voice, and nothing here falls back to Hindi.


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


def sarvam_voice(language: Optional[str]) -> Optional[Voice]:
    """The Sarvam voice (Bulbul's code and speaker) for ``language`` (a code,
    tag or alias), or None when it is not one of the eleven Supported
    Languages. Callers say speech is unavailable; they never substitute Hindi."""
    lang = get_language(language)
    return lang.voice(PROVIDER_SARVAM) if lang is not None else None


def sarvam_stt_language(language: Optional[str]) -> str:
    """The code to ask Saaras to transcribe in: the Supported Language's
    Sarvam code, or "unknown" (Saaras detects the language) for any other."""
    voice = sarvam_voice(language)
    return voice.language if voice is not None else "unknown"


def get_sarvam_config_from_env() -> Dict[str, Any]:
    """
    Load Sarvam configuration from environment variables.

    Environment variables:
    - SARVAM_API_KEY: API key (required)
    - SARVAM_BASE_URL: Base URL (default: https://api.sarvam.ai)
    - SARVAM_DEFAULT_LANGUAGE: Default language (default: hi-IN)
    - SARVAM_TTS_VOICE: Default TTS voice (default: meera)
    - SARVAM_TTS_PACE: TTS pace 0.5-2.0 (default: 1.0)
    - SARVAM_STT_MODEL: STT model (default: saaras:v3; also: saaras:v4)
    - SARVAM_TTS_MODEL: TTS model (default: bulbul:v3; also: bulbul:v4)
    """
    return {
        "api_key": os.getenv("SARVAM_API_KEY", ""),
        "base_url": os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai"),
        "default_language": os.getenv("SARVAM_DEFAULT_LANGUAGE", "hi-IN"),
        "tts_voice": os.getenv("SARVAM_TTS_VOICE", "meera"),
        "tts_pace": float(os.getenv("SARVAM_TTS_PACE", "1.0")),
        "stt_model": get_stt_model(),
        "tts_model": get_tts_model(),
    }
