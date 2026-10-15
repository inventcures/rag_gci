"""
Gemini Live API Configuration

Handles configuration loading from config.yaml and environment variables.
Provides constants for audio formats and voice options. The languages Live
speaks come from the Supported Language table (supported_languages.py).
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

import yaml
import re
from dotenv import load_dotenv

from supported_languages import (
    SUPPORTED_LANGUAGES as TABLE,
    UNSUPPORTED_LANGUAGE_MESSAGE,
    SupportedLanguage,
    get_language,
)

logger = logging.getLogger(__name__)


def _expand_env_vars(value: str) -> str:
    """Expand ${VAR} patterns in string with environment variable values."""
    if not isinstance(value, str):
        return value
    # Match ${VAR} or $VAR patterns
    pattern = r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)'

    def replace(match):
        var_name = match.group(1) or match.group(2)
        return os.getenv(var_name, match.group(0))  # Keep original if not found

    return re.sub(pattern, replace, value)

# Load environment variables
load_dotenv()

# Audio Format Constants (Gemini Live API requirements)
INPUT_SAMPLE_RATE = 16000      # 16kHz for input audio
OUTPUT_SAMPLE_RATE = 24000     # 24kHz for output audio
AUDIO_CHANNELS = 1             # Mono
SAMPLE_WIDTH = 2               # 16-bit (2 bytes)
AUDIO_ENCODING = "pcm_s16le"   # Signed 16-bit little-endian PCM

# Default chunk size for streaming (in bytes)
# 4096 bytes = 2048 samples = ~128ms at 16kHz
DEFAULT_CHUNK_SIZE = 4096

def live_language(code: Optional[str]) -> Optional[SupportedLanguage]:
    """The Supported Language ``code`` names, when Gemini Live can hold a
    conversation in it (the table's ``gemini_live`` voice); None otherwise.
    Languages come from the Supported Language table, never from here."""
    lang = get_language(code)
    return lang if lang is not None and lang.live_supported else None


def live_unavailable_message(code: Optional[str]) -> str:
    """What to tell a participant whose language Live cannot speak: the
    table's "unavailable" message in that language, or the English
    unsupported-language notice for a language outside the eleven."""
    lang = get_language(code)
    return lang.voice_unavailable if lang is not None else UNSUPPORTED_LANGUAGE_MESSAGE


# Live models the service knows how to run. Every one is a full palliative-care
# assistant (RAG, safety, system instruction). Which one runs is not chosen
# here, nor by a client: the study release record names it (release_live_model).
# The speech-to-speech translate model (gemini-3.5-live-translate-preview) is not
# offered: it takes no system instruction and no RAG, so nothing it says has the
# assistant's safety layer (review 3, ticket 12).
SUPPORTED_MODELS: Dict[str, Dict[str, object]] = {
    "gemini-live-2.5-flash-preview-native-audio-09-2025": {
        "label": "Gemini 2.5 Flash Native Audio",
        "description": "Stable conversational model with native audio",
        "kind": "assistant",
        "supports_rag": True,
    },
    "gemini-3.1-flash-live-preview": {
        "label": "Gemini 3.1 Flash Live",
        "description": "Latest low-latency conversational model (90+ languages)",
        "kind": "assistant",
        "supports_rag": True,
    },
}


def get_model_info(model: str) -> Dict[str, object]:
    """Get registry metadata for a model (empty dict if unknown)."""
    return SUPPORTED_MODELS.get(model, {})


def release_live_model() -> str:
    """The Live model the study release approves (`models.voice_primary.model`
    in data/study/release.yaml): the one source of the model every Live
    session opens with. "" when the record names none; the service then
    refuses to start rather than pick one."""
    from study_release import get_study_release

    model = get_study_release().live_model
    if not model:
        logger.error("The study release record names no Live model "
                     "(models.voice_primary.model); Gemini Live will not start")
    return model


def model_supports_rag(model: str) -> bool:
    """True if RAG context can be injected into sessions of this model.

    Unknown models (legacy IDs, env overrides) are treated as assistants.
    """
    return bool(get_model_info(model).get("supports_rag", True))


def model_uses_auto_language(model: str) -> bool:
    """True for models that auto-detect language and reject language_code.

    Native-audio 2.5 models and all Gemini 3.x live models auto-detect the
    spoken language; passing BCP-47 codes like 'en-IN' is rejected.
    """
    return "native-audio" in (model or "") or (model or "").startswith("gemini-3")


def model_uses_realtime_text(model: str) -> bool:
    """True for models where mid-conversation text must use send_realtime_input.

    Gemini 3.x live models only accept send_client_content for seeding initial
    context; after the first model turn, text must go via send_realtime_input.
    """
    return (model or "").startswith("gemini-3")


# Available Voice Options (Gemini Live prebuilt voices)
VOICE_OPTIONS: Dict[str, Dict[str, str]] = {
    "Aoede": {
        "style": "Easy-going",
        "description": "Warm and approachable, ideal for healthcare",
        "recommended_for": "palliative_care",
    },
    "Charon": {
        "style": "Informative",
        "description": "Clear and professional",
        "recommended_for": "medical_info",
    },
    "Kore": {
        "style": "Firm",
        "description": "Confident and reassuring",
        "recommended_for": "instructions",
    },
    "Puck": {
        "style": "Upbeat",
        "description": "Positive and encouraging",
        "recommended_for": "general",
    },
    "Fenrir": {
        "style": "Excitable",
        "description": "Energetic and engaging",
        "recommended_for": "engagement",
    },
    "Zephyr": {
        "style": "Bright",
        "description": "Light and friendly",
        "recommended_for": "casual",
    },
}

# Default voice for palliative care (warm and empathetic)
DEFAULT_VOICE = "Aoede"


@dataclass
class GeminiLiveConfig:
    """
    Configuration for Gemini Live API integration.

    Loads from config.yaml and environment variables.
    Environment variables take precedence.
    """

    # Google Cloud / Vertex AI settings
    project_id: str = ""
    location: str = "us-central1"
    api_key: Optional[str] = None

    # Model settings: the study release record's, never config.yaml's or a
    # client's (release_live_model). GEMINI_LIVE_MODEL overrides it for
    # development only, and the fingerprint then reports RELEASE DRIFT.
    #
    # Verified against the live service on 2026-10-04. The previous default,
    # gemini-live-2.5-flash-preview-native-audio-09-2025, is refused with 1008:
    # it is no longer served for bidiGenerateContent. Real-time voice had therefore
    # never worked against the live endpoint, and no test could have caught it
    # because every test fakes the network.
    #
    # Two models are confirmed working: this one, and
    # gemini-2.5-flash-native-audio-preview-09-2025. A real audio round trip on this
    # one transcribed "What should I give for severe pain at home?" and replied
    # "Managing severe pain at home" in 5.1 seconds.
    model: str = field(default_factory=release_live_model)

    # Voice settings
    default_voice: str = DEFAULT_VOICE
    default_language: str = "en-IN"

    # Session settings
    session_timeout_minutes: int = 14  # Gemini max is 15
    max_sessions_per_user: int = 1

    # Audio settings
    input_sample_rate: int = INPUT_SAMPLE_RATE
    output_sample_rate: int = OUTPUT_SAMPLE_RATE
    chunk_size: int = DEFAULT_CHUNK_SIZE

    # RAG integration
    rag_context_enabled: bool = True
    rag_top_k: int = 3

    # Feature flags
    enabled: bool = False
    fallback_enabled: bool = True  # Fall back to STT+LLM+TTS on error
    transcription_enabled: bool = True  # Capture transcriptions

    # The languages Live speaks: the Supported Language table's, not config.yaml's.
    supported_languages: List[str] = field(
        default_factory=lambda: [lang.tag for lang in TABLE if lang.live_supported]
    )

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "GeminiLiveConfig":
        """
        Load configuration from YAML file and environment variables.

        Args:
            config_path: Path to config.yaml file

        Returns:
            GeminiLiveConfig instance
        """
        config = cls()

        # Load from YAML if exists
        config_file = Path(config_path)
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    yaml_config = yaml.safe_load(f)

                gemini_config = yaml_config.get("gemini_live", {})

                # Map YAML fields to config attributes
                if gemini_config:
                    config.enabled = gemini_config.get("enabled", False)
                    # Expand env vars in project_id (e.g., ${GOOGLE_CLOUD_PROJECT})
                    yaml_project = gemini_config.get("project_id", "")
                    config.project_id = _expand_env_vars(yaml_project) or os.getenv("GOOGLE_CLOUD_PROJECT", "")

                    # Expand env vars in location (e.g., ${VERTEX_AI_LOCATION})
                    yaml_location = gemini_config.get("location", "us-central1")
                    config.location = _expand_env_vars(yaml_location)

                    # The release record names the Live model; config.yaml does not.
                    yaml_model = gemini_config.get("model")
                    if yaml_model and _expand_env_vars(yaml_model) != config.model:
                        logger.warning(
                            "Ignoring gemini_live.model=%r in %s: the study release "
                            "record names the Live model (%r)",
                            yaml_model, config_path, config.model,
                        )
                    config.default_voice = gemini_config.get(
                        "default_voice", DEFAULT_VOICE
                    )
                    config.default_language = gemini_config.get(
                        "default_language", "en-IN"
                    )
                    config.session_timeout_minutes = gemini_config.get(
                        "session_timeout_minutes", 14
                    )
                    config.rag_context_enabled = gemini_config.get(
                        "rag_context_enabled", True
                    )
                    config.rag_top_k = gemini_config.get("rag_top_k", 3)
                    config.fallback_enabled = gemini_config.get(
                        "fallback_enabled", True
                    )

            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        # Override with environment variables (take precedence)
        config.project_id = os.getenv(
            "GOOGLE_CLOUD_PROJECT", config.project_id
        )
        # Check both VERTEX_AI_LOCATION and GOOGLE_CLOUD_LOCATION
        config.location = os.getenv(
            "VERTEX_AI_LOCATION",
            os.getenv("GOOGLE_CLOUD_LOCATION", config.location)
        )
        config.api_key = os.getenv("GEMINI_API_KEY")

        # A development override of the release's model. It is what sessions
        # open with, so the release fingerprint reports it as drift.
        env_model = os.getenv("GEMINI_LIVE_MODEL")
        if env_model:
            logger.warning("GEMINI_LIVE_MODEL=%s overrides the release's Live model %s "
                           "(RELEASE DRIFT)", env_model, config.model)
            config.model = env_model

        # Validate configuration
        config._validate()

        return config

    def _validate(self) -> None:
        """Validate configuration values."""
        # Check if we have authentication
        if self.enabled and not self.project_id and not self.api_key:
            logger.warning(
                "Gemini Live enabled but no GOOGLE_CLOUD_PROJECT or "
                "GEMINI_API_KEY set. Will fail at runtime."
            )

        # Validate voice
        if self.default_voice not in VOICE_OPTIONS:
            logger.warning(
                f"Unknown voice: {self.default_voice}. "
                f"Using default: {DEFAULT_VOICE}"
            )
            self.default_voice = DEFAULT_VOICE

    def get_voice_config(self, voice_name: str = None) -> Dict[str, str]:
        """
        Get configuration for a specific voice.

        Args:
            voice_name: Voice name (e.g., "Aoede")

        Returns:
            Voice configuration dict
        """
        voice = voice_name or self.default_voice
        return VOICE_OPTIONS.get(voice, VOICE_OPTIONS[DEFAULT_VOICE])

    def is_language_supported(self, language_code: str) -> bool:
        """Whether Live can speak ``language_code`` (a code, tag or alias)."""
        return live_language(language_code) is not None

    def to_dict(self) -> Dict:
        """Convert config to dictionary for logging/debugging."""
        return {
            "enabled": self.enabled,
            "project_id": self.project_id[:10] + "..." if self.project_id else None,
            "location": self.location,
            "model": self.model,
            "default_voice": self.default_voice,
            "default_language": self.default_language,
            "supported_languages": self.supported_languages,
            "session_timeout_minutes": self.session_timeout_minutes,
            "rag_context_enabled": self.rag_context_enabled,
            "fallback_enabled": self.fallback_enabled,
        }


def live_enabled(config: GeminiLiveConfig) -> bool:
    """Whether the server starts Gemini Live: config.yaml's `gemini_live.enabled`,
    or GEMINI_LIVE_ENABLED=true (set by `--provider gemini`).

    The server's start-up and the release fingerprint both read this, so the
    release record's provider list cannot disagree with what runs.
    """
    return bool(config.enabled) or os.getenv("GEMINI_LIVE_ENABLED", "").lower() == "true"


# Global config instance (lazy loaded)
_config: Optional[GeminiLiveConfig] = None


def get_config() -> GeminiLiveConfig:
    """Get the global Gemini Live configuration."""
    global _config
    if _config is None:
        _config = GeminiLiveConfig.from_yaml()
    return _config


def reload_config() -> GeminiLiveConfig:
    """Reload configuration from file."""
    global _config
    _config = GeminiLiveConfig.from_yaml()
    return _config
