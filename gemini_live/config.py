"""
Gemini Live API Configuration

Handles configuration loading from config.yaml and environment variables.
Provides constants for audio formats, supported languages, and voice options.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

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

# Supported Languages with their configurations
SUPPORTED_LANGUAGES: Dict[str, Dict[str, str]] = {
    "en-IN": {
        "name": "English (India)",
        "native_name": "English",
        "flag": "🇮🇳",
        "tts_fallback": "en-IN-NeerjaNeural",
    },
    "hi-IN": {
        "name": "Hindi",
        "native_name": "हिन्दी",
        "flag": "🇮🇳",
        "tts_fallback": "hi-IN-SwaraNeural",
    },
    "mr-IN": {
        "name": "Marathi",
        "native_name": "मराठी",
        "flag": "🇮🇳",
        "tts_fallback": "mr-IN-AarohiNeural",
    },
    "ta-IN": {
        "name": "Tamil",
        "native_name": "தமிழ்",
        "flag": "🇮🇳",
        "tts_fallback": "ta-IN-PallaviNeural",
    },
}

# Map short codes to full language codes
LANGUAGE_CODE_MAP: Dict[str, str] = {
    "en": "en-IN",
    "hi": "hi-IN",
    "mr": "mr-IN",
    "ta": "ta-IN",
}

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

    # Model settings
    model: str = "gemini-live-2.5-flash-preview-native-audio-09-2025"

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

    # Supported languages (loaded from config)
    supported_languages: List[str] = field(
        default_factory=lambda: list(SUPPORTED_LANGUAGES.keys())
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
                    config.project_id = gemini_config.get(
                        "project_id",
                        os.getenv("GOOGLE_CLOUD_PROJECT", "")
                    )
                    config.location = gemini_config.get("location", "us-central1")
                    config.model = gemini_config.get("model", config.model)
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

                    # Load supported languages if specified
                    languages = gemini_config.get("supported_languages", [])
                    if languages:
                        config.supported_languages = [
                            lang.get("code", lang) if isinstance(lang, dict)
                            else lang
                            for lang in languages
                        ]

            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        # Override with environment variables (take precedence)
        config.project_id = os.getenv(
            "GOOGLE_CLOUD_PROJECT", config.project_id
        )
        config.location = os.getenv(
            "GOOGLE_CLOUD_LOCATION", config.location
        )
        config.api_key = os.getenv("GEMINI_API_KEY")

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

        # Validate language codes
        for lang in self.supported_languages:
            if lang not in SUPPORTED_LANGUAGES:
                logger.warning(f"Unknown language code: {lang}")

        # Validate voice
        if self.default_voice not in VOICE_OPTIONS:
            logger.warning(
                f"Unknown voice: {self.default_voice}. "
                f"Using default: {DEFAULT_VOICE}"
            )
            self.default_voice = DEFAULT_VOICE

    def get_language_config(self, language_code: str) -> Dict[str, str]:
        """
        Get configuration for a specific language.

        Args:
            language_code: Language code (e.g., "hi-IN" or "hi")

        Returns:
            Language configuration dict
        """
        # Handle short codes
        if language_code in LANGUAGE_CODE_MAP:
            language_code = LANGUAGE_CODE_MAP[language_code]

        return SUPPORTED_LANGUAGES.get(
            language_code,
            SUPPORTED_LANGUAGES["en-IN"]
        )

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
        """Check if a language is supported."""
        if language_code in LANGUAGE_CODE_MAP:
            language_code = LANGUAGE_CODE_MAP[language_code]
        return language_code in self.supported_languages

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
