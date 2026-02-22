"""
Sarvam AI Integration Module for Palli Sahayak Voice AI Helpline

Provides:
- SarvamClient: API client for Sarvam AI (STT, TTS, Translation)
- SarvamStreamingClient: WebSocket streaming for real-time STT/TTS
- SarvamWebhookHandler: Webhook event processing

Documentation: https://docs.sarvam.ai/api-reference-docs/introduction
"""

from .client import SarvamClient, SarvamSTTResult, SarvamTTSResult, SarvamTranslateResult
from .config import (
    get_sarvam_config_from_env,
    SARVAM_SYSTEM_PROMPT,
    SARVAM_LANGUAGE_CONFIGS,
    SARVAM_VOICE_MAP,
    SARVAM_STT_LANGUAGES,
    SARVAM_TTS_LANGUAGES,
)
from .streaming import SarvamStreamingClient
from .webhooks import SarvamWebhookHandler, SarvamCallRecord

__all__ = [
    "SarvamClient",
    "SarvamSTTResult",
    "SarvamTTSResult",
    "SarvamTranslateResult",
    "SarvamStreamingClient",
    "get_sarvam_config_from_env",
    "SARVAM_SYSTEM_PROMPT",
    "SARVAM_LANGUAGE_CONFIGS",
    "SARVAM_VOICE_MAP",
    "SARVAM_STT_LANGUAGES",
    "SARVAM_TTS_LANGUAGES",
    "SarvamWebhookHandler",
    "SarvamCallRecord",
]

__version__ = "1.0.0"
