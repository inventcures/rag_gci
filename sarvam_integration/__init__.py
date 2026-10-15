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
    get_stt_model,
    get_tts_model,
    SARVAM_SYSTEM_PROMPT,
    sarvam_stt_language,
    sarvam_voice,
    SARVAM_STT_MODELS,
    SARVAM_TTS_MODELS,
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
    "get_stt_model",
    "get_tts_model",
    "SARVAM_STT_MODELS",
    "SARVAM_TTS_MODELS",
    "SARVAM_SYSTEM_PROMPT",
    "sarvam_stt_language",
    "sarvam_voice",
    "SarvamWebhookHandler",
    "SarvamCallRecord",
]

__version__ = "1.0.0"
