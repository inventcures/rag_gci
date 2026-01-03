"""
Retell.AI Integration Module for Palli Sahayak Voice AI Helpline

Provides:
- RetellClient: API client for Retell.AI
- RetellWebhookHandler: Webhook event processing
- RetellCustomLLMHandler: WebSocket server for Custom LLM protocol
- VobizConfig: Vobiz.ai telephony configuration

Documentation: https://docs.retellai.com/
"""

from .client import RetellClient, RetellCallResult
from .config import (
    get_palli_sahayak_retell_config,
    get_retell_config_from_env,
    RETELL_SYSTEM_PROMPT,
    RETELL_LANGUAGE_CONFIGS,
    CARTESIA_VOICE_IDS,
    RetellAgentConfig,
)
from .webhooks import RetellWebhookHandler, RetellCallRecord
from .custom_llm_server import RetellCustomLLMHandler, RetellSession
from .vobiz_config import VobizConfig, get_vobiz_config

__all__ = [
    # Client
    "RetellClient",
    "RetellCallResult",
    # Config
    "get_palli_sahayak_retell_config",
    "get_retell_config_from_env",
    "RETELL_SYSTEM_PROMPT",
    "RETELL_LANGUAGE_CONFIGS",
    "CARTESIA_VOICE_IDS",
    "RetellAgentConfig",
    # Webhooks
    "RetellWebhookHandler",
    "RetellCallRecord",
    # Custom LLM
    "RetellCustomLLMHandler",
    "RetellSession",
    # Vobiz
    "VobizConfig",
    "get_vobiz_config",
]

__version__ = "1.0.0"
