"""
Bolna.ai Integration Module for Palli Sahayak Voice AI Helpline

This module provides:
- BolnaClient: API client for Bolna.ai
- BolnaAgentConfig: Agent configuration management
- BolnaWebhookHandler: Webhook event processing

Usage:
    from bolna_integration import BolnaClient, get_palli_sahayak_agent_config

    client = BolnaClient()
    config = get_palli_sahayak_agent_config(server_url="https://your-server.com")
    result = await client.create_agent(config)
"""

from .client import BolnaClient, BolnaCallResult
from .config import (
    get_palli_sahayak_agent_config,
    get_agent_config_from_env,
    PALLI_SAHAYAK_SYSTEM_PROMPT,
    RAG_QUERY_FUNCTION,
)
from .webhooks import BolnaWebhookHandler, CallRecord

__all__ = [
    "BolnaClient",
    "BolnaCallResult",
    "get_palli_sahayak_agent_config",
    "get_agent_config_from_env",
    "PALLI_SAHAYAK_SYSTEM_PROMPT",
    "RAG_QUERY_FUNCTION",
    "BolnaWebhookHandler",
    "CallRecord",
]

__version__ = "1.0.0"
