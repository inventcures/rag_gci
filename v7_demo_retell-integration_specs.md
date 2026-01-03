# V7 Demo: Retell.AI Voice Provider Integration Specification

## Document Version
- **Version**: 1.0.0
- **Date**: January 2026
- **Author**: Claude Code

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Phase 1: Foundation Setup](#3-phase-1-foundation-setup)
4. [Phase 2: Configuration Module](#4-phase-2-configuration-module)
5. [Phase 3: API Client](#5-phase-3-api-client)
6. [Phase 4: Custom LLM WebSocket Server](#6-phase-4-custom-llm-websocket-server)
7. [Phase 5: Webhook Handler](#7-phase-5-webhook-handler)
8. [Phase 6: Vobiz.ai Telephony Configuration](#8-phase-6-vobizai-telephony-configuration)
9. [Phase 7: VoiceRouter Extension](#9-phase-7-voicerouter-extension)
10. [Phase 8: Server Integration](#10-phase-8-server-integration)
11. [Phase 9: Testing Suite](#11-phase-9-testing-suite)
12. [Environment Variables](#12-environment-variables)
13. [Implementation Checklist](#13-implementation-checklist)
14. [References](#14-references)

---

## 1. Executive Summary

### 1.1 Purpose

Add Retell.AI as the 3rd voice provider for the Palli Sahayak Voice AI Helpline, enabling:
- **Custom LLM via WebSocket**: Full RAG integration with palliative care knowledge base
- **Cartesia Sonic-3 TTS**: High-quality Indian language voices
- **Dual Telephony Architecture**:
  - Twilio sandbox for WhatsApp voice calls
  - Vobiz.ai for Indian PSTN inbound (+91 number, no internet required)

### 1.2 CLI Usage

```bash
# Start with Retell.AI provider
python simple_rag_server.py -p r

# Or full form
python simple_rag_server.py --provider retell
```

### 1.3 Provider Comparison

| Feature | Bolna (`-p b`) | Gemini Live (`-p g`) | Retell (`-p r`) |
|---------|----------------|----------------------|-----------------|
| Use Case | Phone helpline | Web voice | Phone + WebRTC |
| LLM Integration | Custom function call | Native context | Custom LLM WebSocket |
| STT | Deepgram Nova-3 | Gemini native | Retell STT |
| TTS | Cartesia Sonic-3 | Gemini native | Cartesia Sonic-3 |
| Telephony | Twilio | N/A | Vobiz.ai (+91) |
| Latency | ~2-3s | <1s | ~1-2s |

---

## 2. Architecture Overview

### 2.1 System Diagram

```
                    +------------------+
                    |    Caller        |
                    | (Regular Phone)  |
                    +--------+---------+
                             |
                             | PSTN Call (+91)
                             v
                    +------------------+
                    |   Vobiz.ai       |
                    | Indian Telephony |
                    | DID: +91XXXXXXXX |
                    +--------+---------+
                             |
                             | SIP Trunk
                             v
                    +------------------+
                    |   Retell.AI      |
                    | Voice Platform   |
                    +--------+---------+
                             |
                             | WebSocket
                             v
            +----------------+----------------+
            |                                 |
            | /ws/retell/llm/{call_id}        |
            |                                 |
            +----------------+----------------+
                             |
                             v
            +----------------+----------------+
            |    RetellCustomLLMHandler       |
            |                                 |
            |  - Receive transcript           |
            |  - Query RAG pipeline           |
            |  - Stream response              |
            +----------------+----------------+
                             |
                             v
            +----------------+----------------+
            |     SimpleRAGPipeline           |
            |                                 |
            |  - ChromaDB vector search       |
            |  - Knowledge Graph (optional)   |
            |  - LLM response generation      |
            +----------------+----------------+
                             |
                             v
            +----------------+----------------+
            |    Cartesia Sonic-3 TTS         |
            |                                 |
            |  - Hindi Narrator Woman         |
            |  - Indian Lady (English)        |
            |  - Hinglish Speaking Lady       |
            +---------------------------------+
```

### 2.2 Data Flow

1. **Inbound Call**: Caller dials +91 Vobiz number
2. **SIP Routing**: Vobiz routes to Retell via SIP trunk
3. **WebSocket Connect**: Retell connects to `/ws/retell/llm/{call_id}`
4. **Transcript Delivery**: Retell sends `response_required` with transcript
5. **RAG Query**: Server queries knowledge base
6. **Response Streaming**: Server sends response chunks
7. **TTS Synthesis**: Retell synthesizes via Cartesia Sonic-3
8. **Audio Playback**: Audio streams to caller

---

## 3. Phase 1: Foundation Setup

### 3.1 Directory Structure

```bash
retell_integration/
├── __init__.py
├── client.py
├── config.py
├── webhooks.py
├── custom_llm_server.py
└── vobiz_config.py
```

### 3.2 Create Directory

```bash
mkdir -p retell_integration
```

### 3.3 `__init__.py`

```python
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
```

### 3.4 Add to requirements.txt

```
retell-sdk>=4.0.0
```

### 3.5 Tests

```bash
python -c "from pathlib import Path; assert Path('retell_integration').exists(); print('OK')"
python -c "from pathlib import Path; assert Path('retell_integration/__init__.py').exists(); print('OK')"
```

### 3.6 Commit

```
feat(retell): Phase 1 - foundation setup

- Create retell_integration/ directory
- Add __init__.py with module exports
- Add retell-sdk to requirements.txt
```

---

## 4. Phase 2: Configuration Module

### 4.1 `config.py`

```python
"""
Retell Agent Configuration for Palli Sahayak Voice AI Helpline

This module provides the agent configuration for the Palli Sahayak
palliative care voice assistant using Retell.AI with Cartesia Sonic-3 TTS.

Documentation: https://docs.retellai.com/api-references
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Retell API Base URL
RETELL_API_BASE = "https://api.retellai.com"

# System prompt for Palli Sahayak voice assistant
RETELL_SYSTEM_PROMPT = """You are Palli Sahayak (पल्ली सहायक), a compassionate palliative care voice assistant for the Palli Sahayak Voice AI Agent Helpline.

## YOUR ROLE
- Provide empathetic support for patients and caregivers dealing with serious illness
- Answer questions about pain management, symptom control, and comfort care
- Offer emotional support and guidance during difficult times
- Help navigate palliative care services and resources

## LANGUAGE GUIDELINES
- Respond in the SAME LANGUAGE the user speaks (Hindi, English, Marathi, or Tamil)
- Use simple, clear language - avoid complex medical jargon
- Be warm, patient, gentle, and understanding at all times
- Acknowledge emotions before providing information

## CRITICAL RULES
1. Use the medical knowledge provided to give accurate, evidence-based responses
2. **NEVER** provide specific medication dosages - always recommend consulting their doctor
3. For emergencies (severe pain, breathing difficulty, unconsciousness), immediately advise calling emergency services (108/112)
4. Keep voice responses concise - 2-3 sentences maximum for easy listening
5. Express empathy and validate feelings before giving medical information
6. **NEVER** use laughter, jokes, or humor - this is a serious palliative care context
7. **DO** use gentle emotional expression to convey empathy and compassion

## CONVERSATION STYLE
- Start with a warm greeting
- Listen carefully to understand the concern
- Provide compassionate, grounded response from the knowledge base
- End by asking if there's anything else you can help with

## SAMPLE RESPONSES

For pain question:
"I understand managing pain can be very difficult. Based on our medical guidelines... [response]. Remember, your doctor can adjust medications based on your specific needs."

For emotional support:
"I hear how challenging this is for you. It's completely natural to feel overwhelmed. Would you like me to share some coping strategies, or would you prefer information about support groups in your area?"
"""

# Cartesia Sonic-3 Voice IDs for Indian languages
# Reference: https://developer.signalwire.com/voice/tts/cartesia/
# Reference: https://cartesia.ai/india
CARTESIA_VOICE_IDS = {
    "hi": {
        "voice_id": "c1abd502-9231-4558-a054-10ac950c356d",
        "voice_name": "Hindi Narrator Woman",
        "model": "sonic-3",
        "description": "Warm, empathetic Hindi female voice",
        "retell_voice_id": "cartesia-hindi-narrator-woman"
    },
    "en": {
        "voice_id": "3b554273-4299-48b9-9aaf-eefd438e3941",
        "voice_name": "Indian Lady",
        "model": "sonic-3",
        "description": "Natural Indian English female voice",
        "retell_voice_id": "cartesia-indian-lady"
    },
    "mr": {
        "voice_id": "c1abd502-9231-4558-a054-10ac950c356d",
        "voice_name": "Hindi Narrator Woman",
        "model": "sonic-3",
        "description": "Hindi voice for Marathi (closest match)",
        "retell_voice_id": "cartesia-hindi-narrator-woman"
    },
    "ta": {
        "voice_id": "3b554273-4299-48b9-9aaf-eefd438e3941",
        "voice_name": "Indian Lady",
        "model": "sonic-3",
        "description": "Indian English voice for Tamil (closest match)",
        "retell_voice_id": "cartesia-indian-lady"
    },
    "hinglish": {
        "voice_id": "95d51f79-c397-46f9-b49a-23763d3eaa2d",
        "voice_name": "Hinglish Speaking Lady",
        "model": "sonic-3",
        "description": "Native Hinglish blend voice",
        "retell_voice_id": "cartesia-hinglish-speaking-lady"
    }
}

# Language-specific configurations
RETELL_LANGUAGE_CONFIGS = {
    "hi": {
        "language_code": "hi",
        "retell_language": "hi-IN",
        "welcome_message": "नमस्ते! मैं पल्ली सहायक हूं, आपका पैलिएटिव केयर सहायक। आज मैं आपकी कैसे मदद कर सकती हूं?",
        "voice": CARTESIA_VOICE_IDS["hi"],
        "name": "Hindi"
    },
    "en": {
        "language_code": "en",
        "retell_language": "en-IN",
        "welcome_message": "Hello! I am Palli Sahayak, your palliative care assistant. How can I help you today?",
        "voice": CARTESIA_VOICE_IDS["en"],
        "name": "English"
    },
    "mr": {
        "language_code": "mr",
        "retell_language": "mr-IN",
        "welcome_message": "नमस्कार! मी पल्ली सहायक आहे, तुमचा पॅलिएटिव केअर सहाय्यक. आज मी तुम्हाला कशी मदत करू शकते?",
        "voice": CARTESIA_VOICE_IDS["mr"],
        "name": "Marathi"
    },
    "ta": {
        "language_code": "ta",
        "retell_language": "ta-IN",
        "welcome_message": "வணக்கம்! நான் பல்லி சகாயக், உங்கள் நோய்த்தடுப்பு பராமரிப்பு உதவியாளர். இன்று நான் உங்களுக்கு எப்படி உதவ முடியும்?",
        "voice": CARTESIA_VOICE_IDS["ta"],
        "name": "Tamil"
    },
    "hinglish": {
        "language_code": "hi",
        "retell_language": "hi-IN",
        "welcome_message": "Namaste! Main Palli Sahayak hoon, aapka palliative care assistant. Aaj main aapki kaise help kar sakti hoon?",
        "voice": CARTESIA_VOICE_IDS["hinglish"],
        "name": "Hinglish"
    }
}


@dataclass
class RetellAgentConfig:
    """Configuration for a Retell agent with Custom LLM."""

    agent_name: str
    llm_websocket_url: str
    voice_id: str
    voice_model: str = "sonic-3"
    voice_temperature: float = 0.7
    voice_speed: float = 1.0
    language: str = "hi-IN"
    welcome_message: str = ""
    webhook_url: str = ""

    # Interaction settings
    responsiveness: float = 0.8
    interruption_sensitivity: float = 0.5
    enable_backchannel: bool = False

    # Call settings
    end_call_after_silence_ms: int = 30000
    max_call_duration_ms: int = 3600000

    # STT settings
    stt_mode: str = "accurate"
    vocab_specialization: str = "medical"

    # Post-call analysis
    enable_post_call_analysis: bool = True
    post_call_analysis_model: str = "gpt-4o-mini"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Retell API create-agent call."""
        config = {
            "agent_name": self.agent_name,
            "llm_websocket_url": self.llm_websocket_url,
            "voice_id": self.voice_id,
            "voice_model": self.voice_model,
            "voice_temperature": self.voice_temperature,
            "voice_speed": self.voice_speed,
            "language": self.language,

            # Interaction
            "responsiveness": self.responsiveness,
            "interruption_sensitivity": self.interruption_sensitivity,
            "enable_backchannel": self.enable_backchannel,

            # Call settings
            "end_call_after_silence_ms": self.end_call_after_silence_ms,
            "max_call_duration_ms": self.max_call_duration_ms,

            # STT
            "stt_mode": self.stt_mode,
            "vocab_specialization": self.vocab_specialization,
        }

        if self.welcome_message:
            config["begin_message"] = self.welcome_message

        if self.webhook_url:
            config["webhook_url"] = self.webhook_url

        if self.enable_post_call_analysis:
            config["post_call_analysis_data"] = [
                {"name": "user_concern", "description": "Main health concern raised"},
                {"name": "language_used", "description": "Language spoken by user"},
                {"name": "emotional_state", "description": "User's emotional state"},
                {"name": "urgency_level", "description": "Urgency: low/medium/high/emergency"},
                {"name": "follow_up_needed", "description": "Whether follow-up is recommended"},
            ]
            config["post_call_analysis_model"] = self.post_call_analysis_model

        return config


def get_palli_sahayak_retell_config(
    llm_websocket_url: str,
    webhook_url: str = "",
    language: str = "hi",
    agent_name: str = "Palli Sahayak"
) -> RetellAgentConfig:
    """
    Get Retell agent configuration for Palli Sahayak.

    Args:
        llm_websocket_url: WebSocket URL for custom LLM
                          (e.g., wss://your-domain.com/ws/retell/llm)
        webhook_url: URL for call webhooks
        language: Primary language (hi, en, mr, ta, hinglish)
        agent_name: Name of the agent

    Returns:
        RetellAgentConfig for creating the agent

    Example:
        config = get_palli_sahayak_retell_config(
            llm_websocket_url="wss://my-server.ngrok.io/ws/retell/llm",
            language="hi"
        )
        result = await retell_client.create_agent(config)
    """
    lang_config = RETELL_LANGUAGE_CONFIGS.get(language, RETELL_LANGUAGE_CONFIGS["hi"])
    voice_config = lang_config["voice"]

    return RetellAgentConfig(
        agent_name=agent_name,
        llm_websocket_url=llm_websocket_url,
        voice_id=voice_config["voice_id"],
        voice_model=voice_config["model"],
        language=lang_config["retell_language"],
        welcome_message=lang_config["welcome_message"],
        webhook_url=webhook_url,
        vocab_specialization="medical"
    )


def get_retell_config_from_env(language: str = "hi") -> RetellAgentConfig:
    """
    Get Retell configuration using environment variables.

    Environment variables:
    - PUBLIC_BASE_URL: Base URL of the server
    - RETELL_AGENT_NAME: Agent name (optional)

    Args:
        language: Primary language code

    Returns:
        RetellAgentConfig for creating the agent
    """
    base_url = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")

    # Convert http to wss for WebSocket
    ws_url = base_url.replace("http://", "wss://").replace("https://", "wss://")
    llm_ws_url = f"{ws_url}/ws/retell/llm"
    webhook_url = f"{base_url}/api/retell/webhook"
    agent_name = os.getenv("RETELL_AGENT_NAME", "Palli Sahayak")

    return get_palli_sahayak_retell_config(
        llm_websocket_url=llm_ws_url,
        webhook_url=webhook_url,
        language=language,
        agent_name=agent_name
    )
```

### 4.2 Tests

```bash
python -c "from retell_integration.config import CARTESIA_VOICE_IDS; print(CARTESIA_VOICE_IDS['hi'])"
python -c "from retell_integration.config import get_palli_sahayak_retell_config; c = get_palli_sahayak_retell_config('wss://test.com/ws', language='hi'); print(c.to_dict())"
```

### 4.3 Commit

```
feat(retell): Phase 2 - configuration module with Cartesia Sonic-3 voices

- Add RETELL_SYSTEM_PROMPT for palliative care
- Add CARTESIA_VOICE_IDS for hi, en, mr, ta, hinglish
- Add RETELL_LANGUAGE_CONFIGS with welcome messages
- Add RetellAgentConfig dataclass with medical STT settings
```

---

## 5. Phase 3: API Client

### 5.1 `client.py`

```python
"""
Retell API Client for Palli Sahayak Voice AI Helpline

This module provides the RetellClient class for interacting with the Retell.AI API.
It handles agent creation, call management, and phone number configuration.

Documentation: https://docs.retellai.com/api-references
"""

import os
import logging
import aiohttp
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .config import RETELL_API_BASE, RetellAgentConfig

logger = logging.getLogger(__name__)


@dataclass
class RetellCallResult:
    """Result of a Retell API call."""
    success: bool
    call_id: Optional[str] = None
    agent_id: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class RetellClient:
    """
    Client for Retell.AI API.

    Handles:
    - Agent creation and management
    - Call initiation (outbound)
    - Phone number management
    - LLM configuration

    Usage:
        client = RetellClient()

        # Create agent with custom LLM
        config = get_palli_sahayak_retell_config(...)
        result = await client.create_agent(config)

        # Initiate outbound call
        result = await client.create_phone_call(
            from_number="+1234567890",
            to_number="+919876543210",
            agent_id="agent-123"
        )
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Retell client.

        Args:
            api_key: Retell API key. If not provided, reads from RETELL_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("RETELL_API_KEY")
        if not self.api_key:
            logger.warning("Retell API key not configured - set RETELL_API_KEY env var")

        self.base_url = RETELL_API_BASE
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def is_available(self) -> bool:
        """Check if Retell client is configured with API key."""
        return bool(self.api_key)

    async def create_agent(self, config: RetellAgentConfig) -> RetellCallResult:
        """
        Create a new Retell agent with Custom LLM.

        Args:
            config: RetellAgentConfig with LLM WebSocket URL

        Returns:
            RetellCallResult with agent_id if successful
        """
        if not self.is_available():
            return RetellCallResult(success=False, error="Retell API key not configured")

        try:
            payload = config.to_dict()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/create-agent",
                    headers=self.headers,
                    json=payload
                ) as response:
                    data = await response.json()

                    if response.status in (200, 201):
                        agent_id = data.get("agent_id")
                        logger.info(f"Created Retell agent: {agent_id}")
                        return RetellCallResult(
                            success=True,
                            agent_id=agent_id,
                            data=data
                        )
                    else:
                        error_msg = data.get("message", data.get("error", f"HTTP {response.status}"))
                        logger.error(f"Failed to create Retell agent: {error_msg}")
                        return RetellCallResult(success=False, error=error_msg)

        except aiohttp.ClientError as e:
            logger.error(f"Network error creating Retell agent: {e}")
            return RetellCallResult(success=False, error=f"Network error: {e}")
        except Exception as e:
            logger.error(f"Failed to create Retell agent: {e}")
            return RetellCallResult(success=False, error=str(e))

    async def get_agent(self, agent_id: str) -> RetellCallResult:
        """
        Get agent details.

        Args:
            agent_id: ID of the agent to retrieve

        Returns:
            RetellCallResult with agent data
        """
        if not self.is_available():
            return RetellCallResult(success=False, error="Retell API key not configured")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/get-agent/{agent_id}",
                    headers=self.headers
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return RetellCallResult(
                            success=True,
                            agent_id=agent_id,
                            data=data
                        )
                    else:
                        return RetellCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to get Retell agent: {e}")
            return RetellCallResult(success=False, error=str(e))

    async def update_agent(self, agent_id: str, updates: Dict[str, Any]) -> RetellCallResult:
        """
        Update an existing agent.

        Args:
            agent_id: ID of the agent to update
            updates: Dictionary of fields to update

        Returns:
            RetellCallResult with updated agent data
        """
        if not self.is_available():
            return RetellCallResult(success=False, error="Retell API key not configured")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.patch(
                    f"{self.base_url}/update-agent/{agent_id}",
                    headers=self.headers,
                    json=updates
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        logger.info(f"Updated Retell agent: {agent_id}")
                        return RetellCallResult(
                            success=True,
                            agent_id=agent_id,
                            data=data
                        )
                    else:
                        return RetellCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to update Retell agent: {e}")
            return RetellCallResult(success=False, error=str(e))

    async def delete_agent(self, agent_id: str) -> RetellCallResult:
        """
        Delete an agent.

        Args:
            agent_id: ID of the agent to delete

        Returns:
            RetellCallResult
        """
        if not self.is_available():
            return RetellCallResult(success=False, error="Retell API key not configured")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.base_url}/delete-agent/{agent_id}",
                    headers=self.headers
                ) as response:
                    if response.status in (200, 204):
                        logger.info(f"Deleted Retell agent: {agent_id}")
                        return RetellCallResult(success=True, agent_id=agent_id)
                    else:
                        data = await response.json()
                        return RetellCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to delete Retell agent: {e}")
            return RetellCallResult(success=False, error=str(e))

    async def list_agents(self) -> RetellCallResult:
        """
        List all agents.

        Returns:
            RetellCallResult with list of agents
        """
        if not self.is_available():
            return RetellCallResult(success=False, error="Retell API key not configured")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/list-agents",
                    headers=self.headers
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return RetellCallResult(success=True, data=data)
                    else:
                        return RetellCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to list Retell agents: {e}")
            return RetellCallResult(success=False, error=str(e))

    async def create_phone_call(
        self,
        from_number: str,
        to_number: str,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        retell_llm_dynamic_variables: Optional[Dict[str, Any]] = None
    ) -> RetellCallResult:
        """
        Create an outbound phone call.

        Args:
            from_number: Phone number to call from (E.164 format)
            to_number: Phone number to call (E.164 format)
            agent_id: ID of the agent to use
            metadata: Optional call metadata
            retell_llm_dynamic_variables: Optional dynamic variables for LLM

        Returns:
            RetellCallResult with call_id if successful
        """
        if not self.is_available():
            return RetellCallResult(success=False, error="Retell API key not configured")

        try:
            payload = {
                "from_number": from_number,
                "to_number": to_number,
                "agent_id": agent_id
            }

            if metadata:
                payload["metadata"] = metadata

            if retell_llm_dynamic_variables:
                payload["retell_llm_dynamic_variables"] = retell_llm_dynamic_variables

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/create-phone-call",
                    headers=self.headers,
                    json=payload
                ) as response:
                    data = await response.json()

                    if response.status in (200, 201):
                        call_id = data.get("call_id")
                        logger.info(f"Created Retell call: {call_id} to {to_number}")
                        return RetellCallResult(
                            success=True,
                            call_id=call_id,
                            agent_id=agent_id,
                            data=data
                        )
                    else:
                        error_msg = data.get("message", f"HTTP {response.status}")
                        logger.error(f"Failed to create call: {error_msg}")
                        return RetellCallResult(success=False, error=error_msg)

        except Exception as e:
            logger.error(f"Failed to create Retell call: {e}")
            return RetellCallResult(success=False, error=str(e))

    async def create_web_call(
        self,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RetellCallResult:
        """
        Create a web call (WebRTC).

        Args:
            agent_id: ID of the agent to use
            metadata: Optional call metadata

        Returns:
            RetellCallResult with call_id and access_token
        """
        if not self.is_available():
            return RetellCallResult(success=False, error="Retell API key not configured")

        try:
            payload = {"agent_id": agent_id}

            if metadata:
                payload["metadata"] = metadata

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/create-web-call",
                    headers=self.headers,
                    json=payload
                ) as response:
                    data = await response.json()

                    if response.status in (200, 201):
                        call_id = data.get("call_id")
                        logger.info(f"Created Retell web call: {call_id}")
                        return RetellCallResult(
                            success=True,
                            call_id=call_id,
                            agent_id=agent_id,
                            data=data
                        )
                    else:
                        return RetellCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to create web call: {e}")
            return RetellCallResult(success=False, error=str(e))

    async def get_call(self, call_id: str) -> RetellCallResult:
        """
        Get call details.

        Args:
            call_id: ID of the call

        Returns:
            RetellCallResult with call data
        """
        if not self.is_available():
            return RetellCallResult(success=False, error="Retell API key not configured")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/get-call/{call_id}",
                    headers=self.headers
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return RetellCallResult(
                            success=True,
                            call_id=call_id,
                            data=data
                        )
                    else:
                        return RetellCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to get call: {e}")
            return RetellCallResult(success=False, error=str(e))

    async def list_calls(
        self,
        agent_id: Optional[str] = None,
        limit: int = 20,
        sort_order: str = "descending"
    ) -> RetellCallResult:
        """
        List calls, optionally filtered by agent.

        Args:
            agent_id: Optional agent ID to filter by
            limit: Maximum number of calls to return
            sort_order: "ascending" or "descending" by start time

        Returns:
            RetellCallResult with list of calls
        """
        if not self.is_available():
            return RetellCallResult(success=False, error="Retell API key not configured")

        try:
            payload = {
                "limit": limit,
                "sort_order": sort_order
            }
            if agent_id:
                payload["filter_criteria"] = [
                    {"member": "agent_id", "operator": "eq", "value": agent_id}
                ]

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/list-calls",
                    headers=self.headers,
                    json=payload
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return RetellCallResult(success=True, data=data)
                    else:
                        return RetellCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to list calls: {e}")
            return RetellCallResult(success=False, error=str(e))

    async def health_check(self) -> bool:
        """
        Check if Retell API is accessible.

        Returns:
            True if API is healthy, False otherwise
        """
        if not self.is_available():
            return False

        try:
            result = await self.list_agents()
            return result.success
        except Exception:
            return False
```

### 5.2 Tests

```bash
python -c "from retell_integration.client import RetellClient; c = RetellClient(); print(f'Available: {c.is_available()}')"
```

### 5.3 Commit

```
feat(retell): Phase 3 - API client implementation

- Add RetellClient class with async aiohttp
- Implement agent CRUD: create, get, update, delete, list
- Implement call management: create_phone_call, create_web_call, get_call, list_calls
- Add health_check method
```

---

## 6. Phase 4: Custom LLM WebSocket Server

### 6.1 `custom_llm_server.py`

```python
"""
Custom LLM WebSocket Server for Retell.AI Integration

This module implements the Retell Custom LLM WebSocket protocol,
receiving transcripts and returning LLM responses with RAG integration.

Protocol Reference: https://docs.retellai.com/api-references/llm-websocket

Message Types from Retell:
- ping_pong: Keepalive
- call_details: Call metadata at start
- update_only: Transcript update, no response needed
- response_required: Need LLM response
- reminder_required: Need reminder response (optional)

Response Types to Retell:
- config: Initial configuration
- ping_pong: Keepalive response
- response: LLM response content
- agent_interrupt: Interrupt user
- update_agent: Update agent config mid-call
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class RetellInteractionType(Enum):
    """Interaction types from Retell."""
    PING_PONG = "ping_pong"
    CALL_DETAILS = "call_details"
    UPDATE_ONLY = "update_only"
    RESPONSE_REQUIRED = "response_required"
    REMINDER_REQUIRED = "reminder_required"


class RetellResponseType(Enum):
    """Response types to send to Retell."""
    CONFIG = "config"
    PING_PONG = "ping_pong"
    RESPONSE = "response"
    AGENT_INTERRUPT = "agent_interrupt"
    UPDATE_AGENT = "update_agent"


@dataclass
class RetellSession:
    """Represents an active Retell LLM session."""
    call_id: str
    websocket: WebSocket
    started_at: datetime = field(default_factory=datetime.now)
    language: str = "hi"
    last_transcript: str = ""
    conversation_history: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_id: str = ""
    from_number: str = ""
    to_number: str = ""

    def get_conversation_context(self, max_turns: int = 5) -> str:
        """Get recent conversation context as string."""
        recent = self.conversation_history[-max_turns * 2:] if self.conversation_history else []
        context_parts = []
        for entry in recent:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            if content:
                prefix = "User" if role == "user" else "Assistant"
                context_parts.append(f"{prefix}: {content}")
        return "\n".join(context_parts)


class RetellCustomLLMHandler:
    """
    Handler for Retell Custom LLM WebSocket protocol.

    Implements:
    - Receiving transcripts from Retell
    - Querying RAG pipeline for context
    - Streaming responses back to Retell

    Protocol Flow:
    1. Retell connects to: wss://your-server/ws/retell/llm/{call_id}
    2. Server sends: config message
    3. Retell sends: call_details with metadata
    4. Retell sends: response_required with transcript
    5. Server sends: response with LLM output
    6. Repeat 4-5 until call ends

    Usage:
        handler = RetellCustomLLMHandler(rag_pipeline=my_rag)

        @app.websocket("/ws/retell/llm/{call_id}")
        async def retell_ws(websocket: WebSocket, call_id: str):
            await handler.handle_websocket(websocket, call_id)
    """

    def __init__(
        self,
        rag_pipeline=None,
        query_classifier=None,
        response_timeout: float = 30.0,
        max_response_tokens: int = 150
    ):
        """
        Initialize the handler.

        Args:
            rag_pipeline: RAG pipeline for querying knowledge base
            query_classifier: Optional QueryClassifier for smart routing
            response_timeout: Timeout for RAG queries in seconds
            max_response_tokens: Max tokens for response (voice should be concise)
        """
        self.rag_pipeline = rag_pipeline
        self.query_classifier = query_classifier
        self.response_timeout = response_timeout
        self.max_response_tokens = max_response_tokens
        self.active_sessions: Dict[str, RetellSession] = {}

    async def handle_websocket(
        self,
        websocket: WebSocket,
        call_id: str
    ) -> None:
        """
        Main WebSocket handler for Retell Custom LLM protocol.

        Args:
            websocket: FastAPI WebSocket connection
            call_id: Unique call identifier from URL path
        """
        await websocket.accept()

        # Create session
        session = RetellSession(call_id=call_id, websocket=websocket)
        self.active_sessions[call_id] = session

        logger.info(f"Retell LLM WebSocket connected: {call_id}")

        # Send initial config
        await self._send_config(websocket)

        try:
            while True:
                # Receive message from Retell
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle different interaction types
                interaction_type = message.get("interaction_type", "")

                if interaction_type == "ping_pong":
                    await self._handle_ping_pong(websocket, message)

                elif interaction_type == "call_details":
                    await self._handle_call_details(session, message)

                elif interaction_type == "update_only":
                    await self._handle_update_only(session, message)

                elif interaction_type in ("response_required", "reminder_required"):
                    await self._handle_response_required(session, message)

                else:
                    logger.warning(f"Unknown Retell interaction type: {interaction_type}")

        except WebSocketDisconnect:
            logger.info(f"Retell LLM WebSocket disconnected: {call_id}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from Retell: {e}")

        except Exception as e:
            logger.error(f"Retell LLM WebSocket error for {call_id}: {e}")

        finally:
            # Cleanup session
            if call_id in self.active_sessions:
                del self.active_sessions[call_id]
                logger.info(f"Cleaned up Retell session: {call_id}")

    async def _send_config(self, websocket: WebSocket) -> None:
        """Send initial configuration to Retell."""
        config = {
            "response_type": "config",
            "config": {
                "auto_reconnect": True,
                "call_details": True
            }
        }
        await websocket.send_json(config)
        logger.debug("Sent config to Retell")

    async def _handle_ping_pong(
        self,
        websocket: WebSocket,
        message: Dict[str, Any]
    ) -> None:
        """Handle ping_pong keepalive."""
        response = {
            "response_type": "ping_pong",
            "timestamp": message.get("timestamp", 0)
        }
        await websocket.send_json(response)

    async def _handle_call_details(
        self,
        session: RetellSession,
        message: Dict[str, Any]
    ) -> None:
        """Handle call_details event with metadata."""
        call = message.get("call", {})

        session.agent_id = call.get("agent_id", "")
        session.from_number = call.get("from_number", "")
        session.to_number = call.get("to_number", "")
        session.metadata = call.get("metadata", {})

        # Detect language from metadata or agent config
        language = session.metadata.get("language", "")
        if language:
            # Map Retell language codes to our short codes
            lang_map = {"hi-IN": "hi", "en-IN": "en", "mr-IN": "mr", "ta-IN": "ta"}
            session.language = lang_map.get(language, language[:2] if len(language) >= 2 else "hi")

        logger.info(
            f"Retell call details - ID: {session.call_id}, "
            f"Agent: {session.agent_id}, "
            f"From: {session.from_number}, "
            f"Language: {session.language}"
        )

    async def _handle_update_only(
        self,
        session: RetellSession,
        message: Dict[str, Any]
    ) -> None:
        """Handle update_only event (transcript update, no response needed)."""
        transcript = message.get("transcript", [])
        if transcript:
            session.last_transcript = self._extract_user_text(transcript)
            session.conversation_history = transcript

    async def _handle_response_required(
        self,
        session: RetellSession,
        message: Dict[str, Any]
    ) -> None:
        """
        Handle response_required event - main LLM response flow.

        This is where RAG integration happens:
        1. Extract user query from transcript
        2. Check for out-of-scope queries
        3. Query RAG pipeline
        4. Send response back to Retell
        """
        response_id = message.get("response_id", 0)
        transcript = message.get("transcript", [])

        # Extract user's latest query
        user_query = self._extract_user_text(transcript)
        session.last_transcript = user_query
        session.conversation_history = transcript

        logger.info("=" * 60)
        logger.info("RETELL CUSTOM LLM - RAG QUERY")
        logger.info("=" * 60)
        logger.info(f"Call ID: {session.call_id}")
        logger.info(f"Query: {user_query[:100]}{'...' if len(user_query) > 100 else ''}")
        logger.info(f"Language: {session.language}")

        # Handle empty or very short queries
        if not user_query or len(user_query.strip()) < 3:
            await self._send_response(
                session.websocket,
                response_id,
                self._get_clarification_message(session.language),
                content_complete=True
            )
            return

        # Check for out-of-scope queries
        if self.query_classifier:
            try:
                is_out_of_scope, keyword = self.query_classifier.is_out_of_scope(user_query)
                if is_out_of_scope:
                    decline_msg = self.query_classifier.get_decline_message(
                        f"{session.language}-IN"
                    )
                    await self._send_response(
                        session.websocket,
                        response_id,
                        decline_msg,
                        content_complete=True
                    )
                    logger.info(f"OUT OF SCOPE query declined (keyword: {keyword})")
                    logger.info("=" * 60)
                    return
            except Exception as e:
                logger.warning(f"Query classifier error: {e}")

        # Query RAG pipeline
        if self.rag_pipeline:
            try:
                # Get conversation context for better responses
                context = session.get_conversation_context(max_turns=3)

                result = await asyncio.wait_for(
                    self.rag_pipeline.query(
                        question=user_query,
                        conversation_id=session.call_id,
                        user_id=session.call_id,
                        source_language=session.language,
                        top_k=3,
                        conversation_context=context
                    ),
                    timeout=self.response_timeout
                )

                if result.get("status") == "success":
                    answer = result.get("answer", "")
                    sources = result.get("sources", [])

                    # Truncate for voice (keep it concise)
                    if len(answer) > 500:
                        answer = answer[:500] + "..."

                    # Log success
                    source_names = ", ".join([
                        s.get("filename", "Unknown")[:30]
                        for s in sources[:3]
                    ])
                    logger.info(f"RAG SUCCESS - Sources: {source_names}")
                    logger.info(f"Answer preview: {answer[:100]}...")

                    # Send response
                    await self._send_response(
                        session.websocket,
                        response_id,
                        answer,
                        content_complete=True
                    )
                else:
                    # RAG query failed - graceful fallback
                    error = result.get("error", "Unknown error")
                    logger.warning(f"RAG query failed: {error}")
                    await self._send_response(
                        session.websocket,
                        response_id,
                        self._get_fallback_message(session.language),
                        content_complete=True
                    )

            except asyncio.TimeoutError:
                logger.error(f"RAG query timeout for call {session.call_id}")
                await self._send_response(
                    session.websocket,
                    response_id,
                    self._get_timeout_message(session.language),
                    content_complete=True
                )

            except Exception as e:
                logger.error(f"RAG query error: {e}")
                await self._send_response(
                    session.websocket,
                    response_id,
                    self._get_error_message(session.language),
                    content_complete=True
                )
        else:
            # No RAG pipeline - basic response
            await self._send_response(
                session.websocket,
                response_id,
                self._get_no_rag_message(session.language),
                content_complete=True
            )

        logger.info("=" * 60)

    def _extract_user_text(self, transcript: list) -> str:
        """
        Extract the latest user utterance from transcript.

        Args:
            transcript: List of transcript entries with role and content

        Returns:
            User's latest message text
        """
        user_utterances = []
        for entry in reversed(transcript):
            role = entry.get("role", "")
            content = entry.get("content", "")

            if role == "user" and content:
                user_utterances.insert(0, content)
            elif role == "agent" and user_utterances:
                # Stop when we hit agent response
                break

        return " ".join(user_utterances).strip()

    async def _send_response(
        self,
        websocket: WebSocket,
        response_id: int,
        content: str,
        content_complete: bool = True,
        end_call: bool = False
    ) -> None:
        """
        Send response to Retell.

        Args:
            websocket: WebSocket connection
            response_id: ID from the request
            content: Response text
            content_complete: Whether this is the final chunk
            end_call: Whether to end the call after this response
        """
        response = {
            "response_type": "response",
            "response_id": response_id,
            "content": content,
            "content_complete": content_complete
        }

        if end_call:
            response["end_call"] = True

        await websocket.send_json(response)
        logger.debug(f"Sent response (id={response_id}, complete={content_complete}, len={len(content)})")

    def _get_clarification_message(self, language: str) -> str:
        """Get clarification request in appropriate language."""
        messages = {
            "hi": "क्षमा करें, मैं समझ नहीं पाया। कृपया दोबारा बताएं।",
            "en": "I'm sorry, I didn't catch that. Could you please repeat?",
            "mr": "माफ करा, मला समजले नाही. कृपया पुन्हा सांगा।",
            "ta": "மன்னிக்கவும், நான் புரிந்து கொள்ளவில்லை. தயவுசெய்து மீண்டும் சொல்லுங்கள்."
        }
        return messages.get(language, messages["en"])

    def _get_fallback_message(self, language: str) -> str:
        """Get fallback message when RAG fails."""
        messages = {
            "hi": "मुझे इस विषय पर जानकारी खोजने में कठिनाई हो रही है। कृपया अपने डॉक्टर से परामर्श करें।",
            "en": "I'm having trouble finding information on this. Please consult your healthcare provider.",
            "mr": "मला या विषयावर माहिती शोधण्यात अडचण येत आहे. कृपया आपल्या डॉक्टरांचा सल्ला घ्या।",
            "ta": "இதற்கான தகவலைக் கண்டறிவதில் சிரமம். உங்கள் மருத்துவரை அணுகவும்."
        }
        return messages.get(language, messages["en"])

    def _get_timeout_message(self, language: str) -> str:
        """Get timeout message."""
        messages = {
            "hi": "जवाब देने में समय लग रहा है। कृपया सरल प्रश्न पूछें।",
            "en": "I'm taking too long to respond. Could you ask a simpler question?",
            "mr": "उत्तर देण्यास वेळ लागतो आहे. कृपया सोपा प्रश्न विचारा।",
            "ta": "பதில் அளிக்க நேரம் எடுக்கிறது. எளிய கேள்வி கேளுங்கள்."
        }
        return messages.get(language, messages["en"])

    def _get_error_message(self, language: str) -> str:
        """Get error message."""
        messages = {
            "hi": "कुछ गलत हो गया। कृपया फिर से प्रयास करें।",
            "en": "Something went wrong. Please try again.",
            "mr": "काहीतरी चूक झाली. कृपया पुन्हा प्रयत्न करा।",
            "ta": "ஏதோ தவறு நடந்தது. மீண்டும் முயற்சிக்கவும்."
        }
        return messages.get(language, messages["en"])

    def _get_no_rag_message(self, language: str) -> str:
        """Get message when RAG pipeline is not available."""
        messages = {
            "hi": "मैं पल्ली सहायक हूं। मेरा ज्ञान आधार अभी तैयार हो रहा है। कृपया कुछ समय बाद कॉल करें।",
            "en": "I am Palli Sahayak. My knowledge base is being set up. Please call back shortly.",
            "mr": "मी पल्ली सहायक आहे. माझा ज्ञान आधार सध्या सेट होत आहे. कृपया थोड्या वेळाने कॉल करा।",
            "ta": "நான் பல்லி சகாயக். என் அறிவுத் தளம் அமைக்கப்படுகிறது. சிறிது நேரம் கழித்து அழைக்கவும்."
        }
        return messages.get(language, messages["en"])

    def get_session(self, call_id: str) -> Optional[RetellSession]:
        """Get an active session by call ID."""
        return self.active_sessions.get(call_id)

    def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        return len(self.active_sessions)

    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        return {
            "active_sessions": self.get_active_session_count(),
            "rag_available": self.rag_pipeline is not None,
            "query_classifier_available": self.query_classifier is not None,
            "response_timeout_seconds": self.response_timeout
        }
```

### 6.2 Tests

```bash
python -c "from retell_integration.custom_llm_server import RetellCustomLLMHandler, RetellInteractionType; print(RetellInteractionType.RESPONSE_REQUIRED.value)"
python -c "from retell_integration.custom_llm_server import RetellSession; s = RetellSession(call_id='test', websocket=None); print(s)"
```

### 6.3 Commit

```
feat(retell): Phase 4 - Custom LLM WebSocket server

- Add RetellCustomLLMHandler with full protocol implementation
- Add RetellSession dataclass for session state
- Implement RAG integration with timeout handling
- Add multi-language error/fallback messages
- Add conversation context extraction
```

---

## 7. Phase 5: Webhook Handler

### 7.1 `webhooks.py`

```python
"""
Webhook Handler for Retell Call Events

This module handles webhook events from Retell.AI, including:
- call_started: Call initiated
- call_ended: Call completed with transcript
- call_analyzed: Post-call analysis ready

Documentation: https://docs.retellai.com/api-references/webhooks
"""

import logging
from typing import Dict, Any, Optional, Callable, Awaitable, List
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetellCallRecord:
    """Record of a Retell call."""
    call_id: str
    agent_id: str
    call_type: str  # "inbound" or "outbound" or "web"
    from_number: str
    to_number: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_ms: int = 0
    transcript: str = ""
    transcript_object: List[Dict] = field(default_factory=list)
    call_summary: str = ""
    user_sentiment: str = ""
    call_successful: bool = True
    disconnection_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    analysis_data: Dict[str, Any] = field(default_factory=dict)
    status: str = "in_progress"

    @property
    def duration_seconds(self) -> int:
        """Get duration in seconds."""
        return self.duration_ms // 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "call_id": self.call_id,
            "agent_id": self.agent_id,
            "call_type": self.call_type,
            "from_number": self.from_number,
            "to_number": self.to_number,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds,
            "transcript": self.transcript,
            "call_summary": self.call_summary,
            "user_sentiment": self.user_sentiment,
            "call_successful": self.call_successful,
            "disconnection_reason": self.disconnection_reason,
            "metadata": self.metadata,
            "analysis_data": self.analysis_data,
            "status": self.status
        }


class RetellWebhookHandler:
    """
    Handler for Retell webhook events.

    Processes:
    - call_started: New call initiated
    - call_ended: Call completed with transcript
    - call_analyzed: Post-call analysis available

    Usage:
        handler = RetellWebhookHandler()

        # Register custom handler for specific events
        handler.register_handler("call_ended", my_callback)

        # Process webhook event
        result = await handler.handle_event(event_data)

        # Get statistics
        stats = handler.get_call_stats()
    """

    def __init__(self, max_completed_calls: int = 1000):
        """
        Initialize webhook handler.

        Args:
            max_completed_calls: Maximum completed calls to keep in memory
        """
        self.active_calls: Dict[str, RetellCallRecord] = {}
        self.completed_calls: Dict[str, RetellCallRecord] = {}
        self.event_handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[None]]] = {}
        self.max_completed_calls = max_completed_calls

    def register_handler(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """
        Register a custom handler for an event type.

        Args:
            event_type: Event type (call_started, call_ended, call_analyzed)
            handler: Async callback function
        """
        self.event_handlers[event_type] = handler
        logger.info(f"Registered handler for Retell event: {event_type}")

    async def handle_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming webhook event from Retell.

        Args:
            event_data: Webhook payload from Retell

        Returns:
            Processing result dictionary
        """
        event_type = event_data.get("event", "unknown")
        call = event_data.get("call", {})
        call_id = call.get("call_id", "unknown")

        logger.info(f"Processing Retell webhook: {event_type} for call {call_id}")

        # Call custom handler if registered
        if event_type in self.event_handlers:
            try:
                await self.event_handlers[event_type](event_data)
            except Exception as e:
                logger.error(f"Custom handler error for {event_type}: {e}")

        # Process standard events
        if event_type == "call_started":
            return await self._handle_call_started(call)

        elif event_type == "call_ended":
            return await self._handle_call_ended(call)

        elif event_type == "call_analyzed":
            return await self._handle_call_analyzed(call)

        else:
            logger.warning(f"Unknown Retell event type: {event_type}")
            return {
                "status": "ignored",
                "reason": "unknown_event",
                "event_type": event_type
            }

    async def _handle_call_started(self, call: Dict[str, Any]) -> Dict[str, Any]:
        """Handle call_started event."""
        call_id = call.get("call_id")

        record = RetellCallRecord(
            call_id=call_id,
            agent_id=call.get("agent_id", ""),
            call_type=call.get("call_type", "inbound"),
            from_number=call.get("from_number", ""),
            to_number=call.get("to_number", ""),
            started_at=datetime.now(),
            status="in_progress",
            metadata=call.get("metadata", {})
        )

        self.active_calls[call_id] = record

        logger.info(
            f"Retell call started: {call_id} "
            f"({record.call_type}) from {record.from_number}"
        )

        return {
            "status": "recorded",
            "call_id": call_id,
            "event": "call_started"
        }

    async def _handle_call_ended(self, call: Dict[str, Any]) -> Dict[str, Any]:
        """Handle call_ended event."""
        call_id = call.get("call_id")

        # Get existing record or create new one
        if call_id in self.active_calls:
            record = self.active_calls.pop(call_id)
        else:
            # Create new record if call_started wasn't received
            start_ts = call.get("start_timestamp")
            started_at = datetime.now()
            if start_ts:
                try:
                    started_at = datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass

            record = RetellCallRecord(
                call_id=call_id,
                agent_id=call.get("agent_id", ""),
                call_type=call.get("call_type", "inbound"),
                from_number=call.get("from_number", ""),
                to_number=call.get("to_number", ""),
                started_at=started_at
            )

        # Update with end data
        record.ended_at = datetime.now()
        record.duration_ms = call.get("duration_ms", 0)
        record.transcript = call.get("transcript", "")
        record.transcript_object = call.get("transcript_object", [])
        record.disconnection_reason = call.get("disconnection_reason", "")
        record.call_successful = call.get("call_successful", True)
        record.status = "completed"

        # Store completed call (with limit)
        self.completed_calls[call_id] = record
        self._enforce_completed_limit()

        logger.info(
            f"Retell call ended: {call_id}, "
            f"duration: {record.duration_seconds}s, "
            f"successful: {record.call_successful}, "
            f"reason: {record.disconnection_reason}"
        )

        return {
            "status": "recorded",
            "call_id": call_id,
            "event": "call_ended",
            "duration_seconds": record.duration_seconds
        }

    async def _handle_call_analyzed(self, call: Dict[str, Any]) -> Dict[str, Any]:
        """Handle call_analyzed event (post-call analysis)."""
        call_id = call.get("call_id")

        if call_id in self.completed_calls:
            record = self.completed_calls[call_id]
            record.call_summary = call.get("call_analysis", {}).get("call_summary", "")
            record.user_sentiment = call.get("call_analysis", {}).get("user_sentiment", "")
            record.analysis_data = call.get("call_analysis", {})

            logger.info(
                f"Retell call analyzed: {call_id}, "
                f"sentiment: {record.user_sentiment}"
            )

            # Check for follow-up needed
            if record.analysis_data.get("follow_up_needed"):
                logger.info(f"Follow-up recommended for call {call_id}")

            # Check for high urgency
            urgency = record.analysis_data.get("urgency_level", "low")
            if urgency in ("high", "emergency"):
                logger.warning(
                    f"HIGH URGENCY call {call_id}: "
                    f"{record.analysis_data.get('user_concern', 'Unknown concern')}"
                )

        return {
            "status": "recorded",
            "call_id": call_id,
            "event": "call_analyzed"
        }

    def _enforce_completed_limit(self) -> None:
        """Remove oldest completed calls if limit exceeded."""
        if len(self.completed_calls) > self.max_completed_calls:
            # Sort by ended_at and remove oldest
            sorted_calls = sorted(
                self.completed_calls.items(),
                key=lambda x: x[1].ended_at or x[1].started_at
            )
            # Remove oldest 10%
            remove_count = len(self.completed_calls) - int(self.max_completed_calls * 0.9)
            for call_id, _ in sorted_calls[:remove_count]:
                del self.completed_calls[call_id]
            logger.info(f"Cleaned up {remove_count} old call records")

    def get_active_call(self, call_id: str) -> Optional[RetellCallRecord]:
        """Get an active call record."""
        return self.active_calls.get(call_id)

    def get_completed_call(self, call_id: str) -> Optional[RetellCallRecord]:
        """Get a completed call record."""
        return self.completed_calls.get(call_id)

    def get_call(self, call_id: str) -> Optional[RetellCallRecord]:
        """Get a call record (active or completed)."""
        return self.active_calls.get(call_id) or self.completed_calls.get(call_id)

    def get_call_stats(self) -> Dict[str, Any]:
        """Get call statistics."""
        total_completed = len(self.completed_calls)
        total_duration = sum(c.duration_ms for c in self.completed_calls.values())

        # Count successful vs failed
        successful = sum(1 for c in self.completed_calls.values() if c.call_successful)

        # Sentiment distribution
        sentiments: Dict[str, int] = {}
        for call in self.completed_calls.values():
            sentiment = call.user_sentiment or "unknown"
            sentiments[sentiment] = sentiments.get(sentiment, 0) + 1

        # Call type distribution
        call_types: Dict[str, int] = {}
        for call in self.completed_calls.values():
            call_type = call.call_type
            call_types[call_type] = call_types.get(call_type, 0) + 1

        # Urgency distribution
        urgency_levels: Dict[str, int] = {}
        for call in self.completed_calls.values():
            urgency = call.analysis_data.get("urgency_level", "unknown")
            urgency_levels[urgency] = urgency_levels.get(urgency, 0) + 1

        return {
            "active_calls": len(self.active_calls),
            "completed_calls": total_completed,
            "successful_calls": successful,
            "failed_calls": total_completed - successful,
            "total_duration_ms": total_duration,
            "total_duration_seconds": total_duration // 1000,
            "average_duration_seconds": (total_duration // 1000) // total_completed if total_completed > 0 else 0,
            "sentiment_distribution": sentiments,
            "call_type_distribution": call_types,
            "urgency_distribution": urgency_levels
        }

    def get_recent_calls(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent completed calls."""
        calls = list(self.completed_calls.values())
        calls.sort(key=lambda x: x.ended_at or x.started_at, reverse=True)
        return [c.to_dict() for c in calls[:limit]]

    def get_high_urgency_calls(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent high urgency calls."""
        high_urgency = [
            c for c in self.completed_calls.values()
            if c.analysis_data.get("urgency_level") in ("high", "emergency")
        ]
        high_urgency.sort(key=lambda x: x.ended_at or x.started_at, reverse=True)
        return [c.to_dict() for c in high_urgency[:limit]]
```

### 7.2 Tests

```bash
python -c "from retell_integration.webhooks import RetellWebhookHandler; h = RetellWebhookHandler(); print(h.get_call_stats())"
```

### 7.3 Commit

```
feat(retell): Phase 5 - webhook handler implementation

- Add RetellWebhookHandler for call events
- Add RetellCallRecord dataclass with analysis data
- Implement call_started, call_ended, call_analyzed handlers
- Add statistics and recent calls retrieval
- Add high urgency call tracking
```

---

## 8. Phase 6: Vobiz.ai Telephony Configuration

### 8.1 `vobiz_config.py`

```python
"""
Vobiz.ai Telephony Configuration for Indian PSTN Integration

Vobiz.ai provides Indian DID numbers and SIP trunking for Retell.AI integration,
allowing callers to reach the Palli Sahayak helpline via regular phone (+91).

No internet required for callers - they use standard cellular/landline phones.

Reference: https://vobiz.ai/
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VobizConfig:
    """
    Configuration for Vobiz.ai Indian telephony integration.

    Vobiz.ai provides:
    - Indian DID numbers (+91 prefixed)
    - SIP trunking to connect with Retell.AI
    - Low latency for real-time voice
    - Compliance with Indian telecom regulations (TRAI)

    Flow:
    1. Caller dials +91 Vobiz number from any phone
    2. Vobiz receives call and routes via SIP
    3. Retell receives call and connects to agent
    4. Agent uses Custom LLM WebSocket for responses
    """

    # Vobiz.ai API credentials
    api_key: str = ""
    api_secret: str = ""

    # Indian DID number for inbound calls
    did_number: str = ""  # Format: +919876543210

    # SIP configuration for Retell integration
    sip_domain: str = ""
    sip_username: str = ""
    sip_password: str = ""
    sip_port: int = 5060

    # Retell SIP trunk settings (obtained after Retell setup)
    retell_sip_trunk_id: str = ""
    retell_inbound_number_id: str = ""

    # Advanced settings
    codec: str = "PCMU"  # G.711 u-law, best for telephony
    dtmf_mode: str = "rfc2833"

    def __post_init__(self):
        """Load from environment if not provided."""
        if not self.api_key:
            self.api_key = os.getenv("VOBIZ_API_KEY", "")
        if not self.api_secret:
            self.api_secret = os.getenv("VOBIZ_API_SECRET", "")
        if not self.did_number:
            self.did_number = os.getenv("VOBIZ_DID_NUMBER", "")
        if not self.sip_domain:
            self.sip_domain = os.getenv("VOBIZ_SIP_DOMAIN", "")
        if not self.sip_username:
            self.sip_username = os.getenv("VOBIZ_SIP_USERNAME", "")
        if not self.sip_password:
            self.sip_password = os.getenv("VOBIZ_SIP_PASSWORD", "")
        if not self.retell_sip_trunk_id:
            self.retell_sip_trunk_id = os.getenv("RETELL_SIP_TRUNK_ID", "")
        if not self.retell_inbound_number_id:
            self.retell_inbound_number_id = os.getenv("RETELL_INBOUND_NUMBER_ID", "")

    def is_configured(self) -> bool:
        """Check if Vobiz is minimally configured."""
        return bool(self.did_number)

    def is_fully_configured(self) -> bool:
        """Check if Vobiz is fully configured with SIP."""
        return bool(
            self.api_key and
            self.did_number and
            self.sip_domain and
            self.sip_username and
            self.sip_password
        )

    def get_sip_uri(self) -> str:
        """Get SIP URI for Retell integration."""
        if self.sip_domain and self.sip_username:
            return f"sip:{self.sip_username}@{self.sip_domain}:{self.sip_port}"
        return ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding secrets)."""
        return {
            "did_number": self.did_number,
            "sip_domain": self.sip_domain,
            "sip_port": self.sip_port,
            "configured": self.is_configured(),
            "fully_configured": self.is_fully_configured(),
            "has_sip_trunk": bool(self.retell_sip_trunk_id),
            "codec": self.codec
        }

    def get_retell_sip_trunk_config(self) -> Dict[str, Any]:
        """
        Get configuration for Retell SIP trunk setup.

        This is used when configuring Retell to receive calls
        from Vobiz.ai via SIP.

        Reference: https://docs.retellai.com/deploy/custom-telephony

        Returns:
            Dictionary for Retell create-sip-trunk API
        """
        config = {
            "inbound_sip_trunk_settings": {
                "termination_uri": self.get_sip_uri()
            }
        }

        if self.sip_username and self.sip_password:
            config["inbound_sip_trunk_settings"]["authentication"] = {
                "type": "credentials",
                "username": self.sip_username,
                "password": self.sip_password
            }

        return config

    def get_retell_phone_number_config(self, agent_id: str) -> Dict[str, Any]:
        """
        Get configuration for importing Vobiz number to Retell.

        Args:
            agent_id: Retell agent ID to assign to this number

        Returns:
            Dictionary for Retell import-phone-number API
        """
        return {
            "phone_number": self.did_number,
            "inbound_agent_id": agent_id,
            "sip_trunk_id": self.retell_sip_trunk_id
        }


def get_vobiz_config() -> VobizConfig:
    """
    Get Vobiz configuration from environment.

    Environment variables:
    - VOBIZ_API_KEY: Vobiz.ai API key
    - VOBIZ_API_SECRET: Vobiz.ai API secret
    - VOBIZ_DID_NUMBER: Indian DID number (+91...)
    - VOBIZ_SIP_DOMAIN: SIP server domain
    - VOBIZ_SIP_USERNAME: SIP authentication username
    - VOBIZ_SIP_PASSWORD: SIP authentication password
    - RETELL_SIP_TRUNK_ID: Retell SIP trunk ID (after setup)
    - RETELL_INBOUND_NUMBER_ID: Retell inbound number ID (after import)

    Returns:
        VobizConfig instance
    """
    return VobizConfig()


# Instructions for setting up Vobiz.ai with Retell
VOBIZ_SETUP_INSTRUCTIONS = """
# Vobiz.ai + Retell.AI Integration Setup Guide

## Overview

This guide explains how to connect a Vobiz.ai Indian phone number (+91)
to Retell.AI for the Palli Sahayak palliative care voice helpline.

## Prerequisites

1. Vobiz.ai account with API access
2. Retell.AI account with Custom Telephony enabled
3. Server with Custom LLM WebSocket endpoint deployed

## Step 1: Get Indian DID from Vobiz.ai

1. Log into Vobiz.ai dashboard (https://dashboard.vobiz.ai)
2. Navigate to Numbers > Buy Number
3. Select India (+91) and choose your preferred city/region
4. Purchase the DID number (e.g., +919876543210)
5. Note down:
   - DID Number
   - SIP Domain (e.g., sip.vobiz.ai)
   - SIP Username
   - SIP Password

## Step 2: Deploy Custom LLM Server

1. Deploy your server with the WebSocket endpoint:
   - wss://your-domain.com/ws/retell/llm

2. Ensure the server is accessible from the internet
   - Use ngrok for development: `ngrok http 8000`
   - Use proper domain for production

3. Test the endpoint is reachable

## Step 3: Create Retell Agent with Custom LLM

```python
from retell_integration import RetellClient, get_palli_sahayak_retell_config

client = RetellClient()
config = get_palli_sahayak_retell_config(
    llm_websocket_url="wss://your-domain.com/ws/retell/llm",
    webhook_url="https://your-domain.com/api/retell/webhook",
    language="hi"
)
result = await client.create_agent(config)
agent_id = result.agent_id
print(f"Created agent: {agent_id}")
```

## Step 4: Create SIP Trunk in Retell

1. Go to Retell Dashboard > Settings > Custom Telephony
2. Click "Add SIP Trunk"
3. Configure:
   - Name: "Vobiz India"
   - Termination URI: sip:username@sip.vobiz.ai:5060
   - Authentication: Username/Password from Vobiz
4. Save and note the SIP Trunk ID

Alternatively, use the API:
```python
# Create SIP trunk via Retell API
trunk_config = vobiz_config.get_retell_sip_trunk_config()
# POST to /create-sip-trunk
```

## Step 5: Import Phone Number to Retell

1. In Retell Dashboard > Phone Numbers
2. Click "Import Number"
3. Enter your Vobiz +91 number
4. Select the SIP trunk created in Step 4
5. Assign the agent created in Step 3

## Step 6: Configure Vobiz Routing

1. In Vobiz Dashboard > Numbers > Your DID
2. Configure inbound routing:
   - Destination Type: SIP
   - SIP URI: (Retell's SIP endpoint - from SIP trunk details)
3. Save configuration

## Step 7: Set Environment Variables

Add to your .env file:

```bash
# Vobiz.ai Configuration
VOBIZ_API_KEY=your_vobiz_api_key
VOBIZ_API_SECRET=your_vobiz_secret
VOBIZ_DID_NUMBER=+919876543210
VOBIZ_SIP_DOMAIN=sip.vobiz.ai
VOBIZ_SIP_USERNAME=your_sip_username
VOBIZ_SIP_PASSWORD=your_sip_password

# Retell Configuration (after setup)
RETELL_API_KEY=your_retell_api_key
RETELL_AGENT_ID=agent_id_from_step_3
RETELL_SIP_TRUNK_ID=trunk_id_from_step_4
RETELL_INBOUND_NUMBER_ID=number_id_from_step_5
```

## Step 8: Test the Integration

1. Start your server:
   ```bash
   python simple_rag_server.py -p r
   ```

2. Call the +91 number from any phone:
   ```
   +919876543210
   ```

3. Verify:
   - Call connects to Retell
   - Welcome message plays
   - Your responses are processed via Custom LLM
   - RAG queries return knowledge base content

## Troubleshooting

### Call Not Connecting
- Check Vobiz dashboard for call logs
- Verify SIP trunk configuration in Retell
- Ensure SIP credentials are correct

### No Audio
- Check codec compatibility (G.711 recommended)
- Verify network firewall allows SIP/RTP traffic

### Custom LLM Not Responding
- Check WebSocket endpoint is accessible
- Review server logs for connection attempts
- Test with: `websocat wss://your-domain.com/ws/retell/llm/test`

### High Latency
- Use a server closer to India
- Consider Vobiz PoP locations
- Check network route to Retell

## Support

- Vobiz.ai: support@vobiz.ai
- Retell.AI: support@retellai.com
- Palli Sahayak: [Your support contact]
"""
```

### 8.2 Tests

```bash
python -c "from retell_integration.vobiz_config import VobizConfig, get_vobiz_config; c = get_vobiz_config(); print(c.to_dict())"
```

### 8.3 Commit

```
feat(retell): Phase 6 - Vobiz.ai telephony configuration

- Add VobizConfig dataclass for Indian PSTN
- Add SIP trunk configuration helpers
- Add Retell phone number import config
- Include comprehensive setup instructions
```

---

## 9. Phase 7: VoiceRouter Extension

### 9.1 Changes to `voice_router.py`

Add the following changes to the existing `voice_router.py`:

```python
# 1. Update VoiceProvider enum (around line 23-27)
class VoiceProvider(Enum):
    """Available voice providers."""
    BOLNA = "bolna"
    GEMINI_LIVE = "gemini_live"
    RETELL = "retell"  # NEW
    FALLBACK_PIPELINE = "fallback_pipeline"


# 2. Add Retell initialization in __init__ (after Gemini initialization ~line 134)
# Initialize Retell client
self.retell_client = None
self.retell_available = False
try:
    from retell_integration import RetellClient
    self.retell_client = RetellClient()
    self.retell_available = self.retell_client.is_available()
    if self.retell_available:
        logger.info("Retell client initialized")
except ImportError:
    logger.warning("Retell integration not available")
except Exception as e:
    logger.warning(f"Failed to initialize Retell client: {e}")


# 3. Update get_available_providers() (after line 151)
if self.retell_available:
    providers.append(VoiceProvider.RETELL)


# 4. Update select_provider() force_provider handling (~line 172-178)
elif force_provider == VoiceProvider.RETELL and self.retell_available:
    return VoiceProvider.RETELL


# 5. Add Retell case in route_voice_request() (~line 237)
elif provider == VoiceProvider.RETELL:
    return await self._handle_retell_request(
        phone_number=phone_number,
        user_id=user_id,
        language=language,
        **kwargs
    )


# 6. Add new method _handle_retell_request()
async def _handle_retell_request(
    self,
    phone_number: Optional[str] = None,
    user_id: Optional[str] = None,
    language: str = "hi",
    **kwargs
) -> VoiceResponse:
    """Handle request via Retell."""
    if not self.retell_available:
        raise RuntimeError("Retell client not available")

    agent_id = os.getenv("RETELL_AGENT_ID")
    if not agent_id:
        raise RuntimeError("RETELL_AGENT_ID not configured")

    session_id = f"retell_{user_id or 'anonymous'}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Create session record
    session = VoiceSession(
        session_id=session_id,
        provider=VoiceProvider.RETELL,
        phone_number=phone_number,
        user_id=user_id,
        language=language,
        metadata={
            "agent_id": agent_id,
            "websocket_path": "/ws/retell/llm"
        }
    )
    self.active_sessions[session_id] = session

    return VoiceResponse(
        success=True,
        provider=VoiceProvider.RETELL,
        session_id=session_id,
        message="Retell agent ready",
        metadata={
            "agent_id": agent_id,
            "websocket_path": "/ws/retell/llm"
        }
    )


# 7. Update get_status() to include retell_available
def get_status(self) -> Dict[str, Any]:
    """Get router status."""
    return {
        "bolna_available": self.bolna_available,
        "gemini_available": self.gemini_available,
        "retell_available": self.retell_available,  # NEW
        "fallback_available": True,
        "preferred_provider": self.preferred_provider.value,
        "active_sessions": self.get_active_session_count(),
        "available_providers": [p.value for p in self.get_available_providers()]
    }


# 8. Update create_voice_router() provider_map (~line 572-578)
provider_map = {
    "bolna": VoiceProvider.BOLNA,
    "b": VoiceProvider.BOLNA,
    "gemini_live": VoiceProvider.GEMINI_LIVE,
    "gemini": VoiceProvider.GEMINI_LIVE,
    "g": VoiceProvider.GEMINI_LIVE,
    "retell": VoiceProvider.RETELL,  # NEW
    "r": VoiceProvider.RETELL,       # NEW
    "fallback": VoiceProvider.FALLBACK_PIPELINE,
    "fallback_pipeline": VoiceProvider.FALLBACK_PIPELINE
}
```

### 9.2 Tests

```bash
python -c "from voice_router import VoiceProvider; print(VoiceProvider.RETELL.value)"
python -c "from voice_router import create_voice_router; r = create_voice_router(); print(r.get_status())"
```

### 9.3 Commit

```
feat(retell): Phase 7 - VoiceRouter extension with Retell provider

- Add VoiceProvider.RETELL to enum
- Add Retell client initialization
- Add _handle_retell_request() method
- Update provider_map with 'r' shortcut
```

---

## 10. Phase 8: Server Integration

### 10.1 Changes to `simple_rag_server.py`

See detailed code in implementation plan. Key additions:

1. **Imports** (~line 52-64): Add Retell module imports
2. **CLI flag** (~line 3245-3250): Add `-p r`/`-p retell` choice
3. **Provider selection** (~line 3256-3268): Handle Retell selection
4. **Global variables** (~line 3327): Add `retell_*` globals
5. **Startup event**: Add `startup_retell()` function
6. **WebSocket endpoint**: Add `/ws/retell/llm/{call_id}`
7. **REST endpoints**: Add `/api/retell/webhook`, `/api/retell/stats`, etc.

### 10.2 Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/ws/retell/llm/{call_id}` | WebSocket | Custom LLM protocol |
| `/api/retell/webhook` | POST | Call event webhooks |
| `/api/retell/health` | GET | Health check |
| `/api/retell/stats` | GET | Call statistics |
| `/api/retell/calls` | GET | Recent calls list |

### 10.3 Tests

```bash
# Start server with Retell provider
python simple_rag_server.py -p r --port 8001 &

# Test health endpoint
curl http://localhost:8001/api/retell/health

# Test stats endpoint
curl http://localhost:8001/api/retell/stats
```

### 10.4 Commit

```
feat(retell): Phase 8 - server integration with endpoints and CLI

- Add -p r/--provider retell CLI option
- Add startup_retell() initialization
- Add WebSocket /ws/retell/llm/{call_id}
- Add REST endpoints for webhook, stats, health
```

---

## 11. Phase 9: Testing Suite

### 11.1 Test Files to Create

- `tests/test_retell_config.py`
- `tests/test_retell_client.py`
- `tests/test_retell_webhooks.py`
- `tests/test_retell_llm_server.py`
- `tests/test_retell_integration.py`

### 11.2 Run Tests

```bash
pytest tests/test_retell_*.py -v
```

### 11.3 Commit

```
test(retell): Phase 9 - comprehensive test suite

- Add unit tests for config module
- Add unit tests for API client
- Add unit tests for webhook handler
- Add unit tests for Custom LLM server
- Add integration tests
```

---

## 12. Environment Variables

### 12.1 Required Variables

```bash
# Retell.AI Configuration
RETELL_API_KEY=your_retell_api_key_here
RETELL_AGENT_ID=agent_id_after_creation
RETELL_AGENT_NAME="Palli Sahayak"
```

### 12.2 Vobiz.ai Variables (for Indian PSTN)

```bash
# Vobiz.ai Telephony
VOBIZ_API_KEY=your_vobiz_api_key
VOBIZ_API_SECRET=your_vobiz_secret
VOBIZ_DID_NUMBER=+919876543210
VOBIZ_SIP_DOMAIN=sip.vobiz.ai
VOBIZ_SIP_USERNAME=your_sip_username
VOBIZ_SIP_PASSWORD=your_sip_password

# Retell SIP Integration (after setup)
RETELL_SIP_TRUNK_ID=trunk_id_from_retell
RETELL_INBOUND_NUMBER_ID=number_id_from_retell
```

### 12.3 Add to `.env.example`

```bash
# ===== RETELL.AI VOICE PROVIDER =====
# Get API key from: https://dashboard.retellai.com/apikey

RETELL_API_KEY=
RETELL_AGENT_ID=
RETELL_AGENT_NAME="Palli Sahayak"

# ===== VOBIZ.AI INDIAN TELEPHONY =====
# For +91 inbound number (callers don't need internet)
# Contact: https://vobiz.ai/

VOBIZ_API_KEY=
VOBIZ_API_SECRET=
VOBIZ_DID_NUMBER=+91
VOBIZ_SIP_DOMAIN=sip.vobiz.ai
VOBIZ_SIP_USERNAME=
VOBIZ_SIP_PASSWORD=

# After Retell SIP trunk setup
RETELL_SIP_TRUNK_ID=
RETELL_INBOUND_NUMBER_ID=
```

---

## 13. Implementation Checklist

```
[ ] Phase 1: Foundation Setup
    [ ] mkdir retell_integration/
    [ ] Create __init__.py
    [ ] Add retell-sdk to requirements.txt
    [ ] Tests pass

[ ] Phase 2: Configuration Module
    [ ] Create config.py
    [ ] CARTESIA_VOICE_IDS for Indian languages
    [ ] RETELL_LANGUAGE_CONFIGS with welcome messages
    [ ] RetellAgentConfig dataclass
    [ ] Tests pass

[ ] Phase 3: API Client
    [ ] Create client.py
    [ ] RetellClient with aiohttp
    [ ] Agent CRUD operations
    [ ] Call management methods
    [ ] Tests pass

[ ] Phase 4: Custom LLM WebSocket
    [ ] Create custom_llm_server.py
    [ ] RetellCustomLLMHandler
    [ ] RAG integration
    [ ] Multi-language responses
    [ ] Tests pass

[ ] Phase 5: Webhook Handler
    [ ] Create webhooks.py
    [ ] RetellWebhookHandler
    [ ] Call event processing
    [ ] Statistics tracking
    [ ] Tests pass

[ ] Phase 6: Vobiz Configuration
    [ ] Create vobiz_config.py
    [ ] VobizConfig dataclass
    [ ] SIP trunk configuration
    [ ] Setup instructions
    [ ] Tests pass

[ ] Phase 7: VoiceRouter Extension
    [ ] Add VoiceProvider.RETELL
    [ ] Add _handle_retell_request()
    [ ] Update provider_map
    [ ] Tests pass

[ ] Phase 8: Server Integration
    [ ] Add CLI -p r flag
    [ ] Add startup_retell()
    [ ] Add WebSocket endpoint
    [ ] Add REST endpoints
    [ ] Tests pass

[ ] Phase 9: Testing
    [ ] Create all test files
    [ ] All tests passing
    [ ] Manual integration test
```

---

## 14. References

### 14.1 Official Documentation

- [Retell.AI API Documentation](https://docs.retellai.com/api-references)
- [Retell Custom LLM WebSocket](https://docs.retellai.com/api-references/llm-websocket)
- [Retell Custom Telephony](https://docs.retellai.com/deploy/custom-telephony)
- [Retell Python SDK](https://github.com/RetellAI/retell-python-sdk)
- [Cartesia Sonic-3 Voices](https://cartesia.ai/india)
- [Vobiz.ai](https://vobiz.ai/)

### 14.2 Internal References

- `bolna_integration/` - Pattern reference
- `gemini_live/` - WebSocket pattern reference
- `voice_router.py` - Core file to extend
- `simple_rag_server.py` - Server integration

---

**Document End**
