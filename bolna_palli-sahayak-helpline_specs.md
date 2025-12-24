# Palli Sahayak Voice AI Agent Helpline - Technical Specifications

## Architecture: Bolna.ai Orchestrator + Gemini Live Fallback

---

## USER PROMPT (Requirements)

```
1. USE GOOGLE GEMINI LIVE AS THE FALLBACK

2. ADD CODE TO USE BOLNA.AI AS THE ORCHESTRATOR

DOCS:
- https://www.bolna.ai/docs/introduction
- https://www.bolna.ai/docs/platform-concepts
- https://www.bolna.ai/docs/tips-and-tricks

3. USE CUSTOM FN CALLS TO INTEGRATE RAG PIPELINE TO GENERATE THE RESPONSE
   TO THE PALLIATIVE HEALTH CARE QUERIES
   https://www.bolna.ai/docs/tool-calling/custom-function-calls

4. THIS SYSTEM IS NOW CALLED "PALLI SAHAYAK VOICE AI AGENT HELPLINE"

5. REGENERATE THE PLAN KEEPING BOLNA AS THE ORCHESTRATOR (CONTROL PLANE)
   FOR THIS NEW PALLIATIVE CARE VOICE AI AGENT HELPLINE AND
   GOOGLE GEMINI LIVE API AS THE FALLBACK

6. THE PLAN SHOULD BE IMPLEMENTABLE WITH 0 BUGS AND OPTIMAL ARCHITECTURAL
   DECISIONS

7. EXISTING CODE BASE IS DOCUMENTED AT:
   https://deepwiki.com/inventcures/rag_gci
```

---

## EXECUTIVE SUMMARY

**Palli Sahayak Voice AI Agent Helpline** is a production-grade voice AI system for palliative care assistance that uses:

- **Primary (Orchestrator)**: Bolna.ai - handles telephony, ASR, TTS, and conversation orchestration
- **Fallback**: Google Gemini Live API - for web-based voice and when Bolna is unavailable
- **Intelligence**: Custom RAG pipeline integration via Bolna's custom function calls
- **Languages**: Indian English (en-IN), Hindi (hi-IN), Marathi (mr-IN), Tamil (ta-IN)

### Why This Architecture?

| Component | Role | Benefit |
|-----------|------|---------|
| **Bolna.ai** | Orchestrator/Control Plane | Production telephony, scalable calls, enterprise features |
| **RAG Pipeline** | Knowledge Engine | Grounded medical responses from verified documents |
| **Gemini Live** | Web Fallback | Browser-based voice when phone not available |

---

## SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PALLI SAHAYAK VOICE AI AGENT HELPLINE                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌──────────────────────────────────────────────────┐  │
│  │   USERS     │     │              BOLNA.AI (ORCHESTRATOR)              │  │
│  │             │     │  ┌─────────┐  ┌─────────┐  ┌─────────┐           │  │
│  │  Phone Call ├────►│  │  ASR    │─►│   LLM   │─►│   TTS   │           │  │
│  │  (Primary)  │     │  │Deepgram │  │ OpenAI  │  │Elevenlabs│          │  │
│  │             │     │  └────┬────┘  └────┬────┘  └─────────┘           │  │
│  └─────────────┘     │       │            │                              │  │
│                      │       │     ┌──────▼──────┐                       │  │
│  ┌─────────────┐     │       │     │  CUSTOM FN  │                       │  │
│  │   USERS     │     │       │     │    CALL     │                       │  │
│  │             │     │       │     └──────┬──────┘                       │  │
│  │  Web Browser│     └───────┼───────────┼──────────────────────────────┘  │
│  │  (Fallback) │             │           │                                  │
│  │             │             │           ▼                                  │
│  └──────┬──────┘     ┌───────┴───────────────────────────────────────────┐  │
│         │            │           RAG PIPELINE SERVER                      │  │
│         │            │  ┌─────────────────────────────────────────────┐  │  │
│         │            │  │  /api/bolna/query                           │  │  │
│         │            │  │  - Receives query from Bolna custom fn      │  │  │
│         │            │  │  - Queries vector DB (ChromaDB)             │  │  │
│         │            │  │  - Returns grounded medical response        │  │  │
│         │            │  └─────────────────────────────────────────────┘  │  │
│         │            └───────────────────────────────────────────────────┘  │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              GEMINI LIVE API (FALLBACK)                               │  │
│  │  - WebSocket /ws/voice                                                │  │
│  │  - Browser-based real-time voice                                      │  │
│  │  - RAG context injection                                              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## PART 1: BOLNA.AI INTEGRATION

### 1.1 Bolna Agent Configuration

Create a Bolna agent with the following configuration:

```json
{
  "agent_name": "Palli Sahayak",
  "agent_type": "free_flowing",
  "agent_welcome_message": "Namaste! I am Palli Sahayak, your palliative care assistant. How can I help you today?",

  "tasks": [
    {
      "task_type": "conversation",
      "toolchain": {
        "execution": "parallel",
        "pipelines": [["transcriber", "llm", "synthesizer"]]
      },
      "tools_config": {
        "llm_agent": {
          "agent_flow_type": "streaming",
          "provider": "openai",
          "request_json": true,
          "model": "gpt-4o-mini",
          "max_tokens": 500,
          "temperature": 0.3,
          "family": "openai"
        },
        "synthesizer": {
          "provider": "elevenlabs",
          "provider_config": {
            "voice": "Rachel",
            "model": "eleven_multilingual_v2",
            "voice_id": "21m00Tcm4TlvDq8ikWAM"
          },
          "stream": true
        },
        "transcriber": {
          "provider": "deepgram",
          "stream": true,
          "language": "hi",
          "model": "nova-2"
        }
      }
    }
  ],

  "agent_prompts": {
    "task_1": {
      "system_prompt": "PALLI_SAHAYAK_SYSTEM_PROMPT"
    }
  }
}
```

### 1.2 System Prompt for Bolna Agent

```
You are Palli Sahayak (पल्ली सहायक), a compassionate palliative care voice assistant.

ROLE:
- Provide empathetic support for patients and caregivers dealing with serious illness
- Answer questions about pain management, symptom control, and comfort care
- Offer emotional support and guidance
- Help navigate palliative care services

LANGUAGE:
- Respond in the same language the user speaks (Hindi, English, Marathi, or Tamil)
- Use simple, clear language avoiding medical jargon
- Be warm, patient, and understanding

IMPORTANT RULES:
1. ALWAYS call the `query_rag_knowledge_base` function FIRST to get accurate medical information
2. Never provide specific medication dosages - recommend consulting a doctor
3. For emergencies, immediately advise calling emergency services
4. Keep responses concise (2-3 sentences for voice)
5. Express empathy before providing information

CONVERSATION FLOW:
1. Greet warmly
2. Listen to the query
3. Call RAG function for accurate information
4. Provide compassionate, grounded response
5. Ask if there's anything else to help with

VARIABLES:
- user_query: The user's current question
- conversation_history: Previous exchanges in this call
- user_language: Detected language (hi/en/mr/ta)
```

### 1.3 Custom Function Call for RAG Integration

This is the critical integration point - Bolna calls our RAG pipeline:

```json
{
  "name": "query_rag_knowledge_base",
  "description": "Query the Palli Sahayak palliative care knowledge base to get accurate, verified medical information. ALWAYS call this function when the user asks any health-related question about palliative care, pain management, symptoms, medications, or caregiving.",
  "parameters": {
    "type": "object",
    "properties": {
      "user_query": {
        "type": "string",
        "description": "The user's health question or query about palliative care"
      },
      "user_language": {
        "type": "string",
        "enum": ["en", "hi", "mr", "ta"],
        "description": "The language the user is speaking"
      },
      "conversation_context": {
        "type": "string",
        "description": "Brief context from the conversation so far"
      }
    },
    "required": ["user_query", "user_language"]
  },
  "value": {
    "method": "POST",
    "url": "https://YOUR_SERVER_URL/api/bolna/query",
    "param": {
      "query": "%(user_query)s",
      "language": "%(user_language)s",
      "context": "%(conversation_context)s",
      "source": "bolna"
    },
    "headers": {
      "Content-Type": "application/json",
      "Authorization": "Bearer %(api_key)s"
    }
  }
}
```

---

## PART 2: RAG PIPELINE API ENDPOINT

### 2.1 New Endpoint: `/api/bolna/query`

Add this endpoint to `simple_rag_server.py`:

```python
# File: simple_rag_server.py
# Add after existing endpoints

from fastapi import Header
from typing import Optional
import hmac
import hashlib

# Bolna webhook secret for verification
BOLNA_WEBHOOK_SECRET = os.getenv("BOLNA_WEBHOOK_SECRET", "")

def verify_bolna_signature(payload: bytes, signature: str) -> bool:
    """Verify Bolna webhook signature."""
    if not BOLNA_WEBHOOK_SECRET:
        return True  # Skip verification if no secret configured

    expected = hmac.new(
        BOLNA_WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected, signature)


@app.post("/api/bolna/query")
async def bolna_query_endpoint(
    request: Request,
    x_bolna_signature: Optional[str] = Header(None)
):
    """
    RAG query endpoint for Bolna custom function calls.

    This endpoint receives queries from Bolna's LLM during voice calls,
    queries the palliative care knowledge base, and returns grounded responses.

    Request Body:
    {
        "query": "How to manage pain for cancer patients?",
        "language": "hi",
        "context": "User asking about pain management",
        "source": "bolna"
    }

    Response:
    {
        "status": "success",
        "answer": "For cancer pain management...",
        "sources": ["WHO Pain Guidelines", "AIIMS Palliative Care Manual"],
        "confidence": 0.92
    }
    """
    try:
        # Get raw body for signature verification
        body = await request.body()

        # Verify signature if provided
        if x_bolna_signature and not verify_bolna_signature(body, x_bolna_signature):
            logger.warning("Invalid Bolna signature")
            raise HTTPException(status_code=401, detail="Invalid signature")

        # Parse request
        data = json.loads(body)
        query = data.get("query", "")
        language = data.get("language", "en")
        context = data.get("context", "")
        source = data.get("source", "unknown")

        logger.info(f"Bolna RAG query: '{query}' (lang={language}, source={source})")

        if not query:
            return JSONResponse({
                "status": "error",
                "answer": "I didn't catch your question. Could you please repeat?",
                "sources": [],
                "confidence": 0.0
            })

        # Query RAG pipeline
        rag_result = await rag_pipeline.query(
            question=query,
            user_id=f"bolna_{source}",
            source_language=language,
            top_k=3
        )

        if rag_result.get("status") == "success":
            answer = rag_result.get("answer", "")

            # Extract source citations
            sources = []
            if "sources" in rag_result:
                sources = [s.get("title", "Unknown") for s in rag_result["sources"][:3]]
            elif "context_used" in rag_result:
                # Extract from context if available
                sources = ["Palliative Care Knowledge Base"]

            # Calculate confidence based on retrieval scores
            confidence = min(0.95, rag_result.get("relevance_score", 0.8))

            # Translate if needed
            if language != "en" and answer:
                translation_result = await rag_pipeline.translate_text(answer, language)
                if translation_result.get("status") == "success":
                    translated_answer = translation_result.get("translated_text", answer)
                    # Return both for Bolna to use
                    return JSONResponse({
                        "status": "success",
                        "answer": translated_answer,
                        "answer_english": answer,
                        "sources": sources,
                        "confidence": confidence,
                        "language": language
                    })

            return JSONResponse({
                "status": "success",
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "language": "en"
            })

        else:
            # RAG query failed - return graceful fallback
            logger.error(f"RAG query failed: {rag_result}")
            return JSONResponse({
                "status": "partial",
                "answer": "I'm having trouble accessing my knowledge base right now. Based on general palliative care principles, I'd recommend consulting with your healthcare provider for specific guidance.",
                "sources": [],
                "confidence": 0.3
            })

    except Exception as e:
        logger.error(f"Bolna query endpoint error: {e}")
        return JSONResponse({
            "status": "error",
            "answer": "I apologize, but I'm experiencing technical difficulties. Please try again or contact the helpline directly.",
            "sources": [],
            "confidence": 0.0
        }, status_code=500)


@app.post("/api/bolna/webhook")
async def bolna_webhook_endpoint(request: Request):
    """
    Webhook endpoint for Bolna call events.

    Receives notifications about:
    - Call started
    - Call ended
    - Call transferred
    - Extraction completed
    """
    try:
        data = await request.json()
        event_type = data.get("event", "unknown")
        call_id = data.get("call_id", "unknown")

        logger.info(f"Bolna webhook: {event_type} for call {call_id}")

        if event_type == "call_ended":
            # Log call summary
            summary = data.get("summary", "")
            duration = data.get("duration_seconds", 0)
            logger.info(f"Call {call_id} ended - Duration: {duration}s, Summary: {summary[:100]}...")

            # Could store in database for analytics

        elif event_type == "extraction_completed":
            # Handle extracted data
            extracted = data.get("extracted_data", {})
            logger.info(f"Call {call_id} extraction: {extracted}")

        return {"status": "received"}

    except Exception as e:
        logger.error(f"Bolna webhook error: {e}")
        return {"status": "error", "message": str(e)}
```

### 2.2 Bolna Configuration Module

Create a new module for Bolna integration:

```python
# File: bolna_integration/__init__.py
"""
Bolna.ai Integration Module for Palli Sahayak Voice AI Helpline

This module provides:
- BolnaClient: API client for Bolna.ai
- BolnaAgentConfig: Agent configuration management
- BolnaWebhookHandler: Webhook event processing
"""

from .client import BolnaClient
from .config import BolnaAgentConfig, PALLI_SAHAYAK_AGENT_CONFIG
from .webhooks import BolnaWebhookHandler

__all__ = [
    "BolnaClient",
    "BolnaAgentConfig",
    "PALLI_SAHAYAK_AGENT_CONFIG",
    "BolnaWebhookHandler",
]

__version__ = "1.0.0"
```

```python
# File: bolna_integration/client.py
"""
Bolna API Client for Palli Sahayak Voice AI Helpline
"""

import os
import logging
import aiohttp
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

BOLNA_API_BASE = "https://api.bolna.ai/v1"


@dataclass
class BolnaCallResult:
    """Result of a Bolna API call."""
    success: bool
    call_id: Optional[str] = None
    agent_id: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class BolnaClient:
    """
    Client for Bolna.ai API.

    Handles:
    - Agent creation and management
    - Call initiation (outbound)
    - Phone number management
    - Webhook configuration
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Bolna client.

        Args:
            api_key: Bolna API key. If not provided, reads from BOLNA_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("BOLNA_API_KEY")
        if not self.api_key:
            logger.warning("Bolna API key not configured")

        self.base_url = BOLNA_API_BASE
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def is_available(self) -> bool:
        """Check if Bolna client is configured."""
        return bool(self.api_key)

    async def create_agent(self, config: Dict[str, Any]) -> BolnaCallResult:
        """
        Create a new Bolna agent.

        Args:
            config: Agent configuration dictionary

        Returns:
            BolnaCallResult with agent_id if successful
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/agents",
                    headers=self.headers,
                    json=config
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return BolnaCallResult(
                            success=True,
                            agent_id=data.get("agent_id"),
                            data=data
                        )
                    else:
                        return BolnaCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )
        except Exception as e:
            logger.error(f"Failed to create Bolna agent: {e}")
            return BolnaCallResult(success=False, error=str(e))

    async def get_agent(self, agent_id: str) -> BolnaCallResult:
        """Get agent details."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/agents/{agent_id}",
                    headers=self.headers
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return BolnaCallResult(
                            success=True,
                            agent_id=agent_id,
                            data=data
                        )
                    else:
                        return BolnaCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )
        except Exception as e:
            logger.error(f"Failed to get Bolna agent: {e}")
            return BolnaCallResult(success=False, error=str(e))

    async def update_agent(self, agent_id: str, config: Dict[str, Any]) -> BolnaCallResult:
        """Update an existing agent."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    f"{self.base_url}/agents/{agent_id}",
                    headers=self.headers,
                    json=config
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return BolnaCallResult(
                            success=True,
                            agent_id=agent_id,
                            data=data
                        )
                    else:
                        return BolnaCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )
        except Exception as e:
            logger.error(f"Failed to update Bolna agent: {e}")
            return BolnaCallResult(success=False, error=str(e))

    async def initiate_call(
        self,
        agent_id: str,
        phone_number: str,
        user_data: Optional[Dict[str, Any]] = None
    ) -> BolnaCallResult:
        """
        Initiate an outbound call.

        Args:
            agent_id: ID of the agent to use
            phone_number: Phone number to call (E.164 format)
            user_data: Optional user context data

        Returns:
            BolnaCallResult with call_id if successful
        """
        try:
            payload = {
                "agent_id": agent_id,
                "phone_number": phone_number
            }

            if user_data:
                payload["user_data"] = user_data

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/calls",
                    headers=self.headers,
                    json=payload
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return BolnaCallResult(
                            success=True,
                            call_id=data.get("call_id"),
                            agent_id=agent_id,
                            data=data
                        )
                    else:
                        return BolnaCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )
        except Exception as e:
            logger.error(f"Failed to initiate Bolna call: {e}")
            return BolnaCallResult(success=False, error=str(e))

    async def get_call_status(self, call_id: str) -> BolnaCallResult:
        """Get status of a call."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/calls/{call_id}",
                    headers=self.headers
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return BolnaCallResult(
                            success=True,
                            call_id=call_id,
                            data=data
                        )
                    else:
                        return BolnaCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )
        except Exception as e:
            logger.error(f"Failed to get call status: {e}")
            return BolnaCallResult(success=False, error=str(e))

    async def list_phone_numbers(self) -> BolnaCallResult:
        """List available phone numbers."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/phone-numbers",
                    headers=self.headers
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return BolnaCallResult(
                            success=True,
                            data=data
                        )
                    else:
                        return BolnaCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )
        except Exception as e:
            logger.error(f"Failed to list phone numbers: {e}")
            return BolnaCallResult(success=False, error=str(e))

    async def configure_webhook(
        self,
        agent_id: str,
        webhook_url: str,
        events: List[str] = None
    ) -> BolnaCallResult:
        """
        Configure webhook for agent events.

        Args:
            agent_id: Agent ID
            webhook_url: URL to receive webhooks
            events: List of events to subscribe to
        """
        if events is None:
            events = ["call_started", "call_ended", "extraction_completed"]

        try:
            payload = {
                "webhook_url": webhook_url,
                "events": events
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/agents/{agent_id}/webhooks",
                    headers=self.headers,
                    json=payload
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return BolnaCallResult(
                            success=True,
                            agent_id=agent_id,
                            data=data
                        )
                    else:
                        return BolnaCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )
        except Exception as e:
            logger.error(f"Failed to configure webhook: {e}")
            return BolnaCallResult(success=False, error=str(e))
```

```python
# File: bolna_integration/config.py
"""
Bolna Agent Configuration for Palli Sahayak
"""

import os
from typing import Dict, Any

# System prompt for Palli Sahayak
PALLI_SAHAYAK_SYSTEM_PROMPT = """You are Palli Sahayak (पल्ली सहायक), a compassionate palliative care voice assistant for the Palli Sahayak Voice AI Agent Helpline.

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
1. **ALWAYS** call the `query_rag_knowledge_base` function FIRST when user asks any health question
2. **NEVER** provide specific medication dosages - always recommend consulting their doctor
3. For emergencies (severe pain, breathing difficulty, unconsciousness), immediately advise calling emergency services (108/112)
4. Keep voice responses concise - 2-3 sentences maximum for easy listening
5. Express empathy and validate feelings before giving medical information

## CONVERSATION STYLE
- Start with a warm greeting
- Listen carefully to understand the concern
- Call the RAG function to get accurate information
- Provide compassionate, grounded response
- End by asking if there's anything else you can help with

## SAMPLE RESPONSES

For pain question:
"I understand managing pain can be very difficult. Let me check our medical knowledge base for you... [RAG response]. Remember, your doctor can adjust medications based on your specific needs."

For emotional support:
"I hear how challenging this is for you. It's completely natural to feel overwhelmed. Would you like me to share some coping strategies, or would you prefer information about support groups in your area?"

## VARIABLES AVAILABLE
- user_query: The user's current question
- conversation_history: Previous exchanges
- user_language: Detected language (hi/en/mr/ta)
"""

# Custom function for RAG integration
RAG_QUERY_FUNCTION = {
    "name": "query_rag_knowledge_base",
    "description": "Query the Palli Sahayak palliative care knowledge base to get accurate, verified medical information about palliative care, pain management, symptom control, caregiving, and end-of-life care. ALWAYS call this function when the user asks ANY health-related question.",
    "parameters": {
        "type": "object",
        "properties": {
            "user_query": {
                "type": "string",
                "description": "The user's health question or query about palliative care, pain, symptoms, or caregiving"
            },
            "user_language": {
                "type": "string",
                "enum": ["en", "hi", "mr", "ta"],
                "description": "The language code the user is speaking in"
            },
            "conversation_context": {
                "type": "string",
                "description": "Brief summary of relevant conversation context"
            }
        },
        "required": ["user_query", "user_language"]
    }
}


def get_palli_sahayak_agent_config(
    server_url: str,
    api_key: str = "",
    language: str = "hi"
) -> Dict[str, Any]:
    """
    Get complete Bolna agent configuration for Palli Sahayak.

    Args:
        server_url: Base URL of the RAG server (e.g., https://your-domain.com)
        api_key: API key for RAG server authentication
        language: Primary language (hi, en, mr, ta)

    Returns:
        Complete agent configuration dictionary
    """

    # Language-specific configurations
    language_configs = {
        "hi": {
            "transcriber_language": "hi",
            "welcome_message": "नमस्ते! मैं पल्ली सहायक हूं, आपका पैलिएटिव केयर सहायक। आज मैं आपकी कैसे मदद कर सकती हूं?",
            "voice": "Rachel"  # Supports Hindi
        },
        "en": {
            "transcriber_language": "en",
            "welcome_message": "Hello! I am Palli Sahayak, your palliative care assistant. How can I help you today?",
            "voice": "Rachel"
        },
        "mr": {
            "transcriber_language": "mr",
            "welcome_message": "नमस्कार! मी पल्ली सहायक आहे, तुमचा पॅलिएटिव केअर सहाय्यक. आज मी तुम्हाला कशी मदत करू शकते?",
            "voice": "Rachel"
        },
        "ta": {
            "transcriber_language": "ta",
            "welcome_message": "வணக்கம்! நான் பல்லி சகாயக், உங்கள் நோய்த்தடுப்பு பராமரிப்பு உதவியாளர். இன்று நான் உங்களுக்கு எப்படி உதவ முடியும்?",
            "voice": "Rachel"
        }
    }

    lang_config = language_configs.get(language, language_configs["hi"])

    # Build custom function with server URL
    custom_function = RAG_QUERY_FUNCTION.copy()
    custom_function["value"] = {
        "method": "POST",
        "url": f"{server_url}/api/bolna/query",
        "param": {
            "query": "%(user_query)s",
            "language": "%(user_language)s",
            "context": "%(conversation_context)s",
            "source": "bolna_call"
        },
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else ""
        }
    }

    return {
        "agent_name": "Palli Sahayak",
        "agent_type": "free_flowing",
        "agent_welcome_message": lang_config["welcome_message"],

        "tasks": [
            {
                "task_type": "conversation",
                "toolchain": {
                    "execution": "parallel",
                    "pipelines": [["transcriber", "llm", "synthesizer"]]
                },
                "tools_config": {
                    "llm_agent": {
                        "agent_flow_type": "streaming",
                        "provider": "openai",
                        "request_json": True,
                        "model": "gpt-4o-mini",
                        "max_tokens": 500,
                        "temperature": 0.3,
                        "family": "openai",
                        "functions": [custom_function]
                    },
                    "synthesizer": {
                        "provider": "elevenlabs",
                        "provider_config": {
                            "voice": lang_config["voice"],
                            "model": "eleven_multilingual_v2",
                            "voice_id": "21m00Tcm4TlvDq8ikWAM"
                        },
                        "stream": True
                    },
                    "transcriber": {
                        "provider": "deepgram",
                        "stream": True,
                        "language": lang_config["transcriber_language"],
                        "model": "nova-2"
                    }
                }
            }
        ],

        "agent_prompts": {
            "task_1": {
                "system_prompt": PALLI_SAHAYAK_SYSTEM_PROMPT
            }
        },

        # Follow-up tasks
        "follow_up_tasks": [
            {
                "task_type": "summary",
                "enabled": True
            },
            {
                "task_type": "extraction",
                "enabled": True,
                "extraction_schema": {
                    "user_concern": "Main health concern or question",
                    "language_used": "Language spoken by user",
                    "emotional_state": "User's emotional state (calm/anxious/distressed)",
                    "follow_up_needed": "Whether follow-up call is recommended (yes/no)",
                    "key_topics": "List of main topics discussed"
                }
            }
        ]
    }


# Pre-configured agent config (template)
PALLI_SAHAYAK_AGENT_CONFIG = get_palli_sahayak_agent_config(
    server_url=os.getenv("PUBLIC_BASE_URL", "http://localhost:8000"),
    api_key=os.getenv("RAG_API_KEY", ""),
    language="hi"
)
```

```python
# File: bolna_integration/webhooks.py
"""
Webhook Handler for Bolna Events
"""

import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CallRecord:
    """Record of a Bolna call."""
    call_id: str
    agent_id: str
    phone_number: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_seconds: int = 0
    summary: str = ""
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    status: str = "in_progress"


class BolnaWebhookHandler:
    """
    Handler for Bolna webhook events.

    Processes:
    - call_started: New call initiated
    - call_ended: Call completed
    - extraction_completed: Data extracted from call
    - transcription: Real-time transcription updates
    """

    def __init__(self):
        self.active_calls: Dict[str, CallRecord] = {}
        self.completed_calls: Dict[str, CallRecord] = {}
        self.event_handlers: Dict[str, Callable] = {}

    def register_handler(self, event_type: str, handler: Callable):
        """Register a custom handler for an event type."""
        self.event_handlers[event_type] = handler

    async def handle_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming webhook event.

        Args:
            event_data: Webhook payload from Bolna

        Returns:
            Processing result
        """
        event_type = event_data.get("event", "unknown")
        call_id = event_data.get("call_id", "unknown")

        logger.info(f"Processing Bolna event: {event_type} for call {call_id}")

        # Call custom handler if registered
        if event_type in self.event_handlers:
            try:
                await self.event_handlers[event_type](event_data)
            except Exception as e:
                logger.error(f"Custom handler error for {event_type}: {e}")

        # Process standard events
        if event_type == "call_started":
            return await self._handle_call_started(event_data)

        elif event_type == "call_ended":
            return await self._handle_call_ended(event_data)

        elif event_type == "extraction_completed":
            return await self._handle_extraction(event_data)

        elif event_type == "transcription":
            return await self._handle_transcription(event_data)

        else:
            logger.warning(f"Unknown event type: {event_type}")
            return {"status": "ignored", "reason": "unknown_event"}

    async def _handle_call_started(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle call started event."""
        call_id = data.get("call_id")

        record = CallRecord(
            call_id=call_id,
            agent_id=data.get("agent_id", ""),
            phone_number=data.get("phone_number", ""),
            started_at=datetime.now(),
            status="in_progress"
        )

        self.active_calls[call_id] = record

        logger.info(f"Call started: {call_id} from {record.phone_number}")

        return {"status": "recorded", "call_id": call_id}

    async def _handle_call_ended(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle call ended event."""
        call_id = data.get("call_id")

        if call_id in self.active_calls:
            record = self.active_calls.pop(call_id)
            record.ended_at = datetime.now()
            record.duration_seconds = data.get("duration_seconds", 0)
            record.summary = data.get("summary", "")
            record.status = "completed"

            self.completed_calls[call_id] = record

            logger.info(
                f"Call ended: {call_id}, duration: {record.duration_seconds}s, "
                f"summary: {record.summary[:100]}..."
            )

        return {"status": "recorded", "call_id": call_id}

    async def _handle_extraction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle extraction completed event."""
        call_id = data.get("call_id")
        extracted = data.get("extracted_data", {})

        # Update record if exists
        if call_id in self.completed_calls:
            self.completed_calls[call_id].extracted_data = extracted
        elif call_id in self.active_calls:
            self.active_calls[call_id].extracted_data = extracted

        logger.info(f"Extraction completed for {call_id}: {extracted}")

        # Check if follow-up is needed
        if extracted.get("follow_up_needed") == "yes":
            logger.info(f"Follow-up recommended for call {call_id}")
            # Could trigger notification or task here

        return {"status": "recorded", "call_id": call_id, "extracted": extracted}

    async def _handle_transcription(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle real-time transcription update."""
        call_id = data.get("call_id")
        text = data.get("text", "")
        role = data.get("role", "user")

        logger.debug(f"Transcription [{call_id}] {role}: {text}")

        return {"status": "received"}

    def get_call_stats(self) -> Dict[str, Any]:
        """Get statistics about calls."""
        total_completed = len(self.completed_calls)
        total_duration = sum(c.duration_seconds for c in self.completed_calls.values())

        return {
            "active_calls": len(self.active_calls),
            "completed_calls": total_completed,
            "total_duration_seconds": total_duration,
            "average_duration_seconds": total_duration / total_completed if total_completed > 0 else 0
        }
```

---

## PART 3: GEMINI LIVE FALLBACK INTEGRATION

The existing Gemini Live implementation serves as the fallback when:
1. Bolna service is unavailable
2. User prefers web-based voice (no phone)
3. Cost optimization during high-load periods

### 3.1 Fallback Logic

```python
# File: voice_router.py
"""
Voice Routing Logic for Palli Sahayak

Routes voice requests to:
1. Bolna.ai (primary) - for phone calls
2. Gemini Live (fallback) - for web voice or when Bolna unavailable
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class VoiceProvider(Enum):
    BOLNA = "bolna"
    GEMINI_LIVE = "gemini_live"
    FALLBACK_STT_TTS = "fallback_stt_tts"


class VoiceRouter:
    """
    Routes voice requests to appropriate provider.

    Priority:
    1. Bolna - phone calls, production telephony
    2. Gemini Live - web voice, real-time conversations
    3. STT+LLM+TTS - ultimate fallback
    """

    def __init__(
        self,
        bolna_client=None,
        gemini_service=None,
        rag_pipeline=None
    ):
        self.bolna_client = bolna_client
        self.gemini_service = gemini_service
        self.rag_pipeline = rag_pipeline

        # Track provider health
        self.provider_health = {
            VoiceProvider.BOLNA: True,
            VoiceProvider.GEMINI_LIVE: True,
            VoiceProvider.FALLBACK_STT_TTS: True
        }

    def get_available_provider(
        self,
        preferred: Optional[VoiceProvider] = None,
        is_phone_call: bool = False
    ) -> VoiceProvider:
        """
        Get the best available voice provider.

        Args:
            preferred: Preferred provider if available
            is_phone_call: Whether this is a phone call (requires Bolna)

        Returns:
            Best available VoiceProvider
        """
        # Phone calls must use Bolna
        if is_phone_call:
            if self._is_bolna_available():
                return VoiceProvider.BOLNA
            else:
                logger.warning("Bolna unavailable for phone call - cannot proceed")
                return None

        # Check preferred provider
        if preferred:
            if preferred == VoiceProvider.BOLNA and self._is_bolna_available():
                return VoiceProvider.BOLNA
            elif preferred == VoiceProvider.GEMINI_LIVE and self._is_gemini_available():
                return VoiceProvider.GEMINI_LIVE

        # Default priority: Gemini Live for web, then fallback
        if self._is_gemini_available():
            return VoiceProvider.GEMINI_LIVE

        # Ultimate fallback
        return VoiceProvider.FALLBACK_STT_TTS

    def _is_bolna_available(self) -> bool:
        """Check if Bolna is available."""
        if not self.bolna_client:
            return False
        return self.bolna_client.is_available() and self.provider_health[VoiceProvider.BOLNA]

    def _is_gemini_available(self) -> bool:
        """Check if Gemini Live is available."""
        if not self.gemini_service:
            return False
        return self.gemini_service.is_available() and self.provider_health[VoiceProvider.GEMINI_LIVE]

    def mark_provider_unhealthy(self, provider: VoiceProvider):
        """Mark a provider as unhealthy after failure."""
        self.provider_health[provider] = False
        logger.warning(f"Marked {provider.value} as unhealthy")

    def mark_provider_healthy(self, provider: VoiceProvider):
        """Mark a provider as healthy."""
        self.provider_health[provider] = True
        logger.info(f"Marked {provider.value} as healthy")

    def get_status(self) -> Dict[str, Any]:
        """Get router status."""
        return {
            "bolna_available": self._is_bolna_available(),
            "gemini_available": self._is_gemini_available(),
            "provider_health": {p.value: h for p, h in self.provider_health.items()},
            "primary_provider": "bolna" if self._is_bolna_available() else "gemini_live"
        }
```

---

## PART 4: CONFIGURATION

### 4.1 Environment Variables

Add to `.env`:

```bash
# =============================================================================
# PALLI SAHAYAK VOICE AI AGENT HELPLINE CONFIGURATION
# =============================================================================

# Bolna.ai Configuration (Primary Orchestrator)
BOLNA_API_KEY=your-bolna-api-key
BOLNA_AGENT_ID=your-agent-id-after-creation
BOLNA_WEBHOOK_SECRET=your-webhook-secret

# Bolna Phone Number (purchase from Bolna dashboard)
BOLNA_PHONE_NUMBER=+91XXXXXXXXXX

# Google Cloud / Gemini Live (Fallback)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=./credentials/gcp-key.json
GEMINI_API_KEY=your-gemini-api-key

# RAG Server
PUBLIC_BASE_URL=https://your-server-url.com
RAG_API_KEY=your-rag-api-key

# Telephony Provider (used by Bolna)
TWILIO_ACCOUNT_SID=your-twilio-sid
TWILIO_AUTH_TOKEN=your-twilio-token
```

### 4.2 config.yaml Updates

```yaml
# =============================================================================
# PALLI SAHAYAK VOICE AI AGENT HELPLINE
# =============================================================================

# Voice AI Configuration
voice_ai:
  system_name: "Palli Sahayak Voice AI Agent Helpline"
  version: "2.0.0"

  # Primary: Bolna.ai Orchestrator
  primary_provider: "bolna"

  # Fallback: Gemini Live
  fallback_provider: "gemini_live"

  # Ultimate fallback: STT + LLM + TTS pipeline
  ultimate_fallback: "stt_llm_tts"

# Bolna.ai Configuration (Primary Orchestrator)
bolna:
  enabled: true
  api_key: ${BOLNA_API_KEY}
  agent_id: ${BOLNA_AGENT_ID}
  webhook_secret: ${BOLNA_WEBHOOK_SECRET}

  # Phone number for inbound calls
  phone_number: ${BOLNA_PHONE_NUMBER}

  # Agent settings
  agent:
    name: "Palli Sahayak"
    type: "free_flowing"

    # LLM settings
    llm:
      provider: "openai"
      model: "gpt-4o-mini"
      temperature: 0.3
      max_tokens: 500

    # Voice synthesis
    synthesizer:
      provider: "elevenlabs"
      voice: "Rachel"
      model: "eleven_multilingual_v2"

    # Speech recognition
    transcriber:
      provider: "deepgram"
      model: "nova-2"
      languages: ["hi", "en", "mr", "ta"]

  # Webhook configuration
  webhooks:
    enabled: true
    endpoint: "/api/bolna/webhook"
    events:
      - "call_started"
      - "call_ended"
      - "extraction_completed"

  # Follow-up tasks
  follow_up:
    summary_enabled: true
    extraction_enabled: true

# Gemini Live Configuration (Fallback)
gemini_live:
  enabled: true
  project_id: ${GOOGLE_CLOUD_PROJECT}
  location: "us-central1"
  model: "gemini-2.0-flash-live-001"

  default_voice: "Aoede"
  default_language: "en-IN"

  supported_languages:
    - code: "en-IN"
      name: "English (India)"
    - code: "hi-IN"
      name: "Hindi"
    - code: "mr-IN"
      name: "Marathi"
    - code: "ta-IN"
      name: "Tamil"

  session_timeout_minutes: 14
  max_sessions_per_user: 1

  # Audio settings
  input_sample_rate: 16000
  output_sample_rate: 24000
  chunk_size: 4096

  # RAG integration
  rag_context_enabled: true
  rag_top_k: 3

  # Fallback to STT+LLM+TTS on error
  fallback_enabled: true

# Supported Languages
languages:
  primary: "hi"  # Hindi as default
  supported:
    - code: "hi"
      name: "Hindi"
      native: "हिंदी"
    - code: "en"
      name: "English"
      native: "English"
    - code: "mr"
      name: "Marathi"
      native: "मराठी"
    - code: "ta"
      name: "Tamil"
      native: "தமிழ்"
```

---

## PART 5: IMPLEMENTATION PHASES

### Phase 1: Bolna Integration Module (Priority: HIGH)

**Files to create:**
```
bolna_integration/
├── __init__.py
├── client.py          # Bolna API client
├── config.py          # Agent configuration
└── webhooks.py        # Webhook handler
```

**Tasks:**
1. Create `bolna_integration/` directory
2. Implement `BolnaClient` class
3. Implement `get_palli_sahayak_agent_config()` function
4. Implement `BolnaWebhookHandler` class
5. Add Bolna API key validation

### Phase 2: RAG API Endpoint (Priority: HIGH)

**Files to modify:**
```
simple_rag_server.py   # Add /api/bolna/query endpoint
```

**Tasks:**
1. Add `/api/bolna/query` POST endpoint
2. Add `/api/bolna/webhook` POST endpoint
3. Implement signature verification
4. Add response translation for non-English queries
5. Add error handling and fallback responses

### Phase 3: Voice Router (Priority: MEDIUM)

**Files to create:**
```
voice_router.py        # Provider routing logic
```

**Tasks:**
1. Implement `VoiceRouter` class
2. Add provider health tracking
3. Implement fallback logic
4. Add status endpoint

### Phase 4: Configuration Updates (Priority: MEDIUM)

**Files to modify:**
```
config.yaml            # Add Bolna configuration
.env.example           # Add new environment variables
```

**Tasks:**
1. Update config.yaml with Bolna settings
2. Create .env.example with all required variables
3. Add configuration validation

### Phase 5: Testing & Deployment (Priority: HIGH)

**Tasks:**
1. Create Bolna agent via API or dashboard
2. Configure webhook URL
3. Test RAG endpoint with sample queries
4. Test end-to-end phone call
5. Test Gemini Live fallback
6. Document deployment steps

---

## PART 6: API REFERENCE

### 6.1 RAG Query Endpoint

**POST /api/bolna/query**

Request:
```json
{
  "query": "How to manage pain for cancer patients?",
  "language": "hi",
  "context": "User is a caregiver asking about pain management",
  "source": "bolna_call"
}
```

Response:
```json
{
  "status": "success",
  "answer": "कैंसर के दर्द के प्रबंधन के लिए...",
  "answer_english": "For cancer pain management...",
  "sources": ["WHO Pain Guidelines", "AIIMS Manual"],
  "confidence": 0.92,
  "language": "hi"
}
```

### 6.2 Webhook Endpoint

**POST /api/bolna/webhook**

Events:
- `call_started`: New call initiated
- `call_ended`: Call completed with summary
- `extraction_completed`: Data extracted from call

---

## PART 7: TESTING CHECKLIST

- [ ] Bolna API key validation
- [ ] Agent creation via API
- [ ] Custom function call to RAG endpoint
- [ ] RAG response in multiple languages
- [ ] Webhook event processing
- [ ] Gemini Live fallback when Bolna unavailable
- [ ] End-to-end phone call test
- [ ] Web voice interface test
- [ ] Error handling and graceful degradation
- [ ] Load testing with concurrent calls

---

## PART 8: COST ESTIMATION

| Component | Cost | Notes |
|-----------|------|-------|
| Bolna.ai | ~$0.10-0.15/min | Includes ASR + LLM + TTS |
| Phone number | ~$1-2/month | Indian number via Twilio/Plivo |
| Gemini Live (fallback) | ~$0.075/min | Web voice only |
| RAG Pipeline | Infrastructure cost | Self-hosted |

**Estimated monthly cost for 1000 calls (avg 5 min each):**
- Bolna: ~$500-750
- Gemini fallback (10%): ~$37.50
- **Total: ~$550-800/month**

---

## SUMMARY

This specification provides a complete blueprint for implementing the **Palli Sahayak Voice AI Agent Helpline** with:

1. **Bolna.ai as Primary Orchestrator**: Production-grade telephony with enterprise features
2. **Custom RAG Integration**: Grounded medical responses via custom function calls
3. **Gemini Live as Fallback**: Web-based voice for non-phone users
4. **Multi-language Support**: Hindi, English, Marathi, Tamil
5. **Comprehensive Monitoring**: Webhooks, call summaries, and data extraction

The architecture ensures high availability through the fallback mechanism while optimizing costs by using the appropriate provider for each use case.

---

*Document Version: 1.0.0*
*Last Updated: December 2024*
*System: Palli Sahayak Voice AI Agent Helpline*