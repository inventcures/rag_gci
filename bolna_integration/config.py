"""
Bolna Agent Configuration for Palli Sahayak Voice AI Helpline

This module provides the agent configuration for the Palli Sahayak
palliative care voice assistant, including:
- System prompt
- Custom function call for RAG integration
- Language-specific settings

Documentation: https://www.bolna.ai/docs/platform-concepts
"""

import os
from typing import Dict, Any

from supported_languages import (
    CARTESIA_VOICES,
    PROVIDER_BOLNA,
    SUPPORTED_LANGUAGES,
    SupportedLanguage,
    Voice,
    get_language,
)

# The codes the caller's language can be reported in: all eleven Supported
# Languages, so the server knows the language even when this agent cannot
# speak it.
_LANGUAGE_CODES = [lang.code for lang in SUPPORTED_LANGUAGES]
_LANGUAGE_LEGEND = ", ".join(f"{lang.code}={lang.name}" for lang in SUPPORTED_LANGUAGES)

# System prompt for Palli Sahayak voice assistant
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
6. **NEVER** use laughter, jokes, or humor - this is a serious palliative care context
7. **DO** use gentle emotional expression to convey empathy and compassion - warmth helps patients feel understood
8. Appropriate emotions: gentle concern, soft reassurance, calm understanding, compassionate support

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
                "enum": _LANGUAGE_CODES,
                "description": f"The language code the user is speaking in ({_LANGUAGE_LEGEND})"
            },
            "conversation_context": {
                "type": "string",
                "description": "Brief summary of relevant conversation context"
            }
        },
        "required": ["user_query", "user_language"]
    }
}

# Languages, voices and greetings come from the Supported Language table
# (supported_languages.py): a row's "bolna" voice is the Deepgram code the agent
# listens in and the Cartesia Sonic-3 voice it speaks with. A language with no
# Bolna voice gets no agent: the old table answered Punjabi and Malayalam in a
# Hindi voice, and any unlisted language with a Hindi agent.
# NOTE: Do NOT use [laughter] tags - inappropriate for palliative care
# DO use empathetic emotion for warmth: gentle, calm, compassionate, reassuring


def bolna_language(language: str) -> SupportedLanguage:
    """The Supported Language a Bolna agent is built for, or ValueError when
    the language is outside the eleven or Bolna has no voice for it."""
    lang = get_language(language)
    if lang is None:
        raise ValueError(f"{language!r} is not a Supported Language; no Bolna agent is built for it")
    if lang.voice(PROVIDER_BOLNA) is None:
        speaks = ", ".join(l.name for l in SUPPORTED_LANGUAGES if l.voice(PROVIDER_BOLNA))
        raise ValueError(f"Bolna has no voice for {lang.name}; it speaks {speaks}")
    return lang


def _get_synthesizer_config(voice: Voice) -> Dict[str, Any]:
    """The Cartesia Sonic-3 synthesizer for a row's Bolna voice."""
    return {
        "provider": "cartesia",
        "audio_format": "wav",
        "stream": True,
        "buffer_size": 100.0,
        "provider_config": {
            "voice": voice.name,
            "voice_id": CARTESIA_VOICES[voice.name],
            "model": "sonic-3"
        }
    }


def get_palli_sahayak_agent_config(
    server_url: str,
    api_key: str = "",
    language: str = "hi",
    agent_name: str = "Palli Sahayak"
) -> Dict[str, Any]:
    """
    Get complete Bolna agent configuration for Palli Sahayak.

    Args:
        server_url: Base URL of the RAG server (e.g., https://your-domain.com)
        api_key: API key for RAG server authentication (optional)
        language: Primary language: a Supported Language Bolna has a voice
                  for (supported_languages.py)
        agent_name: Name of the agent

    Returns:
        Complete agent configuration dictionary for Bolna API

    Raises:
        ValueError: when Bolna cannot speak ``language``; no agent in another
            language is built in its place

    Example:
        config = get_palli_sahayak_agent_config(
            server_url="https://my-server.ngrok.io",
            language="hi"
        )
        result = await bolna_client.create_agent(config)
    """
    lang = bolna_language(language)
    voice = lang.voice(PROVIDER_BOLNA)

    # Build custom function with server URL
    custom_function = RAG_QUERY_FUNCTION.copy()
    custom_function["value"] = {
        "method": "POST",
        "url": f"{server_url.rstrip('/')}/api/bolna/query",
        "param": {
            "query": "%(user_query)s",
            "language": "%(user_language)s",
            "context": "%(conversation_context)s",
            "source": "bolna_call",
            # System variables: one participant per caller, not one for all.
            "from_number": "%(from_number)s",
            "execution_id": "%(execution_id)s"
        },
        "headers": {
            "Content-Type": "application/json"
        }
    }

    # Add authorization header if API key provided
    if api_key:
        custom_function["value"]["headers"]["Authorization"] = f"Bearer {api_key}"

    return {
        "agent_name": agent_name,
        "agent_type": "free_flowing",
        "agent_welcome_message": lang.greeting,

        "tasks": [
            {
                "task_type": "conversation",
                "toolchain": {
                    "execution": "parallel",
                    "pipelines": [["transcriber", "llm", "synthesizer"]]
                },
                "tools_config": {
                    "input": {
                        "format": "wav",
                        "provider": "twilio"
                    },
                    "output": {
                        "format": "wav",
                        "provider": "twilio"
                    },
                    "llm_agent": {
                        "agent_type": "simple_llm_agent",
                        "agent_flow_type": "streaming",
                        "llm_config": {
                            "provider": "openai",
                            "model": "gpt-4o-mini",
                            "request_json": True,
                            "max_tokens": 500,
                            "temperature": 0.3
                        },
                        "functions": [custom_function]
                    },
                    "synthesizer": _get_synthesizer_config(voice),
                    "transcriber": {
                        "provider": "deepgram",
                        "stream": True,
                        "encoding": "linear16",
                        "language": voice.language,
                        "model": "nova-3",
                        "endpointing": 2000  # 2 sec silence tolerance for natural speech
                    }
                },
                "task_config": {
                    "hangup_after_silence": 30.0,
                    # Patient interruption settings for natural speech with fillers (umm, aaa, hmm)
                    "endpointing": 2000,  # 2 sec silence before considering speech done
                    "number_of_words_for_interruption": 8,  # Need 8 words to interrupt user
                    "interruption_backoff_period": 1500,  # 1.5 sec cooldown after speaking
                    "incremental_delay": 600,  # 600ms pause before responding
                    "backchanneling": False,  # No "uh huh" sounds
                    "use_fillers": False  # No filler words from agent
                }
            }
        ],

        "agent_prompts": {
            "task_1": {
                "system_prompt": PALLI_SAHAYAK_SYSTEM_PROMPT
            }
        },

        # Follow-up tasks for call summary and data extraction
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


def get_agent_config_from_env(language: str = "hi") -> Dict[str, Any]:
    """
    Get agent configuration using environment variables.

    Environment variables used:
    - PUBLIC_BASE_URL: Base URL of the RAG server
    - RAG_API_KEY: API key for RAG server (optional)
    - BOLNA_AGENT_NAME: Agent name (optional, default: "Palli Sahayak")

    Args:
        language: Primary language code

    Returns:
        Agent configuration dictionary
    """
    server_url = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")
    api_key = os.getenv("RAG_API_KEY", "")
    agent_name = os.getenv("BOLNA_AGENT_NAME", "Palli Sahayak")

    return get_palli_sahayak_agent_config(
        server_url=server_url,
        api_key=api_key,
        language=language,
        agent_name=agent_name
    )


# Extraction schema for post-call analysis
EXTRACTION_SCHEMA = {
    "user_concern": {
        "type": "string",
        "description": "Main health concern or question raised by the user"
    },
    "language_used": {
        "type": "string",
        "enum": _LANGUAGE_CODES,
        "description": f"Primary language spoken by the user ({_LANGUAGE_LEGEND})"
    },
    "emotional_state": {
        "type": "string",
        "enum": ["calm", "anxious", "distressed", "neutral"],
        "description": "User's emotional state during the call"
    },
    "topics_discussed": {
        "type": "array",
        "items": {"type": "string"},
        "description": "List of main topics discussed"
    },
    "follow_up_needed": {
        "type": "boolean",
        "description": "Whether a follow-up call is recommended"
    },
    "urgency_level": {
        "type": "string",
        "enum": ["low", "medium", "high", "emergency"],
        "description": "Urgency level of the user's concern"
    }
}
