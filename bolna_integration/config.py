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
from typing import Any, Dict, List

from supported_languages import (
    CARTESIA_VOICES,
    PROVIDER_BOLNA,
    SUPPORTED_LANGUAGES,
    SupportedLanguage,
    Voice,
    get_language,
)

# The codes the call's language can be reported in (post-call extraction):
# all eleven Supported Languages, so the server knows the language even when
# this agent cannot speak it.
_LANGUAGE_CODES = [lang.code for lang in SUPPORTED_LANGUAGES]
_LANGUAGE_LEGEND = ", ".join(f"{lang.code}={lang.name}" for lang in SUPPORTED_LANGUAGES)

# The line Bolna fills with the call's system variables ({from_number},
# {execution_id}) before it sends the prompt to the agent's LLM, which is this
# server (/api/bolna/llm/...): the one place a custom LLM learns who is calling.
# bolna_integration.custom_llm.caller_fields() reads it back.
CALLER_LINE = "Caller number: {from_number}; call id: {execution_id}"

_BOLNA_LANGUAGE_NAMES = ", ".join(
    lang.name for lang in SUPPORTED_LANGUAGES if lang.voice(PROVIDER_BOLNA) is not None
)

# The agent's prompt. The agent's LLM is this server, which answers every turn
# with the guarded answer (Dose Boundary) and does not follow this prompt; it
# documents the agent for whoever opens it in Bolna, and carries CALLER_LINE.
PALLI_SAHAYAK_SYSTEM_PROMPT = (
    "You are Palli Sahayak (पल्ली सहायक), a palliative care voice assistant for the "
    "Palli Sahayak Voice AI Agent Helpline.\n\n"
    "Every reply is written by the Palli Sahayak server from its palliative care "
    "knowledge base and has passed its dose check. Speak each reply exactly as given: "
    "do not reword, shorten or add to it.\n\n"
    f"This agent speaks {_BOLNA_LANGUAGE_NAMES}.\n\n"
    f"{CALLER_LINE}\n"
)


def bolna_system_prompt() -> str:
    """The prompt the Bolna agent is built with (CALLER_LINE unfilled)."""
    return PALLI_SAHAYAK_SYSTEM_PROMPT


# The model name this server answers as. The operator registers it in Bolna as
# a custom LLM (Settings, Integrations, LLM, Custom LLM) with the endpoint URL
# below and RAG_API_KEY as its key; Bolna then sends that key as a Bearer.
CUSTOM_LLM_MODEL = "palli-sahayak-guarded"


def bolna_llm_base_url(server_url: str, language: str) -> str:
    """The OpenAI-compatible base URL Bolna appends /chat/completions to."""
    return f"{server_url.rstrip('/')}/api/bolna/llm/{language}/v1"


def bolna_outside_services(config: Dict[str, Any]) -> List[str]:
    """The services outside this server that a Bolna agent config sends the call to.

    The Bolna platform itself, then each provider the toolchain names
    (telephony in and out, transcriber, synthesizer, LLM). A "custom" LLM is
    this server, which is not an outside service. The post-call summary and
    extraction run on Bolna's own hosted LLM, so an enabled follow-up task adds
    `bolna.post_call_analysis`. Sorted, without repeats.
    """
    found = {"bolna"}
    if any(task.get("enabled") for task in config.get("follow_up_tasks") or []):
        found.add("bolna.post_call_analysis")
    for task in config.get("tasks") or []:
        tools = task.get("tools_config") or {}
        for key in ("input", "output", "transcriber", "synthesizer"):
            provider = (tools.get(key) or {}).get("provider")
            if provider:
                found.add(f"bolna.{provider}")
        llm = ((tools.get("llm_agent") or {}).get("llm_config") or {}).get("provider")
        if llm and llm != "custom":
            found.add(f"bolna.{llm}")
    return sorted(found)


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
    language: str = "hi",
    agent_name: str = "Palli Sahayak"
) -> Dict[str, Any]:
    """
    Get complete Bolna agent configuration for Palli Sahayak.

    Args:
        server_url: Base URL of the RAG server (e.g., https://your-domain.com);
                    the agent's LLM is this server's /api/bolna/llm/<language>/v1.
                    Its key (RAG_API_KEY) is registered with the custom LLM in
                    Bolna, not sent in the agent config.
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
                        # This server is the LLM: Bolna speaks the guarded
                        # answer as written, and no hosted LLM rewords it
                        # (decision D11; Bolna gives no hook on its own
                        # LLM's output to pass it through the Dose Boundary).
                        "llm_config": {
                            "provider": "custom",
                            "model": CUSTOM_LLM_MODEL,
                            "base_url": bolna_llm_base_url(server_url, lang.code),
                            "max_tokens": 500,
                            "temperature": 0.0
                        }
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
    - BOLNA_AGENT_NAME: Agent name (optional, default: "Palli Sahayak")

    Args:
        language: Primary language code

    Returns:
        Agent configuration dictionary
    """
    server_url = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")
    agent_name = os.getenv("BOLNA_AGENT_NAME", "Palli Sahayak")

    return get_palli_sahayak_agent_config(
        server_url=server_url,
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
