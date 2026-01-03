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
RETELL_SYSTEM_PROMPT = """You are Palli Sahayak, a compassionate palliative care voice assistant for the Palli Sahayak Voice AI Agent Helpline.

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
