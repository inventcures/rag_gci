"""
Gemini Live API Integration Module

This module provides real-time voice conversation capabilities using
Google's Gemini Live API for the Palliative Care RAG system.

Supports:
- Indian English (en-IN)
- Hindi (hi-IN)
- Marathi (mr-IN)
- Tamil (ta-IN)

Components:
- GeminiLiveConfig: Configuration management
- AudioHandler: Audio format conversion (PCM, MP3, OGG)
- GeminiLiveService: Main service for voice conversations
- GeminiLiveSession: Individual voice conversation session
- SessionManager: Session lifecycle management

Usage:
    from gemini_live import (
        GeminiLiveService,
        SessionManager,
        AudioHandler,
        get_config,
    )

    # Initialize service
    service = GeminiLiveService(rag_pipeline=your_rag_pipeline)
    manager = SessionManager(service)
    await manager.start()

    # Create session for user
    session = await manager.get_or_create_session(
        user_id="user123",
        language="hi-IN"
    )

    # Use session for voice conversation
    await session.send_audio(audio_chunk)
    async for response in session.receive_audio():
        # Process audio response
        pass

    # Cleanup
    await manager.stop()
"""

from .config import (
    GeminiLiveConfig,
    SUPPORTED_LANGUAGES,
    VOICE_OPTIONS,
    LANGUAGE_CODE_MAP,
    DEFAULT_VOICE,
    get_config,
    reload_config,
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    DEFAULT_CHUNK_SIZE,
)
from .audio_handler import (
    AudioHandler,
    AudioHandlerError,
    convert_whatsapp_audio_to_pcm,
    convert_gemini_audio_to_mp3,
)
from .service import (
    GeminiLiveService,
    GeminiLiveSession,
    GeminiLiveError,
)
from .session_manager import (
    SessionManager,
    SessionInfo,
)

__all__ = [
    # Config
    "GeminiLiveConfig",
    "get_config",
    "reload_config",
    "SUPPORTED_LANGUAGES",
    "VOICE_OPTIONS",
    "LANGUAGE_CODE_MAP",
    "DEFAULT_VOICE",
    # Audio constants
    "INPUT_SAMPLE_RATE",
    "OUTPUT_SAMPLE_RATE",
    "DEFAULT_CHUNK_SIZE",
    # Audio handler
    "AudioHandler",
    "AudioHandlerError",
    "convert_whatsapp_audio_to_pcm",
    "convert_gemini_audio_to_mp3",
    # Service
    "GeminiLiveService",
    "GeminiLiveSession",
    "GeminiLiveError",
    # Session management
    "SessionManager",
    "SessionInfo",
]

__version__ = "0.2.0"
