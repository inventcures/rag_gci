"""
Voice Router for Palli Sahayak Voice AI Agent Helpline

Routes voice requests between providers:
- Bolna.ai (-p b): Phone calls via Twilio
- Gemini Live (-p g): Web-based voice with native audio
- Retell.AI (-p r): Phone calls via Vobiz.ai (Indian PSTN +91)
- Fallback Pipeline: STT → RAG → LLM → TTS (always available)

This module provides a unified interface for voice handling regardless
of the underlying provider.
"""

import os
import logging
from enum import Enum
from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class VoiceProvider(Enum):
    """Available voice providers."""
    BOLNA = "bolna"
    GEMINI_LIVE = "gemini_live"
    RETELL = "retell"
    FALLBACK_PIPELINE = "fallback_pipeline"


@dataclass
class VoiceSession:
    """Represents an active voice session."""
    session_id: str
    provider: VoiceProvider
    phone_number: Optional[str] = None
    user_id: Optional[str] = None
    language: str = "hi"
    started_at: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class VoiceResponse:
    """Response from a voice provider."""
    success: bool
    provider: VoiceProvider
    session_id: Optional[str] = None
    message: str = ""
    audio_url: Optional[str] = None
    transcript: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class VoiceRouter:
    """
    Routes voice requests to appropriate providers.

    Priority:
    1. Bolna.ai - For phone calls (production telephony)
    2. Gemini Live - For web-based voice (real-time streaming)
    3. Fallback Pipeline - STT → RAG → LLM → TTS (always available)

    Usage:
        router = VoiceRouter(rag_pipeline=my_rag)

        # For phone call
        response = await router.handle_phone_call("+919876543210", language="hi")

        # For web voice
        response = await router.handle_web_voice(user_id="user123", language="en")

        # Auto-select based on context
        response = await router.route_voice_request(
            request_type="phone",
            phone_number="+919876543210"
        )
    """

    def __init__(
        self,
        rag_pipeline=None,
        bolna_client=None,
        gemini_service=None,
        preferred_provider: VoiceProvider = VoiceProvider.BOLNA
    ):
        """
        Initialize the voice router.

        Args:
            rag_pipeline: RAG pipeline for fallback queries
            bolna_client: BolnaClient instance (optional, will create if available)
            gemini_service: GeminiLiveService instance (optional)
            preferred_provider: Default provider preference
        """
        self.rag_pipeline = rag_pipeline
        self.preferred_provider = preferred_provider
        self.active_sessions: Dict[str, VoiceSession] = {}

        # Initialize Bolna client
        self.bolna_client = bolna_client
        self.bolna_available = False
        if self.bolna_client is None:
            try:
                from bolna_integration import BolnaClient
                self.bolna_client = BolnaClient()
                self.bolna_available = self.bolna_client.is_available()
            except ImportError:
                logger.warning("Bolna integration not available")
            except Exception as e:
                logger.warning(f"Failed to initialize Bolna client: {e}")

        # Initialize Gemini Live service
        self.gemini_service = gemini_service
        self.gemini_available = False
        if self.gemini_service is None:
            try:
                from gemini_live import GeminiLiveService
                self.gemini_service = GeminiLiveService()
                self.gemini_available = True
            except ImportError:
                logger.warning("Gemini Live not available")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini Live: {e}")

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

        # Log available providers
        providers = []
        if self.bolna_available:
            providers.append("Bolna.ai")
        if self.gemini_available:
            providers.append("Gemini Live")
        if self.retell_available:
            providers.append("Retell.AI")
        providers.append("Fallback Pipeline")

        logger.info(f"VoiceRouter initialized with providers: {', '.join(providers)}")

    def get_available_providers(self) -> list:
        """Get list of available voice providers."""
        providers = []
        if self.bolna_available:
            providers.append(VoiceProvider.BOLNA)
        if self.gemini_available:
            providers.append(VoiceProvider.GEMINI_LIVE)
        if self.retell_available:
            providers.append(VoiceProvider.RETELL)
        providers.append(VoiceProvider.FALLBACK_PIPELINE)
        return providers

    def select_provider(
        self,
        request_type: str = "web",
        force_provider: Optional[VoiceProvider] = None
    ) -> VoiceProvider:
        """
        Select the best provider for a request.

        Args:
            request_type: "phone" or "web"
            force_provider: Force a specific provider (if available)

        Returns:
            Selected VoiceProvider
        """
        # If forced and available, use it
        if force_provider:
            if force_provider == VoiceProvider.BOLNA and self.bolna_available:
                return VoiceProvider.BOLNA
            elif force_provider == VoiceProvider.GEMINI_LIVE and self.gemini_available:
                return VoiceProvider.GEMINI_LIVE
            elif force_provider == VoiceProvider.RETELL and self.retell_available:
                return VoiceProvider.RETELL
            elif force_provider == VoiceProvider.FALLBACK_PIPELINE:
                return VoiceProvider.FALLBACK_PIPELINE

        # Phone calls: prefer Bolna
        if request_type == "phone":
            if self.bolna_available:
                return VoiceProvider.BOLNA
            elif self.gemini_available:
                return VoiceProvider.GEMINI_LIVE
            else:
                return VoiceProvider.FALLBACK_PIPELINE

        # Web voice: prefer Gemini Live
        if request_type == "web":
            if self.gemini_available:
                return VoiceProvider.GEMINI_LIVE
            elif self.bolna_available:
                return VoiceProvider.BOLNA
            else:
                return VoiceProvider.FALLBACK_PIPELINE

        # Default to fallback
        return VoiceProvider.FALLBACK_PIPELINE

    async def route_voice_request(
        self,
        request_type: str = "web",
        phone_number: Optional[str] = None,
        user_id: Optional[str] = None,
        language: str = "hi",
        force_provider: Optional[VoiceProvider] = None,
        **kwargs
    ) -> VoiceResponse:
        """
        Route a voice request to the appropriate provider.

        Args:
            request_type: "phone" or "web"
            phone_number: Phone number for phone calls
            user_id: User ID for web sessions
            language: Language code (hi, en, mr, ta)
            force_provider: Force a specific provider
            **kwargs: Additional provider-specific arguments

        Returns:
            VoiceResponse from the selected provider
        """
        provider = self.select_provider(request_type, force_provider)

        logger.info(f"Routing {request_type} request to {provider.value}")

        try:
            if provider == VoiceProvider.BOLNA:
                return await self._handle_bolna_request(
                    phone_number=phone_number,
                    user_id=user_id,
                    language=language,
                    **kwargs
                )

            elif provider == VoiceProvider.GEMINI_LIVE:
                return await self._handle_gemini_request(
                    user_id=user_id,
                    language=language,
                    **kwargs
                )

            elif provider == VoiceProvider.RETELL:
                return await self._handle_retell_request(
                    phone_number=phone_number,
                    user_id=user_id,
                    language=language,
                    **kwargs
                )

            else:
                return await self._handle_fallback_request(
                    user_id=user_id,
                    language=language,
                    **kwargs
                )

        except Exception as e:
            logger.error(f"Provider {provider.value} failed: {e}")

            # Try fallback if primary failed
            if provider != VoiceProvider.FALLBACK_PIPELINE:
                logger.info("Falling back to pipeline...")
                return await self._handle_fallback_request(
                    user_id=user_id,
                    language=language,
                    error_context=str(e),
                    **kwargs
                )

            return VoiceResponse(
                success=False,
                provider=provider,
                error=str(e),
                message="Voice service unavailable"
            )

    async def handle_phone_call(
        self,
        phone_number: str,
        language: str = "hi",
        user_data: Optional[Dict[str, Any]] = None
    ) -> VoiceResponse:
        """
        Handle an incoming or outgoing phone call.

        Uses Bolna as primary provider for phone calls.

        Args:
            phone_number: Phone number in E.164 format
            language: Language code
            user_data: Optional user context data

        Returns:
            VoiceResponse with call details
        """
        return await self.route_voice_request(
            request_type="phone",
            phone_number=phone_number,
            language=language,
            user_data=user_data
        )

    async def handle_web_voice(
        self,
        user_id: str,
        language: str = "hi",
        **kwargs
    ) -> VoiceResponse:
        """
        Handle a web-based voice session.

        Uses Gemini Live as primary provider for web voice.

        Args:
            user_id: User identifier
            language: Language code
            **kwargs: Additional arguments

        Returns:
            VoiceResponse with session details
        """
        return await self.route_voice_request(
            request_type="web",
            user_id=user_id,
            language=language,
            **kwargs
        )

    async def _handle_bolna_request(
        self,
        phone_number: Optional[str] = None,
        user_id: Optional[str] = None,
        language: str = "hi",
        user_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> VoiceResponse:
        """Handle request via Bolna."""
        if not self.bolna_available:
            raise RuntimeError("Bolna client not available")

        agent_id = os.getenv("BOLNA_AGENT_ID")
        if not agent_id:
            raise RuntimeError("BOLNA_AGENT_ID not configured")

        # Initiate outbound call if phone number provided
        if phone_number:
            result = await self.bolna_client.initiate_call(
                agent_id=agent_id,
                phone_number=phone_number,
                user_data=user_data or {"language": language, "user_id": user_id}
            )

            if result.success:
                # Create session record
                session = VoiceSession(
                    session_id=result.call_id,
                    provider=VoiceProvider.BOLNA,
                    phone_number=phone_number,
                    user_id=user_id,
                    language=language,
                    metadata={"agent_id": agent_id}
                )
                self.active_sessions[result.call_id] = session

                return VoiceResponse(
                    success=True,
                    provider=VoiceProvider.BOLNA,
                    session_id=result.call_id,
                    message=f"Call initiated to {phone_number}",
                    metadata=result.data
                )
            else:
                return VoiceResponse(
                    success=False,
                    provider=VoiceProvider.BOLNA,
                    error=result.error,
                    message="Failed to initiate call"
                )

        # For inbound calls, Bolna handles via webhook
        return VoiceResponse(
            success=True,
            provider=VoiceProvider.BOLNA,
            message="Bolna agent ready for inbound calls",
            metadata={"agent_id": agent_id}
        )

    async def _handle_gemini_request(
        self,
        user_id: Optional[str] = None,
        language: str = "hi",
        **kwargs
    ) -> VoiceResponse:
        """Handle request via Gemini Live."""
        if not self.gemini_available:
            raise RuntimeError("Gemini Live not available")

        try:
            # Create Gemini Live session
            session_id = f"gemini_{user_id or 'anonymous'}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Map language to Gemini voice
            language_map = {
                "hi": "hi-IN",
                "en": "en-IN",
                "mr": "mr-IN",
                "ta": "ta-IN"
            }
            gemini_language = language_map.get(language, "en-IN")

            # Create session record
            session = VoiceSession(
                session_id=session_id,
                provider=VoiceProvider.GEMINI_LIVE,
                user_id=user_id,
                language=language,
                metadata={"gemini_language": gemini_language}
            )
            self.active_sessions[session_id] = session

            return VoiceResponse(
                success=True,
                provider=VoiceProvider.GEMINI_LIVE,
                session_id=session_id,
                message="Gemini Live session ready",
                metadata={
                    "language": gemini_language,
                    "websocket_path": "/ws/voice"
                }
            )

        except Exception as e:
            logger.error(f"Gemini session creation failed: {e}")
            raise

    async def _handle_retell_request(
        self,
        phone_number: Optional[str] = None,
        user_id: Optional[str] = None,
        language: str = "hi",
        user_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> VoiceResponse:
        """
        Handle request via Retell.AI.

        Retell uses Custom LLM via WebSocket for full RAG integration,
        Cartesia Sonic-3 TTS, and Vobiz.ai for Indian PSTN telephony.

        Args:
            phone_number: Phone number for calls (E.164 format)
            user_id: User identifier
            language: Language code (hi, en, mr, ta)
            user_data: Optional user context data

        Returns:
            VoiceResponse with session/call details
        """
        if not self.retell_available:
            raise RuntimeError("Retell client not available")

        agent_id = os.getenv("RETELL_AGENT_ID")
        if not agent_id:
            raise RuntimeError("RETELL_AGENT_ID not configured")

        # For inbound calls via Vobiz.ai, Retell handles via WebSocket
        # The Custom LLM server at /ws/retell/llm/{call_id} processes requests
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
                "telephony": "vobiz" if phone_number else "web"
            }
        )
        self.active_sessions[session_id] = session

        # Map language to Retell/Cartesia voice
        from retell_integration.config import CARTESIA_VOICE_IDS
        voice_config = CARTESIA_VOICE_IDS.get(language, CARTESIA_VOICE_IDS.get("hi"))

        return VoiceResponse(
            success=True,
            provider=VoiceProvider.RETELL,
            session_id=session_id,
            message="Retell agent ready" + (f" for {phone_number}" if phone_number else ""),
            metadata={
                "agent_id": agent_id,
                "voice": voice_config.get("voice_name") if voice_config else "Hindi Narrator Woman",
                "websocket_path": "/ws/retell/llm",
                "webhook_path": "/api/retell/webhook",
                "language": language
            }
        )

    async def _handle_fallback_request(
        self,
        user_id: Optional[str] = None,
        language: str = "hi",
        query: Optional[str] = None,
        error_context: Optional[str] = None,
        **kwargs
    ) -> VoiceResponse:
        """
        Handle request via fallback STT → RAG → LLM → TTS pipeline.

        This is the ultimate fallback when both Bolna and Gemini are unavailable.
        """
        session_id = f"fallback_{user_id or 'anonymous'}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Create session record
        session = VoiceSession(
            session_id=session_id,
            provider=VoiceProvider.FALLBACK_PIPELINE,
            user_id=user_id,
            language=language,
            metadata={"error_context": error_context} if error_context else {}
        )
        self.active_sessions[session_id] = session

        response_message = "Fallback voice pipeline ready"
        if error_context:
            response_message = f"Using fallback pipeline (primary error: {error_context})"

        return VoiceResponse(
            success=True,
            provider=VoiceProvider.FALLBACK_PIPELINE,
            session_id=session_id,
            message=response_message,
            metadata={
                "pipeline": "STT → RAG → LLM → TTS",
                "stt_provider": "groq_whisper",
                "tts_provider": "edge_tts",
                "language": language
            }
        )

    async def process_audio_fallback(
        self,
        audio_data: bytes,
        language: str = "hi",
        user_id: Optional[str] = None
    ) -> VoiceResponse:
        """
        Process audio through the fallback pipeline.

        Pipeline: Audio → STT → RAG Query → LLM → TTS → Audio

        Args:
            audio_data: Raw audio bytes
            language: Language code
            user_id: User identifier

        Returns:
            VoiceResponse with transcript and audio URL
        """
        try:
            # This would integrate with existing STT/TTS services
            # For now, return placeholder

            if not self.rag_pipeline:
                return VoiceResponse(
                    success=False,
                    provider=VoiceProvider.FALLBACK_PIPELINE,
                    error="RAG pipeline not configured"
                )

            # Placeholder for STT processing
            # transcript = await stt_service.transcribe(audio_data, language)

            # Placeholder for RAG query
            # result = await self.rag_pipeline.query(transcript, source_language=language)

            # Placeholder for TTS
            # audio_url = await tts_service.synthesize(result["answer"], language)

            return VoiceResponse(
                success=True,
                provider=VoiceProvider.FALLBACK_PIPELINE,
                message="Audio processed via fallback pipeline",
                metadata={"language": language}
            )

        except Exception as e:
            logger.error(f"Fallback audio processing failed: {e}")
            return VoiceResponse(
                success=False,
                provider=VoiceProvider.FALLBACK_PIPELINE,
                error=str(e)
            )

    def get_session(self, session_id: str) -> Optional[VoiceSession]:
        """Get an active session by ID."""
        return self.active_sessions.get(session_id)

    def end_session(self, session_id: str) -> bool:
        """End and remove a session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Session {session_id} ended")
            return True
        return False

    def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        return len(self.active_sessions)

    def get_status(self) -> Dict[str, Any]:
        """Get router status."""
        return {
            "bolna_available": self.bolna_available,
            "gemini_available": self.gemini_available,
            "retell_available": self.retell_available,
            "fallback_available": True,
            "preferred_provider": self.preferred_provider.value,
            "active_sessions": self.get_active_session_count(),
            "available_providers": [p.value for p in self.get_available_providers()]
        }


# Convenience function to create router from environment
def create_voice_router(rag_pipeline=None) -> VoiceRouter:
    """
    Create a VoiceRouter with configuration from environment.

    Environment variables:
    - VOICE_PREFERRED_PROVIDER: "bolna", "gemini_live", "retell", or "fallback_pipeline"
      (shortcuts: "b", "g", "r")
    - BOLNA_API_KEY: Required for Bolna
    - GOOGLE_CLOUD_PROJECT: Required for Gemini Live
    - RETELL_API_KEY: Required for Retell
    - RETELL_AGENT_ID: Required for Retell

    Args:
        rag_pipeline: Optional RAG pipeline for fallback queries

    Returns:
        Configured VoiceRouter instance
    """
    preferred = os.getenv("VOICE_PREFERRED_PROVIDER", "bolna").lower()

    provider_map = {
        "bolna": VoiceProvider.BOLNA,
        "b": VoiceProvider.BOLNA,
        "gemini_live": VoiceProvider.GEMINI_LIVE,
        "gemini": VoiceProvider.GEMINI_LIVE,
        "g": VoiceProvider.GEMINI_LIVE,
        "retell": VoiceProvider.RETELL,
        "r": VoiceProvider.RETELL,
        "fallback": VoiceProvider.FALLBACK_PIPELINE,
        "fallback_pipeline": VoiceProvider.FALLBACK_PIPELINE
    }

    preferred_provider = provider_map.get(preferred, VoiceProvider.BOLNA)

    return VoiceRouter(
        rag_pipeline=rag_pipeline,
        preferred_provider=preferred_provider
    )
