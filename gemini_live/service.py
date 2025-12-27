"""
Gemini Live Service - Full Implementation

Provides real-time voice conversation capabilities using Google's Gemini Live API.

Features:
- WebSocket connection to Gemini Live API via Vertex AI
- Real-time audio streaming (send/receive)
- Session management with resumption support
- RAG context injection for grounded responses
- Multi-language support (en-IN, hi-IN, mr-IN, ta-IN)
"""

import asyncio
import logging
from typing import Optional, Dict, Any, AsyncGenerator, List
from datetime import datetime

from google import genai
from google.genai import types

from .config import (
    GeminiLiveConfig,
    get_config,
    SUPPORTED_LANGUAGES,
    VOICE_OPTIONS,
    DEFAULT_VOICE,
    INPUT_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


class GeminiLiveError(Exception):
    """Exception raised for Gemini Live API errors."""
    pass


class GeminiLiveService:
    """
    Main service for Gemini Live API integration.

    Provides:
    - create_session(): Create new voice conversation session
    - inject_rag_context(): Add RAG context to session
    - Active session management
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model: Optional[str] = None,
        rag_pipeline: Optional[Any] = None,
        config: Optional[GeminiLiveConfig] = None
    ):
        """
        Initialize Gemini Live Service.

        Args:
            project_id: Google Cloud project ID (default from config)
            location: Vertex AI location (default from config)
            model: Gemini model ID (default from config)
            rag_pipeline: Reference to RAG pipeline for context injection
            config: Optional pre-loaded configuration
        """
        self.config = config or get_config()

        self.project_id = project_id or self.config.project_id
        self.location = location or self.config.location or "us-central1"
        self.model = model or self.config.model
        self.rag_pipeline = rag_pipeline

        # Initialize Google GenAI client
        self.client = self._create_client()

        # Active sessions (session_id -> GeminiLiveSession)
        self.active_sessions: Dict[str, "GeminiLiveSession"] = {}

        logger.info(
            f"GeminiLiveService initialized - "
            f"project={self._mask_project_id()}, location={self.location}, "
            f"model={self.model}"
        )

    def _mask_project_id(self) -> str:
        """Mask project ID for logging."""
        if not self.project_id:
            return "(not set)"
        if len(self.project_id) <= 8:
            return "***"
        return f"{self.project_id[:4]}...{self.project_id[-4:]}"

    def _create_client(self) -> Optional[genai.Client]:
        """Create Google GenAI client."""
        try:
            # Check if we have credentials
            if self.config.api_key:
                # Use API key authentication
                client = genai.Client(api_key=self.config.api_key)
                logger.info("GenAI client created with API key")
                return client
            elif self.project_id and not self.project_id.startswith("$"):
                # Use Vertex AI with ADC
                client = genai.Client(
                    vertexai=True,
                    project=self.project_id,
                    location=self.location
                )
                logger.info("GenAI client created with Vertex AI (ADC)")
                return client
            else:
                logger.warning(
                    "No valid credentials for GenAI client. "
                    "Set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT."
                )
                return None
        except Exception as e:
            logger.error(f"Failed to create GenAI client: {e}")
            return None

    def _build_system_instruction(
        self,
        language: str,
        custom_instruction: Optional[str] = None
    ) -> str:
        """
        Build the medical/palliative care system instruction.

        Args:
            language: Language code (e.g., "hi-IN")
            custom_instruction: Optional custom instruction to append

        Returns:
            Complete system instruction string
        """
        language_instructions = {
            "en-IN": "Respond in Indian English with a warm, empathetic tone.",
            "hi-IN": "हिंदी में जवाब दें। गर्मजोशी और सहानुभूति के साथ बात करें।",
            "mr-IN": "मराठीत उत्तर द्या. सहानुभूती आणि काळजी घेणारा स्वर वापरा.",
            "ta-IN": "தமிழில் பதிலளிக்கவும். அன்பான மற்றும் பரிவான தொனியில் பேசுங்கள்."
        }

        lang_instruction = language_instructions.get(
            language,
            language_instructions["en-IN"]
        )

        base_instruction = f"""You are a compassionate palliative care assistant helping patients and caregivers with healthcare queries.

IMPORTANT GUIDELINES:
1. Be warm, empathetic, and supportive in all interactions
2. Provide accurate medical information from the knowledge base when available
3. Always recommend consulting healthcare professionals for serious concerns
4. Use simple, clear language appropriate for patients and families
5. Be culturally sensitive to Indian healthcare contexts
6. If unsure, acknowledge uncertainty and suggest professional consultation
7. Keep responses concise and focused - this is a voice conversation

LANGUAGE INSTRUCTION: {lang_instruction}

SAFETY GUIDELINES:
- Never provide emergency medical advice
- For emergencies, direct users to call emergency services or visit the nearest hospital
- Do not diagnose conditions - only provide general health information
- Always encourage professional medical consultation for specific concerns

CONVERSATION STYLE:
- Speak naturally as in a phone conversation
- Use appropriate pauses
- Confirm understanding when needed
- Be patient with users who may be distressed
"""

        if custom_instruction:
            base_instruction += f"\n\nADDITIONAL INSTRUCTIONS:\n{custom_instruction}"

        return base_instruction

    async def create_session(
        self,
        session_id: str,
        language: str = "en-IN",
        voice: str = "Aoede",
        system_instruction: Optional[str] = None
    ) -> "GeminiLiveSession":
        """
        Create a new Gemini Live session.

        Args:
            session_id: Unique identifier for this session
            language: Language code (en-IN, hi-IN, mr-IN, ta-IN)
            voice: Voice name (Aoede, Puck, Kore, etc.)
            system_instruction: Custom system prompt to append

        Returns:
            GeminiLiveSession object

        Raises:
            GeminiLiveError: If session creation fails
        """
        if not self.client:
            raise GeminiLiveError(
                "GenAI client not initialized. "
                "Check credentials (GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT)."
            )

        # Validate language
        if language not in SUPPORTED_LANGUAGES:
            logger.warning(
                f"Unsupported language {language}, falling back to en-IN"
            )
            language = "en-IN"

        # Validate voice
        if voice not in VOICE_OPTIONS:
            logger.warning(
                f"Unknown voice {voice}, falling back to {DEFAULT_VOICE}"
            )
            voice = DEFAULT_VOICE

        # Build system instruction
        full_instruction = self._build_system_instruction(
            language, system_instruction
        )

        # Build configuration
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice
                    )
                ),
                language_code=language
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=full_instruction)]
            ),
        )

        # Add transcription if enabled
        if self.config.transcription_enabled:
            config.input_audio_transcription = types.AudioTranscriptionConfig()
            config.output_audio_transcription = types.AudioTranscriptionConfig()

        # Create session object
        session = GeminiLiveSession(
            service=self,
            session_id=session_id,
            config=config,
            language=language,
            voice=voice
        )

        # Store in active sessions
        self.active_sessions[session_id] = session

        logger.info(
            f"Created Gemini Live session: {session_id} "
            f"(language={language}, voice={voice})"
        )

        return session

    async def inject_rag_context(
        self,
        session: "GeminiLiveSession",
        query_context: str
    ) -> bool:
        """
        Inject RAG-retrieved context into an active session.

        Queries the RAG pipeline and sends relevant context to the
        Gemini session as a text message for grounding.

        Args:
            session: Active GeminiLiveSession
            query_context: Query to search RAG for relevant context

        Returns:
            True if context was injected, False otherwise
        """
        if not self.rag_pipeline:
            logger.debug("No RAG pipeline configured, skipping context injection")
            return False

        if not self.config.rag_context_enabled:
            logger.debug("RAG context injection disabled")
            return False

        if not session.is_active:
            logger.warning("Cannot inject context into inactive session")
            return False

        try:
            # Query RAG for relevant documents
            result = await self.rag_pipeline.query(
                question=query_context,
                conversation_id=session.session_id,
                user_id=session.session_id,
                top_k=self.config.rag_top_k
            )

            if result.get("status") != "success":
                logger.warning(f"RAG query failed: {result.get('error')}")
                return False

            context_used = result.get("context_used", "")
            if not context_used:
                logger.debug("No relevant RAG context found")
                return False

            # Format context message
            context_message = f"""[MEDICAL KNOWLEDGE BASE CONTEXT]
The following information from verified medical documents may be relevant to the user's query:

{context_used}

Use this information to provide accurate, evidence-based responses.
When using specific information from this context, mention the source.
[END CONTEXT]"""

            # Send to session
            await session.send_text(context_message)

            logger.info(
                f"Injected RAG context into session {session.session_id} "
                f"({len(context_used)} chars)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to inject RAG context: {e}")
            return False

    async def close_session(self, session_id: str) -> None:
        """
        Close and cleanup a session.

        Args:
            session_id: Session ID to close
        """
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            try:
                await session.disconnect()
            except Exception as e:
                logger.error(f"Error closing session {session_id}: {e}")
            finally:
                del self.active_sessions[session_id]
                logger.info(f"Closed session: {session_id}")

    def is_available(self) -> bool:
        """
        Check if Gemini Live service is available.

        Returns:
            True if service is configured and ready
        """
        return (
            self.config.enabled and
            self.client is not None
        )

    def get_status(self) -> Dict[str, Any]:
        """
        Get service status for health checks.

        Returns:
            Status dictionary
        """
        return {
            "service": "GeminiLiveService",
            "status": "ready" if self.is_available() else "not_ready",
            "enabled": self.config.enabled,
            "client_initialized": self.client is not None,
            "project_id": self._mask_project_id(),
            "model": self.model,
            "active_sessions": len(self.active_sessions),
            "supported_languages": self.config.supported_languages,
            "rag_enabled": self.config.rag_context_enabled,
            "fallback_enabled": self.config.fallback_enabled,
        }


class GeminiLiveSession:
    """
    Represents an active Gemini Live session.

    Handles:
    - Audio streaming (send/receive)
    - Text messaging
    - Session lifecycle
    - Transcription capture
    """

    # Special marker bytes for control signals
    TURN_COMPLETE = b"__TURN_COMPLETE__"
    INTERRUPTED = b"__INTERRUPTED__"

    def __init__(
        self,
        service: GeminiLiveService,
        session_id: str,
        config: types.LiveConnectConfig,
        language: str = "en-IN",
        voice: str = "Aoede"
    ):
        """
        Initialize session.

        Args:
            service: Parent GeminiLiveService
            session_id: Unique session identifier
            config: LiveConnectConfig for the session
            language: Session language
            voice: Voice name
        """
        self.service = service
        self.session_id = session_id
        self.config = config
        self.language = language
        self.voice = voice

        # Session state
        self.is_active = False
        self.is_connected = False
        self._session = None  # Actual genai session object

        # Buffers for transcription
        self.transcription_buffer: List[str] = []
        self.response_buffer: List[str] = []

        # Metadata
        self.created_at = datetime.now()
        self.last_activity = self.created_at

        # Session resumption
        self.resumption_handle: Optional[str] = None

        logger.debug(f"GeminiLiveSession created: {session_id}")

    async def connect(self) -> None:
        """
        Establish connection to Gemini Live API.

        Raises:
            GeminiLiveError: If connection fails
        """
        if self.is_connected:
            logger.warning(f"Session {self.session_id} already connected")
            return

        if not self.service.client:
            raise GeminiLiveError("Service client not initialized")

        try:
            # Connect using async context manager
            self._session = await self.service.client.aio.live.connect(
                model=self.service.model,
                config=self.config
            ).__aenter__()

            self.is_connected = True
            self.is_active = True
            self.last_activity = datetime.now()

            logger.info(f"Connected session: {self.session_id}")

        except Exception as e:
            self.is_connected = False
            self.is_active = False
            raise GeminiLiveError(f"Failed to connect: {e}") from e

    async def disconnect(self) -> None:
        """Close the session."""
        if not self.is_connected:
            return

        try:
            if self._session:
                await self._session.__aexit__(None, None, None)
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
        finally:
            self._session = None
            self.is_connected = False
            self.is_active = False
            logger.info(f"Disconnected session: {self.session_id}")

    async def send_audio(self, audio_chunk: bytes) -> None:
        """
        Send audio chunk to Gemini.

        Args:
            audio_chunk: Raw PCM audio (16kHz, 16-bit, mono, little-endian)

        Raises:
            GeminiLiveError: If session not connected or send fails
        """
        if not self.is_active or not self._session:
            raise GeminiLiveError("Session not connected")

        try:
            await self._session.send_realtime_input(
                audio=types.Blob(
                    data=audio_chunk,
                    mime_type=f"audio/pcm;rate={INPUT_SAMPLE_RATE}"
                )
            )
            self.last_activity = datetime.now()

        except Exception as e:
            raise GeminiLiveError(f"Failed to send audio: {e}") from e

    async def send_text(self, text: str) -> None:
        """
        Send text message to Gemini (for context injection or text input).

        Args:
            text: Text message to send

        Raises:
            GeminiLiveError: If session not connected or send fails
        """
        if not self.is_active or not self._session:
            raise GeminiLiveError("Session not connected")

        try:
            await self._session.send_client_content(
                turns=[types.Content(
                    role="user",
                    parts=[types.Part(text=text)]
                )],
                turn_complete=True
            )
            self.last_activity = datetime.now()
            logger.debug(f"Sent text to session {self.session_id}: {text[:50]}...")

        except Exception as e:
            raise GeminiLiveError(f"Failed to send text: {e}") from e

    async def receive_audio(self) -> AsyncGenerator[bytes, None]:
        """
        Receive audio responses from Gemini.

        Yields:
            Raw PCM audio chunks (24kHz, 16-bit, mono, little-endian)
            Special markers: TURN_COMPLETE, INTERRUPTED

        Raises:
            GeminiLiveError: If session not connected
        """
        if not self.is_active or not self._session:
            raise GeminiLiveError("Session not connected")

        try:
            async for message in self._session.receive():
                self.last_activity = datetime.now()

                # Handle server content
                if message.server_content:
                    content = message.server_content

                    # Model turn (audio output)
                    if content.model_turn:
                        for part in content.model_turn.parts:
                            if part.inline_data:
                                yield part.inline_data.data

                    # Input transcription (what user said)
                    if content.input_transcription:
                        text = content.input_transcription.text
                        if text:
                            self.transcription_buffer.append(text)
                            logger.debug(f"User transcription: {text}")

                    # Output transcription (what model said)
                    if content.output_transcription:
                        text = content.output_transcription.text
                        if text:
                            self.response_buffer.append(text)
                            logger.debug(f"Model transcription: {text}")

                    # Turn complete
                    if content.turn_complete:
                        yield self.TURN_COMPLETE

                    # Interrupted (user barged in)
                    if content.interrupted:
                        logger.debug(f"Session {self.session_id} interrupted")
                        yield self.INTERRUPTED

                # Handle go_away (session ending)
                if message.go_away:
                    logger.warning(
                        f"Session {self.session_id} received go_away, "
                        f"time_left={message.go_away.time_left}"
                    )

                # Handle session resumption update
                if message.session_resumption_update:
                    update = message.session_resumption_update
                    if update.resumable and update.new_handle:
                        self.resumption_handle = update.new_handle
                        logger.debug(
                            f"Session {self.session_id} resumption handle updated"
                        )

        except Exception as e:
            logger.error(f"Error receiving audio: {e}")
            raise GeminiLiveError(f"Failed to receive audio: {e}") from e

    def get_transcription(self, clear: bool = True) -> str:
        """
        Get accumulated user transcription.

        Args:
            clear: Whether to clear the buffer after reading

        Returns:
            Concatenated transcription text
        """
        text = " ".join(self.transcription_buffer)
        if clear:
            self.transcription_buffer.clear()
        return text

    def get_response_transcription(self, clear: bool = True) -> str:
        """
        Get accumulated model response transcription.

        Args:
            clear: Whether to clear the buffer after reading

        Returns:
            Concatenated response text
        """
        text = " ".join(self.response_buffer)
        if clear:
            self.response_buffer.clear()
        return text

    def get_status(self) -> Dict[str, Any]:
        """Get session status."""
        return {
            "session_id": self.session_id,
            "language": self.language,
            "voice": self.voice,
            "is_active": self.is_active,
            "is_connected": self.is_connected,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "has_resumption_handle": self.resumption_handle is not None,
            "transcription_buffer_size": len(self.transcription_buffer),
            "response_buffer_size": len(self.response_buffer),
        }
