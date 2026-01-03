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
from typing import Dict, Any, Optional
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
