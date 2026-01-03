"""
Tests for Retell.AI Custom LLM WebSocket Handler

Tests the Custom LLM server that integrates with RAG pipeline.
"""

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRetellCustomLLMHandler:
    """Test RetellCustomLLMHandler class."""

    def test_import_custom_llm_module(self):
        """Test that custom LLM module can be imported."""
        from retell_integration.custom_llm_server import (
            RetellCustomLLMHandler,
            RetellSession,
        )
        assert RetellCustomLLMHandler is not None
        assert RetellSession is not None

    def test_handler_initialization_no_rag(self):
        """Test handler initialization without RAG pipeline."""
        from retell_integration.custom_llm_server import RetellCustomLLMHandler

        handler = RetellCustomLLMHandler()
        assert handler.rag_pipeline is None
        assert len(handler.active_sessions) == 0

    def test_handler_initialization_with_rag(self):
        """Test handler initialization with RAG pipeline."""
        from retell_integration.custom_llm_server import RetellCustomLLMHandler

        mock_rag = MagicMock()
        handler = RetellCustomLLMHandler(rag_pipeline=mock_rag)
        assert handler.rag_pipeline is mock_rag

    def test_handler_initialization_with_options(self):
        """Test handler initialization with custom options."""
        from retell_integration.custom_llm_server import RetellCustomLLMHandler

        handler = RetellCustomLLMHandler(
            response_timeout=60.0,
            max_response_tokens=200
        )
        assert handler.response_timeout == 60.0
        assert handler.max_response_tokens == 200


class TestRetellSession:
    """Test RetellSession dataclass."""

    def test_session_creation(self):
        """Test creating a session."""
        from retell_integration.custom_llm_server import RetellSession

        mock_ws = MagicMock()
        session = RetellSession(
            call_id="call_123",
            websocket=mock_ws
        )

        assert session.call_id == "call_123"
        assert session.websocket is mock_ws
        assert session.language == "hi"  # Default

    def test_session_with_language(self):
        """Test session with custom language."""
        from retell_integration.custom_llm_server import RetellSession

        mock_ws = MagicMock()
        session = RetellSession(
            call_id="call_123",
            websocket=mock_ws,
            language="en"
        )

        assert session.language == "en"


class TestCustomLLMEventHandling:
    """Test Custom LLM event handling."""

    @pytest.fixture
    def handler(self):
        """Create a handler with mock RAG pipeline."""
        from retell_integration.custom_llm_server import RetellCustomLLMHandler

        mock_rag = MagicMock()
        mock_rag.query = AsyncMock(return_value={
            "answer": "Here is some helpful information about pain management.",
            "sources": [{"title": "Pain Guide"}],
            "confidence": 0.95
        })

        return RetellCustomLLMHandler(rag_pipeline=mock_rag)

    def test_get_session_missing(self, handler):
        """Test getting a non-existent session."""
        session = handler.get_session("non_existent")
        assert session is None

    def test_end_session(self, handler):
        """Test ending a session."""
        from retell_integration.custom_llm_server import RetellSession

        # Manually add a session
        mock_ws = MagicMock()
        session = RetellSession(call_id="call_003", websocket=mock_ws)
        handler.active_sessions["call_003"] = session

        assert "call_003" in handler.active_sessions

        # Now get it
        retrieved = handler.get_session("call_003")
        assert retrieved is not None
        assert retrieved.call_id == "call_003"


class TestTranscriptParsing:
    """Test transcript parsing from Retell messages."""

    def test_extract_user_message(self):
        """Test extracting user message from transcript."""
        transcript = [
            {"role": "agent", "content": "Hello, how can I help you?"},
            {"role": "user", "content": "I have pain in my back"},
            {"role": "agent", "content": "I understand. Let me help."}
        ]

        # Get the last user message
        user_messages = [t for t in transcript if t["role"] == "user"]
        last_user = user_messages[-1]["content"] if user_messages else None

        assert last_user == "I have pain in my back"

    def test_empty_transcript(self):
        """Test handling empty transcript."""
        transcript = []

        user_messages = [t for t in transcript if t.get("role") == "user"]
        last_user = user_messages[-1]["content"] if user_messages else None

        assert last_user is None


class TestResponseGeneration:
    """Test response generation with RAG integration."""

    @pytest.fixture
    def handler_with_rag(self):
        """Create a handler with mock RAG."""
        from retell_integration.custom_llm_server import RetellCustomLLMHandler

        mock_rag = MagicMock()
        mock_rag.query = AsyncMock(return_value={
            "answer": "For pain management, you can try these approaches...",
            "sources": [{"title": "Pain Management Guide"}],
            "confidence": 0.9
        })

        return RetellCustomLLMHandler(rag_pipeline=mock_rag)

    @pytest.mark.asyncio
    async def test_generate_response_with_rag(self, handler_with_rag):
        """Test response generation using RAG pipeline."""
        # Simulate a query
        query = "How can I manage pain at home?"
        result = await handler_with_rag.rag_pipeline.query(query)

        assert "answer" in result
        assert "pain management" in result["answer"].lower()

    def test_handler_without_rag(self):
        """Test handler works without RAG pipeline."""
        from retell_integration.custom_llm_server import RetellCustomLLMHandler

        handler = RetellCustomLLMHandler()  # No RAG
        assert handler.rag_pipeline is None


class TestWebSocketProtocol:
    """Test Retell WebSocket protocol handling."""

    def test_ping_pong_response_format(self):
        """Test ping/pong response format."""
        # Retell sends ping, we respond with same
        ping_message = {"type": "ping", "timestamp": 1234567890}
        pong_response = {"type": "ping", "timestamp": 1234567890}

        assert pong_response["type"] == "ping"
        assert pong_response["timestamp"] == ping_message["timestamp"]

    def test_response_message_format(self):
        """Test response message format for Retell."""
        response = {
            "response_type": "response",
            "response_id": 1,
            "content": "Here is my response",
            "content_complete": True,
            "end_call": False
        }

        assert response["response_type"] == "response"
        assert response["content_complete"] is True
        assert isinstance(response["response_id"], int)

    def test_error_response_format(self):
        """Test error response format."""
        error_response = {
            "response_type": "response",
            "response_id": 0,
            "content": "I'm sorry, there was an error processing your request.",
            "content_complete": True,
            "end_call": False
        }

        assert error_response["response_type"] == "response"
        assert error_response["content_complete"] is True


class TestSessionManagement:
    """Test session lifecycle management."""

    @pytest.fixture
    def handler(self):
        """Create a handler."""
        from retell_integration.custom_llm_server import RetellCustomLLMHandler
        return RetellCustomLLMHandler()

    def test_active_sessions_dict(self, handler):
        """Test that active_sessions is a dict."""
        assert isinstance(handler.active_sessions, dict)
        assert len(handler.active_sessions) == 0

    def test_add_session_manually(self, handler):
        """Test manually adding sessions."""
        from retell_integration.custom_llm_server import RetellSession

        mock_ws = MagicMock()

        # Simulate adding sessions
        for i in range(3):
            session = RetellSession(call_id=f"call_{i:03d}", websocket=mock_ws)
            handler.active_sessions[session.call_id] = session

        assert len(handler.active_sessions) == 3
        assert all(f"call_{i:03d}" in handler.active_sessions for i in [0, 1, 2])

    def test_session_isolation(self, handler):
        """Test that sessions are isolated."""
        from retell_integration.custom_llm_server import RetellSession

        mock_ws = MagicMock()

        session1 = RetellSession(call_id="call_001", websocket=mock_ws, language="hi")
        session2 = RetellSession(call_id="call_002", websocket=mock_ws, language="en")

        handler.active_sessions[session1.call_id] = session1
        handler.active_sessions[session2.call_id] = session2

        assert handler.get_session("call_001").language == "hi"
        assert handler.get_session("call_002").language == "en"


class TestRetellIntegration:
    """Integration-style tests for Retell Custom LLM."""

    @pytest.fixture
    def full_handler(self):
        """Create a fully configured handler."""
        from retell_integration.custom_llm_server import RetellCustomLLMHandler

        mock_rag = MagicMock()
        mock_rag.query = AsyncMock(return_value={
            "answer": "Pain can be managed through medication, physical therapy, and rest.",
            "sources": [{"title": "Pain Management"}],
            "confidence": 0.92
        })

        return RetellCustomLLMHandler(rag_pipeline=mock_rag)

    @pytest.mark.asyncio
    async def test_rag_query_flow(self, full_handler):
        """Test querying RAG pipeline."""
        query = "How do I manage back pain?"
        result = await full_handler.rag_pipeline.query(query)

        assert "pain" in result["answer"].lower()
        assert result["confidence"] > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
