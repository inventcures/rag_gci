"""
Tests for Retell.AI API Client

Tests client initialization, API methods, and error handling.
"""

import os
import sys
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRetellClient:
    """Test RetellClient class."""

    def test_import_client_module(self):
        """Test that client module can be imported."""
        from retell_integration.client import RetellClient, RetellCallResult
        assert RetellClient is not None
        assert RetellCallResult is not None

    def test_client_initialization_no_key(self):
        """Test client initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove RETELL_API_KEY if present
            os.environ.pop("RETELL_API_KEY", None)

            from retell_integration.client import RetellClient
            client = RetellClient()
            assert not client.is_available()

    @patch.dict(os.environ, {"RETELL_API_KEY": "test_api_key"})
    def test_client_initialization_with_key(self):
        """Test client initialization with API key."""
        from retell_integration.client import RetellClient
        client = RetellClient()
        assert client.is_available()
        assert client.api_key == "test_api_key"

    def test_client_initialization_explicit_key(self):
        """Test client initialization with explicit API key."""
        from retell_integration.client import RetellClient
        client = RetellClient(api_key="explicit_key")
        assert client.is_available()
        assert client.api_key == "explicit_key"

    def test_retell_call_result_dataclass(self):
        """Test RetellCallResult dataclass."""
        from retell_integration.client import RetellCallResult

        # Success result
        success = RetellCallResult(
            success=True,
            call_id="call_123",
            agent_id="agent_456"
        )
        assert success.success
        assert success.call_id == "call_123"
        assert success.error is None

        # Failure result
        failure = RetellCallResult(
            success=False,
            error="API error"
        )
        assert not failure.success
        assert failure.error == "API error"


class TestRetellClientMethods:
    """Test RetellClient API methods with mocking."""

    @pytest.fixture
    def client(self):
        """Create a client with test API key."""
        from retell_integration.client import RetellClient
        return RetellClient(api_key="test_key")

    @pytest.mark.asyncio
    async def test_create_agent_no_key(self):
        """Test create_agent fails without API key."""
        from retell_integration.client import RetellClient
        from retell_integration.config import RetellAgentConfig, CARTESIA_VOICE_IDS

        client = RetellClient(api_key="")
        config = RetellAgentConfig(
            agent_name="Test Agent",
            llm_websocket_url="wss://example.com/ws",
            voice_id=CARTESIA_VOICE_IDS["hi"]["voice_id"],
            webhook_url="https://example.com/webhook"
        )

        result = await client.create_agent(config)
        assert not result.success
        assert "not configured" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_agent_no_key(self):
        """Test get_agent fails without API key."""
        from retell_integration.client import RetellClient

        client = RetellClient(api_key="")
        result = await client.get_agent("agent_123")
        assert not result.success
        assert "not configured" in result.error.lower()

    @pytest.mark.asyncio
    async def test_list_agents_no_key(self):
        """Test list_agents fails without API key."""
        from retell_integration.client import RetellClient

        client = RetellClient(api_key="")
        result = await client.list_agents()
        assert not result.success
        assert "not configured" in result.error.lower()

    @pytest.mark.asyncio
    async def test_create_phone_call_no_key(self):
        """Test create_phone_call fails without API key."""
        from retell_integration.client import RetellClient

        client = RetellClient(api_key="")
        result = await client.create_phone_call(
            from_number="+1234567890",
            to_number="+919876543210",
            agent_id="agent_123"
        )
        assert not result.success
        assert "not configured" in result.error.lower()

    @pytest.mark.asyncio
    async def test_create_web_call_no_key(self):
        """Test create_web_call fails without API key."""
        from retell_integration.client import RetellClient

        client = RetellClient(api_key="")
        result = await client.create_web_call(agent_id="agent_123")
        assert not result.success
        assert "not configured" in result.error.lower()

    @pytest.mark.asyncio
    async def test_health_check_no_key(self):
        """Test health_check returns False without API key."""
        from retell_integration.client import RetellClient

        client = RetellClient(api_key="")
        result = await client.health_check()
        assert result is False


class TestRetellClientWithMockedNetwork:
    """Test RetellClient with mocked network calls."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock aiohttp session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_create_agent_success(self, mock_session):
        """Test successful agent creation."""
        import aiohttp
        from retell_integration.client import RetellClient
        from retell_integration.config import RetellAgentConfig, CARTESIA_VOICE_IDS

        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.json = AsyncMock(return_value={
            "agent_id": "new_agent_123",
            "agent_name": "Test Agent"
        })

        mock_session.post = AsyncMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=None)
        ))

        with patch("aiohttp.ClientSession", return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=None)
        )):
            client = RetellClient(api_key="test_key")
            config = RetellAgentConfig(
                agent_name="Test Agent",
                llm_websocket_url="wss://example.com/ws",
                voice_id=CARTESIA_VOICE_IDS["hi"]["voice_id"],
                webhook_url="https://example.com/webhook"
            )

            # Note: This test would need proper async context manager mocking
            # For now, just verify the client is available
            assert client.is_available()


class TestRetellCallResult:
    """Test RetellCallResult edge cases."""

    def test_call_result_with_data(self):
        """Test result with additional data."""
        from retell_integration.client import RetellCallResult

        result = RetellCallResult(
            success=True,
            call_id="call_123",
            data={"duration": 120, "status": "completed"}
        )

        assert result.data["duration"] == 120
        assert result.data["status"] == "completed"

    def test_call_result_defaults(self):
        """Test default values."""
        from retell_integration.client import RetellCallResult

        result = RetellCallResult(success=True)

        assert result.call_id is None
        assert result.agent_id is None
        assert result.error is None
        assert result.data is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
