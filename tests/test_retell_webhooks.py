"""
Tests for Retell.AI Webhook Handler

Tests webhook event processing and call record management.
"""

import os
import sys
import pytest
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRetellWebhookHandler:
    """Test RetellWebhookHandler class."""

    def test_import_webhook_module(self):
        """Test that webhook module can be imported."""
        from retell_integration.webhooks import (
            RetellWebhookHandler,
            RetellCallRecord,
        )
        assert RetellWebhookHandler is not None
        assert RetellCallRecord is not None

    def test_handler_initialization(self):
        """Test webhook handler initialization."""
        from retell_integration.webhooks import RetellWebhookHandler

        handler = RetellWebhookHandler()
        assert len(handler.active_calls) == 0
        assert len(handler.completed_calls) == 0

    def test_handler_custom_max_calls(self):
        """Test handler with custom max completed calls."""
        from retell_integration.webhooks import RetellWebhookHandler

        handler = RetellWebhookHandler(max_completed_calls=100)
        assert handler.max_completed_calls == 100


class TestRetellCallRecord:
    """Test RetellCallRecord dataclass."""

    def test_call_record_creation(self):
        """Test creating a call record."""
        from retell_integration.webhooks import RetellCallRecord

        record = RetellCallRecord(
            call_id="call_123",
            agent_id="agent_456",
            call_type="inbound",
            from_number="+919876543210",
            to_number="+1234567890",
            started_at=datetime.now()
        )

        assert record.call_id == "call_123"
        assert record.call_type == "inbound"
        assert record.status == "in_progress"
        assert record.duration_ms == 0

    def test_call_record_duration_seconds(self):
        """Test duration_seconds property."""
        from retell_integration.webhooks import RetellCallRecord

        record = RetellCallRecord(
            call_id="call_123",
            agent_id="agent_456",
            call_type="inbound",
            from_number="+919876543210",
            to_number="+1234567890",
            started_at=datetime.now(),
            duration_ms=65000  # 65 seconds
        )

        assert record.duration_seconds == 65

    def test_call_record_to_dict(self):
        """Test to_dict() method."""
        from retell_integration.webhooks import RetellCallRecord

        record = RetellCallRecord(
            call_id="call_123",
            agent_id="agent_456",
            call_type="inbound",
            from_number="+919876543210",
            to_number="+1234567890",
            started_at=datetime.now(),
            transcript="Hello, how can I help?"
        )

        result = record.to_dict()

        assert isinstance(result, dict)
        assert result["call_id"] == "call_123"
        assert result["call_type"] == "inbound"
        assert result["transcript"] == "Hello, how can I help?"
        assert "started_at" in result


class TestWebhookEvents:
    """Test webhook event handling."""

    @pytest.fixture
    def handler(self):
        """Create a webhook handler."""
        from retell_integration.webhooks import RetellWebhookHandler
        return RetellWebhookHandler()

    @pytest.mark.asyncio
    async def test_handle_call_started(self, handler):
        """Test handling call_started event."""
        event_data = {
            "event": "call_started",
            "call": {
                "call_id": "call_001",
                "agent_id": "agent_123",
                "call_type": "inbound",
                "from_number": "+919876543210",
                "to_number": "+1234567890"
            }
        }

        result = await handler.handle_event(event_data)

        assert result["status"] == "recorded"
        assert result["event"] == "call_started"
        assert "call_001" in handler.active_calls

    @pytest.mark.asyncio
    async def test_handle_call_ended(self, handler):
        """Test handling call_ended event."""
        # First start a call
        start_event = {
            "event": "call_started",
            "call": {
                "call_id": "call_002",
                "agent_id": "agent_123",
                "call_type": "inbound",
                "from_number": "+919876543210",
                "to_number": "+1234567890"
            }
        }
        await handler.handle_event(start_event)

        # Then end it
        end_event = {
            "event": "call_ended",
            "call": {
                "call_id": "call_002",
                "duration_ms": 120000,
                "transcript": "User said hello",
                "call_successful": True,
                "disconnection_reason": "user_hangup"
            }
        }

        result = await handler.handle_event(end_event)

        assert result["status"] == "recorded"
        assert result["event"] == "call_ended"
        assert result["duration_seconds"] == 120
        assert "call_002" not in handler.active_calls
        assert "call_002" in handler.completed_calls

    @pytest.mark.asyncio
    async def test_handle_call_ended_without_start(self, handler):
        """Test handling call_ended without prior call_started."""
        event_data = {
            "event": "call_ended",
            "call": {
                "call_id": "call_003",
                "agent_id": "agent_123",
                "call_type": "inbound",
                "from_number": "+919876543210",
                "to_number": "+1234567890",
                "duration_ms": 60000,
                "call_successful": True
            }
        }

        result = await handler.handle_event(event_data)

        # Should still record successfully
        assert result["status"] == "recorded"
        assert "call_003" in handler.completed_calls

    @pytest.mark.asyncio
    async def test_handle_call_analyzed(self, handler):
        """Test handling call_analyzed event."""
        # First complete a call
        end_event = {
            "event": "call_ended",
            "call": {
                "call_id": "call_004",
                "agent_id": "agent_123",
                "call_type": "inbound",
                "from_number": "+919876543210",
                "to_number": "+1234567890",
                "duration_ms": 180000,
                "call_successful": True
            }
        }
        await handler.handle_event(end_event)

        # Then analyze it
        analyze_event = {
            "event": "call_analyzed",
            "call": {
                "call_id": "call_004",
                "call_analysis": {
                    "call_summary": "User asked about pain management",
                    "user_sentiment": "Neutral",
                    "urgency_level": "low"
                }
            }
        }

        result = await handler.handle_event(analyze_event)

        assert result["status"] == "recorded"
        assert result["event"] == "call_analyzed"

        # Check analysis was stored
        call = handler.completed_calls["call_004"]
        assert call.call_summary == "User asked about pain management"
        assert call.user_sentiment == "Neutral"

    @pytest.mark.asyncio
    async def test_handle_unknown_event(self, handler):
        """Test handling unknown event type."""
        event_data = {
            "event": "unknown_event",
            "call": {"call_id": "call_005"}
        }

        result = await handler.handle_event(event_data)

        assert result["status"] == "ignored"
        assert result["reason"] == "unknown_event"


class TestWebhookHandlerMethods:
    """Test webhook handler utility methods."""

    @pytest.fixture
    def handler_with_calls(self):
        """Create handler with some completed calls."""
        from retell_integration.webhooks import RetellWebhookHandler, RetellCallRecord

        handler = RetellWebhookHandler()

        # Add some completed calls
        for i in range(5):
            record = RetellCallRecord(
                call_id=f"call_{i:03d}",
                agent_id="agent_123",
                call_type="inbound",
                from_number=f"+91987654321{i}",
                to_number="+1234567890",
                started_at=datetime.now(),
                duration_ms=(i + 1) * 60000,
                call_successful=True,
                status="completed"
            )
            record.ended_at = datetime.now()
            handler.completed_calls[record.call_id] = record

        return handler

    def test_get_call_stats(self, handler_with_calls):
        """Test get_call_stats() method."""
        stats = handler_with_calls.get_call_stats()

        assert stats["active_calls"] == 0
        assert stats["completed_calls"] == 5
        assert stats["successful_calls"] == 5
        assert stats["failed_calls"] == 0
        assert "total_duration_seconds" in stats
        assert "average_duration_seconds" in stats

    def test_get_recent_calls(self, handler_with_calls):
        """Test get_recent_calls() method."""
        calls = handler_with_calls.get_recent_calls(limit=3)

        assert len(calls) == 3
        assert all(isinstance(c, dict) for c in calls)

    def test_get_session(self, handler_with_calls):
        """Test get_call() method."""
        call = handler_with_calls.get_call("call_001")
        assert call is not None
        assert call.call_id == "call_001"

        # Non-existent call
        missing = handler_with_calls.get_call("non_existent")
        assert missing is None

    def test_get_active_call(self, handler_with_calls):
        """Test get_active_call() method."""
        # No active calls
        active = handler_with_calls.get_active_call("call_001")
        assert active is None

    def test_get_completed_call(self, handler_with_calls):
        """Test get_completed_call() method."""
        completed = handler_with_calls.get_completed_call("call_001")
        assert completed is not None
        assert completed.call_id == "call_001"


class TestCustomEventHandlers:
    """Test custom event handler registration."""

    @pytest.mark.asyncio
    async def test_register_custom_handler(self):
        """Test registering a custom event handler."""
        from retell_integration.webhooks import RetellWebhookHandler

        handler = RetellWebhookHandler()
        custom_called = {"value": False}

        async def custom_callback(event_data):
            custom_called["value"] = True

        handler.register_handler("call_started", custom_callback)

        event_data = {
            "event": "call_started",
            "call": {
                "call_id": "call_custom",
                "agent_id": "agent_123",
                "call_type": "inbound",
                "from_number": "+919876543210",
                "to_number": "+1234567890"
            }
        }

        await handler.handle_event(event_data)

        assert custom_called["value"] is True


class TestCallLimitEnforcement:
    """Test completed calls limit enforcement."""

    @pytest.mark.asyncio
    async def test_enforce_completed_limit(self):
        """Test that old calls are removed when limit is exceeded."""
        from retell_integration.webhooks import RetellWebhookHandler

        # Create handler with small limit
        handler = RetellWebhookHandler(max_completed_calls=10)

        # Add more calls than the limit
        for i in range(15):
            event = {
                "event": "call_ended",
                "call": {
                    "call_id": f"call_{i:03d}",
                    "agent_id": "agent_123",
                    "call_type": "inbound",
                    "from_number": "+919876543210",
                    "to_number": "+1234567890",
                    "duration_ms": 60000,
                    "call_successful": True
                }
            }
            await handler.handle_event(event)

        # Should have enforced limit (keeping ~90% = 9 calls)
        assert len(handler.completed_calls) <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
