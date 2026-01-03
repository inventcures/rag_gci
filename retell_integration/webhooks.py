"""
Webhook Handler for Retell Call Events

This module handles webhook events from Retell.AI, including:
- call_started: Call initiated
- call_ended: Call completed with transcript
- call_analyzed: Post-call analysis ready

Documentation: https://docs.retellai.com/api-references/webhooks
"""

import logging
from typing import Dict, Any, Optional, Callable, Awaitable, List
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetellCallRecord:
    """Record of a Retell call."""
    call_id: str
    agent_id: str
    call_type: str  # "inbound" or "outbound" or "web"
    from_number: str
    to_number: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_ms: int = 0
    transcript: str = ""
    transcript_object: List[Dict] = field(default_factory=list)
    call_summary: str = ""
    user_sentiment: str = ""
    call_successful: bool = True
    disconnection_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    analysis_data: Dict[str, Any] = field(default_factory=dict)
    status: str = "in_progress"

    @property
    def duration_seconds(self) -> int:
        """Get duration in seconds."""
        return self.duration_ms // 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "call_id": self.call_id,
            "agent_id": self.agent_id,
            "call_type": self.call_type,
            "from_number": self.from_number,
            "to_number": self.to_number,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds,
            "transcript": self.transcript,
            "call_summary": self.call_summary,
            "user_sentiment": self.user_sentiment,
            "call_successful": self.call_successful,
            "disconnection_reason": self.disconnection_reason,
            "metadata": self.metadata,
            "analysis_data": self.analysis_data,
            "status": self.status
        }


class RetellWebhookHandler:
    """
    Handler for Retell webhook events.

    Processes:
    - call_started: New call initiated
    - call_ended: Call completed with transcript
    - call_analyzed: Post-call analysis available

    Usage:
        handler = RetellWebhookHandler()

        # Register custom handler for specific events
        handler.register_handler("call_ended", my_callback)

        # Process webhook event
        result = await handler.handle_event(event_data)

        # Get statistics
        stats = handler.get_call_stats()
    """

    def __init__(self, max_completed_calls: int = 1000):
        """
        Initialize webhook handler.

        Args:
            max_completed_calls: Maximum completed calls to keep in memory
        """
        self.active_calls: Dict[str, RetellCallRecord] = {}
        self.completed_calls: Dict[str, RetellCallRecord] = {}
        self.event_handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[None]]] = {}
        self.max_completed_calls = max_completed_calls

    def register_handler(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """
        Register a custom handler for an event type.

        Args:
            event_type: Event type (call_started, call_ended, call_analyzed)
            handler: Async callback function
        """
        self.event_handlers[event_type] = handler
        logger.info(f"Registered handler for Retell event: {event_type}")

    async def handle_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming webhook event from Retell.

        Args:
            event_data: Webhook payload from Retell

        Returns:
            Processing result dictionary
        """
        event_type = event_data.get("event", "unknown")
        call = event_data.get("call", {})
        call_id = call.get("call_id", "unknown")

        logger.info(f"Processing Retell webhook: {event_type} for call {call_id}")

        # Call custom handler if registered
        if event_type in self.event_handlers:
            try:
                await self.event_handlers[event_type](event_data)
            except Exception as e:
                logger.error(f"Custom handler error for {event_type}: {e}")

        # Process standard events
        if event_type == "call_started":
            return await self._handle_call_started(call)

        elif event_type == "call_ended":
            return await self._handle_call_ended(call)

        elif event_type == "call_analyzed":
            return await self._handle_call_analyzed(call)

        else:
            logger.warning(f"Unknown Retell event type: {event_type}")
            return {
                "status": "ignored",
                "reason": "unknown_event",
                "event_type": event_type
            }

    async def _handle_call_started(self, call: Dict[str, Any]) -> Dict[str, Any]:
        """Handle call_started event."""
        call_id = call.get("call_id")

        record = RetellCallRecord(
            call_id=call_id,
            agent_id=call.get("agent_id", ""),
            call_type=call.get("call_type", "inbound"),
            from_number=call.get("from_number", ""),
            to_number=call.get("to_number", ""),
            started_at=datetime.now(),
            status="in_progress",
            metadata=call.get("metadata", {})
        )

        self.active_calls[call_id] = record

        logger.info(
            f"Retell call started: {call_id} "
            f"({record.call_type}) from {record.from_number}"
        )

        return {
            "status": "recorded",
            "call_id": call_id,
            "event": "call_started"
        }

    async def _handle_call_ended(self, call: Dict[str, Any]) -> Dict[str, Any]:
        """Handle call_ended event."""
        call_id = call.get("call_id")

        # Get existing record or create new one
        if call_id in self.active_calls:
            record = self.active_calls.pop(call_id)
        else:
            # Create new record if call_started wasn't received
            start_ts = call.get("start_timestamp")
            started_at = datetime.now()
            if start_ts:
                try:
                    started_at = datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass

            record = RetellCallRecord(
                call_id=call_id,
                agent_id=call.get("agent_id", ""),
                call_type=call.get("call_type", "inbound"),
                from_number=call.get("from_number", ""),
                to_number=call.get("to_number", ""),
                started_at=started_at
            )

        # Update with end data
        record.ended_at = datetime.now()
        record.duration_ms = call.get("duration_ms", 0)
        record.transcript = call.get("transcript", "")
        record.transcript_object = call.get("transcript_object", [])
        record.disconnection_reason = call.get("disconnection_reason", "")
        record.call_successful = call.get("call_successful", True)
        record.status = "completed"

        # Store completed call (with limit)
        self.completed_calls[call_id] = record
        self._enforce_completed_limit()

        logger.info(
            f"Retell call ended: {call_id}, "
            f"duration: {record.duration_seconds}s, "
            f"successful: {record.call_successful}, "
            f"reason: {record.disconnection_reason}"
        )

        return {
            "status": "recorded",
            "call_id": call_id,
            "event": "call_ended",
            "duration_seconds": record.duration_seconds
        }

    async def _handle_call_analyzed(self, call: Dict[str, Any]) -> Dict[str, Any]:
        """Handle call_analyzed event (post-call analysis)."""
        call_id = call.get("call_id")

        if call_id in self.completed_calls:
            record = self.completed_calls[call_id]
            record.call_summary = call.get("call_analysis", {}).get("call_summary", "")
            record.user_sentiment = call.get("call_analysis", {}).get("user_sentiment", "")
            record.analysis_data = call.get("call_analysis", {})

            logger.info(
                f"Retell call analyzed: {call_id}, "
                f"sentiment: {record.user_sentiment}"
            )

            # Check for follow-up needed
            if record.analysis_data.get("follow_up_needed"):
                logger.info(f"Follow-up recommended for call {call_id}")

            # Check for high urgency
            urgency = record.analysis_data.get("urgency_level", "low")
            if urgency in ("high", "emergency"):
                logger.warning(
                    f"HIGH URGENCY call {call_id}: "
                    f"{record.analysis_data.get('user_concern', 'Unknown concern')}"
                )

        return {
            "status": "recorded",
            "call_id": call_id,
            "event": "call_analyzed"
        }

    def _enforce_completed_limit(self) -> None:
        """Remove oldest completed calls if limit exceeded."""
        if len(self.completed_calls) > self.max_completed_calls:
            # Sort by ended_at and remove oldest
            sorted_calls = sorted(
                self.completed_calls.items(),
                key=lambda x: x[1].ended_at or x[1].started_at
            )
            # Remove oldest 10%
            remove_count = len(self.completed_calls) - int(self.max_completed_calls * 0.9)
            for call_id, _ in sorted_calls[:remove_count]:
                del self.completed_calls[call_id]
            logger.info(f"Cleaned up {remove_count} old call records")

    def get_active_call(self, call_id: str) -> Optional[RetellCallRecord]:
        """Get an active call record."""
        return self.active_calls.get(call_id)

    def get_completed_call(self, call_id: str) -> Optional[RetellCallRecord]:
        """Get a completed call record."""
        return self.completed_calls.get(call_id)

    def get_call(self, call_id: str) -> Optional[RetellCallRecord]:
        """Get a call record (active or completed)."""
        return self.active_calls.get(call_id) or self.completed_calls.get(call_id)

    def get_call_stats(self) -> Dict[str, Any]:
        """Get call statistics."""
        total_completed = len(self.completed_calls)
        total_duration = sum(c.duration_ms for c in self.completed_calls.values())

        # Count successful vs failed
        successful = sum(1 for c in self.completed_calls.values() if c.call_successful)

        # Sentiment distribution
        sentiments: Dict[str, int] = {}
        for call in self.completed_calls.values():
            sentiment = call.user_sentiment or "unknown"
            sentiments[sentiment] = sentiments.get(sentiment, 0) + 1

        # Call type distribution
        call_types: Dict[str, int] = {}
        for call in self.completed_calls.values():
            call_type = call.call_type
            call_types[call_type] = call_types.get(call_type, 0) + 1

        # Urgency distribution
        urgency_levels: Dict[str, int] = {}
        for call in self.completed_calls.values():
            urgency = call.analysis_data.get("urgency_level", "unknown")
            urgency_levels[urgency] = urgency_levels.get(urgency, 0) + 1

        return {
            "active_calls": len(self.active_calls),
            "completed_calls": total_completed,
            "successful_calls": successful,
            "failed_calls": total_completed - successful,
            "total_duration_ms": total_duration,
            "total_duration_seconds": total_duration // 1000,
            "average_duration_seconds": (total_duration // 1000) // total_completed if total_completed > 0 else 0,
            "sentiment_distribution": sentiments,
            "call_type_distribution": call_types,
            "urgency_distribution": urgency_levels
        }

    def get_recent_calls(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent completed calls."""
        calls = list(self.completed_calls.values())
        calls.sort(key=lambda x: x.ended_at or x.started_at, reverse=True)
        return [c.to_dict() for c in calls[:limit]]

    def get_high_urgency_calls(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent high urgency calls."""
        high_urgency = [
            c for c in self.completed_calls.values()
            if c.analysis_data.get("urgency_level") in ("high", "emergency")
        ]
        high_urgency.sort(key=lambda x: x.ended_at or x.started_at, reverse=True)
        return [c.to_dict() for c in high_urgency[:limit]]
