"""
Webhook Handler for Bolna Call Events

This module handles webhook events from Bolna.ai, including:
- Call started/ended events
- Transcription updates
- Data extraction results

Documentation: https://www.bolna.ai/docs/platform-concepts
"""

import logging
from typing import Dict, Any, Optional, Callable, Awaitable
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CallRecord:
    """Record of a Bolna call."""
    call_id: str
    agent_id: str
    phone_number: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_seconds: int = 0
    summary: str = ""
    transcript: str = ""
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    status: str = "in_progress"
    direction: str = "inbound"  # inbound or outbound
    language: str = "hi"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "call_id": self.call_id,
            "agent_id": self.agent_id,
            "phone_number": self.phone_number,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds,
            "summary": self.summary,
            "transcript": self.transcript,
            "extracted_data": self.extracted_data,
            "status": self.status,
            "direction": self.direction,
            "language": self.language
        }


class BolnaWebhookHandler:
    """
    Handler for Bolna webhook events.

    Processes:
    - call_started: New call initiated
    - call_ended: Call completed
    - extraction_completed: Data extracted from call
    - transcription: Real-time transcription updates

    Usage:
        handler = BolnaWebhookHandler()

        # Register custom handler for call_ended
        handler.register_handler("call_ended", my_callback)

        # Process webhook event
        result = await handler.handle_event(event_data)
    """

    def __init__(self):
        self.active_calls: Dict[str, CallRecord] = {}
        self.completed_calls: Dict[str, CallRecord] = {}
        self.event_handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[None]]] = {}

    def register_handler(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any]], Awaitable[None]]
    ):
        """
        Register a custom handler for an event type.

        Args:
            event_type: Event type (call_started, call_ended, etc.)
            handler: Async callback function
        """
        self.event_handlers[event_type] = handler
        logger.info(f"Registered handler for event: {event_type}")

    async def handle_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming webhook event.

        Args:
            event_data: Webhook payload from Bolna

        Returns:
            Processing result dictionary
        """
        event_type = event_data.get("event", event_data.get("type", "unknown"))
        call_id = event_data.get("call_id", event_data.get("id", "unknown"))

        logger.info(f"Processing Bolna event: {event_type} for call {call_id}")

        # Call custom handler if registered
        if event_type in self.event_handlers:
            try:
                await self.event_handlers[event_type](event_data)
            except Exception as e:
                logger.error(f"Custom handler error for {event_type}: {e}")

        # Process standard events
        if event_type == "call_started":
            return await self._handle_call_started(event_data)

        elif event_type == "call_ended":
            return await self._handle_call_ended(event_data)

        elif event_type == "extraction_completed":
            return await self._handle_extraction(event_data)

        elif event_type == "transcription":
            return await self._handle_transcription(event_data)

        elif event_type == "function_call":
            return await self._handle_function_call(event_data)

        else:
            logger.warning(f"Unknown event type: {event_type}")
            return {"status": "ignored", "reason": "unknown_event", "event_type": event_type}

    async def _handle_call_started(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle call started event."""
        call_id = data.get("call_id", data.get("id"))

        record = CallRecord(
            call_id=call_id,
            agent_id=data.get("agent_id", ""),
            phone_number=data.get("phone_number", data.get("from", "")),
            started_at=datetime.now(),
            status="in_progress",
            direction=data.get("direction", "inbound"),
            language=data.get("language", "hi")
        )

        self.active_calls[call_id] = record

        logger.info(f"Call started: {call_id} from {record.phone_number}")

        return {
            "status": "recorded",
            "call_id": call_id,
            "event": "call_started"
        }

    async def _handle_call_ended(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle call ended event."""
        call_id = data.get("call_id", data.get("id"))

        if call_id in self.active_calls:
            record = self.active_calls.pop(call_id)
            record.ended_at = datetime.now()
            record.duration_seconds = data.get("duration_seconds", data.get("duration", 0))
            record.summary = data.get("summary", "")
            record.transcript = data.get("transcript", "")
            record.status = "completed"

            self.completed_calls[call_id] = record

            logger.info(
                f"Call ended: {call_id}, duration: {record.duration_seconds}s"
            )

            # Log summary if available
            if record.summary:
                logger.info(f"Call summary: {record.summary[:200]}...")

        return {
            "status": "recorded",
            "call_id": call_id,
            "event": "call_ended"
        }

    async def _handle_extraction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle extraction completed event."""
        call_id = data.get("call_id", data.get("id"))
        extracted = data.get("extracted_data", data.get("data", {}))

        # Update record if exists
        if call_id in self.completed_calls:
            self.completed_calls[call_id].extracted_data = extracted
        elif call_id in self.active_calls:
            self.active_calls[call_id].extracted_data = extracted

        logger.info(f"Extraction completed for {call_id}: {extracted}")

        # Check if follow-up is needed
        if extracted.get("follow_up_needed") in (True, "yes", "Yes"):
            logger.info(f"Follow-up recommended for call {call_id}")
            # Could trigger notification or task here

        # Check urgency level
        urgency = extracted.get("urgency_level", "low")
        if urgency in ("high", "emergency"):
            logger.warning(f"High urgency call {call_id}: {extracted.get('user_concern', 'Unknown')}")

        return {
            "status": "recorded",
            "call_id": call_id,
            "event": "extraction_completed",
            "extracted": extracted
        }

    async def _handle_transcription(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle real-time transcription update."""
        call_id = data.get("call_id", data.get("id"))
        text = data.get("text", "")
        role = data.get("role", "user")  # user or assistant

        logger.debug(f"Transcription [{call_id}] {role}: {text[:100]}...")

        # Append to transcript if call is active
        if call_id in self.active_calls:
            record = self.active_calls[call_id]
            prefix = "User: " if role == "user" else "Assistant: "
            record.transcript += f"\n{prefix}{text}"

        return {
            "status": "received",
            "call_id": call_id,
            "event": "transcription"
        }

    async def _handle_function_call(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle function call event (for logging/monitoring)."""
        call_id = data.get("call_id", data.get("id"))
        function_name = data.get("function_name", data.get("name", "unknown"))
        parameters = data.get("parameters", data.get("arguments", {}))

        logger.info(f"Function call [{call_id}] {function_name}: {parameters}")

        return {
            "status": "logged",
            "call_id": call_id,
            "event": "function_call",
            "function": function_name
        }

    def get_active_call(self, call_id: str) -> Optional[CallRecord]:
        """Get an active call record."""
        return self.active_calls.get(call_id)

    def get_completed_call(self, call_id: str) -> Optional[CallRecord]:
        """Get a completed call record."""
        return self.completed_calls.get(call_id)

    def get_call(self, call_id: str) -> Optional[CallRecord]:
        """Get a call record (active or completed)."""
        return self.active_calls.get(call_id) or self.completed_calls.get(call_id)

    def get_call_stats(self) -> Dict[str, Any]:
        """Get statistics about calls."""
        total_completed = len(self.completed_calls)
        total_duration = sum(c.duration_seconds for c in self.completed_calls.values())

        # Count by language
        language_counts = {}
        for call in self.completed_calls.values():
            lang = call.language
            language_counts[lang] = language_counts.get(lang, 0) + 1

        # Count follow-ups needed
        follow_ups = sum(
            1 for c in self.completed_calls.values()
            if c.extracted_data.get("follow_up_needed") in (True, "yes", "Yes")
        )

        return {
            "active_calls": len(self.active_calls),
            "completed_calls": total_completed,
            "total_duration_seconds": total_duration,
            "average_duration_seconds": total_duration / total_completed if total_completed > 0 else 0,
            "calls_by_language": language_counts,
            "follow_ups_needed": follow_ups
        }

    def get_recent_calls(self, limit: int = 10) -> list:
        """Get most recent completed calls."""
        calls = list(self.completed_calls.values())
        calls.sort(key=lambda x: x.ended_at or x.started_at, reverse=True)
        return [c.to_dict() for c in calls[:limit]]

    def clear_old_calls(self, hours: int = 24):
        """Clear completed calls older than specified hours."""
        cutoff = datetime.now()
        to_remove = []

        for call_id, record in self.completed_calls.items():
            if record.ended_at:
                age = (cutoff - record.ended_at).total_seconds() / 3600
                if age > hours:
                    to_remove.append(call_id)

        for call_id in to_remove:
            del self.completed_calls[call_id]

        if to_remove:
            logger.info(f"Cleared {len(to_remove)} old call records")

        return len(to_remove)
