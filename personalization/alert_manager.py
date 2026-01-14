"""
Alert Manager

Manages proactive monitoring and alert delivery for palliative care.

Features:
- Alert generation from monitoring rules
- Multi-channel alert delivery (WhatsApp, dashboard, email)
- Alert acknowledgment and resolution tracking
- Caregiver notification coordination
- Compassionate alert messaging

Alerts are delivered with care and empathy, recognizing the sensitive
nature of palliative care communication.
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import json
import aiofiles

from .longitudinal_memory import (
    LongitudinalPatientRecord,
    LongitudinalMemoryManager,
    MonitoringAlert,
    MonitoringRule,
    AlertPriority,
)

logger = logging.getLogger(__name__)


# ============================================================================
# ALERT DELIVERY STATUS
# ============================================================================

class DeliveryStatus(Enum):
    """Status of alert delivery attempts."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    ACKNOWLEDGED = "acknowledged"


# ============================================================================
# ALERT TEMPLATES (Multi-language, Compassionate)
# ============================================================================

ALERT_TEMPLATES = {
    "en-IN": {
        "greeting": "Namaste. Hope you are doing well.",
        "concern_intro": "We wanted to check in with you about",
        "concern_symptom": "your recent symptom reports",
        "concern_medication": "your medication routine",
        "concern_missed": "that we haven't heard from you",
        "action_intro": "It would be good if you could",
        "support_offer": "Please remember we are here to support you.",
        "closing": "Take care.",
        "urgent_header": "⚠️ Important Health Check-in",
        "routine_header": "🌸 Caring Check-in"
    },
    "hi-IN": {
        "greeting": "नमस्ते। आशा है कि आप ठीक से हैं।",
        "concern_intro": "हम आपसे यह बताना चाहते थे कि",
        "concern_symptom": "आपके हाल के लक्षण",
        "concern_medication": "आपकी दवा की दिनचर्या",
        "concern_missed": "हमें आपसे बात नहीं हुई",
        "action_intro": "कृपया करके आप",
        "support_offer": "कृपया याद रखें, हम आपके साथ हैं।",
        "closing": "ध्यान रखें।",
        "urgent_header": "⚠️ महत्वपूर्ण जांच",
        "routine_header": "🌸 प्यार भरी जांच"
    }
}


# ============================================================================
# ALERT DELIVERY RESULT
# ============================================================================

@dataclass
class AlertDeliveryResult:
    """Result of an alert delivery attempt."""
    alert_id: str
    channel: str  # "whatsapp", "email", "dashboard"
    status: DeliveryStatus
    timestamp: datetime
    error_message: Optional[str] = None
    delivery_id: Optional[str] = None  # Provider's delivery ID


# ============================================================================
# ALERT MANAGER
# ============================================================================

class AlertManager:
    """
    Manages proactive monitoring alerts for palliative care.

    Responsibilities:
    1. Generate alerts from monitoring rules
    2. Format alerts compassionately in multiple languages
    3. Deliver alerts through appropriate channels
    4. Track acknowledgment and resolution
    5. Coordinate with care team for escalations
    """

    def __init__(
        self,
        longitudinal_manager: LongitudinalMemoryManager,
        storage_path: str = "data/alerts",
        default_channels: List[str] = None
    ):
        """
        Initialize the alert manager.

        Args:
            longitudinal_manager: Manager for patient records
            storage_path: Path for alert storage
            default_channels: Default delivery channels
        """
        self.longitudinal = longitudinal_manager
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.default_channels = default_channels or ["dashboard", "whatsapp"]

        logger.info("AlertManager initialized")

    def _get_alert_path(self, alert_id: str) -> Path:
        """Get file path for alert."""
        return self.storage_path / f"{alert_id}.json"

    async def generate_alerts_for_patient(
        self,
        patient_id: str,
        force_check: bool = False
    ) -> List[MonitoringAlert]:
        """
        Run monitoring checks and generate alerts for a patient.

        Args:
            patient_id: Patient identifier
            force_check: Force check even if recent check exists

        Returns:
            List of newly generated alerts
        """
        return await self.longitudinal.run_monitoring_checks(patient_id)

    async def send_alert(
        self,
        alert: MonitoringAlert,
        channels: Optional[List[str]] = None,
        language: str = "en-IN"
    ) -> List[AlertDeliveryResult]:
        """
        Send alert through specified channels.

        Args:
            alert: The alert to send
            channels: Channels to use (default: self.default_channels)
            language: Language for alert message

        Returns:
            List of delivery results
        """
        if channels is None:
            channels = self.default_channels

        # For urgent alerts, use multiple channels
        if alert.priority == AlertPriority.URGENT:
            channels = ["whatsapp", "dashboard", "email"]

        results = []

        # Format alert message
        message = self._format_alert_message(alert, language)

        for channel in channels:
            try:
                result = await self._send_via_channel(
                    alert,
                    channel,
                    message,
                    language
                )
                results.append(result)

                # Update alert delivery status
                alert.delivery_channels.append(channel)
                alert.delivery_status[channel] = result.status.value

            except Exception as e:
                logger.error(f"Error sending alert via {channel}: {e}")
                results.append(AlertDeliveryResult(
                    alert_id=alert.alert_id,
                    channel=channel,
                    status=DeliveryStatus.FAILED,
                    timestamp=datetime.now(),
                    error_message=str(e)
                ))

        # Save updated alert
        await self._save_alert(alert)

        return results

    async def _send_via_channel(
        self,
        alert: MonitoringAlert,
        channel: str,
        message: str,
        language: str
    ) -> AlertDeliveryResult:
        """Send alert via a specific channel."""
        timestamp = datetime.now()

        if channel == "whatsapp":
            return await self._send_via_whatsapp(alert, message, language)
        elif channel == "email":
            return await self._send_via_email(alert, message, language)
        elif channel == "dashboard":
            return await self._send_via_dashboard(alert, message)
        else:
            return AlertDeliveryResult(
                alert_id=alert.alert_id,
                channel=channel,
                status=DeliveryStatus.FAILED,
                timestamp=timestamp,
                error_message=f"Unknown channel: {channel}"
            )

    async def _send_via_whatsapp(
        self,
        alert: MonitoringAlert,
        message: str,
        language: str
    ) -> AlertDeliveryResult:
        """
        Send alert via WhatsApp.

        This would integrate with the WhatsApp bot or Twilio API.
        """
        # Placeholder for actual WhatsApp integration
        # In production, this would call the WhatsApp bot's send_message function

        logger.info(f"Would send WhatsApp alert for {alert.patient_id}: {alert.title}")

        # Simulate successful delivery
        return AlertDeliveryResult(
            alert_id=alert.alert_id,
            channel="whatsapp",
            status=DeliveryStatus.SENT,
            timestamp=datetime.now()
        )

    async def _send_via_email(
        self,
        alert: MonitoringAlert,
        message: str,
        language: str
    ) -> AlertDeliveryResult:
        """Send alert via email."""
        # Placeholder for email integration
        logger.info(f"Would send email alert for {alert.patient_id}: {alert.title}")

        return AlertDeliveryResult(
            alert_id=alert.alert_id,
            channel="email",
            status=DeliveryStatus.SENT,
            timestamp=datetime.now()
        )

    async def _send_via_dashboard(
        self,
        alert: MonitoringAlert,
        message: str
    ) -> AlertDeliveryResult:
        """
        Add alert to dashboard for caregiver/clinical team viewing.

        The dashboard is automatically updated when alerts are generated.
        """
        # Dashboard alerts are visible by default when stored
        return AlertDeliveryResult(
            alert_id=alert.alert_id,
            channel="dashboard",
            status=DeliveryStatus.DELIVERED,
            timestamp=datetime.now()
        )

    def _format_alert_message(
        self,
        alert: MonitoringAlert,
        language: str
    ) -> str:
        """
        Format alert message with compassionate tone.

        Messages are:
        - Empathetic and kind
        - Clear about the concern
        - Action-oriented but not alarming
        - Multi-language
        """
        templates = ALERT_TEMPLATES.get(language, ALERT_TEMPLATES["en-IN"])

        # Select header based on priority
        if alert.priority == AlertPriority.URGENT:
            header = templates.get("urgent_header", "⚠️ Important Health Check-in")
        else:
            header = templates.get("routine_header", "🌸 Caring Check-in")

        # Build message parts
        parts = [header]

        # Greeting
        parts.append(templates.get("greeting", "Hello."))

        # Main concern
        concern_intro = templates.get("concern_intro", "We wanted to check in about")
        parts.append(f"{concern_intro} {alert.description}")

        # Suggested actions
        if alert.suggested_actions:
            action_intro = templates.get("action_intro", "Please consider")
            actions = "\n• ".join(alert.suggested_actions[:3])
            parts.append(f"{action_intro}:\n• {actions}")

        # Support offer
        parts.append(templates.get("support_offer", "We are here to support you."))

        # Closing
        parts.append(templates.get("closing", "Take care."))

        return "\n\n".join(parts)

    async def acknowledge_alert(
        self,
        patient_id: str,
        alert_id: str,
        acknowledged_by: str
    ) -> bool:
        """
        Acknowledge an alert (mark as seen).

        Args:
            patient_id: Patient identifier
            alert_id: Alert identifier
            acknowledged_by: Who acknowledged (user ID)

        Returns:
            True if successful
        """
        return await self.longitudinal.acknowledge_alert(patient_id, alert_id, acknowledged_by)

    async def resolve_alert(
        self,
        patient_id: str,
        alert_id: str,
        resolution_notes: Optional[str] = None
    ) -> bool:
        """
        Resolve an alert (mark as handled).

        Args:
            patient_id: Patient identifier
            alert_id: Alert identifier
            resolution_notes: Notes about resolution

        Returns:
            True if successful
        """
        return await self.longitudinal.resolve_alert(patient_id, alert_id, resolution_notes)

    async def get_active_alerts(
        self,
        patient_id: Optional[str] = None,
        priority: Optional[AlertPriority] = None,
        category: Optional[str] = None
    ) -> List[MonitoringAlert]:
        """
        Get active alerts with optional filtering.

        Args:
            patient_id: Filter by patient (None = all patients)
            priority: Filter by priority
            category: Filter by category

        Returns:
            List of active alerts
        """
        alerts = []

        if patient_id:
            record = await self.longitudinal.get_or_create_record(patient_id)
            alerts = [a for a in record.active_alerts if not a.resolved]
        else:
            # Get all alerts across all patients
            storage_path = self.longitudinal.storage_path
            for file_path in storage_path.glob("*_longitudinal.json"):
                try:
                    async with aiofiles.open(file_path, "r") as f:
                        content = await f.read()
                        if content:
                            data = json.loads(content)
                            record = LongitudinalPatientRecord.from_dict(data)
                            alerts.extend([
                                a for a in record.active_alerts
                                if not a.resolved
                            ])
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")

        # Apply filters
        if priority:
            alerts = [a for a in alerts if a.priority == priority]

        if category:
            alerts = [a for a in alerts if a.category == category]

        # Sort by priority and date
        priority_order = {
            AlertPriority.URGENT: 0,
            AlertPriority.HIGH: 1,
            AlertPriority.MEDIUM: 2,
            AlertPriority.LOW: 3
        }

        alerts.sort(key=lambda a: (priority_order.get(a.priority, 4), a.created_at))

        return alerts

    async def get_alert_summary(
        self,
        patient_id: str
    ) -> Dict[str, Any]:
        """
        Get summary of alerts for a patient.

        Args:
            patient_id: Patient identifier

        Returns:
            Summary with counts and categories
        """
        record = await self.longitudinal.get_or_create_record(patient_id)

        active_alerts = [a for a in record.active_alerts if not a.resolved]
        recent_history = [a for a in record.alert_history if a.resolved][-10:]

        # Count by priority
        by_priority = {
            AlertPriority.URGENT: 0,
            AlertPriority.HIGH: 0,
            AlertPriority.MEDIUM: 0,
            AlertPriority.LOW: 0
        }

        for alert in active_alerts:
            by_priority[alert.priority] = by_priority.get(alert.priority, 0) + 1

        # Count by category
        by_category = {}
        for alert in active_alerts:
            by_category[alert.category] = by_category.get(alert.category, 0) + 1

        return {
            "patient_id": patient_id,
            "active_alerts": len(active_alerts),
            "by_priority": {p.value: c for p, c in by_priority.items()},
            "by_category": by_category,
            "recent_resolutions": len(recent_history),
            "last_check": record.last_alert_check.isoformat() if record.last_alert_check else None
        }

    async def create_custom_alert(
        self,
        patient_id: str,
        title: str,
        description: str,
        priority: AlertPriority,
        category: str,
        suggested_actions: List[str],
        created_by: str
    ) -> MonitoringAlert:
        """
        Create a custom alert (manually triggered).

        Args:
            patient_id: Patient identifier
            title: Alert title
            description: Alert description
            priority: Alert priority
            category: Alert category
            suggested_actions: Suggested actions
            created_by: Who created the alert

        Returns:
            Created alert
        """
        import hashlib

        timestamp = datetime.now()
        alert_id = f"alert_{timestamp.strftime('%Y%m%d%H%M%S')}_{hashlib.md5(f'{patient_id}:{timestamp.isoformat()}'.encode()).hexdigest()[:8]}"

        alert = MonitoringAlert(
            alert_id=alert_id,
            patient_id=patient_id,
            created_at=timestamp,
            priority=priority,
            category=category,
            title=title,
            description=description,
            pattern_description=f"Manually created by {created_by}",
            suggested_actions=suggested_actions
        )

        # Add to patient record
        record = await self.longitudinal.get_or_create_record(patient_id)
        record.active_alerts.append(alert)

        await self.longitudinal.save_record(record)

        logger.info(f"Created custom alert {alert_id} for {patient_id}")

        return alert

    async def _save_alert(self, alert: MonitoringAlert) -> None:
        """Save alert to individual file for dashboard access."""
        alert_path = self._get_alert_path(alert.alert_id)

        try:
            async with aiofiles.open(alert_path, "w") as f:
                await f.write(json.dumps(alert.to_dict(), indent=2))
        except Exception as e:
            logger.error(f"Error saving alert {alert.alert_id}: {e}")

    async def run_batch_monitoring(
        self,
        patient_ids: Optional[List[str]] = None,
        max_concurrent: int = 10
    ) -> Dict[str, Any]:
        """
        Run monitoring checks for multiple patients in parallel.

        Args:
            patient_ids: List of patients to check (None = all)
            max_concurrent: Maximum concurrent checks

        Returns:
            Summary with total alerts generated
        """
        # Get all patient IDs if not provided
        if patient_ids is None:
            patient_ids = []
            storage_path = self.longitudinal.storage_path
            for file_path in storage_path.glob("*_longitudinal.json"):
                patient_ids.append(file_path.stem.replace("_longitudinal", ""))

        # Run monitoring checks in batches
        all_new_alerts = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def check_one(patient_id: str) -> List[MonitoringAlert]:
            async with semaphore:
                return await self.generate_alerts_for_patient(patient_id)

        tasks = [check_one(pid) for pid in patient_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in monitoring check: {result}")
            elif isinstance(result, list):
                all_new_alerts.extend(result)

        # Count by priority
        by_priority = {
            AlertPriority.URGENT: 0,
            AlertPriority.HIGH: 0,
            AlertPriority.MEDIUM: 0,
            AlertPriority.LOW: 0
        }

        for alert in all_new_alerts:
            by_priority[alert.priority] = by_priority.get(alert.priority, 0) + 1

        return {
            "patients_checked": len(patient_ids),
            "total_alerts": len(all_new_alerts),
            "by_priority": {p.value: c for p, c in by_priority.items()},
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# ALERT NOTIFICATION COORDINATOR
# ============================================================================

class AlertNotificationCoordinator:
    """
    Coordinates alert notifications with care team.

    Ensures that:
    - Urgent alerts reach appropriate team members
    - Caregivers are notified for their assigned patients
    - No duplicate notifications
    - Notification preferences are respected
    """

    def __init__(
        self,
        alert_manager: AlertManager,
        longitudinal_manager: LongitudinalMemoryManager
    ):
        """
        Initialize the coordinator.

        Args:
            alert_manager: Alert manager instance
            longitudinal_manager: Longitudinal manager instance
        """
        self.alert_manager = alert_manager
        self.longitudinal = longitudinal_manager

    async def notify_care_team(
        self,
        alert: MonitoringAlert,
        notify_primary: bool = True
    ) -> List[str]:
        """
        Notify care team members about an alert.

        Args:
            alert: The alert to notify about
            notify_primary: Whether to notify only primary contact

        Returns:
            List of notified team member IDs
        """
        record = await self.longitudinal.get_or_create_record(alert.patient_id)
        notified = []

        # Determine who to notify
        if notify_primary:
            primary = record.get_primary_contact()
            if primary:
                team_members = [primary]
            else:
                team_members = record.care_team
        else:
            team_members = record.care_team

        for member in team_members:
            # Check notification preferences
            # (For now, just log - in production would send actual notifications)
            logger.info(
                f"Notifying {member.name} ({member.role}) "
                f"about alert {alert.alert_id} for {alert.patient_id}"
            )
            notified.append(member.provider_id)

        return notified

    async def escalate_alert(
        self,
        alert: MonitoringAlert,
        escalation_reason: str
    ) -> MonitoringAlert:
        """
        Escalate an alert to higher priority and wider notification.

        Args:
            alert: The alert to escalate
            escalation_reason: Why escalation is needed

        Returns:
            Escalated alert
        """
        # Increase priority
        old_priority = alert.priority
        if alert.priority == AlertPriority.LOW:
            alert.priority = AlertPriority.MEDIUM
        elif alert.priority == AlertPriority.MEDIUM:
            alert.priority = AlertPriority.HIGH
        elif alert.priority == AlertPriority.HIGH:
            alert.priority = AlertPriority.URGENT

        # Add escalation note
        alert.description = f"[ESCALATED: {escalation_reason}] {alert.description}"
        alert.suggested_actions.insert(0, "Review escalation details immediately")

        # Notify broader care team
        await self.notify_care_team(alert, notify_primary=False)

        logger.info(
            f"Escalated alert {alert.alert_id} from {old_priority.value} to {alert.priority.value}"
        )

        return alert

    async def send_proactive_checkin(
        self,
        patient_id: str,
        language: str = "en-IN",
        message_template: Optional[str] = None
    ) -> bool:
        """
        Send a proactive check-in message to a patient.

        This is not an alert but a caring touchpoint.

        Args:
            patient_id: Patient identifier
            language: Language for message
            message_template: Custom message template

        Returns:
            True if sent successfully
        """
        templates = ALERT_TEMPLATES.get(language, ALERT_TEMPLATES["en-IN"])

        if message_template:
            message = message_template
        else:
            message = f"""{templates['greeting']}

{templates['support_offer']}

How have you been feeling lately? We're here if you need anything.

{templates['closing']}"""

        # Send via WhatsApp
        # In production, this would call the WhatsApp bot
        logger.info(f"Would send proactive check-in to {patient_id}")

        return True
