"""
Alert Manager

Generates, stores and tracks proactive monitoring alerts for palliative care.

Features:
- Alert generation from monitoring rules
- Alert acknowledgment and resolution tracking
- Alert summaries for the admin dashboard

Alerts are shown on the admin dashboard only. Nothing here sends a WhatsApp
message, an email or a care-team notification: no notification path exists
(CONTEXT.md), so the earlier delivery paths that marked alerts "sent" after
writing a log line were removed (review 3, decision D12).
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime
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
# ALERT MANAGER
# ============================================================================

class AlertManager:
    """
    Manages proactive monitoring alerts for palliative care.

    Responsibilities:
    1. Generate alerts from monitoring rules
    2. Store them for the admin dashboard
    3. Track acknowledgment and resolution
    """

    def __init__(
        self,
        longitudinal_manager: LongitudinalMemoryManager,
        storage_path: str = "data/alerts",
    ):
        """
        Initialize the alert manager.

        Args:
            longitudinal_manager: Manager for patient records
            storage_path: Path for alert storage
        """
        self.longitudinal = longitudinal_manager
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

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
