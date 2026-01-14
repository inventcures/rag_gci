"""
User Personalization Module for Palli Sahayak

Provides personalized experiences through:
1. User profiles with role-based context (patient, caregiver, healthcare worker)
2. Patient context memory (conditions, symptoms, medications)
3. Interaction history tracking for continuity
4. Preference management (language, communication style)

NEW: Longitudinal Patient Memory (1-5 year tracking):
5. Longitudinal patient records with temporal trend analysis
6. Cross-modal data aggregation (voice, WhatsApp, documents)
7. Compassionate context injection for personalized responses
8. Proactive monitoring and alert generation
9. Temporal reasoning for symptom progression analysis
"""

# Existing modules
from .user_profile import UserProfile, UserProfileManager, UserRole
from .context_memory import PatientContext, ContextMemory
from .interaction_history import InteractionHistory, ConversationTurn

# New longitudinal memory modules
from .longitudinal_memory import (
    # Data structures
    LongitudinalPatientRecord,
    # Observation types
    TimestampedObservation,
    SymptomObservation,
    MedicationEvent,
    VitalSignObservation,
    FunctionalStatusObservation,
    EmotionalObservation,
    TimeSeriesSummary,
    # Enums
    DataSourceType,
    SeverityLevel,
    TemporalTrend,
    AlertPriority,
    MedicationAction,
    # Monitoring
    MonitoringAlert,
    MonitoringRule,
    # Manager
    LongitudinalMemoryManager,
)

from .context_injector import (
    ContextInjector,
    PromptContextBuilder,
)

from .cross_modal_aggregator import (
    CrossModalAggregator,
    VoiceDataExtractor,
    WhatsAppDataExtractor,
    DocumentDataExtractor,
)

from .temporal_reasoner import (
    TemporalReasoner,
    SymptomProgressionReport,
    MedicationEffectivenessReport,
    CorrelationAnalysis,
)

from .alert_manager import (
    AlertManager,
    AlertNotificationCoordinator,
    DeliveryStatus,
    AlertDeliveryResult,
)

__all__ = [
    # Existing exports
    "UserProfile",
    "UserProfileManager",
    "UserRole",
    "PatientContext",
    "ContextMemory",
    "InteractionHistory",
    "ConversationTurn",
    # Longitudinal memory - Core
    "LongitudinalPatientRecord",
    "LongitudinalMemoryManager",
    # Longitudinal memory - Observations
    "TimestampedObservation",
    "SymptomObservation",
    "MedicationEvent",
    "VitalSignObservation",
    "FunctionalStatusObservation",
    "EmotionalObservation",
    "TimeSeriesSummary",
    # Longitudinal memory - Enums
    "DataSourceType",
    "SeverityLevel",
    "TemporalTrend",
    "AlertPriority",
    "MedicationAction",
    # Longitudinal memory - Monitoring
    "MonitoringAlert",
    "MonitoringRule",
    # Context injection
    "ContextInjector",
    "PromptContextBuilder",
    # Cross-modal aggregation
    "CrossModalAggregator",
    "VoiceDataExtractor",
    "WhatsAppDataExtractor",
    "DocumentDataExtractor",
    # Temporal reasoning
    "TemporalReasoner",
    "SymptomProgressionReport",
    "MedicationEffectivenessReport",
    "CorrelationAnalysis",
    # Alert management
    "AlertManager",
    "AlertNotificationCoordinator",
    "DeliveryStatus",
    "AlertDeliveryResult",
]
