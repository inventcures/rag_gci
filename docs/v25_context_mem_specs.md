# Longitudinal Patient Context Memory System for Palli Sahayak
## Inspired by MedAgentBenchV2

---

## Original Prompt (as requested by user)

```
0. KNOWING EVERYTHING YOU KNOW ABOUT THE PROJECT, ALL THE CODE, THE PLANS, THE .MDS

0.5 AND https://deepwiki.com/inventcures/rag_gci

NEW FEATURE ADDITION

1. DO DEEP ANALYSIS & DEEP RESEARCH TO GROK & UNDERSTAND

@docs/v25/medagentbenchv2.pdf
https://github.com/inventcures/medagentbenchv2
https://deepwiki.com/ericoericochen/medagentbenchv2

TRIGGER A DEEP RESEARCH ON HOW BEST TO INTEGRATE THE CORE THESIS OF
@docs/v25/medagentbenchv2.pdf
IN THIS CODEBASE, TO INCOPRORATE PATIENT CONTEXT, SPANNING MONTHS OR POSSIBLY YEARS USING SOME SORT OF MEMORY COMPONENT

WRITE A DETAILED PLAN AND SPECS IN @docs/v25_context_mem_specs.md
to add this feature in the existing codebase, to reason across the ENTIRE PATIENT CONTEXT, ACROSS MODALITIES, PROVIDERS, THERAPIES/TREATMENTS, , longitudinally over time

write this prompt, as is & your response to it, in the next turn, also as is,
at the top of @docs/v25_context_mem_specs.md

5. ADAPT EVERYTHING FOR PALLI SAHAYAK - compassionate, empathetic kind palliative care
```

---

## Summary of Analysis and Response

### MedAgentBenchV2 Core Thesis (from PDF analysis)

1. **Longitudinal Patient Records**: Uses FHIR-compliant EHR system with Patient, Observation, and MedicationRequest resources spanning multiple visits and time periods

2. **Memory Component**: A learning system that captures failures and improves agent performance from 91.0% to 98.0% success rate

3. **Tool Abstraction Layer**: High-level functions for clinical data access instead of manual HTTP requests

4. **Multi-step Clinical Reasoning**: Tasks require reasoning across time, correlating lab results, medications, and symptoms

5. **Temporal Awareness**: Understanding disease progression, medication changes, and treatment effectiveness over time

### Palli Sahayak Current State (Critical Finding)

**Existing but NOT Integrated:**
- `/personalization/context_memory.py` - PatientContext with symptoms, medications, allergies (90-day expiry)
- `/personalization/interaction_history.py` - Session tracking, conversation turns, 30-day retention
- `/personalization/user_profile.py` - UserProfile with role, language, communication style

**Critical Gap**: The `user_id` parameter in `simple_rag_server.py:query()` is **accepted but NOT used** for personalization. Each query is stateless.

### User Requirements (from clarifying questions)

- **Time Horizon**: 1-5 years (full longitudinal picture)
- **Provider Data**: Hybrid approach - start with manual/document upload, add FHIR later
- **Priorities**: ALL of (1) Symptom progression tracking, (2) Medication adherence, (3) Care continuity, (4) Caregiver coordination
- **Proactive Alerts**: Hybrid - automated pattern detection + caregiver-triggered check-ins

---

# Table of Contents

1. [System Architecture](#system-architecture)
2. [Data Model Design](#data-model-design)
3. [Module Structure](#module-structure)
4. [Integration Points](#integration-points)
5. [Temporal Reasoning](#temporal-reasoning)
6. [Cross-Modal Aggregation](#cross-modal-aggregation)
7. [Care Team Coordination](#care-team-coordination)
8. [Proactive Monitoring](#proactive-monitoring)
9. [Compassionate Design](#compassionate-design)
10. [FHIR Integration Path](#fhir-integration-path)
11. [Implementation Phases](#implementation-phases)
12. [File Storage Schema](#file-storage-schema)

---

## System Architecture

### Current State vs. Target State

```
CURRENT STATE (Each query is stateless):
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   WhatsApp  │     │   Voice     │     │     Web     │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            ▼
                   ┌─────────────────┐
                   │  RAG Pipeline    │
                   │  (stateless)     │
                   └────────┬─────────┘
                            ▼
                   ┌─────────────────┐
                   │  Vector DB      │
                   │  (medical only) │
                   └─────────────────┘
```

```
TARGET STATE (Longitudinal memory integrated):
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   WhatsApp  │     │   Voice     │     │     Web     │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            ▼
                   ┌─────────────────┐
                   │  Cross-Modal    │
                   │  Aggregator     │
                   └────────┬─────────┘
                            ▼
           ┌────────────────────────────────┐
           │     Longitudinal Memory        │
           │  (1-5 year patient record)     │
           │  - Observations                │
           │  - Time-series trends          │
           │  - Care team                   │
           │  - Alerts                      │
           └────────────┬───────────────────┘
                        │
                        ▼
                   ┌─────────────────┐
                   │  Context        │
                   │  Injector       │
                   └────────┬─────────┘
                            ▼
                   ┌─────────────────┐
                   │  RAG Pipeline    │
                   │  (personalized)  │
                   └────────┬─────────┘
                            ▼
                   ┌─────────────────┐
                   │  Vector DB      │
                   │  + Patient      │
                   │  Context        │
                   └─────────────────┘
```

---

## Data Model Design

### Core Hierarchy

```
LongitudinalPatientRecord (1-5 years)
│
├── observations: List[TimestampedObservation]
│   ├── SymptomObservation
│   ├── MedicationEvent
│   ├── VitalSignObservation
│   └── FunctionalStatusObservation
│
├── care_team: List[CareTeamMember]
│
├── active_alerts: List[MonitoringAlert]
│
├── alert_history: List[MonitoringAlert]
│
├── monitoring_rules: List[MonitoringRule]
│
├── conversations_index: Dict[conversation_id, observation_ids]
│
└── documents_index: Dict[document_id, observation_ids]
```

### TimestampedObservation (Base Class)

The fundamental unit of longitudinal tracking - every piece of patient data is an observation with a timestamp.

```python
@dataclass
class TimestampedObservation:
    """A single observation at a point in time."""
    observation_id: str
    timestamp: datetime
    source_type: DataSourceType  # VOICE_CALL, WHATSAPP, UPLOADED_DOCUMENT, etc.
    source_id: str  # Reference to conversation_id, document_id
    reported_by: str  # "patient", "caregiver", "system", "provider"

    # Observation content
    category: str  # "symptom", "medication", "vital_sign", etc.
    entity_name: str  # "pain", "morphine", "blood_pressure"
    value: Any  # SeverityLevel, numeric, or text
    value_text: str  # Human-readable description
    metadata: Dict[str, Any]
```

### SymptomObservation

```python
@dataclass
class SymptomObservation(TimestampedObservation):
    """Symptom with palliative-specific tracking."""
    symptom_name: str
    severity: SeverityLevel  # NONE(0), MILD(1), MODERATE(2), SEVERE(3), VERY_SEVERE(4)
    duration: Optional[str]  # "2 days", "chronic"
    location: Optional[str]  # "lower back", "head"
    aggravating_factors: List[str]
    relieving_factors: List[str]
    impact_on_function: Optional[SeverityLevel]  # How it affects daily life
```

### MedicationEvent

```python
@dataclass
class MedicationEvent(TimestampedObservation):
    """Medication with adherence tracking."""
    medication_name: str
    dosage: str
    action: str  # "started", "stopped", "dose_changed", "taken", "missed"
    effectiveness: Optional[SeverityLevel]
    side_effects: List[str]
    adherence_rate: Optional[float]  # 0.0 to 1.0
```

### TimeSeriesSummary

```python
@dataclass
class TimeSeriesSummary:
    """Aggregated view for temporal reasoning."""
    entity_name: str
    category: str
    start_date: date
    end_date: date
    total_observations: int

    latest_value: Any
    earliest_value: Any
    average_value: Optional[float]
    trend: TemporalTrend  # IMPROVING, STABLE, WORSENING, FLUCTUATING
    trend_confidence: float  # 0.0 to 1.0
    change_per_week: Optional[float]
```

---

## Module Structure

### New Files to Create

| File Path | Purpose |
|-----------|---------|
| `personalization/longitudinal_memory.py` | Core longitudinal data structures, `LongitudinalPatientRecord`, `LongitudinalMemoryManager` |
| `personalization/cross_modal_aggregator.py` | Aggregates data from voice, WhatsApp, documents |
| `personalization/temporal_reasoner.py` | Trend detection, correlation analysis |
| `personalization/context_injector.py` | Integrates context into RAG queries |
| `personalization/alert_manager.py` | Proactive monitoring and alert delivery |
| `personalization/fhir_adapter.py` | FHIR data models for future EHR integration |

### Files to Modify

| File Path | Modifications |
|-----------|---------------|
| `personalization/__init__.py` | Add exports for new modules |
| `personalization/context_memory.py` | Extend with link to longitudinal record |
| `personalization/user_profile.py` | Add care_team association |
| `simple_rag_server.py` | **CRITICAL**: Integrate context injection into `query()` method |
| `bolna_integration/webhooks.py` | Trigger observation extraction after calls |
| `whatsapp_bot.py` | Trigger observation extraction from messages |

---

## Integration Points

### Critical Integration: simple_rag_server.py

The `query()` method at line 973 currently accepts `user_id` but does NOT use it.

**Current Code:**
```python
async def query(self, question: str, conversation_id: Optional[str] = None,
               user_id: Optional[str] = None, top_k: int = 5, source_language: str = "en"):
```

**Required Changes:**

```python
# In SimpleRAGPipeline.__init__
from personalization import (
    UserProfileManager, ContextMemory, InteractionHistory,
    LongitudinalMemoryManager, ContextInjector, CrossModalAggregator
)

self.profile_manager = UserProfileManager()
self.context_memory = ContextMemory()
self.interaction_history = InteractionHistory()
self.longitudinal_manager = LongitudinalMemoryManager()
self.context_injector = ContextInjector(
    self.longitudinal_manager,
    self.profile_manager,
    self.context_memory
)
self.cross_modal_aggregator = CrossModalAggregator(self.longitudinal_manager)

# In query() method - ADD AT THE BEGINNING:
patient_context_summary = ""
if user_id:
    patient_context_summary = await self.context_injector.inject_context(
        user_id=user_id,
        question=question
    )

# In LLM prompt generation - ADD patient_context:
system_prompt = self._build_system_prompt(
    base_prompt=self.system_prompt,
    patient_context=patient_context_summary,
    user_profile=profile if user_id else None
)

# AFTER generating response - EXTRACT and STORE observations:
if user_id and conversation_id:
    asyncio.create_task(
        self.cross_modal_aggregator.process_conversation(
            patient_id=user_id,
            conversation_id=conversation_id,
            transcript=f"{question}\n\n{answer}",
            source_type=source_type,
            metadata={"language": source_language}
        )
    )
```

---

## Temporal Reasoning

### Trend Detection Algorithm

```python
def _calculate_trend(observations: List[TimestampedObservation]) -> Tuple[TemporalTrend, float]:
    """
    Calculate trend using linear regression.

    Returns:
        (trend_direction, confidence)
    """
    if len(observations) < 3:
        return TemporalTrend.UNKNOWN, 0.0

    # Extract numeric values from severity
    numeric_values = []
    for obs in observations:
        if isinstance(obs.value, SeverityLevel):
            numeric_values.append(float(obs.value.value))
        elif isinstance(obs.value, (int, float)):
            numeric_values.append(float(obs.value))

    # Linear regression
    n = len(numeric_values)
    x = list(range(n))
    y = numeric_values

    slope = (n * sum(xy) - sum(x) * sum(y)) / (n * sum(x²) - sum(x)²)

    # R² for confidence
    r_squared = calculate_r_squared(y, y_pred)

    # Classify trend
    if slope > 0.5:
        trend = TemporalTrend.WORSENING
    elif slope < -0.5:
        trend = TemporalTrend.IMPROVING
    elif abs(slope) < 0.2:
        trend = TemporalTrend.STABLE
    else:
        trend = TemporalTrend.FLUCTUATING

    return trend, min(1.0, max(0.0, r_squared))
```

### Symptom Progression Report

```python
@dataclass
class SymptomProgressionReport:
    """Analysis of how a symptom has changed over time."""
    symptom_name: str
    analysis_period_days: int
    total_observations: int

    # Temporal analysis
    trend: TemporalTrend
    trend_confidence: float
    current_severity: SeverityLevel
    baseline_severity: SeverityLevel  # Average of first 30%

    # Pattern detection
    diurnal_pattern: Optional[str]  # "worse_morning", "worse_evening", "none"
    response_to_medication: Optional[str]  # "improves_with", "no_change", "unknown"

    # Recommendations
    clinical_concerns: List[str]
    suggested_actions: List[str]
```

---

## Cross-Modal Aggregation

### Data Sources

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cross-Modal Aggregator                       │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Voice Calls │    │   WhatsApp   │    │  Documents   │
│  (Bolna)     │    │   Messages   │    │  (PDF/Img)   │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Voice      │    │  WhatsApp    │    │  Document    │
│  Extractor   │    │  Extractor   │    │  Extractor   │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  Observations    │
                    │  (unified)       │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Longitudinal    │
                    │  Memory Store    │
                    └──────────────────┘
```

### Extraction Example

```python
class VoiceDataExtractor:
    """Extract medical observations from voice call transcripts."""

    async def extract(self, transcript: str, metadata: Dict) -> List[SymptomObservation]:
        """Use LLM to extract structured medical data."""
        prompt = f"""
Extract from this palliative care conversation:
- Symptoms (with severity: mild/moderate/severe)
- Medications (with dosage)
- Emotional state

Transcript: {transcript}

Return JSON format.
"""
        # Call LLM and parse response
        extracted = await self._llm_extract(prompt)

        observations = []
        for item in extracted.get("symptoms", []):
            observations.append(SymptomObservation(
                symptom_name=item["name"],
                severity=SeverityLevel.from_string(item["severity"]),
                location=item.get("location"),
                # ... other fields
            ))

        return observations
```

---

## Care Team Coordination

### Care Team Member

```python
@dataclass
class CareTeamMember:
    """Member of patient's care team."""
    provider_id: str
    name: str
    role: str  # "doctor", "nurse", "asha_worker", "caregiver", "volunteer"
    organization: Optional[str]
    phone_number: Optional[str]  # Hashed
    primary_contact: bool = False

    # Tracking
    first_contact: datetime
    last_contact: datetime
    total_interactions: int

    # Attribution
    attributed_observations: List[str]  # observation_ids
```

### Provider Attribution

Every observation can be attributed to a specific care team member:

```python
def attribute_observation(
    record: LongitudinalPatientRecord,
    observation_id: str,
    provider_id: str
) -> None:
    """Link observation to care team member."""
    for obs in record.observations:
        if obs.observation_id == observation_id:
            obs.reported_by = provider_id

            # Update provider stats
            for member in record.care_team:
                if member.provider_id == provider_id:
                    member.last_contact = obs.timestamp
                    member.total_interactions += 1
                    member.attributed_observations.append(observation_id)
```

---

## Proactive Monitoring

### Default Palliative Care Rules

| Rule ID | Pattern | Priority | Time Window | Alert Message |
|---------|---------|----------|-------------|---------------|
| `worsening_pain` | Pain trend worsening | HIGH | 7 days | Pain appears to be worsening over the past week. Consider assessment for medication adjustment. |
| `severe_breathlessness` | Severe breathlessness | URGENT | 1 day | Severe breathlessness reported. Immediate clinical assessment recommended. |
| `medication_adherence_decline` | Adherence declining | MEDIUM | 7 days | Medication adherence appears to be declining. Potential causes: side effects, cost, confusion. |
| `missed_checkin` | No contact for 14 days | LOW | 14 days | Patient hasn't had any interaction in 14 days. Proactive check-in recommended. |
| `new_severe_symptom` | New severe symptom | HIGH | 1 day | New severe symptom reported. |
| `opioid_rotation_needed` | Opioid effectiveness declining | HIGH | 14 days | Consider opioid rotation. |

### Alert Structure

```python
@dataclass
class MonitoringAlert:
    """Proactive alert from pattern detection."""
    alert_id: str
    patient_id: str
    created_at: datetime
    priority: AlertPriority  # LOW, MEDIUM, HIGH, URGENT
    category: str
    title: str
    description: str

    # Trigger information
    trigger_observation_ids: List[str]
    pattern_description: str

    # State
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    resolved: bool = False

    # Actions
    suggested_actions: List[str]
    assigned_to: Optional[str]  # care_team_member_id
```

---

## Compassionate Design

### Tone Templates (Multi-Language)

```python
COMPASSIONATE_TEMPLATES = {
    "en-IN": {
        "opening": [
            "I remember that",
            "As we discussed before,",
            "I understand that"
        ],
        "empathy": {
            "worsening": "I'm sorry to hear that things have been difficult",
            "pain": "I know pain can be very challenging",
            "breathlessness": "Breathlessness can feel scary"
        },
        "continuity": [
            "Let's see how you've been since we last spoke",
            "I want to make sure you're comfortable"
        ]
    },
    "hi-IN": {
        "opening": [
            "मैं याद रखता हूं कि",
            "जैसा हमने पहले बात की थी",
            "मैं समझता हूं कि"
        ],
        "empathy": {
            "worsening": "मुझे खेद है कि चीजें मुश्किल हो रही हैं",
            "pain": "मुझे पता है कि दर्द कितना कष्टदायक होता है",
            "breathlessness": "सांस फूलना डरावना महसूस हो सकता है"
        }
    }
}
```

### Context Injection with Compassion

```python
class ContextInjector:
    async def inject_context(self, user_id: str, question: str) -> str:
        """Generate compassionate, personalized context."""
        profile = await self.profiles.get_profile(user_id)
        language = profile.preferences.language

        # Get longitudinal summary
        summary = await self.longitudinal.get_longitudinal_summary(user_id, days=30)

        # Build compassionate context
        context_parts = []

        # Check if returning user (continuity)
        if profile.total_sessions > 1:
            context_parts.append(self._template("welcome_back", language))

        # Recent symptom trends (with empathy)
        for entity, trend_summary in summary.get("summaries", {}).items():
            if "symptom" in entity and trend_summary.trend == TemporalTrend.WORSENING:
                context_parts.append(
                    self._empathize_worsening(entity, language)
                )

        # Active alerts (carefully phrased)
        active_alerts = summary.get("active_alerts", [])
        if active_alerts:
            context_parts.append(
                self._format_alerts_compassionately(active_alerts, language)
            )

        return " ".join(context_parts)
```

---

## FHIR Integration Path

### Phased Approach

| Phase | Description | Timeline |
|-------|-------------|----------|
| **Phase 1** | FHIR data model definitions | Sprint 1-2 |
| **Phase 2** | Import from FHIR JSON files | Sprint 3 |
| **Phase 3** | Export to FHIR format | Sprint 4 |
| **Phase 4** | Live FHIR API integration | Future |
| **Phase 5** | Bidirectional sync | Future |

### FHIR Data Model

```python
@dataclass
class FHIRObservation:
    """FHIR Observation resource."""
    resource_type: str = "Observation"
    status: str = "final"
    category: str = "vital-signs"  # or "symptom"
    code: Dict[str, str] = None  # LOINC/SNOMED
    subject_reference: str = None  # Patient reference
    effective_datetime: datetime = None
    value_codeable_concept: Optional[Dict] = None

class FHIRAdapter:
    """Convert between internal and FHIR formats."""

    @staticmethod
    def to_fhir(observation: TimestampedObservation, patient_fhir_id: str) -> FHIRObservation:
        """Convert internal observation to FHIR format."""
        return FHIRObservation(
            status="final",
            category=observation.category,
            code={"coding": [{"system": "http://snomed.info/sct", "code": observation.entity_name}]},
            subject_reference=f"Patient/{patient_fhir_id}",
            effective_datetime=observation.timestamp,
            value_codeable_concept={"text": observation.value_text}
        )
```

---

## Implementation Phases

### Phase 1: Foundation (Sprint 1)

**Goal**: Create core longitudinal data structures

- [ ] Create `personalization/longitudinal_memory.py`
- [ ] Implement `TimestampedObservation` base class
- [ ] Implement `LongitudinalPatientRecord`
- [ ] Implement `LongitudinalMemoryManager` with file storage
- [ ] Unit tests for data structures
- [ ] Commit: `feat(longitudinal): Phase 1 - core data structures`

### Phase 2: Cross-Modal Aggregation (Sprint 2)

**Goal**: Extract observations from all data sources

- [ ] Create `personalization/cross_modal_aggregator.py`
- [ ] Implement `VoiceDataExtractor` with LLM extraction
- [ ] Implement `WhatsAppDataExtractor`
- [ ] Implement `DocumentDataExtractor`
- [ ] Integration with Bolna webhooks
- [ ] Integration with WhatsApp bot
- [ ] Unit tests
- [ ] Commit: `feat(longitudinal): Phase 2 - cross-modal aggregation`

### Phase 3: Query Pipeline Integration (Sprint 3)

**Goal**: Use patient context in RAG queries

- [ ] Create `personalization/context_injector.py`
- [ ] Modify `simple_rag_server.py` query() method
- [ ] Implement context building for LLM prompts
- [ ] Add multi-language context formatting
- [ ] End-to-end tests
- [ ] Commit: `feat(longitudinal): Phase 3 - query integration`

### Phase 4: Temporal Reasoning (Sprint 4)

**Goal**: Analyze trends and patterns

- [ ] Create `personalization/temporal_reasoner.py`
- [ ] Implement trend detection algorithms
- [ ] Implement medication effectiveness analysis
- [ ] Implement correlation detection
- [ ] Commit: `feat(longitudinal): Phase 4 - temporal reasoning`

### Phase 5: Proactive Monitoring (Sprint 5)

**Goal**: Automated alerts and check-ins

- [ ] Create `personalization/alert_manager.py`
- [ ] Implement monitoring rule engine
- [ ] Create default palliative care rules
- [ ] Implement WhatsApp alert delivery
- [ ] Admin UI for alert management
- [ ] Commit: `feat(longitudinal): Phase 5 - proactive monitoring`

### Phase 6: Care Team Coordination (Sprint 6)

**Goal**: Track and coordinate with care team

- [ ] Extend `LongitudinalPatientRecord` with care team
- [ ] Implement provider attribution
- [ ] Create care team dashboard
- [ ] Implement caregiver notifications
- [ ] Commit: `feat(longitudinal): Phase 6 - care team coordination`

### Phase 7: FHIR Foundation (Sprint 7)

**Goal**: Prepare for EHR integration

- [ ] Create `personalization/fhir_adapter.py`
- [ ] Implement FHIR data models
- [ ] Implement JSON import/export
- [ ] Create FHIR schema validation
- [ ] Commit: `feat(longitudinal): Phase 7 - FHIR foundation`

---

## File Storage Schema

### Directory Structure

```
data/
├── longitudinal_memory/          # NEW - 1-5 year records
│   ├── {patient_id}_longitudinal.json
│   ├── index.json               # Patient index
│   └── alerts/                  # Active alerts
│       ├── {alert_id}.json
│       └── index.json
├── patient_context/             # EXISTING - 90-day current state
│   └── {user_id}_context.json
├── user_profiles/               # EXISTING
│   └── {user_id}.json
├── interaction_history/         # EXISTING - 30-day sessions
│   └── {user_id}_history.json
├── cross_modal_cache/           # NEW - Extraction cache
│   └── {source_id}_extracted.json
└── conversations.json           # EXISTING - monolithic
```

### Longitudinal Record Schema

```json
{
  "patient_id": "user_abc123def456",
  "created_at": "2025-01-01T00:00:00",
  "updated_at": "2025-01-08T12:30:00",
  "observations": [
    {
      "observation_id": "obs_001",
      "timestamp": "2025-01-08T10:00:00",
      "source_type": "voice_call",
      "source_id": "conv_123",
      "reported_by": "patient",
      "category": "symptom",
      "entity_name": "pain",
      "value": "3",
      "value_text": "Severe pain in lower back",
      "metadata": {}
    }
  ],
  "care_team": [
    {
      "provider_id": "dr_xyz",
      "name": "Dr. Sharma",
      "role": "doctor",
      "organization": "City Hospital",
      "primary_contact": true,
      "first_contact": "2025-01-01T00:00:00",
      "last_contact": "2025-01-08T10:00:00",
      "total_interactions": 5
    }
  ],
  "active_alerts": [],
  "alert_history": [],
  "monitoring_rules": [...],
  "conversations_index": {
    "conv_123": ["obs_001", "obs_002"]
  },
  "documents_index": {},
  "data_sources": ["voice_call", "whatsapp"],
  "total_observations": 42,
  "observation_date_range": ["2025-01-01T00:00:00", "2025-01-08T12:30:00"]
}
```

---

## Critical Files Summary

### Must Create

1. **`personalization/longitudinal_memory.py`**
   - `LongitudinalPatientRecord` - Main data structure
   - `TimestampedObservation` - Base observation class
   - `SymptomObservation`, `MedicationEvent` - Specialized observations
   - `TimeSeriesSummary` - Temporal aggregation
   - `MonitoringAlert`, `MonitoringRule` - Proactive monitoring
   - `LongitudinalMemoryManager` - Storage and retrieval

2. **`personalization/context_injector.py`**
   - `ContextInjector` - Integrates patient context into queries
   - Multi-language compassionate formatting
   - Context hierarchy management

3. **`personalization/cross_modal_aggregator.py`**
   - `CrossModalAggregator` - Main coordinator
   - `VoiceDataExtractor`, `WhatsAppDataExtractor`, `DocumentDataExtractor`

4. **`personalization/temporal_reasoner.py`**
   - `TemporalReasoner` - Trend detection
   - `SymptomProgressionReport` - Progression analysis

5. **`personalization/alert_manager.py`**
   - `AlertManager` - Alert delivery
   - Default palliative care monitoring rules

### Must Modify

1. **`simple_rag_server.py`** (Line 973)
   - Integrate `ContextInjector` into `query()` method
   - Add longitudinal managers to `__init__`
   - Trigger observation extraction after queries

2. **`personalization/__init__.py`**
   - Add exports for new modules

3. **`bolna_integration/webhooks.py`**
   - Trigger observation extraction after calls

4. **`whatsapp_bot.py`**
   - Trigger observation extraction from messages

---

## Compassion First: Design Principles

1. **Always acknowledge context**: "I remember that you mentioned..." before providing information

2. **Empathy before advice**: Validate feelings before offering solutions

3. **Continuity matters**: Reference previous conversations naturally

4. **Respect the journey**: Recognize that patients are on a difficult path

5. **Multi-language compassion**: Templates in Hindi, Bengali, Tamil, etc.

6. **Alert with care**: Even concerning news is delivered gently

7. **Caregiver recognition**: Acknowledge the burden on caregivers

---

*Document Version: 1.0.0*
*Created: 2025-01-08*
*For: Palli Sahayak - Compassionate Voice AI for Palliative Care*
