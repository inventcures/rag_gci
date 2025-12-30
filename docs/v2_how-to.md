# Palli Sahayak V2 Features: Complete How-To Guide

This guide provides detailed instructions for using the newly implemented V2 features:
1. **Clinical Validation Pipeline** - Ensure safe, accurate medical responses
2. **User Personalization** - Deliver context-aware, personalized experiences
3. **Real-time Analytics Dashboard** - Monitor system health and impact

---

## Table of Contents

1. [Overview](#1-overview)
2. [Clinical Validation Pipeline](#2-clinical-validation-pipeline)
3. [User Personalization System](#3-user-personalization-system)
4. [Real-time Analytics Dashboard](#4-real-time-analytics-dashboard)
5. [Integration Guide](#5-integration-guide)
6. [API Reference](#6-api-reference)
7. [Best Practices](#7-best-practices)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Overview

### 1.1 What's New in V2

| Feature | Module | Purpose |
|---------|--------|---------|
| Clinical Validation | `clinical_validation/` | Validate medical accuracy, detect hallucinations, track safety |
| User Personalization | `personalization/` | User profiles, patient context, interaction history |
| Analytics Dashboard | `analytics/` | Real-time metrics, usage trends, health monitoring |

### 1.2 Module Structure

```
rag_gci/
├── clinical_validation/
│   ├── __init__.py
│   ├── validator.py          # ClinicalValidator
│   ├── expert_sampling.py    # ExpertSampler
│   ├── feedback.py           # FeedbackCollector
│   └── metrics.py            # ValidationMetrics
│
├── personalization/
│   ├── __init__.py
│   ├── user_profile.py       # UserProfileManager
│   ├── context_memory.py     # ContextMemory
│   └── interaction_history.py # InteractionHistory
│
└── analytics/
    ├── __init__.py
    ├── realtime_metrics.py   # RealtimeMetrics
    ├── usage_analytics.py    # UsageAnalytics
    └── dashboard.py          # AnalyticsDashboard
```

### 1.3 Quick Start

```python
# Import V2 features
from clinical_validation import ClinicalValidator, ExpertSampler, FeedbackCollector, ValidationMetrics
from personalization import UserProfileManager, ContextMemory, InteractionHistory
from analytics import AnalyticsDashboard, RealtimeMetrics, UsageAnalytics

# Initialize components
validator = ClinicalValidator()
profile_manager = UserProfileManager()
dashboard = AnalyticsDashboard()
```

---

## 2. Clinical Validation Pipeline

### 2.1 Overview

The Clinical Validation Pipeline ensures medical accuracy and safety through:
- **Automated Validation**: Dosage checks, safety checks, hallucination detection
- **Expert Sampling**: Human review for quality assurance
- **Feedback Collection**: User ratings and issue reporting
- **Metrics Tracking**: Continuous quality monitoring

### 2.2 ClinicalValidator

#### 2.2.1 Basic Usage

```python
from clinical_validation import ClinicalValidator

# Initialize
validator = ClinicalValidator()

# Validate a response
result = await validator.validate(
    query="What is the dose of morphine for severe pain?",
    response="For severe pain, morphine can be started at 5-10mg orally every 4 hours.",
    sources=[
        {"title": "WHO Pain Guidelines", "content": "Morphine starting dose..."}
    ],
    context={"language": "en-IN", "user_role": "caregiver"}
)

# Check result
if result["is_valid"]:
    print("Response passed validation")
    print(f"Confidence: {result['confidence_score']}")
else:
    print("Validation issues found:")
    for issue in result["issues"]:
        print(f"  - [{issue['level']}] {issue['category']}: {issue['message']}")
```

#### 2.2.2 Validation Result Structure

```python
{
    "is_valid": True,                    # Overall pass/fail
    "confidence_score": 0.85,            # 0.0 to 1.0
    "checks_passed": [                   # List of passed checks
        "medication_dosage",
        "safety_check",
        "citation_check"
    ],
    "issues": [                          # List of issues found
        {
            "category": "hallucination",
            "level": "warning",          # info, warning, error, critical
            "message": "Could not verify claim about...",
            "details": {...}
        }
    ],
    "medications_mentioned": ["morphine"],
    "disclaimers_added": ["Consult your doctor before..."]
}
```

#### 2.2.3 Medication Dosage Validation

The validator includes safe dosage ranges for common palliative care medications:

```python
# Supported medications
MEDICATION_DOSAGES = {
    "morphine": {
        "oral_mg_per_dose": (2.5, 200),
        "iv_mg_per_dose": (1, 100),
        "max_daily_mg": 600
    },
    "oxycodone": {
        "oral_mg_per_dose": (2.5, 80),
        "max_daily_mg": 400
    },
    "fentanyl": {
        "patch_mcg_per_hour": (12, 100),
        "iv_mcg_per_dose": (25, 200)
    },
    "paracetamol": {
        "oral_mg_per_dose": (500, 1000),
        "max_daily_mg": 4000
    },
    # ... and more
}
```

### 2.3 ExpertSampler

#### 2.3.1 Automatic Sampling

```python
from clinical_validation import ExpertSampler

# Initialize with custom rates
sampler = ExpertSampler(
    storage_path="data/expert_samples",
    sample_rate=0.05,           # 5% base sampling
    max_samples_per_day=100
)

# Maybe sample a response (based on priority and rate)
sample = await sampler.maybe_sample(
    query="How to manage breakthrough pain?",
    response="For breakthrough pain, use...",
    language="hi-IN",
    sources=[...],
    validation_result=validation_result,
    session_id="sess_123",
    user_id="user_456"
)

if sample:
    print(f"Response sampled for expert review: {sample.sample_id}")
```

#### 2.3.2 Sampling Rates by Priority

| Priority | Trigger | Sample Rate |
|----------|---------|-------------|
| **Normal** | Standard responses | 5% |
| **High** | Validation warnings, low confidence (<0.7) | 50% |
| **Critical** | Safety issues, validation errors | 100% |

#### 2.3.3 Force Sampling

```python
# Force sample specific responses (e.g., from admin review)
sample = await sampler.force_sample(
    query="...",
    response="...",
    language="en-IN",
    sources=[...],
    validation_result={...},
    reason="admin_review"
)
```

#### 2.3.4 Submit Expert Review

```python
from clinical_validation.expert_sampling import ReviewStatus

# Submit expert review for a sample
updated_sample = await sampler.submit_review(
    sample_id="abc123def456",
    reviewer_id="expert_dr_sharma",
    status=ReviewStatus.APPROVED,
    accuracy_score=9.0,       # 0-10
    completeness_score=8.5,   # 0-10
    safety_score=10.0,        # 0-10
    comments="Response is accurate and appropriate.",
    tags=["pain_management", "opioids"]
)
```

#### 2.3.5 Get Pending Reviews

```python
# Get samples awaiting expert review
pending = await sampler.get_pending_samples(
    limit=50,
    priority=SamplingPriority.CRITICAL  # Optional filter
)

for sample in pending:
    print(f"ID: {sample.sample_id}")
    print(f"Query: {sample.query}")
    print(f"Priority: {sample.priority.value}")
```

### 2.4 FeedbackCollector

#### 2.4.1 Collect User Feedback

```python
from clinical_validation import FeedbackCollector
from clinical_validation.feedback import FeedbackType, FeedbackChannel

collector = FeedbackCollector(
    storage_path="data/feedback",
    auto_prompt_rate=0.2  # Ask for feedback 20% of the time
)

# Collect detailed feedback
feedback = await collector.collect_feedback(
    query="How to give morphine to my mother?",
    response="Morphine should be given...",
    rating=5,                              # 1-5 stars
    feedback_type=FeedbackType.HELPFUL,
    channel=FeedbackChannel.VOICE_PROMPT,
    session_id="sess_123",
    language="hi-IN"
)
```

#### 2.4.2 Quick Yes/No Feedback

```python
# Simple helpful/not helpful feedback
feedback = await collector.collect_quick_feedback(
    query="...",
    response="...",
    is_helpful=True,
    channel=FeedbackChannel.WHATSAPP,
    session_id="sess_123"
)
```

#### 2.4.3 Report Issues

```python
# Report a problem with a response
feedback = await collector.report_issue(
    query="What is the dose of morphine?",
    response="...",
    issue_type=FeedbackType.INCORRECT,
    description="The dosage mentioned is too high",
    channel=FeedbackChannel.ADMIN
)
```

#### 2.4.4 Voice Prompts for Feedback

The collector includes multilingual voice prompts:

```python
# Get prompts for a language
prompts = collector.get_feedback_prompt("hi-IN")

# Use in voice flow:
# 1. Ask: prompts["ask"] - "क्या यह जानकारी उपयोगी थी? हां या नहीं बोलें।"
# 2. On yes: prompts["thanks_positive"]
# 3. On no: prompts["thanks_negative"] + prompts["follow_up"]
```

### 2.5 ValidationMetrics

#### 2.5.1 Track Validation Results

```python
from clinical_validation import ValidationMetrics

metrics = ValidationMetrics(
    storage_path="data/metrics",
    aggregation_interval_minutes=60
)

# Record validation result
await metrics.record_validation(
    validation_result=result,
    response_time_ms=250.5
)

# Record expert review
await metrics.record_expert_review(
    accuracy_score=9.0,
    completeness_score=8.5,
    safety_score=10.0,
    validation_agreed=True
)

# Record user feedback
await metrics.record_user_feedback(rating=5)
```

#### 2.5.2 Get Metrics Dashboard

```python
# Get dashboard-ready data
dashboard_data = await metrics.get_dashboard_data()

print(f"Quality Score: {dashboard_data['quality_score']}")
print(f"Weekly Summary: {dashboard_data['weekly_summary']}")
print(f"Alerts: {dashboard_data['alerts']}")
```

---

## 3. User Personalization System

### 3.1 Overview

The personalization system provides:
- **User Profiles**: Role detection, preferences, session tracking
- **Patient Context**: Conditions, symptoms, medications memory
- **Interaction History**: Conversation continuity, follow-up detection

### 3.2 UserProfileManager

#### 3.2.1 Create/Get User Profile

```python
from personalization import UserProfileManager

profile_manager = UserProfileManager(
    storage_path="data/user_profiles",
    cache_size=100
)

# Get or create profile (phone number is hashed for privacy)
profile = await profile_manager.get_or_create_profile(
    phone_number="+919876543210",
    language="hi-IN"
)

print(f"User ID: {profile.user_id}")
print(f"Role: {profile.role.value}")
print(f"Total Sessions: {profile.total_sessions}")
```

#### 3.2.2 Automatic Role Detection

```python
from personalization.user_profile import UserRole

# Detect role from conversation
text = "I am caring for my mother who has cancer"
detected_role = profile_manager.detect_role(text, language="en-IN")

if detected_role:
    print(f"Detected role: {detected_role.value}")  # "caregiver"

    # Update profile
    await profile_manager.update_role(profile.user_id, detected_role)
```

#### 3.2.3 Role-Based Keywords

| Role | English Keywords | Hindi Keywords |
|------|------------------|----------------|
| Patient | "i have", "my pain", "i feel" | "मुझे", "मेरा दर्द" |
| Caregiver | "my mother", "caring for", "patient" | "मेरी माँ", "देखभाल" |
| Healthcare | "doctor", "nurse", "my patient" | "डॉक्टर", "नर्स" |

#### 3.2.4 Update Preferences

```python
# Update user preferences
await profile_manager.update_preferences(
    user_id=profile.user_id,
    language="hi-IN",
    communication_style="simple",  # simple, detailed, clinical, empathetic
    voice_speed="slow"
)
```

#### 3.2.5 Generate System Context

```python
# Get context for LLM system prompt
context = profile_manager.get_system_context(profile)
# Returns: "The user is a caregiver for a patient. Provide practical care guidance..."
```

### 3.3 ContextMemory

#### 3.3.1 Track Patient Context

```python
from personalization import ContextMemory

context_memory = ContextMemory(
    storage_path="data/patient_context",
    context_expiry_days=90
)

# Get or create context
patient_ctx = await context_memory.get_or_create_context(user_id)
```

#### 3.3.2 Set Patient Condition

```python
# Set primary condition
await context_memory.set_condition(
    user_id=user_id,
    condition="cancer",
    stage="advanced"
)
```

#### 3.3.3 Track Symptoms

```python
# Add symptoms (auto-updates if exists)
await context_memory.add_symptom(
    user_id=user_id,
    name="pain",
    severity="moderate"  # mild, moderate, severe
)

await context_memory.add_symptom(
    user_id=user_id,
    name="nausea",
    severity="mild"
)
```

#### 3.3.4 Track Medications

```python
# Add medication
await context_memory.add_medication(
    user_id=user_id,
    name="morphine",
    dosage="10mg",
    frequency="every 4 hours",
    purpose="pain management"
)
```

#### 3.3.5 Track Allergies

```python
# Add allergy
await context_memory.add_allergy(
    user_id=user_id,
    allergy="ibuprofen"
)
```

#### 3.3.6 Auto-Extract from Conversation

```python
# Automatically extract and update context from conversation
text = "My mother has cancer and is taking morphine for pain"
updated_ctx = await context_memory.update_from_conversation(
    user_id=user_id,
    text=text,
    language="en-IN"
)

# Extracts: condition=cancer, symptom=pain, medication=morphine
```

#### 3.3.7 Get Context Summary for LLM

```python
# Get text summary for injection into prompt
summary = context_memory.get_context_summary(patient_ctx)
# Returns: "Patient context: Patient has cancer (advanced). Current symptoms: pain, nausea. Current medications: morphine."
```

### 3.4 InteractionHistory

#### 3.4.1 Session Management

```python
from personalization import InteractionHistory

history = InteractionHistory(
    storage_path="data/interaction_history",
    max_history_days=30
)

# Start or get existing session
session = await history.get_or_create_session(
    user_id=user_id,
    language="hi-IN",
    channel="voice"  # voice, whatsapp, web
)

print(f"Session ID: {session.session_id}")
```

#### 3.4.2 Record Conversation Turns

```python
# Add a conversation turn
turn = await history.add_turn(
    user_id=user_id,
    query="How to manage pain at night?",
    response="For nighttime pain management...",
    language="hi-IN",
    used_rag=True,
    rag_sources=["WHO_Pain_Guidelines.pdf"],
    response_time_ms=350
)

print(f"Turn ID: {turn.turn_id}")
print(f"Query Type: {turn.query_type.value}")  # symptom_inquiry, medication_question, etc.
```

#### 3.4.3 Record Feedback on Turn

```python
# Record feedback for a specific turn
await history.record_feedback(
    user_id=user_id,
    turn_id=turn.turn_id,
    was_helpful=True
)
```

#### 3.4.4 End Session

```python
# End and save session
session = await history.end_session(user_id)
print(f"Session duration: {session.duration_seconds}s")
print(f"Total turns: {session.turn_count}")
```

#### 3.4.5 Get Recent Context

```python
# Get recent turns for context injection
recent_turns = await history.get_recent_context(
    user_id=user_id,
    max_turns=5
)

# Generate context summary
summary = history.get_context_summary(recent_turns)
```

#### 3.4.6 Detect Follow-up Queries

```python
# Check if query is a follow-up
is_followup = history.is_followup_query(
    query="What about at night?",
    recent_turns=recent_turns
)

if is_followup:
    # Include previous context in RAG query
    last_topic = history.get_last_topic(recent_turns)
```

---

## 4. Real-time Analytics Dashboard

### 4.1 Overview

The analytics system provides:
- **Real-time Metrics**: Latency, throughput, error rates
- **Usage Analytics**: Daily stats, trends, distributions
- **Unified Dashboard**: Combined view with health status

### 4.2 RealtimeMetrics

#### 4.2.1 Record Metrics

```python
from analytics import RealtimeMetrics, MetricType

metrics = RealtimeMetrics(
    window_seconds=300,  # 5-minute rolling window
    enable_alerts=True
)

# Record response latency
await metrics.record_latency(
    MetricType.RESPONSE_LATENCY_MS,
    latency_ms=250.5
)

# Record RAG query
await metrics.record_rag_query(
    success=True,
    latency_ms=120.0,
    sources_count=3
)

# Record validation
await metrics.record_validation(
    passed=True,
    confidence=0.92
)

# Record user feedback
await metrics.record_user_feedback(rating=5)

# Record error
await metrics.record_error(error_type="rag_timeout")
```

#### 4.2.2 Get Latency Statistics

```python
# Get detailed latency stats
stats = await metrics.get_latency_stats(MetricType.RESPONSE_LATENCY_MS)

print(f"Count: {stats['count']}")
print(f"Average: {stats['avg']}ms")
print(f"P50: {stats['p50']}ms")
print(f"P95: {stats['p95']}ms")
print(f"P99: {stats['p99']}ms")
```

#### 4.2.3 Get Health Status

```python
# Get system health
health = await metrics.get_health_status()

print(f"Status: {health['status']}")  # healthy, degraded, unhealthy
print(f"Alerts: {health['alerts']}")
print(f"Summary: {health['summary']}")
```

#### 4.2.4 Alert Thresholds

| Metric | P95 Threshold | P99 Threshold | Avg Threshold |
|--------|---------------|---------------|---------------|
| Response Latency | 2000ms | 5000ms | - |
| Error Rate | - | - | 5% |
| RAG Success Rate | - | - | 90% |
| Validation Pass Rate | - | - | 95% |

### 4.3 UsageAnalytics

#### 4.3.1 Record Usage Events

```python
from analytics import UsageAnalytics

usage = UsageAnalytics(
    storage_path="data/analytics",
    retention_days=90
)

# Record query event
await usage.record_query(
    user_id="user_123",
    language="hi-IN",
    query_type="symptom_inquiry",
    used_rag=True,
    rag_success=True,
    validation_passed=True
)

# Record session end
await usage.record_session(
    user_id="user_123",
    duration_seconds=245.5
)

# Record feedback
await usage.record_feedback(is_positive=True)
```

#### 4.3.2 Get Daily Statistics

```python
# Get today's stats
today_stats = await usage.get_daily_stats("2025-01-15")

print(f"Total Queries: {today_stats.total_queries}")
print(f"Unique Users: {today_stats.unique_users}")
print(f"RAG Success Rate: {today_stats.rag_success / today_stats.rag_queries}")
```

#### 4.3.3 Get Trend Analysis

```python
# Analyze trends over 7 days
trends = await usage.get_trend_analysis(days=7)

print(f"Query Trend: {trends['trends']['queries']}")  # growing, stable, declining
print(f"Peak Hours: {trends['peak_hours']}")
print(f"Language Distribution: {trends['language_distribution']}")
```

#### 4.3.4 Get Language Statistics

```python
# Get language breakdown
lang_stats = await usage.get_language_stats(days=7)

for lang, data in lang_stats["by_language"].items():
    print(f"{lang}: {data['count']} queries ({data['percentage']}%)")
```

### 4.4 AnalyticsDashboard

#### 4.4.1 Complete Dashboard

```python
from analytics import AnalyticsDashboard

dashboard = AnalyticsDashboard(
    metrics_path="data/metrics",
    analytics_path="data/analytics"
)

# Optional: Connect validation metrics and personalization
dashboard.set_validation_metrics(validation_metrics)
dashboard.set_personalization_stats(profile_manager, context_memory)
dashboard.set_interaction_history(interaction_history)
```

#### 4.4.2 Record Complete Query Event

```python
# Record all metrics at once
await dashboard.record_query(
    user_id="user_123",
    query="How to manage pain?",
    response="For pain management...",
    language="hi-IN",
    query_type="symptom_inquiry",
    used_rag=True,
    rag_success=True,
    rag_latency_ms=120.0,
    response_latency_ms=350.0,
    validation_passed=True,
    validation_confidence=0.92,
    sources_count=3
)
```

#### 4.4.3 Get Full Dashboard Data

```python
# Get complete dashboard data
data = await dashboard.get_dashboard_data()

# Structure:
# {
#   "timestamp": "...",
#   "health": {...},           # System health status
#   "sections": {
#     "realtime": {...},       # Real-time metrics
#     "usage": {...},          # Usage analytics
#     "languages": {...},      # Language distribution
#     "hourly_distribution": [...],  # Hourly query distribution
#     "validation": {...},     # Validation metrics (if connected)
#     "personalization": {...}, # User stats (if connected)
#     "interactions": {...}    # Interaction stats (if connected)
#   }
# }
```

#### 4.4.4 Get Lightweight Realtime Snapshot

```python
# For frequent polling (e.g., every 10 seconds)
snapshot = await dashboard.get_realtime_snapshot()

print(f"Status: {snapshot['status']}")
print(f"Queries/min: {snapshot['queries_per_minute']}")
print(f"Avg Latency: {snapshot['avg_latency_ms']}ms")
print(f"Active Sessions: {snapshot['active_sessions']}")
```

#### 4.4.5 Export Analytics Data

```python
# Export for external analysis
export = await dashboard.export_data(days=30, format="json")

# Save to file
import json
with open("analytics_export.json", "w") as f:
    json.dump(export, f, indent=2)
```

---

## 5. Integration Guide

### 5.1 Complete RAG Pipeline Integration

```python
import asyncio
import time
from clinical_validation import ClinicalValidator, ExpertSampler, FeedbackCollector, ValidationMetrics
from personalization import UserProfileManager, ContextMemory, InteractionHistory
from analytics import AnalyticsDashboard

class EnhancedRAGPipeline:
    def __init__(self):
        # Clinical Validation
        self.validator = ClinicalValidator()
        self.sampler = ExpertSampler()
        self.feedback = FeedbackCollector()
        self.val_metrics = ValidationMetrics()

        # Personalization
        self.profiles = UserProfileManager()
        self.context = ContextMemory()
        self.history = InteractionHistory()

        # Analytics
        self.dashboard = AnalyticsDashboard()
        self.dashboard.set_validation_metrics(self.val_metrics)
        self.dashboard.set_personalization_stats(self.profiles, self.context)
        self.dashboard.set_interaction_history(self.history)

    async def process_query(
        self,
        phone_number: str,
        query: str,
        language: str = "en-IN"
    ):
        start_time = time.time()

        # 1. Get/Create User Profile
        profile = await self.profiles.get_or_create_profile(phone_number, language)

        # 2. Get Patient Context
        patient_ctx = await self.context.get_or_create_context(profile.user_id)

        # 3. Get Session & Recent Context
        session = await self.history.get_or_create_session(
            profile.user_id, language, "voice"
        )
        recent_turns = await self.history.get_recent_context(profile.user_id)

        # 4. Detect Role (if unknown)
        if profile.role.value == "unknown":
            detected_role = self.profiles.detect_role(query, language)
            if detected_role:
                await self.profiles.update_role(profile.user_id, detected_role)

        # 5. Update Patient Context from Query
        await self.context.update_from_conversation(
            profile.user_id, query, language
        )

        # 6. Build Enhanced Context
        system_context = self.profiles.get_system_context(profile)
        patient_summary = self.context.get_context_summary(patient_ctx)
        history_summary = self.history.get_context_summary(recent_turns)

        enhanced_prompt = f"""
        {system_context}
        {patient_summary}
        {history_summary}
        """

        # 7. Query RAG (your existing pipeline)
        rag_start = time.time()
        rag_result = await self.query_rag(query, enhanced_prompt)
        rag_latency = (time.time() - rag_start) * 1000

        response = rag_result["answer"]
        sources = rag_result["sources"]

        # 8. Validate Response
        validation = await self.validator.validate(
            query=query,
            response=response,
            sources=sources,
            context={"language": language, "user_role": profile.role.value}
        )

        # 9. Maybe Sample for Expert Review
        await self.sampler.maybe_sample(
            query=query,
            response=response,
            language=language,
            sources=sources,
            validation_result=validation,
            session_id=session.session_id,
            user_id=profile.user_id
        )

        # 10. Record Interaction
        turn = await self.history.add_turn(
            user_id=profile.user_id,
            query=query,
            response=response,
            language=language,
            used_rag=True,
            rag_sources=[s.get("title", "") for s in sources],
            response_time_ms=(time.time() - start_time) * 1000
        )

        # 11. Record Metrics
        await self.val_metrics.record_validation(
            validation, response_time_ms=(time.time() - start_time) * 1000
        )

        # 12. Record to Dashboard
        await self.dashboard.record_query(
            user_id=profile.user_id,
            query=query,
            response=response,
            language=language,
            query_type=turn.query_type.value,
            used_rag=True,
            rag_success=len(sources) > 0,
            rag_latency_ms=rag_latency,
            response_latency_ms=(time.time() - start_time) * 1000,
            validation_passed=validation["is_valid"],
            validation_confidence=validation["confidence_score"],
            sources_count=len(sources)
        )

        return {
            "response": response,
            "sources": sources,
            "validation": validation,
            "turn_id": turn.turn_id,
            "session_id": session.session_id
        }

    async def query_rag(self, query: str, context: str):
        # Your existing RAG implementation
        pass
```

### 5.2 Voice Flow Integration

```python
async def handle_voice_interaction(phone_number: str, audio_input: bytes):
    pipeline = EnhancedRAGPipeline()

    # 1. Transcribe audio
    query = await transcribe(audio_input)
    language = detect_language(query)

    # 2. Process with enhanced pipeline
    result = await pipeline.process_query(phone_number, query, language)

    # 3. Check if should ask for feedback
    if pipeline.feedback.should_prompt_feedback():
        prompts = pipeline.feedback.get_feedback_prompt(language)
        result["feedback_prompt"] = prompts["ask"]

    # 4. Synthesize response
    audio_response = await synthesize(result["response"], language)

    return audio_response, result
```

### 5.3 WhatsApp Bot Integration

```python
async def handle_whatsapp_message(from_number: str, message: str):
    pipeline = EnhancedRAGPipeline()

    # Get language from message or profile
    profile = await pipeline.profiles.get_or_create_profile(from_number)
    language = profile.preferences.language

    # Process query
    result = await pipeline.process_query(from_number, message, language)

    # Format response for WhatsApp
    response_text = result["response"]

    # Add source citations
    if result["sources"]:
        response_text += "\n\n📚 Sources: "
        response_text += ", ".join(s["title"] for s in result["sources"][:3])

    # Add validation disclaimer if needed
    if not result["validation"]["is_valid"]:
        response_text += "\n\n⚠️ Please verify this information with a healthcare provider."

    return response_text
```

---

## 6. API Reference

### 6.1 Clinical Validation

#### ClinicalValidator
```python
class ClinicalValidator:
    async def validate(
        query: str,
        response: str,
        sources: List[Dict],
        context: Optional[Dict] = None
    ) -> Dict[str, Any]
```

#### ExpertSampler
```python
class ExpertSampler:
    async def maybe_sample(...) -> Optional[SampleRecord]
    async def force_sample(...) -> SampleRecord
    async def get_pending_samples(limit: int, priority: Optional[SamplingPriority]) -> List[SampleRecord]
    async def submit_review(...) -> Optional[SampleRecord]
    async def get_statistics() -> Dict[str, Any]
```

#### FeedbackCollector
```python
class FeedbackCollector:
    async def collect_feedback(...) -> UserFeedback
    async def collect_quick_feedback(...) -> UserFeedback
    async def report_issue(...) -> UserFeedback
    async def get_statistics(days: int) -> Dict[str, Any]
```

#### ValidationMetrics
```python
class ValidationMetrics:
    async def record_validation(validation_result: Dict, response_time_ms: float)
    async def record_expert_review(accuracy, completeness, safety, agreed: bool)
    async def record_user_feedback(rating: int)
    async def get_dashboard_data() -> Dict[str, Any]
    async def get_summary(days: int) -> Dict[str, Any]
```

### 6.2 Personalization

#### UserProfileManager
```python
class UserProfileManager:
    async def get_or_create_profile(phone_number: str, language: str) -> UserProfile
    def detect_role(text: str, language: str) -> Optional[UserRole]
    async def update_role(user_id: str, role: UserRole) -> Optional[UserProfile]
    async def update_preferences(user_id: str, **preferences) -> Optional[UserProfile]
    def get_system_context(profile: UserProfile) -> str
```

#### ContextMemory
```python
class ContextMemory:
    async def get_or_create_context(user_id: str) -> PatientContext
    async def set_condition(user_id, condition, stage: Optional[str])
    async def add_symptom(user_id, name, severity)
    async def add_medication(user_id, name, dosage, frequency, purpose)
    async def add_allergy(user_id, allergy)
    async def update_from_conversation(user_id, text, language) -> PatientContext
    def get_context_summary(context: PatientContext) -> str
```

#### InteractionHistory
```python
class InteractionHistory:
    async def get_or_create_session(user_id, language, channel) -> Session
    async def add_turn(user_id, query, response, ...) -> ConversationTurn
    async def end_session(user_id) -> Optional[Session]
    async def get_recent_context(user_id, max_turns: int) -> List[ConversationTurn]
    def is_followup_query(query, recent_turns) -> bool
    def get_context_summary(turns) -> str
```

### 6.3 Analytics

#### RealtimeMetrics
```python
class RealtimeMetrics:
    async def record_latency(metric_type: MetricType, latency_ms: float)
    async def record_rag_query(success: bool, latency_ms: float, sources_count: int)
    async def record_validation(passed: bool, confidence: float)
    async def record_user_feedback(rating: int)
    async def record_error(error_type: str)
    async def get_latency_stats(metric_type) -> Dict
    async def get_health_status() -> Dict
```

#### UsageAnalytics
```python
class UsageAnalytics:
    async def record_query(user_id, language, query_type, ...)
    async def record_session(user_id, duration_seconds)
    async def record_feedback(is_positive: bool)
    async def get_daily_stats(date: str) -> Optional[DailyStats]
    async def get_trend_analysis(days: int) -> Dict
    async def get_language_stats(days: int) -> Dict
```

#### AnalyticsDashboard
```python
class AnalyticsDashboard:
    async def record_query(...)  # Records to all metrics
    async def get_dashboard_data() -> Dict
    async def get_realtime_snapshot() -> Dict
    async def get_alerts() -> List[Dict]
    async def export_data(days: int, format: str) -> Dict
```

---

## 7. Best Practices

### 7.1 Clinical Validation

1. **Always validate before returning responses** - Never skip validation for medical content
2. **Handle validation failures gracefully** - Add disclaimers, suggest professional consultation
3. **Review sampled responses regularly** - Set up a weekly expert review process
4. **Monitor hallucination rates** - Alert if rate exceeds 2%

### 7.2 Personalization

1. **Request minimal data** - Only collect what's needed for personalization
2. **Respect privacy** - Hash phone numbers, don't store raw PII
3. **Set appropriate expiry** - Patient context expires after 90 days of inactivity
4. **Handle role detection carefully** - Don't assume role without evidence

### 7.3 Analytics

1. **Use lightweight snapshots for frequent polling** - Avoid full dashboard calls
2. **Set appropriate alert thresholds** - Tune based on your traffic patterns
3. **Clean up old data** - Run `cleanup()` periodically
4. **Export data for external analysis** - Don't rely solely on built-in analytics

### 7.4 Performance

1. **Use caching** - Profile manager and context memory have built-in caching
2. **Batch metric recording** - Metrics are aggregated, not stored per-event
3. **Async everything** - All operations are async for non-blocking performance

---

## 8. Troubleshooting

### 8.1 Common Issues

#### Validation always failing
```python
# Check if sources are being passed correctly
result = await validator.validate(query, response, sources=[])  # Empty sources = lower confidence

# Ensure sources have required fields
sources = [{"title": "...", "content": "..."}]  # Both required
```

#### Profile not persisting
```python
# Check storage path exists and is writable
profile_manager = UserProfileManager(storage_path="data/user_profiles")
# Creates directory if not exists, but parent must exist
```

#### Metrics not showing
```python
# Force flush metrics (usually auto-flushed every hour)
await metrics.force_flush()

# Check if enough data for percentile calculation
# Need at least 1 data point for avg, 3+ for reliable percentiles
```

### 8.2 Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# All modules log to their respective loggers
# clinical_validation.validator
# personalization.user_profile
# analytics.realtime_metrics
```

### 8.3 Data Recovery

```python
# Profiles stored in: data/user_profiles/user_*.json
# Context stored in: data/patient_context/*_context.json
# History stored in: data/interaction_history/*_history.json
# Metrics stored in: data/metrics/metrics_*.json
# Analytics stored in: data/analytics/stats_*.json
```

---

## Conclusion

The V2 features provide a comprehensive framework for building a safe, personalized, and observable palliative care voice AI system. By integrating clinical validation, user personalization, and real-time analytics, Palli Sahayak can deliver trustworthy, context-aware responses while maintaining full operational visibility.

For questions or issues, please refer to the main documentation or open an issue on GitHub.

---

*Document Version: 2.0*
*Last Updated: December 2025*
*Authors: Palli Sahayak Development Team*
