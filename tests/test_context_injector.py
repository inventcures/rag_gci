"""
Unit Tests for Context Injector Module

Tests the compassionate, multi-language context injection
for personalized RAG queries in palliative care.
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from personalization.context_injector import (
    COMPASSIONATE_TEMPLATES,
    SYMPTOM_TRANSLATIONS,
    ContextInjector,
    PromptContextBuilder,
)
from personalization.longitudinal_memory import (
    LongitudinalMemoryManager,
    LongitudinalPatientRecord,
    SymptomObservation,
    MedicationEvent,
    SeverityLevel,
    DataSourceType,
    MedicationAction,
    TemporalTrend,
    AlertPriority,
    MonitoringAlert,
)
from personalization.user_profile import (
    UserProfileManager,
    UserProfile,
    UserPreferences,
    UserRole,
    CommunicationStyle,
)
from personalization.context_memory import (
    ContextMemory,
    PatientContext,
    Symptom,
    Medication,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_longitudinal_manager():
    """Create mock longitudinal memory manager."""
    manager = AsyncMock(spec=LongitudinalMemoryManager)

    # Default summary response
    manager.get_longitudinal_summary.return_value = {
        "patient_id": "patient_001",
        "total_observations": 10,
        "summaries": {
            "symptom:pain": {
                "entity_name": "pain",
                "trend": "worsening",
                "observations": 5
            }
        },
        "active_symptoms": [
            {"name": "pain", "severity": 3, "trend": "worsening"}
        ],
        "active_alerts": []
    }

    return manager


@pytest.fixture
def mock_user_profile_manager():
    """Create mock user profile manager."""
    manager = AsyncMock(spec=UserProfileManager)

    # Default profile
    profile = UserProfile(
        user_id="patient_001",
        role=UserRole.PATIENT,
        preferences=UserPreferences(
            language="en-IN",
            communication_style=CommunicationStyle.EMPATHETIC
        ),
        total_sessions=5,
        last_interaction=datetime.now()
    )
    manager.get_profile.return_value = profile

    return manager


@pytest.fixture
def mock_context_memory():
    """Create mock context memory."""
    memory = AsyncMock(spec=ContextMemory)

    # Default context
    context = PatientContext(
        user_id="patient_001",
        primary_condition="cancer",
        condition_stage="advanced",
        symptoms=[
            Symptom(name="pain", severity="severe", notes="2 weeks duration"),
            Symptom(name="fatigue", severity="moderate", notes="1 month duration")
        ],
        medications=[
            Medication(name="morphine", dosage="10mg", frequency="every 4 hours", purpose="pain"),
            Medication(name="ondansetron", dosage="4mg", frequency="as needed", purpose="nausea")
        ],
        has_caregiver=True,
        care_location="home"
    )
    memory.get_or_create_context.return_value = context

    return memory


@pytest.fixture
def context_injector(mock_longitudinal_manager, mock_user_profile_manager, mock_context_memory):
    """Create ContextInjector with mock dependencies."""
    return ContextInjector(
        longitudinal_manager=mock_longitudinal_manager,
        user_profile_manager=mock_user_profile_manager,
        context_memory=mock_context_memory
    )


# ============================================================================
# TEMPLATE TESTS
# ============================================================================

class TestCompassionateTemplates:
    """Test compassionate language templates."""

    def test_english_templates_exist(self):
        """Test English templates are defined."""
        assert "en-IN" in COMPASSIONATE_TEMPLATES
        templates = COMPASSIONATE_TEMPLATES["en-IN"]

        assert "welcome_back" in templates
        assert "empathy_worsening" in templates
        assert "symptom_trend" in templates
        assert "emotional_support" in templates

    def test_hindi_templates_exist(self):
        """Test Hindi templates are defined."""
        assert "hi-IN" in COMPASSIONATE_TEMPLATES
        templates = COMPASSIONATE_TEMPLATES["hi-IN"]

        assert "welcome_back" in templates
        assert "empathy_worsening" in templates

    def test_regional_language_templates(self):
        """Test regional language templates exist."""
        regional_languages = ["bn-IN", "ta-IN", "te-IN", "mr-IN", "gu-IN", "kn-IN", "ml-IN"]

        for lang in regional_languages:
            assert lang in COMPASSIONATE_TEMPLATES, f"Missing templates for {lang}"
            templates = COMPASSIONATE_TEMPLATES[lang]
            assert "welcome_back" in templates
            assert "symptom_trend" in templates

    def test_symptom_trend_templates(self):
        """Test symptom trend templates for all languages."""
        for lang, templates in COMPASSIONATE_TEMPLATES.items():
            if "symptom_trend" in templates:
                trend_templates = templates["symptom_trend"]
                assert "improving" in trend_templates
                assert "stable" in trend_templates
                assert "worsening" in trend_templates


class TestSymptomTranslations:
    """Test symptom name translations."""

    def test_pain_translations(self):
        """Test pain translations exist for all languages."""
        assert "pain" in SYMPTOM_TRANSLATIONS
        pain_translations = SYMPTOM_TRANSLATIONS["pain"]

        assert "hi-IN" in pain_translations
        assert pain_translations["hi-IN"] == "दर्द"

    def test_common_symptoms_translated(self):
        """Test common palliative symptoms are translated."""
        expected_symptoms = [
            "pain", "breathlessness", "nausea", "fatigue",
            "constipation", "anxiety"
        ]

        for symptom in expected_symptoms:
            assert symptom in SYMPTOM_TRANSLATIONS, f"Missing translation for {symptom}"


# ============================================================================
# CONTEXT INJECTOR TESTS
# ============================================================================

class TestContextInjector:
    """Test ContextInjector class."""

    @pytest.mark.asyncio
    async def test_inject_context_returns_string(self, context_injector):
        """Test that inject_context returns a string."""
        context = await context_injector.inject_context(
            user_id="patient_001",
            question="How can I manage my pain?"
        )

        assert isinstance(context, str)
        assert len(context) > 0

    @pytest.mark.asyncio
    async def test_inject_context_includes_welcome_back(self, context_injector):
        """Test welcome back message for returning users."""
        context = await context_injector.inject_context(
            user_id="patient_001"
        )

        # Should include welcome back for users with >1 session
        assert any(phrase in context for phrase in [
            "Welcome back",
            "वापसी पर स्वागत है"
        ])

    @pytest.mark.asyncio
    async def test_inject_context_includes_condition(self, context_injector):
        """Test primary condition is included."""
        context = await context_injector.inject_context(
            user_id="patient_001"
        )

        assert "cancer" in context.lower() or "managing" in context.lower()

    @pytest.mark.asyncio
    async def test_inject_context_includes_medications(self, context_injector):
        """Test current medications are included."""
        context = await context_injector.inject_context(
            user_id="patient_001"
        )

        assert "morphine" in context.lower() or "taking" in context.lower()

    @pytest.mark.asyncio
    async def test_inject_context_respects_max_length(self, context_injector):
        """Test context respects maximum length."""
        context = await context_injector.inject_context(
            user_id="patient_001",
            max_length=100
        )

        assert len(context) <= 100 or len(context) < 150  # Allow some overflow for word boundaries

    @pytest.mark.asyncio
    async def test_inject_context_new_user(
        self,
        mock_longitudinal_manager,
        mock_context_memory
    ):
        """Test context for new user (no profile)."""
        mock_profile_manager = AsyncMock(spec=UserProfileManager)
        mock_profile_manager.get_profile.return_value = None

        injector = ContextInjector(
            longitudinal_manager=mock_longitudinal_manager,
            user_profile_manager=mock_profile_manager,
            context_memory=mock_context_memory
        )

        context = await injector.inject_context(user_id="new_user")

        # Should return empty for new users
        assert context == ""

    @pytest.mark.asyncio
    async def test_inject_context_hindi(
        self,
        mock_longitudinal_manager,
        mock_context_memory
    ):
        """Test context injection in Hindi."""
        mock_profile_manager = AsyncMock(spec=UserProfileManager)
        profile = UserProfile(
            user_id="patient_001",
            role=UserRole.PATIENT,
            preferences=UserPreferences(
                language="hi-IN",
                communication_style=CommunicationStyle.EMPATHETIC
            ),
            total_sessions=5,
            last_interaction=datetime.now()
        )
        mock_profile_manager.get_profile.return_value = profile

        injector = ContextInjector(
            longitudinal_manager=mock_longitudinal_manager,
            user_profile_manager=mock_profile_manager,
            context_memory=mock_context_memory
        )

        context = await injector.inject_context(user_id="patient_001")

        # Should contain Hindi text
        assert any(char >= '\u0900' and char <= '\u097F' for char in context)

    @pytest.mark.asyncio
    async def test_inject_context_with_alerts(
        self,
        mock_longitudinal_manager,
        mock_user_profile_manager,
        mock_context_memory
    ):
        """Test context includes urgent alerts."""
        # Update mock to include alerts
        mock_longitudinal_manager.get_longitudinal_summary.return_value = {
            "patient_id": "patient_001",
            "total_observations": 10,
            "summaries": {},
            "active_symptoms": [],
            "active_alerts": [
                {
                    "priority": "high",
                    "title": "Pain Worsening",
                    "description": "Pain has been increasing over the past 3 days"
                }
            ]
        }

        injector = ContextInjector(
            longitudinal_manager=mock_longitudinal_manager,
            user_profile_manager=mock_user_profile_manager,
            context_memory=mock_context_memory
        )

        context = await injector.inject_context(user_id="patient_001")

        # Should include alert information
        assert "note" in context.lower() or "pain" in context.lower()

    @pytest.mark.asyncio
    async def test_inject_context_with_caregiver(self, context_injector):
        """Test context includes caregiver information."""
        context = await context_injector.inject_context(user_id="patient_001")

        # Check for family support mention
        assert "family" in context.lower() or "support" in context.lower()


class TestContextInjectorHelperMethods:
    """Test ContextInjector helper methods."""

    def test_get_template_english(self, context_injector):
        """Test getting English template."""
        template = context_injector._get_template("welcome_back", "en-IN")
        assert template is not None
        assert "Welcome" in template

    def test_get_template_hindi(self, context_injector):
        """Test getting Hindi template."""
        template = context_injector._get_template("welcome_back", "hi-IN")
        assert template is not None
        # Should be Hindi
        assert any(char >= '\u0900' and char <= '\u097F' for char in template)

    def test_get_template_fallback(self, context_injector):
        """Test template fallback to English."""
        template = context_injector._get_template("welcome_back", "unknown-XX")
        assert template is not None
        assert "Welcome" in template

    def test_format_condition(self, context_injector):
        """Test condition formatting."""
        result = context_injector._format_condition("cancer", "en-IN")
        assert "cancer" in result.lower()

        result_hi = context_injector._format_condition("cancer", "hi-IN")
        assert "cancer" in result_hi.lower()

    def test_localize_symptom_english(self, context_injector):
        """Test symptom localization stays English."""
        result = context_injector._localize_symptom("pain", "en-IN")
        assert result == "pain"

    def test_localize_symptom_hindi(self, context_injector):
        """Test symptom localization to Hindi."""
        result = context_injector._localize_symptom("pain", "hi-IN")
        assert result == "दर्द"

    def test_localize_symptom_unknown(self, context_injector):
        """Test symptom localization for unknown symptom."""
        result = context_injector._localize_symptom("unknown_symptom", "hi-IN")
        assert result == "unknown_symptom"  # Falls back to original

    def test_trim_context_short(self, context_injector):
        """Test context trimming for short text."""
        short_text = "This is short."
        result = context_injector._trim_context_intelligently(short_text, 100, "en-IN")
        assert result == short_text

    def test_trim_context_long(self, context_injector):
        """Test context trimming for long text."""
        long_text = "This is a very long sentence. " * 10
        result = context_injector._trim_context_intelligently(long_text, 100, "en-IN")
        assert len(result) <= 110  # Allow some overflow for word boundaries


class TestGetContextForMonitoring:
    """Test get_context_for_monitoring method."""

    @pytest.mark.asyncio
    async def test_returns_dict(self, context_injector):
        """Test monitoring context returns dictionary."""
        context = await context_injector.get_context_for_monitoring("patient_001")

        assert isinstance(context, dict)
        assert "user_id" in context
        assert "profile" in context
        assert "current_condition" in context
        assert "longitudinal" in context

    @pytest.mark.asyncio
    async def test_includes_profile_info(self, context_injector):
        """Test monitoring context includes profile info."""
        context = await context_injector.get_context_for_monitoring("patient_001")

        profile = context["profile"]
        assert "role" in profile
        assert "language" in profile
        assert "total_sessions" in profile

    @pytest.mark.asyncio
    async def test_includes_condition_info(self, context_injector):
        """Test monitoring context includes condition info."""
        context = await context_injector.get_context_for_monitoring("patient_001")

        condition = context["current_condition"]
        assert "primary" in condition
        assert "symptoms" in condition
        assert "medications" in condition


# ============================================================================
# PROMPT CONTEXT BUILDER TESTS
# ============================================================================

class TestPromptContextBuilder:
    """Test PromptContextBuilder class."""

    @pytest.fixture
    def prompt_builder(self, context_injector):
        """Create PromptContextBuilder."""
        return PromptContextBuilder(context_injector)

    @pytest.mark.asyncio
    async def test_build_prompt_with_context(self, prompt_builder):
        """Test building prompt with patient context."""
        base_prompt = "You are a helpful palliative care assistant."

        enhanced = await prompt_builder.build_prompt_with_context(
            base_system_prompt=base_prompt,
            user_id="patient_001",
            question="How can I manage my pain?"
        )

        assert base_prompt in enhanced
        assert "Patient Context" in enhanced
        assert "compassion" in enhanced.lower() or "empathy" in enhanced.lower()

    @pytest.mark.asyncio
    async def test_build_prompt_no_context(
        self,
        mock_longitudinal_manager,
        mock_context_memory
    ):
        """Test building prompt for new user with no context."""
        mock_profile_manager = AsyncMock(spec=UserProfileManager)
        mock_profile_manager.get_profile.return_value = None

        injector = ContextInjector(
            longitudinal_manager=mock_longitudinal_manager,
            user_profile_manager=mock_profile_manager,
            context_memory=mock_context_memory
        )
        builder = PromptContextBuilder(injector)

        base_prompt = "You are a helpful assistant."

        enhanced = await builder.build_prompt_with_context(
            base_system_prompt=base_prompt,
            user_id="new_user",
            question="Hello"
        )

        # Should return base prompt unchanged
        assert enhanced == base_prompt

    @pytest.mark.asyncio
    async def test_build_user_context_summary(self, prompt_builder):
        """Test building user context summary."""
        summary = await prompt_builder.build_user_context_summary("patient_001")

        assert isinstance(summary, str)
        assert len(summary) > 0
        # Should contain key information (condition, symptoms, etc)
        assert "condition" in summary.lower() or "patient" in summary.lower() or "session" in summary.lower()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestContextInjectorIntegration:
    """Integration tests for context injection."""

    @pytest.mark.asyncio
    async def test_full_injection_flow(
        self,
        mock_longitudinal_manager,
        mock_user_profile_manager,
        mock_context_memory
    ):
        """Test complete context injection flow."""
        # Setup with realistic data
        mock_longitudinal_manager.get_longitudinal_summary.return_value = {
            "patient_id": "patient_001",
            "total_observations": 25,
            "summaries": {
                "symptom:pain": {
                    "entity_name": "pain",
                    "trend": "worsening",
                    "observations": 10,
                    "latest_severity": 3
                },
                "symptom:fatigue": {
                    "entity_name": "fatigue",
                    "trend": "stable",
                    "observations": 8,
                    "latest_severity": 2
                }
            },
            "active_symptoms": [
                {"name": "pain", "severity": 3, "trend": "worsening"},
                {"name": "fatigue", "severity": 2, "trend": "stable"}
            ],
            "active_alerts": [
                {
                    "priority": "medium",
                    "title": "Missed Medication",
                    "description": "Morphine doses missed in last 2 days"
                }
            ]
        }

        injector = ContextInjector(
            longitudinal_manager=mock_longitudinal_manager,
            user_profile_manager=mock_user_profile_manager,
            context_memory=mock_context_memory
        )

        context = await injector.inject_context(
            user_id="patient_001",
            question="My pain is getting worse, what should I do?"
        )

        # Should be comprehensive but not too long
        assert len(context) > 50
        assert len(context) < 2500

        # Should include key elements
        assert "welcome" in context.lower() or "वापसी" in context


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling in context injection."""

    @pytest.mark.asyncio
    async def test_handles_profile_error(
        self,
        mock_longitudinal_manager,
        mock_context_memory
    ):
        """Test graceful handling of profile errors."""
        mock_profile_manager = AsyncMock(spec=UserProfileManager)
        mock_profile_manager.get_profile.side_effect = Exception("Database error")

        injector = ContextInjector(
            longitudinal_manager=mock_longitudinal_manager,
            user_profile_manager=mock_profile_manager,
            context_memory=mock_context_memory
        )

        # Should return empty string on error, not raise
        context = await injector.inject_context(user_id="patient_001")
        assert context == ""

    @pytest.mark.asyncio
    async def test_handles_memory_error(
        self,
        mock_user_profile_manager,
        mock_context_memory
    ):
        """Test graceful handling of memory errors."""
        mock_longitudinal = AsyncMock(spec=LongitudinalMemoryManager)
        mock_longitudinal.get_longitudinal_summary.side_effect = Exception("Storage error")

        injector = ContextInjector(
            longitudinal_manager=mock_longitudinal,
            user_profile_manager=mock_user_profile_manager,
            context_memory=mock_context_memory
        )

        # Should return empty string on error
        context = await injector.inject_context(user_id="patient_001")
        assert context == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
