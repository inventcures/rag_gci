#!/usr/bin/env python3
"""
Test Suite for Palli Sahayak Safety Enhancements
=================================================
Tests the 5 quick-win features:
1. Evidence Badges
2. Emergency Detection & Escalation
3. Medication Reminder Scheduler
4. Response Length Optimization
5. Human Handoff System
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Try to import pytest, fallback to manual testing if not available
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Create a simple pytest.mark.asyncio decorator fallback
    class MockPytest:
        class mark:
            @staticmethod
            def asyncio(func):
                return func
    pytest = MockPytest()

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_enhancements import (
    EvidenceBadgeSystem,
    EvidenceLevel,
    EvidenceBadge,
    EmergencyDetectionSystem,
    EmergencyLevel,
    EmergencyAlert,
    MedicationReminderScheduler,
    MedicationReminder,
    ResponseLengthOptimizer,
    UserComprehensionLevel,
    HumanHandoffSystem,
    HandoffReason,
    HandoffRequest,
    SafetyEnhancementsManager,
)


# ============================================================================
# 1. EVIDENCE BADGES TESTS
# ============================================================================

class TestEvidenceBadges:
    """Test evidence badge system"""
    
    def test_evidence_level_enum(self):
        """Test evidence level values"""
        assert EvidenceLevel.A.value == "A"
        assert EvidenceLevel.B.value == "B"
        assert EvidenceLevel.C.value == "C"
        assert EvidenceLevel.D.value == "D"
        assert EvidenceLevel.E.value == "E"
    
    def test_evidence_badge_creation(self):
        """Test creating evidence badges"""
        badge = EvidenceBadge(
            level=EvidenceLevel.A,
            confidence_score=0.95,
            source_quality="Excellent",
            recommendation="Strong evidence",
            consult_physician=False
        )
        
        assert badge.level == EvidenceLevel.A
        assert badge.confidence_score == 0.95
        assert badge.get_badge_emoji() == "🟢"
    
    def test_evidence_badge_emojis(self):
        """Test correct emoji for each level"""
        for level, expected_emoji in [
            (EvidenceLevel.A, "🟢"),
            (EvidenceLevel.B, "🟡"),
            (EvidenceLevel.C, "🟠"),
            (EvidenceLevel.D, "🔵"),
            (EvidenceLevel.E, "🔴"),
        ]:
            badge = EvidenceBadge(
                level=level,
                confidence_score=0.5,
                source_quality="Test",
                recommendation="Test",
                consult_physician=False
            )
            assert badge.get_badge_emoji() == expected_emoji
    
    def test_evidence_badge_formatting(self):
        """Test badge formatting for display"""
        badge = EvidenceBadge(
            level=EvidenceLevel.A,
            confidence_score=0.95,
            source_quality="Excellent - Authoritative medical sources",
            recommendation="Strong evidence supports this",
            consult_physician=False
        )
        
        formatted = badge.format_for_user("en")
        assert "95%" in formatted or "0.95" in formatted
        assert "🟢" in formatted
    
    def test_evidence_system_initialization(self):
        """Test evidence system initialization"""
        system = EvidenceBadgeSystem()
        assert system is not None
    
    def test_source_quality_analysis(self):
        """Test source quality analysis"""
        system = EvidenceBadgeSystem()
        
        # Test with high-quality sources
        high_quality_sources = [
            {"filename": "WHO_guidelines.pdf"},
            {"filename": "clinical_trials_2024.pdf"},
        ]
        score = system._analyze_source_quality(high_quality_sources)
        assert score > 0.6
        
        # Test with low-quality sources
        low_quality_sources = [
            {"filename": "blog_post.txt"},
            {"filename": "forum_discussion.txt"},
        ]
        score = system._analyze_source_quality(low_quality_sources)
        assert score < 0.5
    
    def test_high_stakes_detection(self):
        """Test detection of high-stakes queries"""
        system = EvidenceBadgeSystem()
        
        high_stakes = [
            "I have severe chest pain",
            "Can't breathe properly",
            "Patient is unconscious",
        ]
        
        for query in high_stakes:
            assert system._is_high_stakes_query(query), f"Failed for: {query}"
        
        low_stakes = [
            "What is palliative care?",
            "How to manage mild pain?",
            "Tell me about hospice services",
        ]
        
        for query in low_stakes:
            assert not system._is_high_stakes_query(query), f"Failed for: {query}"
    
    def test_uncertainty_detection(self):
        """Test detection of uncertainty in answers"""
        system = EvidenceBadgeSystem()
        
        uncertain_answer = "This may help but I'm not sure. It depends on your condition."
        uncertainty = system._detect_uncertainty(uncertain_answer)
        assert uncertainty > 0
        
        confident_answer = "Based on clinical evidence, this treatment is effective."
        uncertainty = system._detect_uncertainty(confident_answer)
        assert uncertainty < 0.5


# ============================================================================
# 2. EMERGENCY DETECTION TESTS
# ============================================================================

class TestEmergencyDetection:
    """Test emergency detection system"""
    
    def test_emergency_detection_initialization(self):
        """Test emergency system initialization"""
        system = EmergencyDetectionSystem()
        assert system is not None
    
    def test_critical_emergency_detection_english(self):
        """Test detection of critical emergencies in English"""
        system = EmergencyDetectionSystem()
        
        critical_queries = [
            "I can't breathe",
            "Patient is unconscious",
            "Severe chest pain",
            "Having a heart attack",
        ]
        
        for query in critical_queries:
            alert = system.detect_emergency(query, "user123", "en")
            assert alert is not None, f"Failed to detect: {query}"
            assert alert.level == EmergencyLevel.CRITICAL
            assert alert.contact_emergency_services
    
    def test_critical_emergency_detection_hindi(self):
        """Test detection of critical emergencies in Hindi"""
        system = EmergencyDetectionSystem()
        
        critical_queries = [
            "सांस नहीं आ रही",
            "होश नहीं",
            "छाती में दर्द",
        ]
        
        for query in critical_queries:
            alert = system.detect_emergency(query, "user123", "hi")
            assert alert is not None, f"Failed to detect: {query}"
            assert alert.level == EmergencyLevel.CRITICAL
    
    def test_high_priority_emergency_detection(self):
        """Test detection of high-priority emergencies"""
        system = EmergencyDetectionSystem()
        
        high_queries = [
            "Severe pain that won't stop",
            "High fever and vomiting",
            "Can't move my legs",
        ]
        
        for query in high_queries:
            alert = system.detect_emergency(query, "user123", "en")
            assert alert is not None, f"Failed to detect: {query}"
            assert alert.level == EmergencyLevel.HIGH
    
    def test_no_emergency(self):
        """Test that normal queries don't trigger emergencies"""
        system = EmergencyDetectionSystem()
        
        normal_queries = [
            "What is palliative care?",
            "How to manage pain?",
            "Tell me about morphine dosage",
        ]
        
        for query in normal_queries:
            alert = system.detect_emergency(query, "user123", "en")
            assert alert is None, f"False positive for: {query}"
    
    def test_caregiver_registration(self):
        """Test caregiver registration"""
        system = EmergencyDetectionSystem()
        
        system.register_caregiver("user123", "+919876543210")
        assert "user123" in system.caregiver_contacts
        assert "+919876543210" in system.caregiver_contacts["user123"]


# ============================================================================
# 3. MEDICATION REMINDER TESTS
# ============================================================================

class TestMedicationReminders:
    """Test medication reminder scheduler"""
    
    def test_reminder_scheduler_initialization(self):
        """Test scheduler initialization"""
        scheduler = MedicationReminderScheduler(storage_path="/tmp/test_reminders")
        assert scheduler is not None
    
    def test_create_reminder(self):
        """Test creating a medication reminder"""
        scheduler = MedicationReminderScheduler(storage_path="/tmp/test_reminders")
        
        reminder = scheduler.create_reminder(
            user_id="user123",
            medication_name="Paracetamol",
            dosage="500mg",
            frequency="daily",
            times=["08:00", "20:00"],
            instructions="Take with food",
            language="en"
        )
        
        assert reminder is not None
        assert reminder.medication_name == "Paracetamol"
        assert reminder.dosage == "500mg"
        assert len(reminder.scheduled_times) == 2
        assert "08:00" in reminder.scheduled_times
        assert reminder.active
    
    def test_get_user_reminders(self):
        """Test retrieving user reminders"""
        scheduler = MedicationReminderScheduler(storage_path="/tmp/test_reminders")
        
        # Create some reminders
        scheduler.create_reminder(
            user_id="user123",
            medication_name="Med1",
            dosage="10mg",
            frequency="daily",
            times=["08:00"],
            language="en"
        )
        scheduler.create_reminder(
            user_id="user123",
            medication_name="Med2",
            dosage="20mg",
            frequency="daily",
            times=["20:00"],
            language="en"
        )
        
        reminders = scheduler.get_user_reminders("user123")
        assert len(reminders) >= 2
    
    def test_mark_taken(self):
        """Test marking medication as taken"""
        scheduler = MedicationReminderScheduler(storage_path="/tmp/test_reminders")
        
        reminder = scheduler.create_reminder(
            user_id="user123",
            medication_name="TestMed",
            dosage="10mg",
            frequency="daily",
            times=["08:00"],
            language="en"
        )
        
        success = scheduler.mark_taken(reminder.reminder_id)
        assert success
        
        # Verify it was recorded
        updated = scheduler.reminders[reminder.reminder_id]
        assert len(updated.taken_history) > 0
    
    def test_delete_reminder(self):
        """Test deleting a reminder"""
        scheduler = MedicationReminderScheduler(storage_path="/tmp/test_reminders")
        
        reminder = scheduler.create_reminder(
            user_id="user123",
            medication_name="ToDelete",
            dosage="10mg",
            frequency="daily",
            times=["08:00"],
            language="en"
        )
        
        success = scheduler.delete_reminder(reminder.reminder_id)
        assert success
        assert reminder.reminder_id not in scheduler.reminders
    
    def test_reminder_message_generation(self):
        """Test reminder message generation"""
        scheduler = MedicationReminderScheduler(storage_path="/tmp/test_reminders")
        
        reminder = MedicationReminder(
            reminder_id="test123",
            user_id="user123",
            medication_name="Paracetamol",
            dosage="500mg",
            frequency="daily",
            scheduled_times=["08:00"],
            start_date=datetime.now(),
            end_date=None,
            instructions="Take with food",
            language="en"
        )
        
        message = scheduler.get_reminder_message(reminder)
        assert "Paracetamol" in message
        assert "500mg" in message
        assert "TAKEN" in message


# ============================================================================
# 4. RESPONSE LENGTH OPTIMIZATION TESTS
# ============================================================================

class TestResponseLengthOptimization:
    """Test response length optimizer"""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        optimizer = ResponseLengthOptimizer(storage_path="/tmp/test_profiles")
        assert optimizer is not None
    
    def test_analyze_user_message(self):
        """Test user message analysis"""
        optimizer = ResponseLengthOptimizer(storage_path="/tmp/test_profiles")
        
        # Simple message
        optimizer.analyze_user_message("user123", "What is pain?")
        profile = optimizer.profiles["user123"]
        assert profile is not None
        
        # More complex message
        optimizer.analyze_user_message("user123", "Can you explain the mechanism of action of morphine in palliative care settings?")
        # Level should have been updated
    
    def test_get_optimization_prompt(self):
        """Test getting optimization prompts"""
        optimizer = ResponseLengthOptimizer(storage_path="/tmp/test_profiles")
        
        # Simple user
        optimizer.profiles["simple_user"] = type('obj', (object,), {
            'level': UserComprehensionLevel.SIMPLE
        })()
        prompt = optimizer.get_optimization_prompt("simple_user")
        assert "Maximum 4 short sentences" in prompt
        
        # Detailed user
        optimizer.profiles["detailed_user"] = type('obj', (object,), {
            'level': UserComprehensionLevel.DETAILED
        })()
        prompt = optimizer.get_optimization_prompt("detailed_user")
        assert "comprehensive" in prompt.lower()
    
    def test_response_adaptation(self):
        """Test response adaptation"""
        optimizer = ResponseLengthOptimizer(storage_path="/tmp/test_profiles")
        
        optimizer.profiles["user123"] = type('obj', (object,), {
            'level': UserComprehensionLevel.SIMPLE
        })()
        
        long_response = "A" * 3000
        adapted = optimizer.adapt_response(long_response, "user123")
        assert len(adapted) <= 550  # 500 max + truncation notice


# ============================================================================
# 5. HUMAN HANDOFF TESTS
# ============================================================================

class TestHumanHandoff:
    """Test human handoff system"""
    
    def test_handoff_system_initialization(self):
        """Test handoff system initialization"""
        system = HumanHandoffSystem(storage_path="/tmp/test_handoffs")
        assert system is not None
    
    def test_check_handoff_needed(self):
        """Test detecting when handoff is needed"""
        system = HumanHandoffSystem(storage_path="/tmp/test_handoffs")
        
        # User requesting human
        reason = system.check_handoff_needed("I want to talk to a human")
        assert reason == HandoffReason.USER_REQUEST
        
        # Emergency
        reason = system.check_handoff_needed("This is an emergency!")
        assert reason == HandoffReason.EMERGENCY
        
        # Normal query
        reason = system.check_handoff_needed("What is palliative care?")
        assert reason is None
    
    def test_create_handoff_request(self):
        """Test creating handoff requests"""
        system = HumanHandoffSystem(storage_path="/tmp/test_handoffs")
        
        request = system.create_handoff_request(
            user_id="user123",
            reason=HandoffReason.USER_REQUEST,
            context="User wants to speak with someone",
            conversation_history=[],
            priority="medium"
        )
        
        assert request is not None
        assert request.user_id == "user123"
        assert request.reason == HandoffReason.USER_REQUEST
        assert request.status == "pending"
        assert request.request_id is not None
    
    def test_assign_caregiver(self):
        """Test caregiver assignment"""
        system = HumanHandoffSystem(storage_path="/tmp/test_handoffs")
        
        request = system.create_handoff_request(
            user_id="user123",
            reason=HandoffReason.USER_REQUEST,
            context="Test",
            conversation_history=[]
        )
        
        success = system.assign_caregiver(request.request_id, "caregiver456")
        assert success
        
        updated = system.pending_requests[request.request_id]
        assert updated.assigned_to == "caregiver456"
        assert updated.status == "assigned"
    
    def test_resolve_request(self):
        """Test resolving handoff requests"""
        system = HumanHandoffSystem(storage_path="/tmp/test_handoffs")
        
        request = system.create_handoff_request(
            user_id="user123",
            reason=HandoffReason.USER_REQUEST,
            context="Test",
            conversation_history=[]
        )
        
        success = system.resolve_request(request.request_id, "Issue resolved")
        assert success
        
        assert request.request_id not in system.pending_requests
        assert request.request_id in system.resolved_requests


# ============================================================================
# 6. INTEGRATION TESTS
# ============================================================================

class TestSafetyManagerIntegration:
    """Test the main safety manager integration"""
    
    def test_manager_initialization(self):
        """Test safety manager initialization"""
        manager = SafetyEnhancementsManager()
        assert manager is not None
        assert manager.evidence_system is not None
        assert manager.emergency_system is not None
        assert manager.reminder_scheduler is not None
        assert manager.response_optimizer is not None
        assert manager.handoff_system is not None
    
    @pytest.mark.asyncio
    async def test_process_query_with_emergency(self):
        """Test processing a query with emergency"""
        manager = SafetyEnhancementsManager()
        
        result = await manager.process_query(
            user_id="user123",
            query="I can't breathe",
            language="en",
            conversation_history=[]
        )
        
        assert result["emergency_alert"] is not None
        assert not result["should_respond"]
        assert "emergency" in result["response_additions"]
    
    @pytest.mark.asyncio
    async def test_process_query_normal(self):
        """Test processing a normal query"""
        manager = SafetyEnhancementsManager()
        
        result = await manager.process_query(
            user_id="user123",
            query="What is palliative care?",
            language="en",
            conversation_history=[]
        )
        
        assert result["should_respond"]
        assert result["emergency_alert"] is None
        assert result["modified_prompt"] is not None


# ============================================================================
# MAIN
# ============================================================================

def run_tests():
    """Run all tests manually if pytest is not available"""
    print("=" * 70)
    print("PALLI SAHAYAK SAFETY ENHANCEMENTS - TEST SUITE")
    print("=" * 70)
    
    test_classes = [
        TestEvidenceBadges,
        TestEmergencyDetection,
        TestMedicationReminders,
        TestResponseLengthOptimization,
        TestHumanHandoff,
        TestSafetyManagerIntegration,
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n{'─' * 70}")
        print(f"Testing: {test_class.__name__}")
        print(f"{'─' * 70}")
        
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        
        for method_name in methods:
            total_tests += 1
            method = getattr(instance, method_name)
            try:
                if asyncio.iscoroutinefunction(method):
                    asyncio.run(method())
                else:
                    method()
                print(f"  ✅ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ❌ {method_name}: {e}")
    
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print(f"{'=' * 70}")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    if HAS_PYTEST:
        # Run with pytest
        import subprocess
        subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"])
    else:
        # Run manual tests
        success = run_tests()
        sys.exit(0 if success else 1)
