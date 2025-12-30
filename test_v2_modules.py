#!/usr/bin/env python3
"""
Test script for V2 modules:
- Clinical Validation Pipeline
- User Personalization
- Real-time Analytics Dashboard
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_clinical_validation():
    """Test clinical validation pipeline."""
    print("\n" + "="*60)
    print("TESTING: Clinical Validation Pipeline")
    print("="*60)

    from clinical_validation import ClinicalValidator, ExpertSampler, FeedbackCollector, ValidationMetrics
    from clinical_validation.feedback import FeedbackType, FeedbackChannel
    from clinical_validation.expert_sampling import ReviewStatus, SamplingPriority

    # Test 1: ClinicalValidator
    print("\n[1/4] Testing ClinicalValidator...")
    validator = ClinicalValidator()

    # Test valid response (validate is NOT async)
    result = validator.validate(
        query="What is the starting dose of morphine for pain?",
        response="For opioid-naive patients, morphine can be started at 5-10mg orally every 4 hours. Always consult your doctor before starting any medication.",
        sources=[
            {"title": "WHO Pain Guidelines", "content": "Morphine starting dose for opioid-naive patients is 5-10mg oral every 4 hours."}
        ],
        context=None  # context is a string, not a dict
    )

    print(f"   Valid: {result.is_valid}")
    print(f"   Confidence: {result.confidence_score:.2f}")
    print(f"   Checks passed: {result.checks_passed}")
    print(f"   Issues: {len(result.issues)}")

    # Test response with safety issue
    result2 = validator.validate(
        query="Can I take as much morphine as I want?",
        response="You can take as much morphine as you need for pain relief.",
        sources=[],
        context=None
    )
    print(f"\n   Safety test - Valid: {result2.is_valid}")
    print(f"   Issues found: {[i.category.value for i in result2.issues]}")

    print("   ✓ ClinicalValidator working")

    # Test 2: ExpertSampler
    print("\n[2/4] Testing ExpertSampler...")
    sampler = ExpertSampler(
        storage_path="data/test_expert_samples",
        sample_rate=1.0,  # 100% for testing
        max_samples_per_day=10
    )

    sample = await sampler.maybe_sample(
        query="How to manage breakthrough pain?",
        response="For breakthrough pain, use immediate-release morphine at 10-15% of total daily dose.",
        language="en-IN",
        sources=[{"title": "Palliative Care Guidelines"}],
        validation_result=result.to_dict(),  # Convert to dict
        session_id="test_session_001",
        user_id="test_user_001"
    )

    if sample:
        print(f"   Sample created: {sample.sample_id}")
        print(f"   Priority: {sample.priority.value}")
        print(f"   Status: {sample.status.value}")

    # Get pending samples
    pending = await sampler.get_pending_samples(limit=5)
    print(f"   Pending samples: {len(pending)}")

    # Get statistics
    stats = await sampler.get_statistics()
    print(f"   Total samples: {stats['total_samples']}")

    print("   ✓ ExpertSampler working")

    # Test 3: FeedbackCollector
    print("\n[3/4] Testing FeedbackCollector...")
    collector = FeedbackCollector(
        storage_path="data/test_feedback",
        auto_prompt_rate=0.5
    )

    feedback = await collector.collect_feedback(
        query="How to give morphine?",
        response="Morphine should be given orally...",
        rating=5,
        feedback_type=FeedbackType.HELPFUL,
        channel=FeedbackChannel.VOICE_PROMPT,
        session_id="test_session_001",
        language="hi-IN"
    )

    print(f"   Feedback ID: {feedback.feedback_id}")
    print(f"   Rating: {feedback.rating}")

    # Test voice prompts
    prompts = collector.get_feedback_prompt("hi-IN")
    print(f"   Hindi prompt: {prompts['ask'][:50]}...")

    # Get statistics
    stats = await collector.get_statistics(days=7)
    print(f"   Total feedback: {stats['total_feedback']}")

    print("   ✓ FeedbackCollector working")

    # Test 4: ValidationMetrics
    print("\n[4/4] Testing ValidationMetrics...")
    metrics = ValidationMetrics(
        storage_path="data/test_metrics",
        aggregation_interval_minutes=1
    )

    await metrics.record_validation(result.to_dict(), response_time_ms=250.0)
    await metrics.record_user_feedback(rating=5)
    await metrics.record_rag_retrieval(success=True)

    current = await metrics.get_current_metrics()
    print(f"   Validation confidence: {current['validation_confidence']}")
    print(f"   Pass rate count: {current['validation_pass_rate']['count']}")

    print("   ✓ ValidationMetrics working")

    print("\n✅ Clinical Validation Pipeline: ALL TESTS PASSED")
    return True


async def test_personalization():
    """Test personalization system."""
    print("\n" + "="*60)
    print("TESTING: User Personalization System")
    print("="*60)

    from personalization import UserProfileManager, ContextMemory, InteractionHistory
    from personalization.user_profile import UserRole, CommunicationStyle

    # Test 1: UserProfileManager
    print("\n[1/3] Testing UserProfileManager...")
    profile_manager = UserProfileManager(
        storage_path="data/test_user_profiles",
        cache_size=50
    )

    # Create profile
    profile = await profile_manager.get_or_create_profile(
        phone_number="+919876543210",
        language="hi-IN"
    )

    print(f"   User ID: {profile.user_id}")
    print(f"   Role: {profile.role.value}")
    print(f"   Language: {profile.preferences.language}")
    print(f"   Sessions: {profile.total_sessions}")

    # Test role detection
    text = "I am caring for my mother who has cancer"
    detected_role = profile_manager.detect_role(text, "en-IN")
    print(f"   Detected role from '{text[:30]}...': {detected_role.value if detected_role else 'None'}")

    # Update role
    if detected_role:
        await profile_manager.update_role(profile.user_id, detected_role)

    # Get system context
    context = profile_manager.get_system_context(profile)
    print(f"   System context: {context[:60]}...")

    # Get statistics
    stats = await profile_manager.get_statistics()
    print(f"   Total users: {stats['total_users']}")

    print("   ✓ UserProfileManager working")

    # Test 2: ContextMemory
    print("\n[2/3] Testing ContextMemory...")
    context_memory = ContextMemory(
        storage_path="data/test_patient_context",
        context_expiry_days=90
    )

    # Get/create context
    patient_ctx = await context_memory.get_or_create_context(profile.user_id)
    print(f"   Context created for: {patient_ctx.user_id}")

    # Set condition
    await context_memory.set_condition(
        profile.user_id,
        condition="cancer",
        stage="advanced"
    )

    # Add symptoms
    await context_memory.add_symptom(profile.user_id, "pain", "moderate")
    await context_memory.add_symptom(profile.user_id, "nausea", "mild")

    # Add medication
    await context_memory.add_medication(
        profile.user_id,
        name="morphine",
        dosage="10mg",
        frequency="every 4 hours",
        purpose="pain management"
    )

    # Add allergy
    await context_memory.add_allergy(profile.user_id, "ibuprofen")

    # Get updated context
    patient_ctx = await context_memory.get_or_create_context(profile.user_id)
    print(f"   Condition: {patient_ctx.primary_condition}")
    print(f"   Symptoms: {[s.name for s in patient_ctx.symptoms]}")
    print(f"   Medications: {[m.name for m in patient_ctx.medications]}")
    print(f"   Allergies: {patient_ctx.allergies}")

    # Test auto-extraction
    text = "The patient is experiencing fatigue and taking paracetamol"
    await context_memory.update_from_conversation(profile.user_id, text, "en-IN")

    # Get context summary
    summary = context_memory.get_context_summary(patient_ctx)
    print(f"   Context summary: {summary[:80]}...")

    # Get statistics
    stats = await context_memory.get_statistics()
    print(f"   Total contexts: {stats['total_contexts']}")

    print("   ✓ ContextMemory working")

    # Test 3: InteractionHistory
    print("\n[3/3] Testing InteractionHistory...")
    history = InteractionHistory(
        storage_path="data/test_interaction_history",
        max_history_days=30
    )

    # Create session
    session = await history.get_or_create_session(
        user_id=profile.user_id,
        language="hi-IN",
        channel="voice"
    )
    print(f"   Session ID: {session.session_id}")

    # Add turns
    turn1 = await history.add_turn(
        user_id=profile.user_id,
        query="How to manage pain at night?",
        response="For nighttime pain, consider...",
        language="hi-IN",
        used_rag=True,
        rag_sources=["WHO_Guidelines.pdf"],
        response_time_ms=350
    )
    print(f"   Turn 1 - Type: {turn1.query_type.value}")

    turn2 = await history.add_turn(
        user_id=profile.user_id,
        query="What about nausea?",
        response="For nausea management...",
        language="hi-IN",
        used_rag=True,
        response_time_ms=280
    )
    print(f"   Turn 2 - Type: {turn2.query_type.value}")

    # Get recent context
    recent = await history.get_recent_context(profile.user_id, max_turns=5)
    print(f"   Recent turns: {len(recent)}")

    # Test follow-up detection
    is_followup = history.is_followup_query("What about at night?", recent)
    print(f"   Is follow-up: {is_followup}")

    # Get context summary
    ctx_summary = history.get_context_summary(recent)
    print(f"   History summary: {ctx_summary[:80]}...")

    # End session
    ended_session = await history.end_session(profile.user_id)
    if ended_session:
        print(f"   Session duration: {ended_session.duration_seconds:.1f}s")
        print(f"   Total turns: {ended_session.turn_count}")

    # Get statistics
    stats = await history.get_global_statistics()
    print(f"   Total sessions: {stats['total_sessions']}")

    print("   ✓ InteractionHistory working")

    print("\n✅ User Personalization System: ALL TESTS PASSED")
    return True


async def test_analytics():
    """Test analytics dashboard."""
    print("\n" + "="*60)
    print("TESTING: Real-time Analytics Dashboard")
    print("="*60)

    from analytics import RealtimeMetrics, UsageAnalytics, AnalyticsDashboard, MetricType

    # Test 1: RealtimeMetrics
    print("\n[1/3] Testing RealtimeMetrics...")
    metrics = RealtimeMetrics(
        window_seconds=300,
        enable_alerts=True
    )

    # Record various metrics
    for i in range(10):
        await metrics.record_latency(MetricType.RESPONSE_LATENCY_MS, 200 + i * 50)
        await metrics.record_latency(MetricType.RAG_LATENCY_MS, 100 + i * 20)

    await metrics.record_query()
    await metrics.record_query()
    await metrics.record_query()

    await metrics.record_rag_query(success=True, latency_ms=120, sources_count=3)
    await metrics.record_rag_query(success=True, latency_ms=150, sources_count=2)
    await metrics.record_rag_query(success=False, latency_ms=200, sources_count=0)

    await metrics.record_validation(passed=True, confidence=0.92)
    await metrics.record_validation(passed=True, confidence=0.88)
    await metrics.record_validation(passed=False, confidence=0.45)

    await metrics.record_user_feedback(rating=5)
    await metrics.record_user_feedback(rating=4)

    await metrics.record_error("test_error")

    # Get latency stats
    latency_stats = await metrics.get_latency_stats(MetricType.RESPONSE_LATENCY_MS)
    print(f"   Response latency - Avg: {latency_stats['avg']}ms, P95: {latency_stats['p95']}ms")

    # Get success rates
    rag_rate = await metrics.get_success_rate(MetricType.RAG_SUCCESS_RATE)
    print(f"   RAG success rate: {rag_rate['rate']*100:.1f}%")

    val_rate = await metrics.get_success_rate(MetricType.VALIDATION_PASS_RATE)
    print(f"   Validation pass rate: {val_rate['rate']*100:.1f}%")

    # Get health status
    health = await metrics.get_health_status()
    print(f"   System status: {health['status']}")
    print(f"   Alerts: {len(health['alerts'])}")

    # Get all metrics
    all_metrics = await metrics.get_all_metrics()
    print(f"   Total queries recorded: {all_metrics['counters'].get('total_queries', 0)}")

    print("   ✓ RealtimeMetrics working")

    # Test 2: UsageAnalytics
    print("\n[2/3] Testing UsageAnalytics...")
    usage = UsageAnalytics(
        storage_path="data/test_analytics",
        retention_days=90
    )

    # Record usage events
    for i in range(5):
        await usage.record_query(
            user_id=f"user_{i}",
            language="hi-IN" if i % 2 == 0 else "en-IN",
            query_type="symptom_inquiry" if i % 3 == 0 else "medication_question",
            used_rag=True,
            rag_success=i % 4 != 0,
            validation_passed=i % 5 != 0
        )

    await usage.record_session("user_1", duration_seconds=180)
    await usage.record_session("user_2", duration_seconds=240)

    await usage.record_feedback(is_positive=True)
    await usage.record_feedback(is_positive=True)
    await usage.record_feedback(is_positive=False)

    # Get summary
    summary = await usage.get_summary()
    print(f"   Today's queries: {summary['today']['queries']}")
    print(f"   Today's unique users: {summary['today']['unique_users']}")

    # Get language stats
    lang_stats = await usage.get_language_stats(days=1)
    print(f"   Languages: {list(lang_stats['by_language'].keys())}")

    # Get hourly distribution
    hourly = await usage.get_hourly_distribution(days=1)
    print(f"   Hours with queries: {len([h for h, c in hourly.items() if c > 0])}")

    print("   ✓ UsageAnalytics working")

    # Test 3: AnalyticsDashboard
    print("\n[3/3] Testing AnalyticsDashboard...")
    dashboard = AnalyticsDashboard(
        metrics_path="data/test_metrics",
        analytics_path="data/test_analytics"
    )

    # Record a complete query event
    await dashboard.record_query(
        user_id="test_user",
        query="How to manage pain?",
        response="For pain management...",
        language="hi-IN",
        query_type="symptom_inquiry",
        used_rag=True,
        rag_success=True,
        rag_latency_ms=120,
        response_latency_ms=350,
        validation_passed=True,
        validation_confidence=0.92,
        sources_count=3
    )

    # Get full dashboard
    dashboard_data = await dashboard.get_dashboard_data()
    print(f"   Dashboard sections: {list(dashboard_data['sections'].keys())}")
    print(f"   Health status: {dashboard_data['health']['status']}")

    # Get realtime snapshot
    snapshot = await dashboard.get_realtime_snapshot()
    print(f"   Snapshot status: {snapshot['status']}")
    print(f"   Queries/min: {snapshot['queries_per_minute']}")

    # Get alerts
    alerts = await dashboard.get_alerts()
    print(f"   Active alerts: {len(alerts)}")

    # Export data
    export = await dashboard.export_data(days=1)
    print(f"   Export keys: {list(export.keys())}")

    print("   ✓ AnalyticsDashboard working")

    print("\n✅ Real-time Analytics Dashboard: ALL TESTS PASSED")
    return True


async def cleanup_test_data():
    """Clean up test data directories."""
    import shutil

    test_dirs = [
        "data/test_expert_samples",
        "data/test_feedback",
        "data/test_metrics",
        "data/test_user_profiles",
        "data/test_patient_context",
        "data/test_interaction_history",
        "data/test_analytics"
    ]

    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"   Cleaned: {dir_path}")


async def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("#" + " "*20 + "V2 MODULE TESTS" + " "*23 + "#")
    print("#"*60)

    all_passed = True

    try:
        # Test Clinical Validation
        if not await test_clinical_validation():
            all_passed = False
    except Exception as e:
        print(f"\n❌ Clinical Validation FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        # Test Personalization
        if not await test_personalization():
            all_passed = False
    except Exception as e:
        print(f"\n❌ Personalization FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        # Test Analytics
        if not await test_analytics():
            all_passed = False
    except Exception as e:
        print(f"\n❌ Analytics FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    if all_passed:
        print("\n🎉 ALL V2 MODULE TESTS PASSED!")
        print("\nModules tested:")
        print("  ✅ clinical_validation/")
        print("     - ClinicalValidator")
        print("     - ExpertSampler")
        print("     - FeedbackCollector")
        print("     - ValidationMetrics")
        print("  ✅ personalization/")
        print("     - UserProfileManager")
        print("     - ContextMemory")
        print("     - InteractionHistory")
        print("  ✅ analytics/")
        print("     - RealtimeMetrics")
        print("     - UsageAnalytics")
        print("     - AnalyticsDashboard")
    else:
        print("\n⚠️ SOME TESTS FAILED - Check output above")

    # Cleanup
    print("\n" + "-"*60)
    print("Cleaning up test data...")
    await cleanup_test_data()
    print("Done!")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
