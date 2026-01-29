#!/usr/bin/env python3
"""
Test Voice Safety Integration for All Voice Providers
=====================================================

Tests safety enhancements in:
- Gemini Live API
- Bolna.ai
- Retell.AI
- Voice Router

"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from voice_safety_wrapper import (
    VoiceSafetyWrapper,
    get_voice_safety_wrapper,
    VoiceSafetyEvent,
    GeminiLiveSafetyIntegration,
    RetellSafetyIntegration,
    BolnaSafetyIntegration,
)


class TestVoiceSafetyWrapper:
    """Test the voice safety wrapper"""
    
    def test_initialization(self):
        """Test wrapper initialization"""
        wrapper = VoiceSafetyWrapper()
        assert wrapper is not None
        print("✅ VoiceSafetyWrapper initialized")
    
    async def test_emergency_detection_voice(self):
        """Test emergency detection in voice"""
        wrapper = VoiceSafetyWrapper()
        
        # Critical emergency
        result = await wrapper.check_voice_query(
            user_id="test_user",
            transcript="I can't breathe",
            language="en"
        )
        
        assert result.should_escalate
        assert result.event_type == VoiceSafetyEvent.EMERGENCY_DETECTED
        assert result.emergency_alert is not None
        print("✅ Critical emergency detected in voice")
    
    async def test_hindi_emergency_detection(self):
        """Test Hindi emergency detection"""
        wrapper = VoiceSafetyWrapper()
        
        result = await wrapper.check_voice_query(
            user_id="test_user",
            transcript="सांस नहीं आ रही",
            language="hi"
        )
        
        assert result.should_escalate
        print("✅ Hindi emergency detected")
    
    async def test_handoff_detection(self):
        """Test human handoff detection"""
        wrapper = VoiceSafetyWrapper()
        
        result = await wrapper.check_voice_query(
            user_id="test_user",
            transcript="I want to talk to a human",
            language="en"
        )
        
        assert result.should_escalate
        assert result.event_type == VoiceSafetyEvent.HUMAN_HANDOFF_TRIGGERED
        print("✅ Handoff request detected")
    
    async def test_normal_query(self):
        """Test normal query passes through"""
        wrapper = VoiceSafetyWrapper()
        
        result = await wrapper.check_voice_query(
            user_id="test_user",
            transcript="What is palliative care?",
            language="en"
        )
        
        assert not result.should_escalate
        assert result.should_proceed
        print("✅ Normal query passes safety check")
    
    def test_voice_optimization(self):
        """Test response optimization for voice"""
        wrapper = VoiceSafetyWrapper()
        
        long_response = "A" * 3000
        optimized = wrapper.optimize_for_voice(
            long_response,
            user_id="test_user",
            language="en",
            max_duration_seconds=30
        )
        
        # Should be significantly shorter
        assert len(optimized) < len(long_response)
        print("✅ Voice optimization works")
    
    def test_clean_for_voice(self):
        """Test text cleaning for voice"""
        wrapper = VoiceSafetyWrapper()
        
        text = "This is **bold** and `code` with [link](http://example.com)"
        cleaned = wrapper._clean_for_voice(text)
        
        assert "**" not in cleaned
        assert "`" not in cleaned
        assert "http" not in cleaned
        print("✅ Text cleaned for voice")


class TestProviderIntegrations:
    """Test provider-specific integrations"""
    
    async def test_gemini_live_integration(self):
        """Test Gemini Live safety integration"""
        # Mock gemini service
        class MockGeminiService:
            pass
        
        integration = GeminiLiveSafetyIntegration(MockGeminiService())
        
        # Test emergency detection
        result = await integration.on_transcript(
            session_id="test_session",
            transcript="I can't breathe",
            language="en"
        )
        
        assert result["override"] == True
        assert result["escalate"] == True
        print("✅ Gemini Live safety integration works")
    
    async def test_retell_integration(self):
        """Test Retell safety integration"""
        # Mock retell handler
        class MockRetellHandler:
            active_sessions = {}
        
        integration = RetellSafetyIntegration(MockRetellHandler())
        
        # Test handoff detection
        result = await integration.on_transcript(
            call_id="test_call",
            transcript="I want to speak to a doctor",
            language="en"
        )
        
        assert result["override"] == True
        print("✅ Retell safety integration works")
    
    async def test_bolna_integration(self):
        """Test Bolna safety integration"""
        # Mock webhook handler
        class MockWebhookHandler:
            pass
        
        integration = BolnaSafetyIntegration(MockWebhookHandler())
        
        # Test normal query
        result = await integration.on_transcript(
            call_id="test_call",
            transcript="What is morphine?",
            phone_number="+919876543210",
            language="hi"
        )
        
        assert result["override"] == False
        print("✅ Bolna safety integration works")


def run_tests():
    """Run all tests"""
    print("=" * 70)
    print("VOICE SAFETY INTEGRATION TESTS")
    print("=" * 70)
    
    # Run synchronous tests
    wrapper_tests = TestVoiceSafetyWrapper()
    
    sync_tests = [
        ("Initialization", wrapper_tests.test_initialization),
        ("Voice Optimization", wrapper_tests.test_voice_optimization),
        ("Clean for Voice", wrapper_tests.test_clean_for_voice),
    ]
    
    for name, test in sync_tests:
        try:
            test()
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    # Run async tests
    async_tests = [
        ("Emergency Detection", wrapper_tests.test_emergency_detection_voice),
        ("Hindi Emergency", wrapper_tests.test_hindi_emergency_detection),
        ("Handoff Detection", wrapper_tests.test_handoff_detection),
        ("Normal Query", wrapper_tests.test_normal_query),
    ]
    
    for name, test in async_tests:
        try:
            asyncio.run(test())
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    # Run provider integration tests
    provider_tests = TestProviderIntegrations()
    
    provider_async_tests = [
        ("Gemini Live Integration", provider_tests.test_gemini_live_integration),
        ("Retell Integration", provider_tests.test_retell_integration),
        ("Bolna Integration", provider_tests.test_bolna_integration),
    ]
    
    for name, test in provider_async_tests:
        try:
            asyncio.run(test())
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    print("\n" + "=" * 70)
    print("VOICE SAFETY TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
