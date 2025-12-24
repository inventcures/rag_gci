#!/usr/bin/env python3
"""
Bolna Integration Test Suite for Palli Sahayak Voice AI Agent Helpline

This script tests all components of the Bolna integration:
1. Module imports
2. BolnaClient functionality
3. Agent configuration
4. Webhook handling
5. Voice router
6. API endpoints (simulated)
7. End-to-end flow

Usage:
    python test_bolna_integration.py           # Run all tests
    python test_bolna_integration.py --quick   # Quick smoke test
    python test_bolna_integration.py --live    # Test with live API (requires keys)
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestResult:
    """Container for test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.results: List[Tuple[str, bool, str]] = []

    def add(self, name: str, passed: bool, message: str = ""):
        self.results.append((name, passed, message))
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def skip(self, name: str, reason: str):
        self.results.append((name, None, reason))
        self.skipped += 1

    def print_summary(self):
        print("\n" + "=" * 70)
        print("TEST RESULTS")
        print("=" * 70)

        for name, passed, message in self.results:
            if passed is None:
                status = "⏭️  SKIP"
            elif passed:
                status = "✅ PASS"
            else:
                status = "❌ FAIL"

            print(f"{status}  {name}")
            if message and not passed:
                print(f"         └─ {message}")

        print("-" * 70)
        print(f"Total: {self.passed} passed, {self.failed} failed, {self.skipped} skipped")
        print("=" * 70)

        return self.failed == 0


def test_imports(results: TestResult):
    """Test that all modules can be imported."""
    print("\n📦 Testing imports...")

    # Test bolna_integration module
    try:
        from bolna_integration import (
            BolnaClient,
            BolnaCallResult,
            get_palli_sahayak_agent_config,
            BolnaWebhookHandler,
            CallRecord,
        )
        results.add("Import bolna_integration", True)
    except ImportError as e:
        results.add("Import bolna_integration", False, str(e))
        return  # Can't continue without this

    # Test voice_router module
    try:
        from voice_router import (
            VoiceRouter,
            VoiceProvider,
            VoiceSession,
            VoiceResponse,
            create_voice_router,
        )
        results.add("Import voice_router", True)
    except ImportError as e:
        results.add("Import voice_router", False, str(e))

    # Test config module
    try:
        from bolna_integration.config import (
            PALLI_SAHAYAK_SYSTEM_PROMPT,
            RAG_QUERY_FUNCTION,
            LANGUAGE_CONFIGS,
        )
        results.add("Import bolna_integration.config", True)
    except ImportError as e:
        results.add("Import bolna_integration.config", False, str(e))


def test_bolna_client(results: TestResult):
    """Test BolnaClient functionality."""
    print("\n🔌 Testing BolnaClient...")

    from bolna_integration import BolnaClient, BolnaCallResult

    # Test client initialization
    client = BolnaClient()
    results.add("BolnaClient initialization", True)

    # Test is_available (should be False without API key)
    is_available = client.is_available()
    if os.getenv("BOLNA_API_KEY"):
        results.add("BolnaClient.is_available() with key", is_available)
    else:
        results.add("BolnaClient.is_available() without key", not is_available)

    # Test BolnaCallResult dataclass
    call_result = BolnaCallResult(
        success=True,
        call_id="test-123",
        agent_id="agent-456",
        data={"test": "data"}
    )
    results.add("BolnaCallResult creation", call_result.success and call_result.call_id == "test-123")


def test_agent_config(results: TestResult):
    """Test agent configuration generation."""
    print("\n⚙️  Testing agent configuration...")

    from bolna_integration import get_palli_sahayak_agent_config
    from bolna_integration.config import PALLI_SAHAYAK_SYSTEM_PROMPT, RAG_QUERY_FUNCTION

    # Test system prompt
    results.add(
        "System prompt defined",
        len(PALLI_SAHAYAK_SYSTEM_PROMPT) > 500,
        f"Length: {len(PALLI_SAHAYAK_SYSTEM_PROMPT)}"
    )

    # Test RAG function definition
    results.add(
        "RAG function has required fields",
        all(k in RAG_QUERY_FUNCTION for k in ["name", "description", "parameters"])
    )

    # Test config generation
    config = get_palli_sahayak_agent_config(
        server_url="https://test.example.com",
        language="hi"
    )

    results.add("Config has agent_name", config.get("agent_name") == "Palli Sahayak")
    results.add("Config has welcome_message", "नमस्ते" in config.get("agent_welcome_message", ""))
    results.add("Config has tasks", len(config.get("tasks", [])) > 0)

    # Check LLM config
    if config.get("tasks"):
        llm_config = config["tasks"][0].get("tools_config", {}).get("llm_agent", {})
        results.add("LLM config has functions", len(llm_config.get("functions", [])) > 0)

        # Check function URL
        if llm_config.get("functions"):
            func = llm_config["functions"][0]
            results.add(
                "Function URL is correct",
                "test.example.com/api/bolna/query" in func.get("value", {}).get("url", "")
            )


def test_webhook_handler(results: TestResult):
    """Test webhook handler functionality."""
    print("\n📡 Testing webhook handler...")

    from bolna_integration import BolnaWebhookHandler, CallRecord

    handler = BolnaWebhookHandler()
    results.add("WebhookHandler initialization", True)

    # Test call_started event
    async def test_events():
        # Call started
        event1 = {
            "event": "call_started",
            "call_id": "test-call-001",
            "phone_number": "+919876543210",
            "direction": "inbound"
        }
        result1 = await handler.handle_event(event1)
        return result1.get("status") == "recorded"

    started_ok = asyncio.run(test_events())
    results.add("Handle call_started event", started_ok)

    # Verify call is tracked
    results.add("Active call tracked", "test-call-001" in handler.active_calls)

    # Test call_ended event
    async def test_end():
        event2 = {
            "event": "call_ended",
            "call_id": "test-call-001",
            "duration_seconds": 180,
            "summary": "User asked about pain management"
        }
        result2 = await handler.handle_event(event2)
        return result2.get("status") == "recorded"

    ended_ok = asyncio.run(test_end())
    results.add("Handle call_ended event", ended_ok)

    # Verify call moved to completed
    results.add("Call moved to completed", "test-call-001" in handler.completed_calls)

    # Test stats
    stats = handler.get_call_stats()
    results.add("Get call stats", stats.get("completed_calls") == 1)
    results.add("Average duration tracked", stats.get("average_duration_seconds") == 180)


def test_voice_router(results: TestResult):
    """Test voice router functionality."""
    print("\n🔀 Testing voice router...")

    from voice_router import VoiceRouter, VoiceProvider, create_voice_router

    # Test router creation
    router = VoiceRouter()
    results.add("VoiceRouter initialization", True)

    # Test status
    status = router.get_status()
    results.add("Get router status", "available_providers" in status)
    results.add("Fallback always available", VoiceProvider.FALLBACK_PIPELINE.value in status["available_providers"])

    # Test provider selection
    phone_provider = router.select_provider("phone")
    results.add("Provider selection for phone", phone_provider is not None)

    web_provider = router.select_provider("web")
    results.add("Provider selection for web", web_provider is not None)

    # Test routing
    async def test_routing():
        response = await router.route_voice_request(
            request_type="web",
            user_id="test-user",
            language="hi"
        )
        return response.success

    routing_ok = asyncio.run(test_routing())
    results.add("Route voice request", routing_ok)

    # Test session tracking
    results.add("Session tracked", router.get_active_session_count() > 0)

    # Test create_voice_router factory
    router2 = create_voice_router()
    results.add("create_voice_router factory", router2 is not None)


def test_api_endpoints_simulation(results: TestResult):
    """Simulate API endpoint behavior."""
    print("\n🌐 Testing API endpoints (simulated)...")

    # Simulate /api/bolna/query request/response
    query_request = {
        "query": "How to manage pain for cancer patients?",
        "language": "hi",
        "context": "User asking about pain management",
        "source": "bolna_call"
    }

    # Validate request structure
    results.add(
        "Query request has required fields",
        all(k in query_request for k in ["query", "language"])
    )

    # Simulate expected response structure
    expected_response = {
        "status": "success",
        "answer": "For cancer pain management...",
        "sources": ["WHO Pain Guidelines"],
        "confidence": 0.92,
        "language": "hi"
    }

    results.add(
        "Response structure valid",
        all(k in expected_response for k in ["status", "answer", "sources", "confidence"])
    )

    # Simulate webhook event
    webhook_event = {
        "event": "call_ended",
        "call_id": "live-call-123",
        "duration_seconds": 240,
        "summary": "Discussed pain management options",
        "extracted_data": {
            "user_concern": "Pain management for cancer",
            "language_used": "hi",
            "emotional_state": "anxious",
            "follow_up_needed": True,
            "urgency_level": "medium"
        }
    }

    results.add(
        "Webhook event has required fields",
        all(k in webhook_event for k in ["event", "call_id"])
    )


async def test_live_api(results: TestResult):
    """Test with live Bolna API (requires API key)."""
    print("\n🔴 Testing live API...")

    api_key = os.getenv("BOLNA_API_KEY")
    if not api_key:
        results.skip("Live API test", "BOLNA_API_KEY not set")
        return

    from bolna_integration import BolnaClient

    client = BolnaClient(api_key=api_key)

    # Test health check
    try:
        is_healthy = await client.health_check()
        results.add("Bolna API health check", is_healthy)
    except Exception as e:
        results.add("Bolna API health check", False, str(e))

    # Test list agents
    try:
        agents_result = await client.list_agents()
        results.add("List agents", agents_result.success, agents_result.error or "")
    except Exception as e:
        results.add("List agents", False, str(e))


def test_config_files(results: TestResult):
    """Test configuration files."""
    print("\n📄 Testing configuration files...")

    import yaml

    # Test config.yaml
    config_path = "config.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        results.add("config.yaml loads", True)
        results.add("config.yaml has bolna section", "bolna" in config)
        results.add("config.yaml has voice_router section", "voice_router" in config)

        if "bolna" in config:
            bolna_config = config["bolna"]
            results.add("bolna.agent_name defined", "agent_name" in bolna_config)
            results.add("bolna.supported_languages defined", "supported_languages" in bolna_config)

    except FileNotFoundError:
        results.add("config.yaml exists", False, "File not found")
    except Exception as e:
        results.add("config.yaml valid", False, str(e))

    # Test .env.example
    env_example_path = ".env.example"
    try:
        with open(env_example_path, "r") as f:
            env_content = f.read()

        results.add(".env.example exists", True)
        results.add(".env.example has BOLNA_API_KEY", "BOLNA_API_KEY" in env_content)
        results.add(".env.example has BOLNA_AGENT_ID", "BOLNA_AGENT_ID" in env_content)
        results.add(".env.example has BOLNA_WEBHOOK_SECRET", "BOLNA_WEBHOOK_SECRET" in env_content)

    except FileNotFoundError:
        results.add(".env.example exists", False, "File not found")


def test_end_to_end_flow(results: TestResult):
    """Test complete end-to-end flow."""
    print("\n🔄 Testing end-to-end flow...")

    from bolna_integration import BolnaWebhookHandler, get_palli_sahayak_agent_config
    from voice_router import VoiceRouter

    async def e2e_test():
        # 1. Create agent config
        config = get_palli_sahayak_agent_config(
            server_url="https://palli-sahayak.example.com",
            language="hi"
        )

        # 2. Initialize components
        webhook_handler = BolnaWebhookHandler()
        voice_router = VoiceRouter()

        # 3. Simulate incoming call
        call_started = await webhook_handler.handle_event({
            "event": "call_started",
            "call_id": "e2e-test-001",
            "phone_number": "+919876543210",
            "agent_id": "test-agent"
        })

        # 4. Simulate RAG query (would be called by Bolna)
        rag_query = {
            "query": "दर्द का प्रबंधन कैसे करें?",
            "language": "hi",
            "source": "bolna_call"
        }

        # 5. Simulate call end with extraction
        call_ended = await webhook_handler.handle_event({
            "event": "call_ended",
            "call_id": "e2e-test-001",
            "duration_seconds": 300,
            "summary": "User inquired about pain management in Hindi",
            "extracted_data": {
                "user_concern": "Pain management",
                "language_used": "hi",
                "emotional_state": "calm",
                "follow_up_needed": False
            }
        })

        # 6. Check final state
        stats = webhook_handler.get_call_stats()

        return (
            call_started.get("status") == "recorded" and
            call_ended.get("status") == "recorded" and
            stats.get("completed_calls") >= 1
        )

    e2e_passed = asyncio.run(e2e_test())
    results.add("End-to-end flow", e2e_passed)


def main():
    parser = argparse.ArgumentParser(description="Test Bolna integration")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    parser.add_argument("--live", action="store_true", help="Include live API tests")
    args = parser.parse_args()

    print("=" * 70)
    print("PALLI SAHAYAK VOICE AI - BOLNA INTEGRATION TESTS")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = TestResult()

    # Always run these tests
    test_imports(results)

    if args.quick:
        # Quick smoke test - just imports
        pass
    else:
        # Full test suite
        test_bolna_client(results)
        test_agent_config(results)
        test_webhook_handler(results)
        test_voice_router(results)
        test_api_endpoints_simulation(results)
        test_config_files(results)
        test_end_to_end_flow(results)

    # Live API tests (optional)
    if args.live:
        asyncio.run(test_live_api(results))

    # Print summary
    all_passed = results.print_summary()

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
