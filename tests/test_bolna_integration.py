"""
The Bolna phone path is configured and tracks calls, without a key or the network.

Moved from the root `test_bolna_integration.py` (round-3 ticket 20). That script
collected pass/fail flags in its own result object, never raised, and was never
collected by pytest; its `--live` mode called the Bolna API. The offline checks
are kept here as assertions; nothing below reaches the network.

What these pin down:

  * without BOLNA_API_KEY the client reports itself unavailable, so the server
    does not offer a phone path it cannot serve
  * the agent config makes this server the agent's LLM (/api/bolna/llm/hi/v1),
    in Hindi with a Hindi welcome (ticket 17: no hosted LLM rewords the answer)
  * a call moves from active to completed, and the stats report its duration
"""

import asyncio
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bolna_integration import (  # noqa: E402
    BolnaClient,
    BolnaWebhookHandler,
    get_palli_sahayak_agent_config,
)

SERVER = "https://helpline.example.org"


class TheClientNeedsAKey(unittest.TestCase):
    def test_without_a_key_the_client_is_unavailable(self):
        with mock.patch.dict(os.environ, {"BOLNA_API_KEY": ""}):
            self.assertFalse(BolnaClient().is_available())

    def test_with_a_key_the_client_is_available(self):
        self.assertTrue(BolnaClient(api_key="test-key").is_available())


class TheAgentConfigPointsAtThisServer(unittest.TestCase):
    def setUp(self):
        self.config = get_palli_sahayak_agent_config(server_url=SERVER, language="hi")

    def llm(self):
        tasks = self.config.get("tasks") or []
        self.assertTrue(tasks, "the agent config has no tasks")
        return tasks[0]["tools_config"]["llm_agent"]["llm_config"]

    def test_the_agents_llm_is_this_servers_bolna_llm_route(self):
        self.assertEqual(self.llm()["provider"], "custom")
        self.assertEqual(self.llm()["base_url"], f"{SERVER}/api/bolna/llm/hi/v1")

    def test_a_hindi_agent_greets_in_hindi(self):
        self.assertIn("नमस्ते", self.config.get("agent_welcome_message", ""))


class CallsAreTracked(unittest.TestCase):
    def test_a_call_moves_from_active_to_completed_with_its_duration(self):
        handler = BolnaWebhookHandler()

        started = asyncio.run(handler.handle_event({
            "event": "call_started", "call_id": "call-001",
            "phone_number": "+910000000000", "direction": "inbound",
        }))
        self.assertEqual(started.get("status"), "recorded")
        self.assertIn("call-001", handler.active_calls)

        ended = asyncio.run(handler.handle_event({
            "event": "call_ended", "call_id": "call-001", "duration_seconds": 180,
        }))
        self.assertEqual(ended.get("status"), "recorded")
        self.assertNotIn("call-001", handler.active_calls)
        self.assertIn("call-001", handler.completed_calls)

        stats = handler.get_call_stats()
        self.assertEqual(stats.get("completed_calls"), 1)
        self.assertEqual(stats.get("average_duration_seconds"), 180)


if __name__ == "__main__":
    unittest.main()
