"""
Provider routes verify who is calling (round-3 final review, findings 4 and 5).

Before this:

  * POST /api/medication/voice-reminder/callback was declared "own
    verification" but checked nothing. Anyone with a call id could mark a
    reminder call completed or confirmed, and set off the confirmation or
    missed-call WhatsApp messages, which corrupts the adherence record. No
    provider is configured to call it (Bolna and Retell report through their own
    signed webhooks), so it is now an admin route like the rest of
    /api/medication/*.
  * WS /ws/retell/llm/{call_id} checked only the release. Once Retell was
    released, any socket could claim any caller's number in its call_details
    frame and be answered from that caller's patient context and history, and
    write study rows as them. The socket now asks Retell for the call (keyed by
    RETELL_API_KEY) and takes the caller from that record; a call Retell does
    not know as live is refused, and the frame's numbers are ignored.

Driven through the app create_app() builds; Retell's API is faked at
RetellClient.get_call, the one call that leaves the server.
"""

import asyncio
import os
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fastapi.testclient import TestClient  # noqa: E402

import test_route_policy as harness  # noqa: E402
from test_phone_providers_residency import (  # noqa: E402
    Release as ReleaseHarness,
    retell_services,
)

CALLBACK = "/api/medication/voice-reminder/callback"
REAL_CALLER = "+919800000111"
FORGED_CALLER = "+919800000999"
QUESTION = "How can I help my mother sleep at night?"
SAFE_ANSWER = "Keep the room quiet and dim, and tell the palliative team if pain wakes her."


def setUpModule():
    harness.setUpModule()


def tearDownModule():
    harness.tearDownModule()


# ---------------------------------------------------------------------------
# Finding 4: the reminder callback is an admin route
# ---------------------------------------------------------------------------


class TheReminderCallbackNeedsAnAdmin(harness.ServerHarness):
    BODY = {"call_id": "call_123", "status": "completed", "duration": 45,
            "patient_response": "yes"}

    def test_an_anonymous_callback_is_refused(self):
        response = self.client.post(CALLBACK, json=self.BODY)
        self.assertEqual(response.status_code, 401, response.text)

    def test_a_participant_cannot_mark_a_reminder_call(self):
        response = self.client.post(CALLBACK, json=self.BODY, headers=self.participant)
        self.assertEqual(response.status_code, 403, response.text)

    def test_an_admin_can_still_record_a_call_outcome(self):
        response = self.client.post(CALLBACK, json=self.BODY, headers=self.admin)
        self.assertEqual(response.status_code, 200, response.text)

    def test_the_policy_table_names_it_admin(self):
        from auth.policy import Access, policy_for

        self.assertIs(policy_for(CALLBACK).access, Access.ADMIN)


# ---------------------------------------------------------------------------
# Finding 5: the Retell socket takes the caller from Retell, not the client
# ---------------------------------------------------------------------------


def retell_call(call_id: str, status: str = "ongoing", from_number: str = REAL_CALLER):
    """What Retell's get-call returns for a call on this account."""
    from retell_integration.client import RetellCallResult

    return RetellCallResult(success=True, call_id=call_id, data={
        "call_id": call_id, "call_status": status, "agent_id": "agent_ps",
        "from_number": from_number, "to_number": "+918000000000",
        "metadata": {"language": "en-IN"},
    })


class RetellSocket(ReleaseHarness):
    """The real create_app() with Retell enabled and released, its startup run."""

    app = None

    @classmethod
    def setUpClass(cls):
        if RetellSocket.app is None:
            RetellSocket.app = harness.build_server_app()
        cls.client = TestClient(RetellSocket.app, raise_server_exceptions=False)

    def setUp(self):
        super().setUp()
        self.release_listing(retell_services())
        self.enable(RETELL_ENABLED="true")
        for handler in self.app.router.on_startup:
            if handler.__name__ == "startup_retell":
                asyncio.run(handler())

        self.asked: List[Dict[str, Any]] = []
        asked = self.asked

        async def query(pipeline, question, conversation_id=None, user_id=None,
                        top_k=5, source_language="en", **kwargs):
            asked.append({"question": question, "user_id": user_id})
            return {"status": "success", "answer": SAFE_ANSWER,
                    "sources": [{"filename": "handbook.pdf"}]}

        patch = mock.patch.object(harness._FakePipeline, "query", query, create=True)
        patch.start()
        self.addCleanup(patch.stop)
        self.lookups: List[str] = []

    def retell_knows(self, result_for) -> None:
        """Fake Retell's get-call; `result_for(call_id)` gives its answer."""
        from retell_integration.client import RetellClient

        lookups = self.lookups

        async def get_call(client, call_id: str):
            lookups.append(call_id)
            return result_for(call_id)

        patch = mock.patch.object(RetellClient, "get_call", get_call)
        patch.start()
        self.addCleanup(patch.stop)

    def first_message(self, call_id: str) -> Dict[str, Any]:
        with self.client.websocket_connect(f"/ws/retell/llm/{call_id}") as socket:
            return socket.receive_json()

    def ask(self, call_id: str, frame_from: Optional[str] = FORGED_CALLER) -> str:
        """Open the socket, send call_details claiming `frame_from`, ask, return the answer."""
        with self.client.websocket_connect(f"/ws/retell/llm/{call_id}") as socket:
            self.assertEqual(socket.receive_json()["response_type"], "config")
            socket.send_json({"interaction_type": "call_details",
                              "call": {"call_id": call_id, "from_number": frame_from,
                                       "metadata": {"language": "en-IN"}}})
            socket.send_json({"interaction_type": "response_required", "response_id": 1,
                              "transcript": [{"role": "user", "content": QUESTION}]})
            parts = []
            while True:
                message = socket.receive_json()
                if message.get("response_type") != "response":
                    continue
                parts.append(message.get("content", ""))
                if message.get("content_complete"):
                    return "".join(parts)

    def test_a_call_retell_does_not_know_is_refused(self):
        from retell_integration.client import RetellCallResult

        self.retell_knows(lambda call_id: RetellCallResult(success=False, error="Call not found"))

        message = self.first_message("forged-call")

        self.assertEqual(message["response_type"], "error", message)
        self.assertEqual(self.lookups, ["forged-call"])
        self.assertEqual(self.asked, [])

    def test_an_ended_call_is_refused(self):
        self.retell_knows(lambda call_id: retell_call(call_id, status="ended"))

        self.assertEqual(self.first_message("old-call")["response_type"], "error")

    def test_the_caller_is_the_number_retell_reports_not_the_frame(self):
        self.retell_knows(lambda call_id: retell_call(call_id))

        answer = self.ask("live-call", frame_from=FORGED_CALLER)

        self.assertIn("quiet", answer)
        self.assertEqual([a["user_id"] for a in self.asked], [REAL_CALLER])

    def test_without_a_retell_key_the_socket_is_refused(self):
        self.retell_knows(lambda call_id: retell_call(call_id))
        os.environ.pop("RETELL_API_KEY", None)

        self.assertEqual(self.first_message("live-call")["response_type"], "error")


if __name__ == "__main__":
    unittest.main()
