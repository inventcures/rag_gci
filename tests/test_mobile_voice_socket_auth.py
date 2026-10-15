"""
The app's voice socket needs a signed-in participant (review 3, ticket 22).

`WS /api/mobile/v1/ws/voice` accepted any connection, because the Android app
opened it without a token. The app now signs in with a PIN and sends its access
token on the socket's `Authorization` header, so the route refuses a
connection that has no access token and attributes each turn to the person
the token names.

Driven through the mobile router on a FastAPI app with a TestClient. Only the
network (Sarvam) and the study log's writer are faked.
"""

import base64
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

import auth.pin_auth as pin_auth_module  # noqa: E402
import mobile_api.router as router_module  # noqa: E402
from auth.middleware import init_auth  # noqa: E402
from auth.policy import Access, policy_for  # noqa: E402
from mobile_api.dependencies import init_dependencies  # noqa: E402
from mobile_api.router import mobile_router  # noqa: E402
from safety_enhancements import SafetyEnhancementsManager  # noqa: E402

BASE = "/api/mobile/v1"
SOCKET = f"{BASE}/ws/voice?language=hi-IN"
SECRET = "test-secret-for-the-voice-socket"
PIN = "4821"
PCM = b"\x00\x01\x02\x03" * 64
WS_POLICY_VIOLATION = 1008


class _Rag:
    """Same signature as SimpleRAGPipeline.query; answers without a dose."""

    async def query(self, question, conversation_id=None, user_id=None,
                    top_k: int = 5, source_language: str = "en", care_role: str = ""):
        return {"answer": "Keep the patient comfortable and call the nurse.",
                "sources": [{"filename": "handbook.pdf"}]}


class _Sarvam:
    """Stands in for Sarvam's speech calls, which are the network."""

    async def speech_to_text(self, audio, language):
        return type("R", (), {"transcript": "the patient is in pain"})()

    async def text_to_speech(self, text, language):
        return type("R", (), {"audio_base64": None})()


class VoiceSocketNeedsASignedInParticipant(unittest.TestCase):
    """The socket opens for an access token and for nothing else."""

    def setUp(self) -> None:
        devices = tempfile.mkdtemp(prefix="devices-")
        self.recorded = []

        async def fake_record(**kwargs):
            self.recorded.append(kwargs)

        for patcher in (
            mock.patch.object(pin_auth_module, "DATA_DIR", Path(devices)),
            mock.patch("sarvam_integration.SarvamClient", return_value=_Sarvam()),
            mock.patch.object(router_module._study_logger, "record", new=fake_record),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        init_auth(jwt_secret=SECRET)
        init_dependencies(_Rag(), SafetyEnhancementsManager(), None, None)
        app = FastAPI()
        app.include_router(mobile_router)
        self.client = TestClient(app)
        response = self.client.post(f"{BASE}/auth/register", json={
            "name": "Asha A", "phone_hash": "hash-a", "role": "asha_worker",
            "language": "hi", "site_id": "site-1", "pin": PIN,
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.session = response.json()

    def _refused(self, headers=None) -> int:
        """Open the socket and return the close code it is refused with."""
        with self.assertRaises(WebSocketDisconnect) as refused:
            with self.client.websocket_connect(SOCKET, headers=headers or {}) as ws:
                ws.receive_json()
        return refused.exception.code

    def test_a_connection_without_a_token_is_refused(self) -> None:
        self.assertEqual(self._refused(), WS_POLICY_VIOLATION)

    def test_a_refresh_token_does_not_open_the_socket(self) -> None:
        headers = {"Authorization": f"Bearer {self.session['refresh_token']}"}
        self.assertEqual(self._refused(headers), WS_POLICY_VIOLATION)

    def test_an_invalid_token_is_refused(self) -> None:
        self.assertEqual(self._refused({"Authorization": "Bearer not-a-token"}),
                         WS_POLICY_VIOLATION)

    def test_a_signed_in_participant_gets_a_turn_recorded_under_their_id(self) -> None:
        headers = {"Authorization": f"Bearer {self.session['token']}"}
        with self.client.websocket_connect(SOCKET, headers=headers) as ws:
            ready = ws.receive_json()
            ws.send_text(json.dumps({
                "type": "audio", "language": "hi-IN",
                "audio_base64": base64.b64encode(PCM).decode(),
            }))
            reply = ws.receive_json()

        self.assertEqual(ready["type"], "ready")
        self.assertEqual(reply["type"], "response", reply)
        self.assertTrue(self.recorded, "the turn was not recorded")
        self.assertEqual(self.recorded[-1]["user_id"], self.session["user_id"])
        self.assertEqual(self.recorded[-1]["site_id"], "site-1")

    def test_the_route_table_declares_the_socket_for_participants(self) -> None:
        self.assertIs(policy_for(f"{BASE}/ws/voice").access, Access.PARTICIPANT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
