"""
One Care Role vocabulary across server and app (review 3, ticket 25).

Care Role had three vocabularies: the profile's PATIENT / CAREGIVER /
HEALTHCARE_WORKER (guessed from words like "nurse" and "doctor"), a free
`role: str` on the mobile schemas, and a hard-coded `"patient"` speaker role on
the answer path. The study row recorded whatever string the registration token
carried, never the Active Role the person holding the phone had switched to.

What these pin down, driven through the mobile router on a FastAPI app with a
TestClient and the real study logger writing to a temp dir:

  * the schemas refuse a role outside ASHA Worker, Family Caregiver, Patient
  * a question is recorded with the caller's Active Role, falling back to the
    registered Care Role when the app sends none, on text, voice and the socket
  * a study row never carries a role outside the vocabulary
  * the answer path's observations are reported by the speaker's role, not
    "patient"
  * no code path assigns a role from keywords, and the app's wire names are the
    server's values
"""

import asyncio
import base64
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import auth.pin_auth as pin_auth_module  # noqa: E402
import mobile_api.router as router_module  # noqa: E402
from auth.middleware import get_jwt_handler, init_auth  # noqa: E402
from mobile_api.dependencies import init_dependencies  # noqa: E402
from mobile_api.router import mobile_router  # noqa: E402
from safety_enhancements import SafetyEnhancementsManager  # noqa: E402
from study_logging import StudyInteractionLogger, read_entries  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASE = "/api/mobile/v1"
SECRET = "test-secret-for-the-care-role-vocabulary"
PIN = "4821"
PCM = b"\x00\x01\x02\x03" * 64
QUESTION = "How do I help her sleep when the pain wakes her?"
ANDROID_CARE_ROLE = (ROOT / "android-app" / "core" / "data" / "src" / "main" / "kotlin" / "org"
                     / "inventcures" / "pallisahayak" / "data" / "model" / "CareRole.kt")


class _Rag:
    """Same call shape as SimpleRAGPipeline.query; answers without a dose and
    remembers the role it was asked under."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def query(self, question, conversation_id=None, user_id=None, top_k: int = 5,
                    source_language: str = "en", care_role: str = ""):
        self.calls.append({"user_id": user_id, "care_role": care_role})
        return {"answer": "Keep her warm and settle her before the evening dose time.",
                "sources": [{"filename": "handbook.pdf"}]}


class _Sarvam:
    """Stands in for Sarvam's speech calls, which are the network."""

    async def speech_to_text(self, audio, language):
        return type("R", (), {"transcript": QUESTION})()

    async def text_to_speech(self, text, language):
        return type("R", (), {"audio_base64": None})()


class MobileHarness(unittest.TestCase):
    """The mobile router on its own app, the study log in a fresh directory."""

    def setUp(self) -> None:
        self.log_dir = Path(tempfile.mkdtemp(prefix="care-role-rows-"))
        study_log = StudyInteractionLogger(
            log_dir=self.log_dir, use_analytics=False, use_expert_sampling=False)
        for patcher in (
            mock.patch.object(pin_auth_module, "DATA_DIR", Path(tempfile.mkdtemp(prefix="devices-"))),
            mock.patch("sarvam_integration.SarvamClient", return_value=_Sarvam()),
            mock.patch.object(router_module, "_study_logger", study_log),
            mock.patch.dict(os.environ, {"RETRIEVAL_AUDIT_DIR": tempfile.mkdtemp()}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        init_auth(jwt_secret=SECRET)
        self.rag = _Rag()
        init_dependencies(self.rag, SafetyEnhancementsManager(), None, None)
        app = FastAPI()
        app.include_router(mobile_router)
        self.client = TestClient(app)

    def register(self, role: str):
        return self.client.post(f"{BASE}/auth/register", json={
            "name": "Sunita", "phone_hash": "hash-s", "role": role,
            "language": "hi-IN", "site_id": "site-1", "pin": PIN,
        })

    def signed_in(self, role: str) -> Dict[str, str]:
        response = self.register(role)
        self.assertEqual(response.status_code, 200, response.text)
        return {"Authorization": f"Bearer {response.json()['token']}"}

    def ask(self, headers: Dict[str, str], **extra: Any):
        return self.client.post(f"{BASE}/query", headers=headers,
                                json={"query": QUESTION, "language": "hi-IN", **extra})

    def ask_by_voice(self, headers: Dict[str, str], **params: Any):
        return self.client.post(f"{BASE}/query/voice", headers=headers,
                                params={"language": "hi-IN", **params},
                                files={"audio": ("q.wav", PCM, "audio/wav")})

    def the_one_row(self) -> Dict[str, Any]:
        rows = read_entries(log_dir=self.log_dir)
        self.assertEqual(len(rows), 1, rows)
        return rows[0]


class SchemasRejectAnUnknownRole(MobileHarness):
    def test_registration_refuses_a_role_outside_the_care_role_vocabulary(self) -> None:
        for role in ("nurse", "doctor", "asha", "caregiver", "healthcare_worker", "admin", ""):
            with self.subTest(role=role):
                self.assertEqual(self.register(role).status_code, 422)

    def test_registration_accepts_each_care_role(self) -> None:
        for role in ("asha_worker", "family_caregiver", "patient"):
            with self.subTest(role=role):
                self.assertEqual(self.register(role).status_code, 200)

    def test_a_question_under_an_unknown_active_role_is_refused(self) -> None:
        headers = self.signed_in("asha_worker")
        self.assertEqual(self.ask(headers, active_role="nurse").status_code, 422)
        self.assertEqual(read_entries(log_dir=self.log_dir), [])

    def test_a_voice_question_under_an_unknown_active_role_is_refused(self) -> None:
        headers = self.signed_in("asha_worker")
        self.assertEqual(self.ask_by_voice(headers, active_role="doctor").status_code, 422)


class StudyRowRecordsTheActiveRole(MobileHarness):
    def test_an_asha_workers_question_is_recorded_with_the_asha_worker_role(self) -> None:
        response = self.ask(self.signed_in("asha_worker"))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.the_one_row()["care_role"], "asha_worker")

    def test_a_switched_active_role_is_the_speaker_role_in_the_study_row(self) -> None:
        # A Family Caregiver's phone, handed to the visiting ASHA Worker.
        response = self.ask(self.signed_in("family_caregiver"), active_role="asha_worker")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.the_one_row()["care_role"], "asha_worker")

    def test_the_answer_path_is_asked_under_the_active_role(self) -> None:
        self.ask(self.signed_in("family_caregiver"), active_role="patient")
        self.assertEqual(self.rag.calls[-1]["care_role"], "patient")

    def test_a_voice_question_is_recorded_with_the_active_role(self) -> None:
        response = self.ask_by_voice(self.signed_in("asha_worker"), active_role="patient")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.the_one_row()["care_role"], "patient")
        self.assertEqual(self.rag.calls[-1]["care_role"], "patient")

    def test_a_voice_question_without_an_active_role_is_recorded_with_the_registered_role(self) -> None:
        response = self.ask_by_voice(self.signed_in("family_caregiver"))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.the_one_row()["care_role"], "family_caregiver")

    def test_a_voice_socket_turn_is_recorded_with_the_active_role(self) -> None:
        headers = self.signed_in("patient")
        with self.client.websocket_connect(
                f"{BASE}/ws/voice?language=hi-IN&active_role=family_caregiver",
                headers=headers) as ws:
            ws.receive_json()
            ws.send_text(json.dumps({"type": "audio", "language": "hi-IN",
                                     "audio_base64": base64.b64encode(PCM).decode()}))
            reply = ws.receive_json()
        self.assertEqual(reply["type"], "response", reply)
        self.assertEqual(self.the_one_row()["care_role"], "family_caregiver")

    def test_a_row_never_records_a_role_outside_the_vocabulary(self) -> None:
        # A token minted before the vocabulary was enforced.
        token = get_jwt_handler().create_token_pair("old-1", "nurse", "site-1").access_token
        response = self.ask({"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.the_one_row()["care_role"], "")


class ObservationsAreReportedByTheSpeaker(unittest.TestCase):
    """The answer path's post-reply observations name who spoke."""

    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {"RETRIEVAL_AUDIT_DIR": tempfile.mkdtemp()})
        patcher.start()
        self.addCleanup(patcher.stop)

    def observed_metadata(self, **query_kwargs: Any) -> Dict[str, Any]:
        import post_work
        from retrieval.settings import RetrievalSettings, set_active_settings
        from test_answer_path_retrieval import make_pipeline

        seen: List[Dict[str, Any]] = []

        class Aggregator:
            async def process_conversation(self, **kwargs):
                seen.append(kwargs["metadata"])

        pipeline = make_pipeline()
        pipeline.configure_answer_path(set_active_settings(RetrievalSettings()))
        pipeline.cross_modal_aggregator = Aggregator()

        async def ask() -> None:
            await pipeline.query(QUESTION, user_id="asha-1", **query_kwargs)
            await post_work.drain()

        asyncio.run(ask())
        self.assertEqual(len(seen), 1, seen)
        return seen[0]

    def test_an_asha_workers_question_is_observed_as_the_asha_workers_report(self) -> None:
        self.assertEqual(self.observed_metadata(care_role="asha_worker")["speaker_role"], "asha_worker")

    def test_a_question_with_no_role_is_not_attributed_to_the_patient(self) -> None:
        self.assertNotEqual(self.observed_metadata()["speaker_role"], "patient")


class NoRoleIsGuessed(unittest.TestCase):
    def test_care_role_has_three_values(self) -> None:
        from care_role import CareRole

        self.assertEqual({r.value for r in CareRole}, {"asha_worker", "family_caregiver", "patient"})

    def test_the_app_and_the_server_share_one_care_role_vocabulary(self) -> None:
        from care_role import CareRole

        wire = re.findall(r'^\s*[A-Z_]+\("([a-z_]+)",', ANDROID_CARE_ROLE.read_text(), re.M)
        self.assertEqual(sorted(wire), sorted(r.value for r in CareRole))

    def test_the_profile_never_guesses_a_role_from_what_was_said(self) -> None:
        from personalization.user_profile import UserProfileManager

        manager = UserProfileManager(storage_path=tempfile.mkdtemp(prefix="profiles-"))
        self.assertFalse(hasattr(manager, "detect_role"))
        self.assertFalse(hasattr(UserProfileManager, "ROLE_KEYWORDS"))

    def test_a_stored_profile_role_outside_the_vocabulary_reads_as_unknown(self) -> None:
        from personalization.user_profile import UserProfile

        stored = {"user_id": "u1", "role": "healthcare_worker",
                  "first_interaction": "2026-01-01T00:00:00",
                  "last_interaction": "2026-01-01T00:00:00"}
        self.assertIsNone(UserProfile.from_dict(stored).role)
        stored["role"] = "family_caregiver"
        self.assertEqual(UserProfile.from_dict(stored).role.value, "family_caregiver")

    def test_no_module_scores_a_role_from_keywords(self) -> None:
        offenders = []
        for path in ROOT.glob("**/*.py"):
            parts = path.relative_to(ROOT).parts
            if parts[0] in {"kotaemon-main", "tests", ".claude", "cache", "out"} or "site-packages" in parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"\bROLE_KEYWORDS\b|\bdef detect_role\b", text):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
