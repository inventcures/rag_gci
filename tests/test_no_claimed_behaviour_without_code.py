"""
A study release contains only code that can run (review 3, ticket 26, D12).

Before this:

  * the emergency reply told a caller in English and Hindi that "a human
    caregiver has been notified", and the voice escalation recorded a
    "caregiver_notification_queued" action, though nothing notifies anyone
    (CONTEXT.md: no notification path exists);
  * the care-team "notify" route and admin tab answered "Notification sent to N
    care team members" and listed each member as notified, queued or logged,
    and the alert manager marked WhatsApp and email alerts SENT after writing
    a log line;
  * start-up built Context-1, the memory agents and the self-modifying meta
    agent, which no answer path uses;
  * /feedback paid for an LLM call that graded an empty response;
  * medication adherence was looked up by the patient id in a table keyed by
    the household's user id, so it was always empty.
"""

import asyncio
import os
import sys
import tempfile
import time
import types
import unittest
import uuid
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from tests.test_route_policy import (  # noqa: E402,F401
    build_server_app,
    setUpModule,
    tearDownModule,
)

BASE = "/api/mobile/v1"
NOTIFIED_CLAIMS = ("notified", "सूचित", "notification sent", "queued")


class TheEmergencyReplyClaimsNoNotification(unittest.TestCase):
    """The 108 guidance stays; the claim that someone was told goes."""

    QUESTIONS = {
        "en": ("he is not breathing", "she has unbearable pain"),
        "hi": ("वह बेहोश है", "बहुत दर्द हो रहा है"),
    }

    def test_the_emergency_actions_say_no_one_was_notified(self):
        from safety_enhancements import EmergencyDetectionSystem, EmergencyLevel

        detector = EmergencyDetectionSystem()
        for language, (critical, high) in self.QUESTIONS.items():
            for question, level in ((critical, EmergencyLevel.CRITICAL), (high, EmergencyLevel.HIGH)):
                with self.subTest(language=language, level=level.value):
                    alert = detector.detect_emergency(question, "caller-1", language)
                    self.assertIsNotNone(alert)
                    self.assertEqual(alert.level, level)
                    for claim in NOTIFIED_CLAIMS:
                        self.assertNotIn(claim, alert.action_required.lower())
                    if level is EmergencyLevel.CRITICAL:
                        self.assertIn("108", alert.action_required)

    def test_the_spoken_emergency_reply_and_its_escalation_claim_no_notification(self):
        from voice_safety_wrapper import VoiceSafetyWrapper

        wrapper = VoiceSafetyWrapper()
        result = asyncio.run(wrapper.check_voice_query("caller-1", "he is not breathing", "en"))
        self.assertIn("108", result.safety_message)
        for claim in NOTIFIED_CLAIMS:
            self.assertNotIn(claim, result.safety_message.lower())
        escalation = asyncio.run(wrapper.handle_voice_escalation(result, provider="bolna"))
        self.assertNotIn("caregiver_notification_queued", escalation["actions_taken"])


class NoNotificationIsRecordedWithoutADelivery(unittest.TestCase):
    def test_the_care_team_notify_route_is_gone(self):
        from auth.middleware import init_auth
        from auth.policy import issue_admin_token
        from tests.test_route_policy import MOBILE_SECRET

        app = build_server_app()
        init_auth(jwt_secret=MOBILE_SECRET)
        client = TestClient(app, raise_server_exceptions=False)
        admin = {"Authorization": f"Bearer {issue_admin_token('dr-ashish')}"}
        response = client.post("/api/careteam/pt-1/notify", headers=admin,
                               json={"message": "Patient needs follow-up"})
        self.assertIn(response.status_code, (404, 405), response.text)
        self.assertNotIn("notification sent", response.text.lower())

    def test_the_alert_manager_has_no_send_path(self):
        import personalization
        from personalization.alert_manager import AlertManager

        self.assertFalse(hasattr(AlertManager, "send_alert"))
        self.assertFalse(hasattr(personalization, "AlertNotificationCoordinator"))
        self.assertFalse(hasattr(personalization, "DeliveryStatus"))

    def test_the_admin_panel_has_no_notify_handler(self):
        import simple_rag_server as srv

        self.assertFalse(hasattr(srv.SimpleAdminUI, "_handle_notify_care_team"))


class StartUpBuildsNoUnusedAgent(unittest.TestCase):
    def test_start_up_constructs_no_memory_agent_meta_agent_or_context1_agent(self):
        recorded = {}
        modules = {}
        for name, classes in {
            "context1_integration": ("Context1Config", "Context1RetrievalAgent", "QueryComplexityClassifier"),
            "memory_agents": ("MemoryStore", "PatientIngestAgent", "PatientQueryAgent", "ConsolidationScheduler"),
            "meta_agent": ("ClinicalMetaAgent", "ResponseEvaluator"),
        }.items():
            module = types.ModuleType(name)
            for cls in classes:
                recorded[cls] = mock.MagicMock(name=cls)
                setattr(module, cls, recorded[cls])
            modules[name] = module

        build_server_app(modules=modules)

        built = sorted(name for name, cls in recorded.items() if cls.called or cls.from_dict.called)
        self.assertEqual(built, [], "start-up constructed an agent no answer path uses")


class FeedbackMakesNoLLMCall(unittest.TestCase):
    def setUp(self):
        from auth.middleware import get_jwt_handler, init_auth
        from mobile_api.router import mobile_router

        init_auth(jwt_secret="test-secret-for-feedback-without-an-llm")
        app = FastAPI()
        app.include_router(mobile_router)
        self.client = TestClient(app)
        token = get_jwt_handler().create_token_pair("asha-1", "asha", "site-1").access_token
        self.headers = {"Authorization": f"Bearer {token}"}

    def test_feedback_is_recorded_without_grading(self):
        import mobile_api.dependencies as deps
        from clinical_validation import FeedbackCollector
        from meta_agent.evaluator import ResponseEvaluator

        workdir = tempfile.mkdtemp(prefix="feedback-")
        llm = mock.MagicMock(name="llm")
        llm.generate = mock.AsyncMock(return_value=None)
        collector = FeedbackCollector(storage_path=os.path.join(workdir, "feedback"))
        with mock.patch("meta_agent.evaluator.get_llm_client", return_value=llm), \
                mock.patch.object(deps, "_meta_evaluator",
                                  ResponseEvaluator(evaluations_dir=os.path.join(workdir, "eval")), create=True), \
                mock.patch.object(deps, "_feedback_collector", collector, create=True):
            response = self.client.post(f"{BASE}/feedback", headers=self.headers,
                                        json={"query_id": "q-17", "positive": False, "comment": "too long"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(llm.generate.await_count, 0, "feedback paid for an LLM call")
        self.assertEqual(response.json()["status"], "accepted")
        stored = asyncio.run(collector.get_feedback())
        self.assertEqual([(f.query, f.user_id, f.comment) for f in stored], [("q-17", "asha-1", "too long")])


class AdherenceIsCountedForThePatient(unittest.TestCase):
    """The household's reminders for a patient count towards that patient."""

    def setUp(self):
        from auth.middleware import get_jwt_handler, init_auth
        from medication_voice_reminders import MedicationVoiceReminderSystem
        from mobile_api.dependencies import init_dependencies
        from mobile_api.router import mobile_router
        from personalization.longitudinal_memory import LongitudinalMemoryManager

        workdir = tempfile.mkdtemp(prefix="adherence-")
        for name in ("_init_providers", "_start_scheduler", "_schedule_call"):
            patcher = mock.patch.object(MedicationVoiceReminderSystem, name)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.reminders = MedicationVoiceReminderSystem(storage_path=os.path.join(workdir, "reminders"))
        memory = LongitudinalMemoryManager(storage_path=os.path.join(workdir, "longitudinal"))
        init_auth(jwt_secret="test-secret-for-adherence-by-patient")
        init_dependencies(None, None, memory, self.reminders)
        app = FastAPI()
        app.include_router(mobile_router)
        self.client = TestClient(app)
        token = get_jwt_handler().create_token_pair(str(uuid.uuid4()), "caregiver", "site-1").access_token
        self.headers = {"Authorization": f"Bearer {token}"}
        for patient in ("pt-1", "pt-2"):
            response = self.client.post(f"{BASE}/observation", headers=self.headers, json={
                "patient_id": patient, "category": "symptom", "entity_name": "pain", "severity": 2})
            self.assertEqual(response.status_code, 200, response.text)

    def remind(self, patient: str, medicine: str) -> str:
        response = self.client.post(f"{BASE}/medication/reminder", headers=self.headers, json={
            "patient_id": patient, "medication_name": medicine, "scheduled_time": time.time() - 60,
            "language": "en", "phone_number": "+910000000000"})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["reminder_id"]

    def test_a_confirmed_reminder_counts_for_its_patient_only(self):
        confirmed = self.remind("pt-1", "Paracetamol")
        self.remind("pt-1", "Senna")
        self.remind("pt-2", "Morphine")
        self.reminders.reminders[confirmed].patient_confirmed = True

        stats = self.client.get(f"{BASE}/medication/adherence/pt-1", headers=self.headers).json()
        self.assertEqual((stats["total_reminders"], stats["confirmed"], stats["adherence_rate"]), (2, 1, 50.0))
        self.assertEqual(stats["patient_id"], "pt-1")
        other = self.client.get(f"{BASE}/medication/adherence/pt-2", headers=self.headers).json()
        self.assertEqual((other["total_reminders"], other["confirmed"]), (1, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
