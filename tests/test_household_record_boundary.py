"""
The Household Record boundary holds on reminders, sync and export (review 3, ticket 09).

Before this:

  * GET /medication/reminders took a `user_id` query and read anyone's list (it
    also awaited a method that did not exist, so it always answered []).
  * Sync pull awaited a synchronous reminder query, so reminders never pulled;
    that query ignored the user, so the leak opened the day pull worked.
  * Sync push awaited a synchronous upsert: the reminder was saved, then counted
    as rejected. The upsert took its owner from the payload and changed any
    household's reminder whose id it was given.
  * POST /medication/reminder never checked that the patient was the caller's.
  * /evaluation/export gave any participant everyone's SUS scores, vignette
    answers and interaction logs.
  * Interaction logs and vignettes that failed to save were dropped with no log
    line, and a pushed log or vignette id could overwrite another household's.
  * /auth/login built a file path from the client's user_id, so "../x" read a
    file outside the device store.

What these pin down, over HTTP with two households, A and B: A gets nothing of
B's from the reminder list, sync pull, push upsert, reminder creation or the
export; push acknowledges every item truthfully; an auditor (admin token) gets
the whole export.
"""

import json
import os
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import auth.pin_auth as pin_auth_module  # noqa: E402
from auth.middleware import get_jwt_handler, init_auth  # noqa: E402
from auth.policy import ADMIN_SECRET_ENV, issue_admin_token  # noqa: E402
from medication_voice_reminders import MedicationVoiceReminderSystem  # noqa: E402
from mobile_api.dependencies import init_dependencies  # noqa: E402
from mobile_api.router import mobile_router  # noqa: E402
from personalization.longitudinal_memory import LongitudinalMemoryManager  # noqa: E402

BASE = "/api/mobile/v1"
ADMIN_SECRET = "an-admin-secret-for-household-boundary-tests-0123456789"
EVALUATION_DIRS = ("sus_scores", "vignette_responses", "interaction_logs", "vignette_assignments")


class HouseholdRecordBoundary(unittest.TestCase):
    """Household A and household B, each with a patient and a reminder."""

    def setUp(self) -> None:
        # The evaluation stores write under ./data/evaluation; run in a scratch
        # directory so the test reads only what it wrote.
        workdir = tempfile.mkdtemp(prefix="household-")
        previous = os.getcwd()
        os.chdir(workdir)
        self.addCleanup(os.chdir, previous)
        for name in EVALUATION_DIRS:
            Path("data/evaluation", name).mkdir(parents=True, exist_ok=True)

        for name in ("_init_providers", "_start_scheduler", "_schedule_call"):
            patcher = mock.patch.object(MedicationVoiceReminderSystem, name)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.reminders = MedicationVoiceReminderSystem(storage_path=os.path.join(workdir, "reminders"))
        self.memory = LongitudinalMemoryManager(storage_path=os.path.join(workdir, "longitudinal"))

        init_auth(jwt_secret="test-secret-for-household-record-boundary")
        init_dependencies(None, None, self.memory, self.reminders)
        import mobile_api.router as router_module

        vignettes = mock.patch.object(router_module._vignette_manager, "assign_vignettes", return_value=[])
        vignettes.start()
        self.addCleanup(vignettes.stop)

        app = FastAPI()
        app.include_router(mobile_router)
        self.client = TestClient(app)
        handler = get_jwt_handler()
        self.user = {h: str(uuid.uuid4()) for h in ("A", "B")}
        self.headers = {
            h: {"Authorization": f"Bearer {handler.create_token_pair(uid, 'caregiver', 'site-1').access_token}"}
            for h, uid in self.user.items()
        }
        self.patient = {"A": "pt-household-a", "B": "pt-household-b"}
        for household, patient_id in self.patient.items():
            result = self.push(household, observations=[{
                "observation_id": f"{patient_id}-pain", "patient_id": patient_id,
                "category": "symptom", "entity_name": "pain", "severity": 2, "timestamp": time.time(),
            }])
            self.assertEqual(result["accepted"], 1, result)
        self.reminder_id = {
            "A": self.create_reminder("A", "Paracetamol"),
            "B": self.create_reminder("B", "Morphine"),
        }

    # -- helpers ------------------------------------------------------------

    def create_reminder(self, household: str, medicine: str, patient: str = None) -> str:
        response = self.client.post(f"{BASE}/medication/reminder", headers=self.headers[household], json={
            "patient_id": patient or self.patient[household], "medication_name": medicine,
            "dosage": "500mg", "scheduled_time": time.time() + 3600, "language": "en",
            "phone_number": "+910000000000",
        })
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["reminder_id"]

    def push(self, household: str, **items) -> dict:
        response = self.client.post(f"{BASE}/sync/push", headers=self.headers[household],
                                    json={"device_timestamp": time.time(), **items})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def pull(self, household: str, since: float = 0.0) -> dict:
        response = self.client.post(f"{BASE}/sync/pull?last_sync_at={since}", headers=self.headers[household])
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def listed(self, household: str, query: str = "") -> set:
        response = self.client.get(f"{BASE}/medication/reminders{query}", headers=self.headers[household])
        self.assertEqual(response.status_code, 200, response.text)
        return {r["medication_name"] for r in response.json()["reminders"]}

    def export(self, headers: dict) -> dict:
        response = self.client.get(f"{BASE}/evaluation/export", headers=headers)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def submit_study_records(self, household: str) -> None:
        sus = self.client.post(f"{BASE}/evaluation/sus", headers=self.headers[household], json={
            "scores": [4, 2, 4, 2, 4, 2, 4, 2, 4, 2], "site_id": "site-1",
            "language": "en", "completed_at": time.time()})
        self.assertEqual(sus.status_code, 200, sus.text)
        vignette = self.client.post(f"{BASE}/evaluation/vignette", headers=self.headers[household], json={
            "vignette_id": f"v-{household}", "with_tool": True, "response_text": f"answer {household}",
            "started_at": time.time() - 60, "completed_at": time.time()})
        self.assertEqual(vignette.status_code, 200, vignette.text)
        logs = self.client.post(f"{BASE}/evaluation/logs", headers=self.headers[household], json={
            "device_id": f"device-{household}", "batch_timestamp": time.time(),
            "logs": [{"log_id": f"log-{household}", "event_type": "query", "timestamp": time.time()}]})
        self.assertEqual(logs.status_code, 200, logs.text)

    # -- reminder list ------------------------------------------------------

    def test_reminder_list_shows_only_the_callers_household(self) -> None:
        self.assertEqual(self.listed("A"), {"Paracetamol"})
        self.assertEqual(self.listed("B"), {"Morphine"})

    def test_reminder_list_ignores_another_households_user_id(self) -> None:
        self.assertEqual(self.listed("A", f"?user_id={self.user['B']}"), {"Paracetamol"})

    def test_reminders_sent_to_the_app_carry_no_dose(self) -> None:
        listed = self.client.get(f"{BASE}/medication/reminders", headers=self.headers["A"]).json()
        pulled = self.pull("A")["medication_reminders"]
        self.assertNotIn("500mg", json.dumps(listed))
        self.assertNotIn("500mg", json.dumps(pulled))

    # -- reminder creation --------------------------------------------------

    def test_reminder_for_another_households_patient_is_not_created(self) -> None:
        response = self.client.post(f"{BASE}/medication/reminder", headers=self.headers["B"], json={
            "patient_id": self.patient["A"], "medication_name": "Haloperidol",
            "dosage": "", "scheduled_time": time.time() + 60, "language": "en",
            "phone_number": "+910000000000"})
        self.assertEqual(response.status_code, 404, response.text)
        self.assertNotIn("Haloperidol", self.listed("A") | self.listed("B"))

    # -- sync pull ----------------------------------------------------------

    def test_sync_pull_returns_the_callers_reminders_and_none_of_another_households(self) -> None:
        pulled = self.pull("A")["medication_reminders"]
        self.assertEqual({r["reminder_id"] for r in pulled}, {self.reminder_id["A"]})

    def test_sync_pull_returns_only_reminders_changed_since_the_last_sync(self) -> None:
        last_sync = self.pull("A")["server_timestamp"]
        self.assertEqual(self.pull("A", since=last_sync)["medication_reminders"], [])

        self.push("A", medication_reminders=[{"reminder_id": self.reminder_id["A"], "patient_confirmed": True}])
        changed = self.pull("A", since=last_sync)["medication_reminders"]
        self.assertEqual([r["reminder_id"] for r in changed], [self.reminder_id["A"]])
        self.assertTrue(changed[0]["patient_confirmed"])

    # -- sync push ----------------------------------------------------------

    def test_push_cannot_change_another_households_reminder(self) -> None:
        result = self.push("B", medication_reminders=[
            {"reminder_id": self.reminder_id["A"], "call_status": "confirmed", "patient_confirmed": True}])
        self.assertEqual((result["accepted"], result["rejected"]), (0, 1))
        theirs = self.reminders.reminders[self.reminder_id["A"]]
        self.assertFalse(theirs.patient_confirmed)
        self.assertEqual(theirs.call_status.value, "scheduled")

    def test_pushed_reminder_belongs_to_the_pusher_whatever_its_payload_says(self) -> None:
        result = self.push("B", medication_reminders=[{
            "reminder_id": "planted-1", "user_id": self.user["A"], "medication_name": "Planted",
            "scheduled_time": time.time() + 60}])
        self.assertEqual(result["accepted"], 1, result)
        self.assertNotIn("Planted", self.listed("A"))
        self.assertIn("Planted", self.listed("B"))

    def test_a_saved_reminder_is_acknowledged_as_accepted(self) -> None:
        result = self.push("A", medication_reminders=[{"reminder_id": self.reminder_id["A"], "patient_confirmed": True}])
        self.assertEqual((result["accepted"], result["rejected"]), (1, 0))
        self.assertEqual(result["acknowledgements"], [
            {"kind": "medication_reminder", "index": 0, "id": self.reminder_id["A"], "status": "accepted", "reason": None}])
        self.assertTrue(self.reminders.reminders[self.reminder_id["A"]].patient_confirmed)

    def test_push_acknowledges_each_item_as_accepted_or_rejected(self) -> None:
        result = self.push(
            "A",
            observations=[
                {"observation_id": "obs-mine", "patient_id": self.patient["A"], "category": "symptom",
                 "entity_name": "nausea", "severity": 1, "timestamp": time.time()},
                {"observation_id": "obs-theirs", "patient_id": self.patient["B"], "category": "symptom",
                 "entity_name": "nausea", "severity": 1, "timestamp": time.time()},
            ],
            medication_reminders=[
                {"reminder_id": self.reminder_id["A"], "patient_confirmed": True},
                {"reminder_id": self.reminder_id["B"], "patient_confirmed": True},
                {"medication_name": "No id"},
            ],
        )
        acks = {(a["kind"], a["index"]): (a["id"], a["status"], a["reason"]) for a in result["acknowledgements"]}
        self.assertEqual(acks, {
            ("observation", 0): ("obs-mine", "accepted", None),
            ("observation", 1): ("obs-theirs", "rejected", "not_owned"),
            ("medication_reminder", 0): (self.reminder_id["A"], "accepted", None),
            ("medication_reminder", 1): (self.reminder_id["B"], "rejected", "not_owned"),
            ("medication_reminder", 2): (None, "rejected", "invalid"),
        })
        self.assertEqual((result["accepted"], result["rejected"]), (2, 3))

    def test_push_cannot_overwrite_another_households_interaction_log_or_vignette(self) -> None:
        self.push("A", interaction_logs=[{"log_id": "log-shared", "event_type": "mine", "timestamp": time.time()}],
                  vignette_responses=[{"response_id": "resp-shared", "vignette_id": "v1", "with_tool": True,
                                       "started_at": 1.0, "completed_at": 2.0}])
        result = self.push("B", interaction_logs=[{"log_id": "log-shared", "event_type": "forged", "timestamp": time.time()}],
                           vignette_responses=[{"response_id": "resp-shared", "vignette_id": "v1", "with_tool": False,
                                                "started_at": 1.0, "completed_at": 2.0}])
        self.assertEqual((result["accepted"], result["rejected"]), (0, 2))
        stored = list(Path("data/evaluation/interaction_logs").rglob("log-shared.json"))
        self.assertEqual([json.loads(p.read_text())["user_id"] for p in stored], [self.user["A"]])
        vignette = json.loads(Path("data/evaluation/vignette_responses/resp-shared.json").read_text())
        self.assertEqual(vignette["user_id"], self.user["A"])

    def test_a_pushed_id_cannot_name_a_file_outside_the_store(self) -> None:
        result = self.push("A", interaction_logs=[{"log_id": "../../escaped", "event_type": "x", "timestamp": time.time()}],
                           vignette_responses=[{"response_id": "../escaped", "vignette_id": "v1"}])
        self.assertEqual((result["accepted"], result["rejected"]), (0, 2))
        self.assertEqual(list(Path("data").rglob("escaped.json")), [])

    def test_interaction_logs_and_vignettes_that_fail_to_save_are_logged_and_reported(self) -> None:
        # Each store's directory is a file: nothing can be written beneath it.
        for name in ("interaction_logs", "vignette_responses"):
            store = Path("data/evaluation", name)
            store.rmdir()
            store.write_text("not a directory")
        with self.assertLogs("sync.batch_processor", level="WARNING") as logs:
            result = self.push("A",
                               interaction_logs=[{"log_id": "log-lost", "event_type": "q", "timestamp": time.time()}],
                               vignette_responses=[{"response_id": "resp-lost", "vignette_id": "v1"}])
        statuses = {(a["kind"], a["id"]): (a["status"], a["reason"]) for a in result["acknowledgements"]}
        self.assertEqual(statuses, {("interaction_log", "log-lost"): ("rejected", "save_failed"),
                                    ("vignette_response", "resp-lost"): ("rejected", "save_failed")})
        joined = "\n".join(logs.output)
        self.assertIn("log-lost", joined)
        self.assertIn("resp-lost", joined)

    # -- evaluation export --------------------------------------------------

    def test_export_returns_only_the_callers_own_records(self) -> None:
        self.submit_study_records("A")
        self.submit_study_records("B")
        exported = json.dumps(self.export(self.headers["A"]))
        self.assertIn(self.user["A"], exported)
        self.assertNotIn(self.user["B"], exported)
        self.assertNotIn("answer B", exported)
        self.assertNotIn("log-B", exported)

    def test_an_auditor_exports_every_households_records(self) -> None:
        self.submit_study_records("A")
        self.submit_study_records("B")
        with mock.patch.dict(os.environ, {ADMIN_SECRET_ENV: ADMIN_SECRET}):
            auditor = {"Authorization": f"Bearer {issue_admin_token('auditor', secret=ADMIN_SECRET)}"}
            exported = json.dumps(self.export(auditor))
        self.assertIn(self.user["A"], exported)
        self.assertIn(self.user["B"], exported)

    def test_export_without_a_token_is_refused(self) -> None:
        self.assertEqual(self.client.get(f"{BASE}/evaluation/export").status_code, 401)

    def test_export_refuses_a_refresh_token(self) -> None:
        refresh = get_jwt_handler().create_token_pair(self.user["A"], "caregiver", "site-1").refresh_token
        response = self.client.get(f"{BASE}/evaluation/export", headers={"Authorization": f"Bearer {refresh}"})
        self.assertEqual(response.status_code, 401)


class LoginUserIdIsNotAPath(unittest.TestCase):
    """/auth/login reads the device record only for a well-formed user id."""

    def setUp(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="pin-traversal-"))
        devices = root / "devices"
        devices.mkdir()
        patcher = mock.patch.object(pin_auth_module, "DATA_DIR", devices)
        patcher.start()
        self.addCleanup(patcher.stop)
        init_auth(jwt_secret="test-secret-for-pin-traversal")
        app = FastAPI()
        app.include_router(mobile_router)
        self.client = TestClient(app)
        # A record planted outside the device store, with a PIN the attacker knows.
        authenticator = pin_auth_module.PinAuthenticator()
        salt = "planted-salt"
        self.planted = root / "planted.json"
        self.planted.write_text(json.dumps({
            "user_id": "planted", "role": "asha", "site_id": "site-1",
            "pin_hash": authenticator._hash_pin("1234", salt), "pin_salt": salt, "failed_attempts": 0,
        }))

    def test_a_path_in_the_user_id_is_refused(self) -> None:
        response = self.client.post(f"{BASE}/auth/login", json={"user_id": "../planted", "pin": "1234"})
        self.assertEqual(response.status_code, 401, response.text)

    def test_a_refused_path_is_not_written(self) -> None:
        before = self.planted.read_text()
        self.client.post(f"{BASE}/auth/login", json={"user_id": "../planted", "pin": "0000"})
        self.assertEqual(self.planted.read_text(), before)


if __name__ == "__main__":
    unittest.main()
