#!/usr/bin/env python3
"""
A medication reminder names the medicine and the time, never a dose.

Before this, every reminder path read the stored dose back to the participant:
the voice-call script ("This is a reminder to take your Paracetamol. 500mg."),
the WhatsApp scheduler message ("Dosage: 500mg"), the WhatsApp /remind
confirmation, /myreminders, the missed-call fallback and the /help examples
("/remind Morphine 08:00 10mg"). POST /api/medication/voice-reminder took any
dose and phone number, and the mobile reminder route scheduled an outbound call
to an empty phone number. Decision D10: the safer reading of ADR 0004 ships
until the clinical lead decides; the dose a person entered stays in the data.

What these pin down:

  * the reminder text from every path holds no dose, in every reminder
    language, even when the stored dosage is "500mg" or the medicine name
    carries one; the stored dosage field keeps what was entered
  * a reminder saved by the old code is re-worded when it is loaded
  * POST /api/medication/voice-reminder answers 401 without a token
  * a reminder with no phone number is rejected and never scheduled, on the
    server endpoint, the mobile route and in the reminder system itself
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from typing import List
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from dose_detector import detect_doses  # noqa: E402
from tests.test_text_path_dose_guard import FakeSender  # noqa: E402

REMINDER_LANGUAGES = ("en", "hi", "bn", "ta", "gu")
STORED_DOSE = "500mg"
PHONE = "+919800000001"
AT_EIGHT = datetime(2026, 10, 16, 8, 0)


def assert_dose_free(test: unittest.TestCase, text: str, where: str) -> None:
    """No dose shape, and none of the dose the participant entered."""
    test.assertTrue(text, f"{where}: no reminder text")
    test.assertEqual(detect_doses(text), [], f"{where} states a dose: {text!r}")
    test.assertNotIn("500", text, f"{where} repeats the stored dose: {text!r}")
    test.assertNotIn("10mg", text.replace(" ", "").lower(), f"{where}: {text!r}")


def voice_system(storage: str):
    """A real MedicationVoiceReminderSystem with no thread, provider or global job."""
    from medication_voice_reminders import MedicationVoiceReminderSystem

    patches = [
        mock.patch.object(MedicationVoiceReminderSystem, "_start_scheduler"),
        mock.patch.object(MedicationVoiceReminderSystem, "_init_providers"),
    ]
    for patcher in patches:
        patcher.start()
    try:
        system = MedicationVoiceReminderSystem(storage_path=storage)
    finally:
        for patcher in patches:
            patcher.stop()
    system._schedule_call = mock.MagicMock(name="_schedule_call")
    return system


def reminder_scheduler(storage: str):
    """A real MedicationReminderScheduler with no thread and no global job."""
    from safety_enhancements import MedicationReminderScheduler

    with mock.patch.object(MedicationReminderScheduler, "_start_scheduler"):
        scheduler = MedicationReminderScheduler(storage_path=storage)
    scheduler._schedule_reminder = mock.MagicMock(name="_schedule_reminder")
    return scheduler


class VoiceReminderCallsNeverStateADose(unittest.TestCase):
    def setUp(self):
        self.system = voice_system(tempfile.mkdtemp(prefix="voice-reminders-"))

    def create(self, **overrides):
        args = dict(user_id="asha-1", phone_number=PHONE, medication_name="Paracetamol",
                    dosage=STORED_DOSE, reminder_time=AT_EIGHT, language="en")
        args.update(overrides)
        return self.system.create_voice_reminder(**args)

    def test_the_call_script_names_the_medicine_and_the_time_in_every_reminder_language(self):
        for language in REMINDER_LANGUAGES:
            with self.subTest(language=language):
                reminder = self.create(language=language)
                assert_dose_free(self, reminder.tts_message, f"voice call ({language})")
                self.assertIn("Paracetamol", reminder.tts_message)
                self.assertIn("08:00", reminder.tts_message)

    def test_the_call_script_is_spoken_whole_in_every_reminder_language(self):
        # The last-resort dose strip must never fire on the templates themselves:
        # a template line the detector reads as a dose would be cut mid-sentence.
        for language in REMINDER_LANGUAGES:
            with self.subTest(language=language):
                with self.assertNoLogs("reminder_text", "ERROR"):
                    reminder = self.create(language=language)
                closing = self.system.VOICE_TEMPLATES[language]["closing"]
                self.assertIn(closing, reminder.tts_message)

    def test_the_stored_dose_stays_in_the_data(self):
        reminder = self.create()
        self.assertEqual(reminder.dosage, STORED_DOSE)
        self.assertEqual(reminder.to_dict()["dosage"], STORED_DOSE)

    def test_a_dose_written_into_the_medicine_name_is_not_spoken(self):
        reminder = self.create(medication_name="Paracetamol 500mg twice daily")
        assert_dose_free(self, reminder.tts_message, "voice call with a dosed name")
        self.assertIn("Paracetamol", reminder.tts_message)

    def test_a_medicine_name_that_is_only_a_dose_becomes_your_medicine(self):
        reminder = self.create(medication_name="2 tablets")
        assert_dose_free(self, reminder.tts_message, "voice call with a dose for a name")
        self.assertIn("your medicine", reminder.tts_message)

    def test_a_reminder_saved_with_a_spoken_dose_is_reworded_on_load(self):
        storage = tempfile.mkdtemp(prefix="voice-reminders-old-")
        old = self.create().to_dict()
        old["tts_message"] = "This is a reminder to take your Paracetamol. 500mg after food."
        with open(os.path.join(storage, "voice_reminders.json"), "w") as handle:
            json.dump([old], handle)

        loaded = voice_system(storage).reminders[old["reminder_id"]]

        assert_dose_free(self, loaded.tts_message, "reloaded voice reminder")
        self.assertEqual(loaded.dosage, STORED_DOSE)

    def test_a_reminder_with_no_phone_number_is_rejected_and_not_scheduled(self):
        for phone in ("", "   ", None):
            with self.subTest(phone=phone):
                with self.assertRaises(ValueError):
                    self.create(phone_number=phone)
        self.system._schedule_call.assert_not_called()
        self.assertEqual(self.system.reminders, {})

    def test_a_synced_reminder_with_no_phone_number_is_never_called(self):
        self.system.upsert_reminder_from_sync({"reminder_id": "r-sync", "user_id": "asha-1",
                                               "medication_name": "Paracetamol", "dosage": STORED_DOSE})
        bolna = mock.AsyncMock(name="_make_bolna_call")
        self.system.bolna_client = object()
        with mock.patch.object(self.system, "_make_bolna_call", bolna):
            asyncio.run(self.system._make_call("r-sync"))
        bolna.assert_not_called()


class WhatsAppRemindersNeverStateADose(unittest.TestCase):
    def setUp(self):
        from tests.test_text_path_dose_guard import make_bot

        self.scheduler = reminder_scheduler(tempfile.mkdtemp(prefix="wa-reminders-"))
        self.bot = make_bot(rag=None)
        self.bot.reminder_scheduler = self.scheduler
        self.bot.voice_reminder_system = voice_system(tempfile.mkdtemp(prefix="wa-voice-"))
        self.bot.safety_manager = None

    def sent(self) -> List[str]:
        return self.bot.twilio_api.texts

    def say(self, text: str) -> None:
        asyncio.run(self.bot._handle_twilio_text_message(PHONE, text))

    def test_the_scheduled_message_names_the_medicine_and_the_time_in_every_reminder_language(self):
        for language in REMINDER_LANGUAGES:
            with self.subTest(language=language):
                reminder = self.scheduler.create_reminder(
                    user_id="asha-1", medication_name="Paracetamol", dosage=STORED_DOSE,
                    frequency="daily", times=["08:00"], instructions="2 tablets after food",
                    language=language)
                message = self.scheduler.get_reminder_message(reminder)
                assert_dose_free(self, message, f"WhatsApp reminder ({language})")
                self.assertIn("Paracetamol", message)
                self.assertIn("08:00", message)
                self.assertEqual(reminder.dosage, STORED_DOSE)

    def test_setting_a_reminder_confirms_it_without_the_dose(self):
        self.say("/remind Paracetamol 08:00,20:00 500mg after food")

        self.assertTrue(self.sent(), "nothing was sent")
        for text in self.sent():
            assert_dose_free(self, text, "the /remind confirmation")
        self.assertIn("Paracetamol", self.sent()[-1])
        stored = self.scheduler.get_user_reminders(PHONE)
        self.assertEqual([r.dosage for r in stored], ["500mg after food"])
        voice = list(self.bot.voice_reminder_system.reminders.values())
        self.assertEqual(len(voice), 2)
        for reminder in voice:
            assert_dose_free(self, reminder.tts_message, "the voice call /remind sets up")

    def test_listing_reminders_does_not_read_the_dose_back(self):
        self.say("/remind Paracetamol 08:00 500mg after food")
        self.bot.twilio_api.texts.clear()

        self.say("/myreminders")

        self.assertTrue(self.sent())
        for text in self.sent():
            assert_dose_free(self, text, "/myreminders")
        self.assertIn("Paracetamol", self.sent()[-1])

    def test_the_missed_call_message_does_not_state_the_dose(self):
        reminder = self.bot.voice_reminder_system.create_voice_reminder(
            user_id=PHONE, phone_number=PHONE, medication_name="Paracetamol",
            dosage=STORED_DOSE, reminder_time=AT_EIGHT)

        asyncio.run(self.bot._on_medication_call_failed(reminder))

        assert_dose_free(self, self.sent()[-1], "the missed-call message")
        self.assertIn("Paracetamol", self.sent()[-1])

    def test_the_help_and_usage_text_carry_no_example_dose(self):
        self.say("/help")
        self.say("/remind")
        self.assertEqual(len(self.sent()), 2)
        for text in self.sent():
            assert_dose_free(self, text, "help text")

    def test_deleting_a_reminder_reaches_the_sender(self):
        reminder = self.scheduler.create_reminder(
            user_id=PHONE, medication_name="Paracetamol", dosage=STORED_DOSE,
            frequency="daily", times=["08:00"])

        self.say(f"/deletereminder {reminder.reminder_id}")

        self.assertEqual(self.scheduler.get_user_reminders(PHONE), [])
        self.assertIn("deleted", self.sent()[-1])


class MobileReminderRoute(unittest.TestCase):
    BASE = "/api/mobile/v1"

    def setUp(self):
        from auth.middleware import get_jwt_handler, init_auth
        from mobile_api.dependencies import init_dependencies
        from mobile_api.router import mobile_router

        self.system = voice_system(tempfile.mkdtemp(prefix="mobile-reminders-"))
        init_auth(jwt_secret="test-secret-for-mobile-reminder-checks")
        init_dependencies(None, None, None, self.system)
        self.addCleanup(init_dependencies, None, None, None, None)
        app = FastAPI()
        app.include_router(mobile_router)
        self.client = TestClient(app)
        token = get_jwt_handler().create_token_pair("asha-1", "asha", "site-1").access_token
        self.headers = {"Authorization": f"Bearer {token}"}

    def post(self, **fields):
        body = {"patient_id": "p-1", "medication_name": "Paracetamol", "dosage": STORED_DOSE,
                "scheduled_time": AT_EIGHT.timestamp(), "language": "hi", **fields}
        return self.client.post(f"{self.BASE}/medication/reminder", headers=self.headers, json=body)

    def test_a_reminder_with_no_phone_number_is_rejected_and_not_scheduled(self):
        response = self.post()

        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.system.reminders, {})
        self.system._schedule_call.assert_not_called()

    def test_a_reminder_from_the_app_is_spoken_without_its_dose(self):
        response = self.post(phone_number=PHONE)

        self.assertEqual(response.status_code, 200, response.text)
        reminder = self.system.reminders[response.json()["reminder_id"]]
        assert_dose_free(self, reminder.tts_message, "mobile reminder")
        self.assertIn("Paracetamol", reminder.tts_message)
        self.assertEqual(reminder.dosage, STORED_DOSE)

    def test_the_app_need_not_send_a_dose(self):
        response = self.post(phone_number=PHONE, dosage=None)
        self.assertEqual(response.status_code, 200, response.text)


class ServerVoiceReminderEndpoint(unittest.TestCase):
    PATH = "/api/medication/voice-reminder"

    @classmethod
    def setUpClass(cls):
        import medication_voice_reminders
        from tests import test_route_policy as harness

        harness.setUpModule()
        cls.addClassCleanup(harness.tearDownModule)
        cls.system = voice_system(tempfile.mkdtemp(prefix="server-reminders-"))
        singleton = mock.patch.object(medication_voice_reminders, "_voice_reminder_system", cls.system)
        singleton.start()
        cls.addClassCleanup(singleton.stop)
        cls.client = TestClient(harness.build_server_app(), raise_server_exceptions=False)

    def setUp(self):
        from auth.policy import issue_admin_token

        self.system.reminders.clear()
        self.system._schedule_call.reset_mock()
        self.admin = {"Authorization": f"Bearer {issue_admin_token('dr-ashish')}"}

    def post(self, headers=None, **fields):
        body = {"user_id": "asha-1", "phone_number": PHONE, "medication_name": "Paracetamol",
                "dosage": "500mg after food", "reminder_time": AT_EIGHT.isoformat(),
                "language": "en", **fields}
        return self.client.post(self.PATH, headers=headers or {}, json=body)

    def test_creating_a_voice_reminder_needs_a_token(self):
        response = self.post()

        self.assertEqual(response.status_code, 401, response.text)
        self.assertEqual(self.system.reminders, {})

    def test_a_reminder_with_no_phone_number_is_rejected_and_not_scheduled(self):
        for phone in ("", None):
            with self.subTest(phone=phone):
                response = self.post(headers=self.admin, phone_number=phone)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertEqual(response.json()["status"], "error")
        self.assertEqual(self.system.reminders, {})
        self.system._schedule_call.assert_not_called()

    def test_an_admin_reminder_is_spoken_without_its_dose(self):
        response = self.post(headers=self.admin)

        self.assertEqual(response.status_code, 200, response.text)
        reminder = self.system.reminders[response.json()["reminder_id"]]
        assert_dose_free(self, reminder.tts_message, "server voice reminder")
        self.assertEqual(reminder.dosage, "500mg after food")


if __name__ == "__main__":
    unittest.main()
