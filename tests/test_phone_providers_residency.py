"""
Phone providers outside India are off unless the release lists them
(review round 3, ticket 17; decision D11).

Before this:

  * Bolna sent call audio to Deepgram, the answer to OpenAI gpt-4o-mini and
    the reply to Cartesia, over Twilio; Retell sent the call to its own
    platform, Cartesia and a gpt-4o-mini post-call analysis. None of them was
    in the release record, the fingerprint did not list them, and every Bolna
    and Retell route served whatever the record said;
  * Bolna's OpenAI LLM reworded the guarded answer that /api/bolna/query
    returned, and nothing checked the reworded text for a dose before it was
    spoken.

These drive the seams a caller and a researcher meet: the server's Bolna and
Retell routes (signed, through the app create_app() builds), the release
fingerprint and its drift, the agent configurations sent to the providers, and
the medication reminder call.
"""

import asyncio
import hashlib
import hmac
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fastapi.testclient import TestClient  # noqa: E402

import study_release  # noqa: E402
import test_route_policy as harness  # noqa: E402
from dose_detector import contains_dose  # noqa: E402

RAG_KEY = "rag-api-key-ticket-17"
BOLNA_SECRET = "bolna-signing-secret-ticket-17"
RETELL_KEY = "key_retell_ticket_17"
SECRETS = {"RAG_API_KEY": RAG_KEY, "BOLNA_WEBHOOK_SECRET": BOLNA_SECRET, "RETELL_API_KEY": RETELL_KEY}

DOSE_ANSWER = (
    "Morphine is an opioid used for moderate to severe pain in palliative care. "
    "It often causes constipation, so a laxative is usually started at the same time. "
    "Give morphine 10 mg every 4 hours. "
    "Tell the doctor at once if the patient becomes very drowsy or confused."
)
LLM_ROUTE = "/api/bolna/llm/en/v1/chat/completions"


def bolna_services() -> List[str]:
    from phone_residency import services

    return services("bolna")


def retell_services() -> List[str]:
    from phone_residency import services

    return services("retell")


def chat_request(text: str, stream: bool = False, system: str = "") -> Dict[str, Any]:
    """The body Bolna's OpenAI SDK sends to a custom LLM."""
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": text})
    return {"model": "palli-sahayak", "messages": messages, "stream": stream}


def spoken(response) -> str:
    """The text a Bolna custom-LLM reply makes the agent speak (streamed or not)."""
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        parts = []
        for line in response.text.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[len("data: "):])["choices"][0].get("delta", {})
            parts.append(delta.get("content") or "")
        return "".join(parts)
    return response.json()["choices"][0]["message"]["content"]


def signed_bolna(body: bytes) -> Dict[str, str]:
    signature = hmac.new(BOLNA_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return {"X-Bolna-Signature": signature, "Authorization": f"Bearer {RAG_KEY}",
            "Content-Type": "application/json"}


def signed_retell(body: bytes) -> Dict[str, str]:
    at = int(time.time() * 1000)
    digest = hmac.new(RETELL_KEY.encode(), body + str(at).encode(), hashlib.sha256).hexdigest()
    return {"X-Retell-Signature": f"v={at},d={digest}", "Content-Type": "application/json"}


class Release(unittest.TestCase):
    """A release record on disk, and the environment the phone providers read."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="phone-residency-"))
        self.addCleanup(study_release.reload_study_release)
        env = mock.patch.dict(os.environ, {**SECRETS, "BOLNA_ENABLED": "", "RETELL_ENABLED": "",
                                           "RETELL_POST_CALL_ANALYSIS": ""})
        env.start()
        self.addCleanup(env.stop)
        # The tracked config.yaml decides Bolna's default; keep it out of reach.
        import phone_residency

        config = self.dir / "config.yaml"
        config.write_text("bolna:\n  enabled: false\n", encoding="utf-8")
        patch = mock.patch.object(phone_residency, "CONFIG_PATH", config)
        patch.start()
        self.addCleanup(patch.stop)

    def enable(self, **flags: str) -> None:
        for name, value in flags.items():
            os.environ[name] = value

    def default_release(self) -> study_release.StudyRelease:
        """The tracked record, data/study/release.yaml."""
        return study_release.reload_study_release(study_release.DEFAULT_RELEASE_PATH)

    def release_listing(self, providers: List[str]) -> study_release.StudyRelease:
        record = study_release.load_yaml(study_release.DEFAULT_RELEASE_PATH)
        record.setdefault("providers", {})["outside_india"] = providers
        path = self.dir / "release.yaml"
        path.write_text(yaml.safe_dump(record), encoding="utf-8")
        return study_release.reload_study_release(path)


def setUpModule():
    harness.setUpModule()


def tearDownModule():
    harness.tearDownModule()


class Server(Release):
    """The real create_app(), built once, with the pipeline stubbed."""

    app = None

    @classmethod
    def setUpClass(cls):
        if Server.app is None:
            Server.app = harness.build_server_app()
        cls.client = TestClient(Server.app, raise_server_exceptions=False)

    def setUp(self):
        super().setUp()
        self.asked: List[Dict[str, Any]] = []
        asked = self.asked

        async def query(pipeline, question, conversation_id=None, user_id=None,
                        top_k=5, source_language="en"):
            asked.append({"question": question, "user_id": user_id, "language": source_language})
            return {"status": "success", "answer": DOSE_ANSWER,
                    "sources": [{"filename": "handbook.pdf"}]}

        patch = mock.patch.object(harness._FakePipeline, "query", query, create=True)
        patch.start()
        self.addCleanup(patch.stop)

    def ask_bolna(self, text: str = "How is pain managed at home?", stream: bool = False,
                  system: str = "", route: str = LLM_ROUTE):
        return self.client.post(route, json=chat_request(text, stream=stream, system=system),
                                headers={"Authorization": f"Bearer {RAG_KEY}"})

    def bolna_webhook(self):
        body = json.dumps({"event": "call_started", "call_id": "c-17",
                           "phone_number": "+910000000017"}).encode()
        return self.client.post("/api/bolna/webhook", content=body, headers=signed_bolna(body))

    def retell_webhook(self):
        body = json.dumps({"event": "call_started", "call": {"call_id": "r-17"}}).encode()
        return self.client.post("/api/retell/webhook", content=body, headers=signed_retell(body))

    def retell_socket_message(self) -> Dict[str, Any]:
        with self.client.websocket_connect("/ws/retell/llm/r-17") as socket:
            return socket.receive_json()


# ---------------------------------------------------------------------------
# With the default release, the Bolna and Retell routes refuse
# ---------------------------------------------------------------------------


class TheDefaultReleaseRefusesThePhoneRoutes(Server):
    def setUp(self):
        super().setUp()
        self.default_release()
        # Turned on by the operator: the record still decides.
        self.enable(BOLNA_ENABLED="true", RETELL_ENABLED="true")

    def assert_clear_refusal(self, response, provider: str) -> None:
        self.assertEqual(response.status_code, 503, response.text)
        body = response.json()
        self.assertEqual(body["status"], "error")
        self.assertIn(provider, body["error"])
        self.assertIn("outside India", body["error"])
        self.assertIn("providers.outside_india", body["error"])

    def test_a_bolna_turn_is_refused_with_a_clear_message(self):
        self.assert_clear_refusal(self.ask_bolna(), "Bolna")
        self.assertEqual(self.asked, [], "the question reached the pipeline")

    def test_the_bolna_webhook_is_refused_with_a_clear_message(self):
        self.assert_clear_refusal(self.bolna_webhook(), "Bolna")

    def test_the_retell_webhook_is_refused_with_a_clear_message(self):
        self.assert_clear_refusal(self.retell_webhook(), "Retell")

    def test_the_retell_socket_is_refused_with_a_clear_message(self):
        message = self.retell_socket_message()
        self.assertEqual(message["response_type"], "error")
        self.assertIn("outside India", message["content"])
        self.assertIn("providers.outside_india", message["content"])

    def test_the_refusal_names_every_service_the_record_leaves_out(self):
        error = self.ask_bolna().json()["error"]
        for service in bolna_services():
            self.assertIn(service, error)

    def test_listing_only_part_of_a_provider_still_refuses(self):
        services = bolna_services()
        self.release_listing(["gemini_live", *services[:-1]])
        response = self.ask_bolna()
        self.assertEqual(response.status_code, 503, response.text)
        self.assertIn(services[-1], response.json()["error"])

    def test_a_listed_provider_that_is_not_enabled_is_refused(self):
        self.release_listing(["gemini_live", *bolna_services()])
        self.enable(BOLNA_ENABLED="")
        response = self.ask_bolna()
        self.assertEqual(response.status_code, 503, response.text)
        self.assertIn("not enabled", response.json()["error"])

    def test_an_unsigned_request_is_still_rejected_before_the_release_check(self):
        response = self.client.post(LLM_ROUTE, json=chat_request("pain?"))
        self.assertIn(response.status_code, (401, 403), response.text)


class AMedicationReminderCallNeedsTheRelease(Release):
    def setUp(self):
        super().setUp()
        from test_reminders_never_state_a_dose import voice_system

        self.system = voice_system(tempfile.mkdtemp(prefix="voice-reminders-17-"))
        from datetime import datetime

        self.reminder = self.system.create_voice_reminder(
            user_id="asha-1", phone_number="+910000000017", medication_name="Paracetamol",
            dosage="", reminder_time=datetime(2026, 10, 16, 8, 0), language="en",
            preferred_provider="bolna")
        self.system.bolna_client = object()

    def test_with_the_default_release_no_bolna_reminder_call_is_placed(self):
        from medication_voice_reminders import CallStatus

        self.default_release()
        self.enable(BOLNA_ENABLED="true")
        call = mock.AsyncMock(name="_make_bolna_call", return_value={"success": True})
        with mock.patch.object(self.system, "_make_bolna_call", call):
            asyncio.run(self.system._make_call(self.reminder.reminder_id))
        call.assert_not_called()
        self.assertEqual(self.system.reminders[self.reminder.reminder_id].call_status,
                         CallStatus.FAILED)


# ---------------------------------------------------------------------------
# Enabling a provider without listing it shows as drift
# ---------------------------------------------------------------------------


class EnablingAPhoneProviderShowsAsDrift(Release):
    def setUp(self):
        super().setUp()
        manifest = self.dir / "document_metadata.json"
        manifest.write_text("{}", encoding="utf-8")
        live = study_release.StudyRelease(record={}, corpus_manifest=manifest).live_fingerprint()
        self.manifest = manifest
        self.record = {
            "release_name": "evah-pilot-17",
            "knowledge": {"snapshot_sha256": live["corpus.snapshot"]},
            "models": {"voice_primary": {"model": live["voice.live.model"]}},
            "providers": {"outside_india": [
                p for p in live["providers.outside_india"].split(",") if p != "none"]},
        }

    def release(self) -> study_release.StudyRelease:
        release = study_release.StudyRelease(record=self.record, corpus_manifest=self.manifest)
        return release

    def provider_drift(self):
        return {f.component: f for f in self.release().check_drift()}.get("providers.outside_india")

    def test_the_matching_record_has_no_provider_drift(self):
        self.assertIsNone(self.provider_drift())

    def test_enabling_bolna_without_listing_it_shows_as_drift(self):
        self.enable(BOLNA_ENABLED="true")
        drift = self.provider_drift()
        self.assertIsNotNone(drift, "Bolna turned on without the record saying so")
        for service in bolna_services():
            self.assertIn(service, drift.actual.split(","))

    def test_enabling_bolna_in_the_config_file_shows_as_drift_too(self):
        import phone_residency

        phone_residency.CONFIG_PATH.write_text("bolna:\n  enabled: true\n", encoding="utf-8")
        self.assertIsNotNone(self.provider_drift())

    def test_enabling_retell_without_listing_it_shows_as_drift(self):
        self.enable(RETELL_ENABLED="true")
        drift = self.provider_drift()
        self.assertIsNotNone(drift)
        self.assertIn("retell", drift.actual.split(","))
        self.assertNotIn("retell.post_call_analysis", drift.actual)

    def test_retells_post_call_analysis_is_a_provider_of_its_own(self):
        self.enable(RETELL_ENABLED="true", RETELL_POST_CALL_ANALYSIS="true")
        self.assertIn("retell.post_call_analysis", self.provider_drift().actual.split(","))

    def test_the_bolna_services_are_read_from_the_agent_bolna_is_given(self):
        from bolna_integration.config import bolna_outside_services, get_palli_sahayak_agent_config

        config = get_palli_sahayak_agent_config(server_url="https://helpline.example.org")
        tools = config["tasks"][0]["tools_config"]
        expected = {"bolna", f"bolna.{tools['transcriber']['provider']}",
                    f"bolna.{tools['synthesizer']['provider']}", f"bolna.{tools['input']['provider']}",
                    # the summary and extraction follow-up tasks: Bolna's hosted LLM
                    "bolna.post_call_analysis"}
        self.assertEqual(set(bolna_outside_services(config)), expected)
        self.assertEqual(set(bolna_services()), expected)

    def test_without_follow_up_tasks_bolna_runs_no_post_call_analysis(self):
        from bolna_integration.config import bolna_outside_services, get_palli_sahayak_agent_config

        config = get_palli_sahayak_agent_config(server_url="https://helpline.example.org")
        for task in config["follow_up_tasks"]:
            task["enabled"] = False
        self.assertNotIn("bolna.post_call_analysis", bolna_outside_services(config))


# ---------------------------------------------------------------------------
# A Bolna turn whose text contains a dose is not spoken as written
# ---------------------------------------------------------------------------


class BolnaSpeaksTheGuardedTextVerbatim(unittest.TestCase):
    """Bolna's hosted LLM rewords whatever a function returns, and Bolna gives
    no hook on its output. So the agent's LLM is this server: what Bolna speaks
    is the guarded answer, and no outside LLM is left to reword it."""

    def setUp(self):
        from bolna_integration.config import get_palli_sahayak_agent_config

        self.config = get_palli_sahayak_agent_config(server_url="https://helpline.example.org",
                                                     language="hi")
        self.agent = self.config["tasks"][0]["tools_config"]["llm_agent"]

    def test_the_agents_llm_is_this_server(self):
        llm = self.agent["llm_config"]
        self.assertEqual(llm["provider"], "custom")
        self.assertEqual(llm["base_url"], "https://helpline.example.org/api/bolna/llm/hi/v1")

    def test_no_function_hands_the_answer_to_another_llm(self):
        self.assertFalse(self.agent.get("functions"))
        self.assertNotIn("openai", repr(self.config).lower())

    def test_the_prompt_carries_the_callers_system_variables(self):
        prompt = self.config["agent_prompts"]["task_1"]["system_prompt"]
        self.assertIn("{from_number}", prompt)
        self.assertIn("{execution_id}", prompt)


class ABolnaTurnWithADoseIsNotSpokenAsWritten(Server):
    def setUp(self):
        super().setUp()
        self.release_listing(["gemini_live", *bolna_services()])
        self.enable(BOLNA_ENABLED="true")

    def assert_guarded(self, response) -> str:
        self.assertEqual(response.status_code, 200, response.text)
        text = spoken(response)
        self.assertTrue(text.strip(), "Bolna was given nothing to say")
        self.assertFalse(contains_dose(text), f"Bolna would speak a dose: {text!r}")
        self.assertNotEqual(text, DOSE_ANSWER)
        return text

    def test_a_streamed_turn_speaks_the_guarded_answer(self):
        response = self.ask_bolna(stream=True)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertTrue(response.text.rstrip().endswith("data: [DONE]"))
        text = self.assert_guarded(response)
        self.assertIn("laxative", text)

    def test_a_whole_turn_speaks_the_guarded_answer(self):
        self.assert_guarded(self.ask_bolna(stream=False))

    def test_the_question_is_the_callers_last_words_in_the_agents_language(self):
        route = "/api/bolna/llm/ta/v1/chat/completions"
        self.ask_bolna("வலி எப்படி?", route=route)
        self.assertEqual(self.asked[-1]["question"], "வலி எப்படி?")
        self.assertEqual(self.asked[-1]["language"], "ta")

    def test_the_caller_is_read_from_the_system_variables_in_the_prompt(self):
        from bolna_integration.config import bolna_system_prompt

        self.ask_bolna(system=bolna_system_prompt().replace("{from_number}", "+919812345678")
                       .replace("{execution_id}", "e-17"))
        self.assertEqual(self.asked[-1]["user_id"], "bolna_+919812345678")

    def test_an_unfilled_system_variable_is_not_an_identity(self):
        from bolna_integration.config import bolna_system_prompt

        self.ask_bolna(system=bolna_system_prompt())
        self.assertTrue(self.asked[-1]["user_id"].startswith("session:bolna:anon_"),
                        self.asked[-1]["user_id"])

    def test_the_old_function_route_is_gone(self):
        response = self.client.post("/api/bolna/query", json={"query": "pain?"},
                                    headers=signed_bolna(b"{}"))
        self.assertIn(response.status_code, (404, 405), response.text)


# ---------------------------------------------------------------------------
# Retell: the analysis is off unless asked for; the prompt names the table's languages
# ---------------------------------------------------------------------------


class RetellSendsNoAnalysisUnlessAskedFor(Release):
    def config(self) -> Dict[str, Any]:
        from retell_integration.config import get_palli_sahayak_retell_config

        return get_palli_sahayak_retell_config("wss://helpline.example.org/ws", language="hi").to_dict()

    def test_the_agent_has_no_post_call_analysis_by_default(self):
        config = self.config()
        self.assertNotIn("post_call_analysis_model", config)
        self.assertNotIn("post_call_analysis_data", config)
        self.assertNotIn("retell.post_call_analysis", retell_services())

    def test_asking_for_the_analysis_adds_it_to_the_agent_and_to_the_services(self):
        self.enable(RETELL_POST_CALL_ANALYSIS="true")
        self.assertIn("post_call_analysis_model", self.config())
        self.assertIn("retell.post_call_analysis", retell_services())

    def test_the_phone_prompts_name_the_languages_the_table_gives_them(self):
        from bolna_integration.config import bolna_system_prompt
        from retell_integration.config import RETELL_SYSTEM_PROMPT
        from supported_languages import PROVIDER_BOLNA, PROVIDER_RETELL, SUPPORTED_LANGUAGES

        for prompt, provider in ((RETELL_SYSTEM_PROMPT, PROVIDER_RETELL),
                                 (bolna_system_prompt(), PROVIDER_BOLNA)):
            with self.subTest(provider=provider):
                for lang in SUPPORTED_LANGUAGES:
                    if lang.voice(provider) is not None:
                        self.assertIn(lang.name, prompt)
                    else:
                        self.assertNotIn(lang.name, prompt)


if __name__ == "__main__":
    unittest.main()
