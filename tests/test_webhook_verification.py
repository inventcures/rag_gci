"""
Webhooks are accepted only when verified, and fail closed.

Before this, the Bolna, Retell, Sarvam and Twilio webhooks took any request.
Bolna's signature check passed whenever BOLNA_WEBHOOK_SECRET was unset and was
skipped when no signature was sent; its Bearer key was never checked. Anyone
could post a forged call event or WhatsApp message, and /api/set_language let
anyone set any phone number's language.

What these pin down, over HTTP:

  * for each provider: a bad signature, a missing signature and an unset
    secret are each rejected with 401 or 403, and a valid one is accepted
  * Bolna needs both its signature and its Bearer key
  * the real WhatsApp app: an unsigned message is never accepted, and an
    unverified /api/set_language changes nothing
  * every webhook route in create_app() declares its verifier
"""

import ast
import base64
import hashlib
import hmac
import inspect
import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import Depends, FastAPI, Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import webhook_auth  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BODY = json.dumps({"event": "call_ended", "call_id": "c1"}).encode()

BOLNA_SECRET = "bolna-signing-secret"
BOLNA_KEY = "rag-api-key"
RETELL_KEY = "key_retell_test"
SARVAM_SECRET = "sarvam-signing-secret"
TWILIO_TOKEN = "twilio-auth-token"
PUBLIC_BASE = "https://palli.example.org"
REJECTED = (401, 403)


def hex_hmac_sha256(secret: str, payload: bytes) -> str:
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def retell_signature(key: str, body: bytes, at_ms: int = None) -> str:
    stamp = str(at_ms if at_ms is not None else int(time.time() * 1000))
    return f"v={stamp},d={hex_hmac_sha256(key, body + stamp.encode())}"


def twilio_signature(token: str, url: str, params: dict) -> str:
    """Twilio's documented scheme: HMAC-SHA1 over the URL plus sorted key+value pairs."""
    payload = url + "".join(k + params[k] for k in sorted(params))
    return base64.b64encode(hmac.new(token.encode(), payload.encode(), hashlib.sha1).digest()).decode()


def json_app(verifier) -> TestClient:
    """A one-route app guarded the way the server's webhook routes are."""
    app = FastAPI()

    @app.post("/hook", dependencies=[Depends(verifier)])
    async def hook(request: Request):
        return {"got": (await request.json())["event"]}

    return TestClient(app)


ALL_SECRETS = {
    "BOLNA_WEBHOOK_SECRET": BOLNA_SECRET,
    "RAG_API_KEY": BOLNA_KEY,
    "RETELL_API_KEY": RETELL_KEY,
    "SARVAM_WEBHOOK_SECRET": SARVAM_SECRET,
    "TWILIO_AUTH_TOKEN": TWILIO_TOKEN,
    "PUBLIC_BASE_URL": PUBLIC_BASE,
}


class Secrets(unittest.TestCase):
    """Every test starts with every secret set; a test unsets the one it needs to."""

    def setUp(self):
        env = mock.patch.dict(os.environ, ALL_SECRETS)
        env.start()
        self.addCleanup(env.stop)

    def unset(self, name: str):
        os.environ.pop(name, None)


# ---------------------------------------------------------------------------
# Bolna: signature and Bearer key
# ---------------------------------------------------------------------------


class BolnaWebhookIsVerified(Secrets):
    def setUp(self):
        super().setUp()
        self.client = json_app(webhook_auth.require_bolna)

    def post(self, signature=None, bearer=BOLNA_KEY):
        headers = {"Content-Type": "application/json"}
        if signature is not None:
            headers["X-Bolna-Signature"] = signature
        if bearer is not None:
            headers["Authorization"] = f"Bearer {bearer}"
        return self.client.post("/hook", content=BODY, headers=headers)

    def test_a_valid_signature_and_key_are_accepted(self):
        response = self.post(hex_hmac_sha256(BOLNA_SECRET, BODY))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"got": "call_ended"})

    def test_a_bad_signature_is_rejected(self):
        self.assertIn(self.post(hex_hmac_sha256("wrong", BODY)).status_code, REJECTED)

    def test_a_missing_signature_is_rejected(self):
        self.assertIn(self.post(None).status_code, REJECTED)

    def test_an_unset_secret_is_rejected(self):
        self.unset("BOLNA_WEBHOOK_SECRET")
        self.assertIn(self.post(hex_hmac_sha256("", BODY)).status_code, REJECTED)

    def test_a_valid_signature_without_the_bearer_key_is_rejected(self):
        self.assertIn(self.post(hex_hmac_sha256(BOLNA_SECRET, BODY), bearer=None).status_code, REJECTED)

    def test_a_wrong_bearer_key_is_rejected(self):
        self.assertIn(self.post(hex_hmac_sha256(BOLNA_SECRET, BODY), bearer="guess").status_code, REJECTED)

    def test_an_unset_bearer_key_is_rejected(self):
        self.unset("RAG_API_KEY")
        self.assertIn(self.post(hex_hmac_sha256(BOLNA_SECRET, BODY), bearer="").status_code, REJECTED)


# ---------------------------------------------------------------------------
# Retell: v=<ms>,d=<hmac> keyed by the API key
# ---------------------------------------------------------------------------


class RetellWebhookIsVerified(Secrets):
    def setUp(self):
        super().setUp()
        self.client = json_app(webhook_auth.require_retell)

    def post(self, signature=None):
        headers = {"Content-Type": "application/json"}
        if signature is not None:
            headers["X-Retell-Signature"] = signature
        return self.client.post("/hook", content=BODY, headers=headers)

    def test_a_valid_signature_is_accepted(self):
        response = self.post(retell_signature(RETELL_KEY, BODY))
        self.assertEqual(response.status_code, 200, response.text)

    def test_a_bad_signature_is_rejected(self):
        self.assertIn(self.post(retell_signature("wrong", BODY)).status_code, REJECTED)

    def test_a_missing_signature_is_rejected(self):
        self.assertIn(self.post(None).status_code, REJECTED)

    def test_an_unset_secret_is_rejected(self):
        self.unset("RETELL_API_KEY")
        self.assertIn(self.post(retell_signature("", BODY)).status_code, REJECTED)

    def test_a_replayed_old_signature_is_rejected(self):
        an_hour_ago = int(time.time() * 1000) - 3_600_000
        self.assertIn(self.post(retell_signature(RETELL_KEY, BODY, an_hour_ago)).status_code, REJECTED)


# ---------------------------------------------------------------------------
# Sarvam: HMAC-SHA256 of the body
# ---------------------------------------------------------------------------


class SarvamWebhookIsVerified(Secrets):
    def setUp(self):
        super().setUp()
        self.client = json_app(webhook_auth.require_sarvam)

    def post(self, signature=None):
        headers = {"Content-Type": "application/json"}
        if signature is not None:
            headers["X-Sarvam-Signature"] = signature
        return self.client.post("/hook", content=BODY, headers=headers)

    def test_a_valid_signature_is_accepted(self):
        self.assertEqual(self.post(hex_hmac_sha256(SARVAM_SECRET, BODY)).status_code, 200)

    def test_a_bad_signature_is_rejected(self):
        self.assertIn(self.post(hex_hmac_sha256("wrong", BODY)).status_code, REJECTED)

    def test_a_missing_signature_is_rejected(self):
        self.assertIn(self.post(None).status_code, REJECTED)

    def test_an_unset_secret_is_rejected(self):
        self.unset("SARVAM_WEBHOOK_SECRET")
        self.assertIn(self.post(hex_hmac_sha256("", BODY)).status_code, REJECTED)


# ---------------------------------------------------------------------------
# Twilio: X-Twilio-Signature, through the real WhatsApp app
# ---------------------------------------------------------------------------


MESSAGE = {"From": "whatsapp:+919800000001", "To": "whatsapp:+1", "Body": "breathless", "MessageSid": "SM1"}


class TwilioWebhookIsVerified(Secrets):
    def setUp(self):
        super().setUp()
        from tests.test_whatsapp_fast import bot_with_dedupe
        from tests.test_text_path_dose_guard import RealisticRag

        self.bot = bot_with_dedupe(RealisticRag())
        self.accepted = []
        accept = mock.patch.object(self.bot, "accept_twilio_message",
                                   side_effect=lambda *a: self.accepted.append(a) or True)
        accept.start()
        self.addCleanup(accept.stop)
        self.client = TestClient(self.bot.create_webhook_app())

    def post(self, signature=None, params=MESSAGE):
        headers = {} if signature is None else {"X-Twilio-Signature": signature}
        return self.client.post("/webhook", data=params, headers=headers)

    def test_a_valid_signature_is_accepted(self):
        response = self.post(twilio_signature(TWILIO_TOKEN, f"{PUBLIC_BASE}/webhook", MESSAGE))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(self.accepted), 1)

    def test_a_bad_signature_is_rejected(self):
        response = self.post(twilio_signature("wrong", f"{PUBLIC_BASE}/webhook", MESSAGE))
        self.assertIn(response.status_code, REJECTED)
        self.assertEqual(self.accepted, [])

    def test_a_signature_for_other_params_is_rejected(self):
        forged = dict(MESSAGE, Body="something else")
        response = self.post(twilio_signature(TWILIO_TOKEN, f"{PUBLIC_BASE}/webhook", MESSAGE), forged)
        self.assertIn(response.status_code, REJECTED)
        self.assertEqual(self.accepted, [])

    def test_a_missing_signature_is_rejected(self):
        self.assertIn(self.post(None).status_code, REJECTED)
        self.assertEqual(self.accepted, [])

    def test_an_unset_secret_is_rejected(self):
        self.unset("TWILIO_AUTH_TOKEN")
        response = self.post(twilio_signature("", f"{PUBLIC_BASE}/webhook", MESSAGE))
        self.assertIn(response.status_code, REJECTED)
        self.assertEqual(self.accepted, [])


class SetLanguageIsVerified(Secrets):
    PATH = "/api/set_language?phone_number=%2B919800000001&language=hi"

    def setUp(self):
        super().setUp()
        from tests.test_whatsapp_fast import bot_with_dedupe
        from tests.test_text_path_dose_guard import RealisticRag

        self.bot = bot_with_dedupe(RealisticRag())
        self.bot.stt_service.supported_languages = {"hi": "Hindi", "en": "English"}
        self.client = TestClient(self.bot.create_webhook_app())

    def test_without_verification_nothing_changes(self):
        response = self.client.post(self.PATH)
        self.assertIn(response.status_code, REJECTED)
        self.assertEqual(self.bot.user_preferences, {})

    def test_with_a_bad_signature_nothing_changes(self):
        response = self.client.post(self.PATH, headers={
            "X-Twilio-Signature": twilio_signature("wrong", PUBLIC_BASE + self.PATH, {})})
        self.assertIn(response.status_code, REJECTED)
        self.assertEqual(self.bot.user_preferences, {})

    def test_with_an_unset_secret_nothing_changes(self):
        self.unset("TWILIO_AUTH_TOKEN")
        response = self.client.post(self.PATH, headers={
            "X-Twilio-Signature": twilio_signature("", PUBLIC_BASE + self.PATH, {})})
        self.assertIn(response.status_code, REJECTED)
        self.assertEqual(self.bot.user_preferences, {})

    def test_a_verified_request_sets_the_language(self):
        response = self.client.post(self.PATH, headers={
            "X-Twilio-Signature": twilio_signature(TWILIO_TOKEN, PUBLIC_BASE + self.PATH, {})})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.bot.user_preferences, {"+919800000001": {"language": "hi"}})


# ---------------------------------------------------------------------------
# The server's own routes declare their verifier
# ---------------------------------------------------------------------------


class ServerRoutesDeclareTheirVerifier(unittest.TestCase):
    """create_app() builds the whole pipeline, so read its routes from the source."""

    EXPECTED = {
        "/api/bolna/query": "require_bolna",
        "/api/bolna/webhook": "require_bolna",
        "/api/retell/webhook": "require_retell",
        "/api/sarvam/webhook": "require_sarvam",
        "/webhook": "require_twilio",
    }

    def verifiers_by_route(self):
        import simple_rag_server

        tree = ast.parse(inspect.getsource(simple_rag_server.create_app))
        found = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            for deco in node.decorator_list:
                if (isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute)
                        and deco.func.attr == "post" and deco.args
                        and isinstance(deco.args[0], ast.Constant)
                        and deco.args[0].value in self.EXPECTED):
                    found[deco.args[0].value] = ast.unparse(deco)
        return found

    def test_every_webhook_route_depends_on_its_verifier(self):
        found = self.verifiers_by_route()
        self.assertEqual(set(found), set(self.EXPECTED))
        for path, verifier in self.EXPECTED.items():
            self.assertIn(f"Depends({verifier})", found[path], path)

    def test_the_fail_open_bolna_check_is_gone(self):
        source = (ROOT / "simple_rag_server.py").read_text(encoding="utf-8")
        self.assertNotIn("Skip verification if no secret configured", source)


if __name__ == "__main__":
    unittest.main()
