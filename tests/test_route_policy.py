#!/usr/bin/env python3
"""
One authentication policy for every route on the main server.

Before this, no route in simple_rag_server.py required authentication and CORS
allowed any origin: call transcripts with phone numbers, care-team records, FHIR
exports, medication adherence, the study interaction browser, document upload and
the voice-provider override were open to anyone. The override took its "actor"
from the request body, and a failed audit write kept the change.

What these pin down, over HTTP against the app create_app() builds:

  * every route has a declared policy (public, participant or admin); a new
    route without one fails the route-table test
  * every non-public route answers 401 without a token, and every admin route
    answers 403 to a participant token
  * the named identifiable-data routes answer 401 without a token
  * a mobile token claiming role "admin" is still not an administrator
  * the Gradio admin panel is gated too
  * CORS allows only configured origins, and none by default
  * the override records the admin token's subject as the actor, not a body
    field, and a failed audit write leaves the provider unchanged
"""

import base64
import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Dict, List, Optional
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.routing import APIRoute, APIWebSocketRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.responses import PlainTextResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402

MOBILE_SECRET = "route-policy-test-mobile-secret-0123456789"
ADMIN_SECRET = "route-policy-test-admin-secret-0123456789abcdef"
EVIL_ORIGIN = "https://evil.example"
STUDY_ORIGIN = "https://study.pallisahayak.example"


class _FakePipeline:
    """Stands in for SimpleRAGPipeline: no models, no Chroma."""

    def __init__(self, *args, **kwargs):
        self.longitudinal_manager = None
        self.vector_db = None

    def configure_answer_path(self, *args, **kwargs):
        return None

    def get_index_stats(self):
        return {"stats": {}}

    def __getattr__(self, name):
        return mock.MagicMock(name=f"pipeline.{name}")


_ABSENT = object()


def _fake_mount_gradio_app(app, blocks, path):
    """Mount a stand-in for the Gradio admin panel where the real one goes."""
    panel = Starlette(routes=[Route("/{rest:path}", lambda request: PlainTextResponse("gradio admin panel"))])
    app.mount(path, panel)
    return app


def build_server_app(extra_env: Optional[Dict[str, str]] = None):
    """The real create_app(), with the pipeline, Gradio and WhatsApp bot stubbed.

    Twilio and the mobile API are configured so their conditional routes exist and
    are covered by the route table too."""
    import simple_rag_server as srv

    env = {
        "MOBILE_JWT_SECRET": MOBILE_SECRET,
        "ADMIN_JWT_SECRET": ADMIN_SECRET,
        "TWILIO_ACCOUNT_SID": "ACtest-route-policy",
        "TWILIO_AUTH_TOKEN": "test-token",
        "CORS_ALLOWED_ORIGINS": "",
        **(extra_env or {}),
    }
    whatsapp = types.ModuleType("whatsapp_bot")
    whatsapp.EnhancedWhatsAppBot = mock.MagicMock(name="EnhancedWhatsAppBot")
    whatsapp.EnhancedSTTService = mock.MagicMock(name="EnhancedSTTService")
    whatsapp.EnhancedTTSService = mock.MagicMock(name="EnhancedTTSService")
    # Built-but-unused subsystems that write under data/ at construction.
    modules = {"whatsapp_bot": whatsapp, "memory_agents": None, "meta_agent": None,
               "context1_integration": None}
    saved = {name: sys.modules.get(name, _ABSENT) for name in modules}
    sys.modules.update(modules)
    try:
        with mock.patch.dict(os.environ, env), \
            mock.patch.object(srv, "SimpleRAGPipeline", _FakePipeline), \
            mock.patch.object(srv, "SimpleAdminUI", mock.MagicMock(name="SimpleAdminUI")), \
            mock.patch.object(srv.gr, "mount_gradio_app", _fake_mount_gradio_app):
            return srv.create_app(srv._parse_args(["--no-ngrok"]))
    finally:
        # Only these entries: patch.dict(sys.modules) would also drop the modules
        # the server imported, so later patches would land on fresh copies.
        for name, module in saved.items():
            if module is _ABSENT:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def concrete(path: str) -> str:
    """A request path for a route template: every {param} becomes 'x'."""
    return re.sub(r"\{[^}]+\}", "x", path)


def server_routes(app) -> List[tuple]:
    """(path, method) for every HTTP route, and (path, 'WS') for every socket."""
    from auth.policy import iter_route_paths

    out = []
    for path, route in iter_route_paths(app.routes):
        if isinstance(route, APIRoute):
            out.extend((path, method) for method in sorted(route.methods))
        elif isinstance(route, APIWebSocketRoute):
            out.append((path, "WS"))
    return out


_module_patches: List = []


def setUpModule():
    """The admin key, and no release probe: it opens the real Chroma store and
    writes drift files under data/."""
    import study_release

    _module_patches.extend([
        mock.patch.dict(os.environ, {"ADMIN_JWT_SECRET": ADMIN_SECRET}),
        mock.patch.object(study_release.StudyRelease, "verify_on_startup", return_value=[]),
        mock.patch.object(study_release.StudyRelease, "_live_embedding_model",
                          staticmethod(lambda: "unavailable")),
    ])
    for patcher in _module_patches:
        patcher.start()


def tearDownModule():
    while _module_patches:
        _module_patches.pop().stop()


class ServerHarness(unittest.TestCase):
    """Builds the server once; tokens are minted per test from the live handlers."""

    app = None

    @classmethod
    def setUpClass(cls):
        if ServerHarness.app is None:
            ServerHarness.app = build_server_app()
        cls.client = TestClient(ServerHarness.app, raise_server_exceptions=False)

    def setUp(self):
        from auth.middleware import get_jwt_handler, init_auth
        from auth.policy import issue_admin_token

        # Other tests re-point the mobile handler; put the server's back.
        init_auth(jwt_secret=MOBILE_SECRET)
        handler = get_jwt_handler()
        participant = handler.create_token_pair("asha-1", "asha", "site-1")
        self.participant = {"Authorization": f"Bearer {participant.access_token}"}
        self.participant_refresh = {"Authorization": f"Bearer {participant.refresh_token}"}
        claims_admin = handler.create_token_pair("mallory", "admin", "site-1").access_token
        self.self_registered_admin = {"Authorization": f"Bearer {claims_admin}"}
        self.admin_token = issue_admin_token("dr-ashish")
        self.admin = {"Authorization": f"Bearer {self.admin_token}"}

    def call(self, method: str, path: str, headers: Optional[Dict[str, str]] = None):
        return self.client.request(method, concrete(path), headers=headers or {})


class EveryRouteDeclaresAPolicy(ServerHarness):
    def test_the_route_table_covers_every_route_and_mount(self):
        from auth.policy import undeclared_routes

        self.assertEqual(undeclared_routes(self.app), [],
                         "declare these routes in auth/policy.py ROUTE_POLICIES")

    def test_a_route_without_a_declared_policy_is_reported(self):
        from auth.policy import ROUTE_POLICIES, undeclared_routes

        table = {k: v for k, v in ROUTE_POLICIES.items() if k != "/api/bolna/calls"}
        self.assertIn("/api/bolna/calls", undeclared_routes(self.app, table))

    def test_a_route_without_a_declared_policy_is_admin_only(self):
        from auth.policy import ROUTE_POLICIES

        trimmed = {k: v for k, v in ROUTE_POLICIES.items() if k != "/api/retell/vobiz"}
        with mock.patch.dict("auth.policy.ROUTE_POLICIES", trimmed, clear=True):
            self.assertEqual(self.call("GET", "/api/retell/vobiz").status_code, 401)
            self.assertEqual(self.call("GET", "/api/retell/vobiz", self.participant).status_code, 403)

    def test_the_route_table_has_the_conditional_routes(self):
        from auth.policy import iter_route_paths

        paths = {path for path, _ in iter_route_paths(self.app.routes)}
        for path in ("/webhook", "/api/mobile/v1/query", "/admin/study/api/interactions", "/admin"):
            self.assertIn(path, paths)


class EveryNonPublicRouteNeedsAToken(ServerHarness):
    def non_public(self, access):
        from auth.policy import Access, policy_for

        return [(path, method) for path, method in server_routes(self.app)
                if method != "WS" and policy_for(path).access in access]

    def test_every_non_public_route_answers_401_without_a_token(self):
        from auth.policy import Access

        routes = self.non_public({Access.PARTICIPANT, Access.ADMIN})
        self.assertGreater(len(routes), 60)
        for path, method in routes:
            with self.subTest(route=f"{method} {path}"):
                self.assertEqual(self.call(method, path).status_code, 401)

    def test_every_admin_route_answers_403_to_a_participant_token(self):
        from auth.policy import Access

        for path, method in self.non_public({Access.ADMIN}):
            with self.subTest(route=f"{method} {path}"):
                self.assertEqual(self.call(method, path, self.participant).status_code, 403)

    def test_a_mobile_token_claiming_the_admin_role_is_not_an_administrator(self):
        # Anyone can register on the app with role "admin".
        for path in ("/api/bolna/calls", "/admin/study/api/interactions", "/api/upload/x"):
            with self.subTest(path=path):
                self.assertEqual(self.call("GET", path, self.self_registered_admin).status_code, 403)

    def test_a_refresh_token_is_not_accepted_as_a_session(self):
        self.assertEqual(self.call("POST", "/api/query", self.participant_refresh).status_code, 401)
        self.assertEqual(self.call("GET", "/api/mobile/v1/sync/status", self.participant_refresh).status_code, 401)

    def test_an_admin_token_signed_with_another_key_is_refused(self):
        from auth.policy import issue_admin_token

        forged = issue_admin_token("mallory", secret="some-other-key-that-is-long-enough-123456")
        self.assertEqual(self.call("GET", "/api/bolna/calls", {"Authorization": f"Bearer {forged}"}).status_code, 401)

    def test_admin_sign_in_is_off_when_the_admin_key_is_unset(self):
        with mock.patch.dict(os.environ, {"ADMIN_JWT_SECRET": ""}):
            self.assertEqual(self.call("GET", "/api/bolna/calls", self.admin).status_code, 401)


class IdentifiableDataNeedsAToken(ServerHarness):
    ROUTES = [
        ("GET", "/api/bolna/calls"),
        ("GET", "/api/retell/calls"),
        ("GET", "/api/careteam/{patient_id}"),
        ("POST", "/api/careteam/{patient_id}/add"),
        ("POST", "/api/fhir/export/{patient_id}"),
        ("GET", "/api/medication/adherence/{user_id}"),
        ("GET", "/admin/study/api/interactions"),
        ("GET", "/admin/study/"),
        ("POST", "/api/upload"),
        ("GET", "/api/upload/{job_id}"),
        ("POST", "/admin/study/api/voice/provider"),
    ]

    def test_call_lists_care_team_fhir_adherence_study_browser_and_upload_answer_401(self):
        for method, path in self.ROUTES:
            with self.subTest(route=f"{method} {path}"):
                response = self.call(method, path)
                self.assertEqual(response.status_code, 401, response.text)
                self.assertNotIn("call_id", response.text)

    def test_an_admin_token_reaches_the_study_browser(self):
        response = self.call("GET", "/admin/study/health", self.admin)
        self.assertEqual(response.status_code, 200, response.text)

    def test_a_browser_can_sign_in_with_the_admin_token_as_basic_auth_password(self):
        basic = base64.b64encode(f"anyone:{self.admin_token}".encode()).decode()
        response = self.call("GET", "/admin/study/health", {"Authorization": f"Basic {basic}"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("Basic", self.call("GET", "/admin/study/health").headers.get("www-authenticate", ""))


class PublicRoutesStayOpen(ServerHarness):
    def test_health_release_gate_and_webhooks_need_no_token(self):
        for method, path in (("GET", "/health"), ("GET", "/api/mobile/v1/release/status"),
                             ("GET", "/api/mobile/v1/health")):
            with self.subTest(route=f"{method} {path}"):
                self.assertEqual(self.call(method, path).status_code, 200)
        for method, path in (("POST", "/api/bolna/webhook"), ("POST", "/api/retell/webhook"),
                             ("POST", "/api/sarvam/webhook"), ("POST", "/webhook")):
            with self.subTest(route=f"{method} {path}"):
                self.assertNotIn(self.call(method, path).status_code, (401, 403))


class TheAdminPanelIsGated(ServerHarness):
    def test_the_gradio_panel_needs_an_admin(self):
        self.assertEqual(self.call("GET", "/admin/").status_code, 401)
        self.assertEqual(self.call("GET", "/admin/config", self.participant).status_code, 403)
        response = self.call("GET", "/admin/", self.admin)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "gradio admin panel")

    def test_the_web_voice_client_stays_public(self):
        self.assertNotIn(self.call("GET", "/voice/").status_code, (401, 403))


class CorsAllowsOnlyConfiguredOrigins(unittest.TestCase):
    def preflight(self, client, origin):
        return client.options("/api/query", headers={
            "Origin": origin, "Access-Control-Request-Method": "POST"})

    def test_the_default_is_no_origin(self):
        client = TestClient(ServerHarness.app or build_server_app())
        for origin in (EVIL_ORIGIN, STUDY_ORIGIN):
            response = self.preflight(client, origin)
            self.assertNotIn("access-control-allow-origin", response.headers)
            simple = client.get("/health", headers={"Origin": origin})
            self.assertNotIn("access-control-allow-origin", simple.headers)

    def test_a_configured_origin_is_allowed_and_no_other(self):
        client = TestClient(build_server_app({"CORS_ALLOWED_ORIGINS": f"{STUDY_ORIGIN}, *"}))
        allowed = self.preflight(client, STUDY_ORIGIN)
        self.assertEqual(allowed.headers.get("access-control-allow-origin"), STUDY_ORIGIN)
        refused = self.preflight(client, EVIL_ORIGIN)
        self.assertNotIn("access-control-allow-origin", refused.headers)

    def test_origins_come_from_config_when_the_environment_is_silent(self):
        from auth.policy import cors_allowed_origins

        config = {"server": {"cors_allowed_origins": [STUDY_ORIGIN + "/", "*"]}}
        self.assertEqual(cors_allowed_origins({}, config), [STUDY_ORIGIN])
        self.assertEqual(cors_allowed_origins({}, {}), [])

    def test_the_shipped_config_allows_no_origin(self):
        from auth.policy import cors_allowed_origins

        self.assertEqual(cors_allowed_origins({}, config_path=Path(__file__).resolve().parents[1] / "config.yaml"), [])


class TheOverrideRecordsTheTokensSubject(ServerHarness):
    def setUp(self):
        super().setUp()
        from voice_provider import VoiceProviderControl

        folder = Path(tempfile.mkdtemp(prefix="route-policy-voice-"))
        self.control = VoiceProviderControl(path=folder / "setting.json", audit_path=folder / "audit.jsonl")
        patcher = mock.patch("study_dashboard.get_voice_provider_control", return_value=self.control)
        patcher.start()
        self.addCleanup(patcher.stop)

    def override(self, headers, **body):
        return self.client.post("/admin/study/api/voice/provider", headers=headers,
                                json={"provider": "sarvam", "reason": "incident 412", **body})

    def test_the_actor_in_the_audit_record_is_the_tokens_subject_not_a_body_field(self):
        response = self.override(self.admin, actor="someone-else")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["updated_by"], "dr-ashish")
        self.assertEqual(self.control.history()[0]["updated_by"], "dr-ashish")

    def test_the_override_needs_no_actor_field(self):
        response = self.override(self.admin)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.control.history()[0]["updated_by"], "dr-ashish")

    def test_a_participant_cannot_change_the_voice_provider(self):
        self.assertEqual(self.override(self.participant, actor="asha-1").status_code, 403)
        self.assertTrue(self.control.current().live_enabled)

    def test_a_failed_audit_write_leaves_the_voice_provider_unchanged(self):
        self.control.audit_path = Path("/proc/definitely/not/writable/audit.jsonl")
        response = self.override(self.admin)
        self.assertEqual(response.status_code, 503, response.text)
        self.assertTrue(self.control.current().live_enabled)


if __name__ == "__main__":
    unittest.main()
