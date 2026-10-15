"""
Mobile tokens can't be misused, and the PIN lockout lifts (review 3, ticket 08).

Before this, `require_auth` never read the token's type, so the long-lived
refresh token worked as an access token on every mobile route. `/auth/refresh`
took any bearer token, so an access token renewed itself, and a session never
needed the PIN again. Five wrong PINs locked a participant out for good.

What these pin down, over HTTP:

  * a refresh token is refused (401) on every route that needs an access token
  * /auth/refresh refuses an access token and renews from a refresh token
  * a renewed session does not outlive the PIN sign-in that started it
  * a locked account refuses even the right PIN until the cooling-off period
    passes, then accepts it
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jwt  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import auth.pin_auth as pin_auth_module  # noqa: E402
from auth.middleware import require_auth, init_auth  # noqa: E402
from mobile_api.dependencies import init_dependencies  # noqa: E402
from mobile_api.router import mobile_router  # noqa: E402

BASE = "/api/mobile/v1"
SECRET = "test-secret-for-mobile-token-types"
PIN = "4821"
WRONG_PIN = "0000"
ONE_DAY = 24 * 3600.0  # well past any cooling-off period


def _needs_access_token(route: APIRoute) -> bool:
    """True when the route's dependency tree includes require_auth."""
    stack = list(route.dependant.dependencies)
    while stack:
        dep = stack.pop()
        if dep.call is require_auth:
            return True
        stack.extend(dep.dependencies)
    return False


def _concrete_path(path: str) -> str:
    """Fill every path parameter with a placeholder value."""
    out = []
    for part in path.split("/"):
        out.append("placeholder" if part.startswith("{") and part.endswith("}") else part)
    return "/".join(out)


class MobileTokens(unittest.TestCase):
    """Access and refresh tokens are each accepted only where they belong."""

    def setUp(self) -> None:
        devices = tempfile.mkdtemp(prefix="devices-")
        patcher = mock.patch.object(pin_auth_module, "DATA_DIR", Path(devices))
        patcher.start()
        self.addCleanup(patcher.stop)
        init_auth(jwt_secret=SECRET)
        init_dependencies(None, None, None, None)
        app = FastAPI()
        app.include_router(mobile_router)
        self.app = app
        self.client = TestClient(app)
        self.session = self.register()

    def register(self) -> dict:
        response = self.client.post(f"{BASE}/auth/register", json={
            "name": "Asha A", "phone_hash": "hash-a", "role": "asha_worker",
            "language": "hi", "site_id": "site-1", "pin": PIN,
        })
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def bearer(self, token: str) -> dict:
        return {"Authorization": f"Bearer {token}"}

    def protected_routes(self) -> list:
        routes = [r for r in mobile_router.routes if isinstance(r, APIRoute) and _needs_access_token(r)]
        self.assertGreater(len(routes), 20, "the route walk found too few protected routes")
        return routes

    def test_refresh_token_is_rejected_on_every_authenticated_mobile_route(self) -> None:
        refresh = self.session["refresh_token"]
        accepted = []
        for route in self.protected_routes():
            for method in route.methods:
                response = self.client.request(method, _concrete_path(route.path), headers=self.bearer(refresh))
                if response.status_code != 401:
                    accepted.append(f"{method} {route.path} -> {response.status_code}")
        self.assertEqual(accepted, [], "a refresh token was accepted as an access token")

    def test_access_token_is_accepted_on_an_authenticated_route(self) -> None:
        response = self.client.get(f"{BASE}/sync/status", headers=self.bearer(self.session["token"]))
        self.assertEqual(response.status_code, 200, response.text)

    def test_token_without_a_type_is_rejected(self) -> None:
        untyped = jwt.encode({"user_id": self.session["user_id"], "role": "asha", "site_id": "site-1",
                              "exp": time.time() + 3600}, SECRET, algorithm="HS256")
        response = self.client.get(f"{BASE}/sync/status", headers=self.bearer(untyped))
        self.assertEqual(response.status_code, 401)

    def test_refresh_rejects_an_access_token(self) -> None:
        access = self.session["token"]
        response = self.client.post(f"{BASE}/auth/refresh", headers=self.bearer(access),
                                    json={"refresh_token": access})
        self.assertEqual(response.status_code, 401, response.text)

    def test_refresh_token_renews_the_session(self) -> None:
        response = self.client.post(f"{BASE}/auth/refresh",
                                    json={"refresh_token": self.session["refresh_token"]})
        self.assertEqual(response.status_code, 200, response.text)
        renewed = response.json()
        self.assertEqual(renewed["user_id"], self.session["user_id"])

        status = self.client.get(f"{BASE}/sync/status", headers=self.bearer(renewed["token"]))
        self.assertEqual(status.status_code, 200, status.text)
        claims = jwt.decode(renewed["token"], options={"verify_signature": False})
        self.assertEqual((claims["role"], claims["site_id"]), ("asha_worker", "site-1"))

    def test_renewed_session_does_not_outlive_the_pin_sign_in(self) -> None:
        first = self.session["refresh_token"]
        original_expiry = jwt.decode(first, options={"verify_signature": False})["exp"]

        renewed = self.client.post(f"{BASE}/auth/refresh", json={"refresh_token": first}).json()
        again = self.client.post(f"{BASE}/auth/refresh", json={"refresh_token": renewed["refresh_token"]}).json()

        for pair in (renewed, again):
            self.assertLessEqual(jwt.decode(pair["refresh_token"], options={"verify_signature": False})["exp"],
                                 original_expiry)
            self.assertLessEqual(pair["expires_at"], original_expiry)

    def test_refresh_fails_for_an_unregistered_device(self) -> None:
        refresh = self.session["refresh_token"]
        (pin_auth_module.DATA_DIR / f"{self.session['user_id']}.json").unlink()
        response = self.client.post(f"{BASE}/auth/refresh", json={"refresh_token": refresh})
        self.assertEqual(response.status_code, 401, response.text)


class PinLockout(unittest.TestCase):
    """Five wrong PINs lock the account for a cooling-off period, not for good."""

    def setUp(self) -> None:
        devices = tempfile.mkdtemp(prefix="devices-")
        patcher = mock.patch.object(pin_auth_module, "DATA_DIR", Path(devices))
        patcher.start()
        self.addCleanup(patcher.stop)
        init_auth(jwt_secret=SECRET)
        app = FastAPI()
        app.include_router(mobile_router)
        self.client = TestClient(app)
        response = self.client.post(f"{BASE}/auth/register", json={
            "name": "Caregiver B", "phone_hash": "hash-b", "role": "family_caregiver",
            "language": "mr", "site_id": "site-1", "pin": PIN,
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.user_id = response.json()["user_id"]

    def login(self, pin: str, later_by: float = 0.0) -> int:
        real_time = time.time
        with mock.patch.object(pin_auth_module.time, "time", side_effect=lambda: real_time() + later_by):
            response = self.client.post(f"{BASE}/auth/login", json={"user_id": self.user_id, "pin": pin})
        return response.status_code

    def lock(self) -> None:
        for _ in range(5):
            self.assertEqual(self.login(WRONG_PIN), 401)

    def test_locked_account_refuses_the_correct_pin_during_the_cooling_off_period(self) -> None:
        self.lock()
        self.assertEqual(self.login(PIN), 401)

    def test_after_the_cooling_off_period_a_locked_account_accepts_the_correct_pin(self) -> None:
        self.lock()
        after_cooling_off = ONE_DAY
        self.assertEqual(self.login(PIN, later_by=after_cooling_off), 200)

    def test_a_wrong_pin_after_the_cooling_off_period_locks_again(self) -> None:
        self.lock()
        after_cooling_off = ONE_DAY
        self.assertEqual(self.login(WRONG_PIN, later_by=after_cooling_off), 401)
        self.assertEqual(self.login(PIN, later_by=after_cooling_off + 1), 401)


if __name__ == "__main__":
    unittest.main()
