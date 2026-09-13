"""Authenticated API and strict-boundary integration tests."""

import hashlib
import json
from urllib.parse import quote

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from clinical_governance.api import GovernanceRuntime, create_app, install_governance
import test_clinical_governance as fixtures
from test_clinical_governance import FUTURE

actors, store, released = fixtures.actors, fixtures.store, fixtures.released


CLINICAL_TOKEN = "synthetic-clinical-token-for-tests-only"
USER_TOKEN = "synthetic-requester-token-for-tests-only"


@pytest.fixture
def runtime(store, actors):
    return GovernanceRuntime(
        store,
        {
            hashlib.sha256(CLINICAL_TOKEN.encode()).hexdigest(): actors["clinical"],
            hashlib.sha256(USER_TOKEN.encode()).hexdigest(): actors["user"],
        },
    )


def auth(token=CLINICAL_TOKEN):
    return {"Authorization": "Bearer " + token}


def test_api_rejects_mobile_or_unknown_tokens(runtime):
    with TestClient(create_app(runtime)) as client:
        assert client.post("/api/governance/snapshot", json={}).status_code == 401
        assert (
            client.post(
                "/api/governance/snapshot",
                json={},
                headers=auth("unrelated-mobile-token"),
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/api/governance/snapshot", json={}, headers=auth(USER_TOKEN)
            ).status_code
            == 403
        )
        result = client.post("/api/governance/snapshot", json={}, headers=auth())
        assert (
            result.status_code == 200
            and result.json()["actor"]["clinical_authority"] is True
        )


def test_strict_boundary_blocks_every_legacy_route_and_socket(runtime):
    runtime.mode = "strict"
    app = FastAPI()
    called = []

    @app.post("/api/query")
    async def old_query():
        called.append(True)
        return {"unsafe": "bypassed"}

    install_governance(app, runtime)
    with TestClient(app) as client:
        for path in [
            "/api/query",
            "/api/graphrag/query",
            "/api/bolna/query",
            "/webhook",
            "/admin",
        ]:
            assert client.post(path, json={}).status_code == 409
        assert not called
        assert (
            client.post(
                "/api/governance/catalog", json={}, headers=auth(USER_TOKEN)
            ).status_code
            == 200
        )
        assert client.get("/governance").status_code == 200
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect) as error:
            with client.websocket_connect("/ws/voice"):
                pass
        assert error.value.code == 1008


def test_parallel_mode_preserves_legacy_routes(runtime):
    app = FastAPI()

    @app.get("/legacy")
    async def legacy():
        return {"mode": "legacy"}

    install_governance(app, runtime)
    with TestClient(app) as client:
        assert client.get("/legacy").json() == {"mode": "legacy"}
        assert (
            client.get("/api/governance/health").json()["legacy_routes_guarded"]
            is False
        )


def test_authenticated_ingestion_preserves_indic_title(runtime):
    metadata = {
        "title": "योजना",
        "version": "1",
        "rights": {
            "authorized": True,
            "authorization_reference": "synthetic fixture",
            "purposes": ["information"],
            "valid_until": FUTURE,
        },
    }
    with TestClient(create_app(runtime)) as client:
        response = client.post(
            "/api/governance/ingest",
            content="केवल परीक्षण".encode(),
            headers={
                **auth(),
                "Content-Type": "text/plain",
                "X-Source-Metadata": quote(json.dumps(metadata, ensure_ascii=False)),
                "X-Source-Metadata-Format": "uri",
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["title"] == "योजना"


def test_full_authenticated_session_to_answer(runtime, released):
    with TestClient(create_app(runtime)) as client:
        session = released[3]
        response = client.post(
            "/api/governance/retrieve",
            headers=auth(USER_TOKEN),
            json={
                "session_id": session["id"],
                "epoch": 0,
                "request_id": "http-test",
                "query": "plan",
                "context": {"synthetic_encounter": True},
                "language": "en",
            },
        )
        assert response.status_code == 200, response.text
        answer = client.post(
            "/api/governance/answer",
            headers=auth(USER_TOKEN),
            json={"id": response.json()["id"]},
        )
        assert (
            answer.status_code == 200
            and answer.json()["generation"] == "reviewed-wording-v1"
        )
        assert answer.headers["cache-control"] == "no-store"


def test_request_limit_and_bad_json_are_denied(runtime):
    with TestClient(create_app(runtime)) as client:
        assert (
            client.post(
                "/api/governance/catalog", headers=auth(), content="x" * (257 * 1024)
            ).status_code
            == 413
        )
        assert (
            client.post(
                "/api/governance/catalog", headers=auth(), content="{"
            ).status_code
            == 422
        )
        audit = client.post("/api/governance/snapshot", headers=auth(), json={}).json()[
            "audit"
        ]
        assert any(row["operation"] == "request_failure" for row in audit)


def test_ui_security_headers_and_unknown_asset(runtime):
    with TestClient(create_app(runtime)) as client:
        page = client.get("/governance")
        assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
        assert page.headers["x-content-type-options"] == "nosniff"
        assert client.get("/governance/app.js").status_code == 200
        assert client.get("/governance/private.txt").status_code == 404


def test_missing_environment_config_fails_closed(monkeypatch):
    monkeypatch.setenv("CLINICAL_GOVERNANCE_MODE", "strict")
    monkeypatch.delenv("CLINICAL_GOVERNANCE_PRINCIPALS_JSON", raising=False)
    with pytest.raises(RuntimeError, match="Invalid governance configuration"):
        create_app()
