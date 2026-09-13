"""Opt-in governance API, isolated from self-registered mobile roles."""

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
from typing import Dict, Optional
from urllib.parse import unquote

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from .ingestion import ingest_pdf, ingest_text, render_pdf_page
from .models import Actor, GovernanceError
from .store import GovernanceStore


ASSET_DIR = Path(__file__).with_name("web")
CSP = "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob:; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
ALLOWED_ROLES = {
    "editor",
    "clinician",
    "publisher",
    "operator",
    "auditor",
    "requester",
    "billing",
    "evaluator",
    "language_reviewer",
    "usability_reviewer",
}
OPERATIONS = {
    "catalog",
    "get_retrieval",
    "propose",
    "edit",
    "review",
    "publish",
    "revoke",
    "conflict",
    "policy",
    "session",
    "interrupt",
    "close_session",
    "retrieve",
    "review_retrieval",
    "answer",
    "delivery",
    "snapshot",
    "history",
    "purge_payloads",
    "request_payload",
    "audit_verify",
    "budget_configure",
    "budget_rate",
    "budget_reserve",
    "budget_reconcile",
    "budget_status",
    "validation_record",
    "validation_status",
}


@dataclass
class GovernanceRuntime:
    store: GovernanceStore
    principals: Dict[str, Actor]  # SHA256 bearer token -> provisioned actor.
    mode: str = "parallel"

    def actor(self, authorization: str) -> Actor:
        if not authorization.startswith("Bearer ") or len(authorization) > 1000:
            raise GovernanceError("governance_authentication_required", 401)
        candidate = hashlib.sha256(authorization[7:].encode()).hexdigest()
        for expected, actor in self.principals.items():
            if hmac.compare_digest(candidate, expected):
                return actor
        raise GovernanceError("invalid_governance_credentials", 401)

    @classmethod
    def from_environment(cls) -> Optional["GovernanceRuntime"]:
        mode = os.environ.get("CLINICAL_GOVERNANCE_MODE", "off")
        if mode not in ("off", "parallel", "strict"):
            raise RuntimeError(
                "CLINICAL_GOVERNANCE_MODE must be off, parallel or strict"
            )
        if mode == "off":
            return None
        try:
            config = json.loads(os.environ["CLINICAL_GOVERNANCE_PRINCIPALS_JSON"])
            key = os.environ["CLINICAL_GOVERNANCE_AUDIT_KEY"].encode()
            if len(key) < 32 or not isinstance(config, list) or not config:
                raise ValueError
            principals = {}
            for item in config:
                token_hash = item["token_sha256"]
                if (
                    not isinstance(token_hash, str)
                    or not re.fullmatch(r"[a-f0-9]{64}", token_hash)
                    or token_hash in principals
                ):
                    raise ValueError
                roles = frozenset(item["roles"])
                if not roles or not roles <= ALLOWED_ROLES:
                    raise ValueError
                for field in ("actor_id", "tenant_id"):
                    if (
                        not isinstance(item[field], str)
                        or not 0 < len(item[field]) <= 100
                    ):
                        raise ValueError
                purposes = frozenset(
                    item.get("purposes", ["information", "quality_review"])
                )
                if not purposes or not all(
                    isinstance(p, str) and 0 < len(p) <= 80 for p in purposes
                ):
                    raise ValueError
                principals[token_hash] = Actor(
                    item["actor_id"],
                    item["tenant_id"],
                    roles,
                    purposes,
                    clinical_authority=item.get("clinical_authority") is True,
                )
            store = GovernanceStore(
                os.environ.get(
                    "CLINICAL_GOVERNANCE_DB",
                    "data/clinical_governance/governance.sqlite3",
                ),
                key,
            )
            return cls(store, principals, mode)
        except (KeyError, ValueError, TypeError, GovernanceError):
            raise RuntimeError(
                "Invalid governance configuration; provision principals and an independent audit key"
            ) from None


class StrictGovernanceBoundary:
    """When explicitly enabled, do not expose legacy HTTP/voice bypasses."""

    def __init__(self, app, mode, runtime=None):
        self.app, self.mode, self.runtime = app, mode, runtime

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        allowed = (
            path == "/governance"
            or path.startswith("/governance/")
            or path.startswith("/api/governance/")
            or path == "/health"
        )
        if (
            self.mode == "strict"
            and scope["type"] in ("http", "websocket")
            and not allowed
        ):
            audit_error = None
            try:
                await run_in_threadpool(
                    self.runtime.store.execute,
                    transport_actor(),
                    "boundary_denial",
                    {"reason": "legacy_route_blocked", "transport": scope["type"]},
                )
            except GovernanceError as error:
                audit_error = error
            if scope["type"] == "websocket":
                await send(
                    {
                        "type": "websocket.close",
                        "code": 1011 if audit_error else 1008,
                        "reason": "Use an authenticated governed session",
                    }
                )
            else:
                await JSONResponse(
                    {
                        "status": "governance_required",
                        "detail": audit_error.code
                        if audit_error
                        else "Legacy routes are disabled in strict mode. Use /governance.",
                    },
                    status_code=503 if audit_error else 409,
                )(scope, receive, send)
            return
        await self.app(scope, receive, send)


async def limited_body(request: Request, maximum: int) -> bytes:
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > maximum:
            raise GovernanceError("request_size_limit", 413)
        chunks.append(chunk)
    return b"".join(chunks)


def transport_actor():
    return Actor(
        "governance-transport", "_governance_transport", frozenset({"transport"})
    )


def legacy_query_guard():
    """Block direct legacy pipeline calls when strict governance is selected."""
    if os.environ.get("CLINICAL_GOVERNANCE_MODE", "off") != "strict":
        return None
    runtime = GovernanceRuntime.from_environment()
    runtime.store.execute(
        transport_actor(),
        "boundary_denial",
        {"reason": "legacy_pipeline_blocked", "transport": "internal"},
    )
    return {
        "status": "governance_required",
        "answer": "",
        "sources": [],
        "error": "Use the authenticated governance API and a pinned release.",
    }


def install_governance(app: FastAPI, runtime: Optional[GovernanceRuntime] = None):
    """Register an isolated API. No inference or paid call occurs on startup."""
    runtime = runtime or GovernanceRuntime.from_environment()
    app.state.clinical_governance = runtime
    app.add_middleware(
        StrictGovernanceBoundary,
        mode=runtime.mode if runtime else "off",
        runtime=runtime,
    )
    router = APIRouter(prefix="/api/governance", tags=["clinical-governance"])

    async def authenticate(request):
        if runtime is None:
            raise GovernanceError("governance_not_configured", 503)
        try:
            return runtime.actor(request.headers.get("authorization", ""))
        except GovernanceError:
            await run_in_threadpool(
                runtime.store.execute,
                transport_actor(),
                "boundary_denial",
                {"reason": "authentication_failed", "transport": "http"},
            )
            raise

    async def invoke(actor, operation, data):
        return await run_in_threadpool(runtime.store.execute, actor, operation, data)

    def failure(error):
        return JSONResponse(
            {"status": "error", "error": error.code},
            status_code=error.http_status,
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @router.get("/health")
    async def health():
        return {
            "configured": runtime is not None,
            "mode": runtime.mode if runtime else "off",
            "clinical_validation": "not_established",
            "legacy_routes_guarded": bool(runtime and runtime.mode == "strict"),
        }

    @router.post("/ingest")
    async def ingest(request: Request):
        actor = None
        try:
            actor = await authenticate(request)
            actor.require("editor", "clinician")
            content = await limited_body(request, 20 * 1024 * 1024)
            header = request.headers.get("x-source-metadata", "{}")
            if request.headers.get("x-source-metadata-format") == "uri":
                header = unquote(header, errors="strict")
            metadata = json.loads(header)
            if not isinstance(metadata, dict):
                raise GovernanceError("invalid_source_metadata", 422)
            media_type = request.headers.get("content-type", "").split(";")[0]
            if media_type == "application/zip":
                result = await invoke(
                    actor,
                    "import_bundle",
                    {
                        "content": content,
                        "rights": metadata["rights"],
                        "update_import_id": metadata.get("update_import_id"),
                    },
                )
            elif media_type in ("application/pdf", "text/plain", "text/markdown"):
                # Parsing failures are recorded without retaining the untrusted bytes.
                await invoke(
                    actor,
                    "ingestion_attempt",
                    {
                        "content_sha256": hashlib.sha256(content).hexdigest(),
                        "media_type": media_type,
                    },
                )
                extracted = (
                    await run_in_threadpool(ingest_pdf, content)
                    if media_type == "application/pdf"
                    else await run_in_threadpool(ingest_text, content, media_type)
                )
                result = await invoke(actor, "source", {**metadata, **extracted})
            else:
                raise GovernanceError("unsupported_source_media_type", 415)
            return JSONResponse(result, headers={"Cache-Control": "no-store"})
        except (ValueError, KeyError, TypeError) as error:
            error = (
                error
                if isinstance(error, GovernanceError)
                else GovernanceError("invalid_source_request", 422)
            )
            if actor and runtime:
                try:
                    await invoke(actor, "ingestion_failure", {"reason": error.code})
                except GovernanceError as audit_error:
                    return failure(audit_error)
            return failure(error)

    @router.get("/source/{source_id}/page/{page}")
    async def source_page(source_id: str, page: int, request: Request):
        try:
            actor = await authenticate(request)
            result = await invoke(actor, "source_bytes", {"id": source_id})
            if result["media_type"] != "application/pdf":
                raise GovernanceError("pdf_preview_unavailable", 404)
            png = await run_in_threadpool(render_pdf_page, result["content"], page)
            return Response(
                png,
                media_type="image/png",
                headers={
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                },
            )
        except GovernanceError as error:
            return failure(error)

    @router.post("/{operation}")
    async def operate(operation: str, request: Request):
        actor, invoked = None, False
        try:
            actor = await authenticate(request)
            if operation not in OPERATIONS:
                raise GovernanceError("unknown_operation", 404)
            body = await limited_body(request, 256 * 1024)
            data = json.loads(body)
            if not isinstance(data, dict):
                raise GovernanceError("request_object_required", 422)
            invoked = True
            result = await invoke(actor, operation, data)
            return JSONResponse(result, headers={"Cache-Control": "no-store"})
        except (ValueError, KeyError, TypeError) as error:
            if actor and not invoked:
                try:
                    await invoke(
                        actor,
                        "request_failure",
                        {
                            "reason": "invalid_transport_request",
                            "operation": operation
                            if operation in OPERATIONS
                            else "unknown",
                        },
                    )
                except GovernanceError as audit_error:
                    return failure(audit_error)
            return failure(
                error
                if isinstance(error, GovernanceError)
                else GovernanceError("invalid_json_request", 422)
            )

    app.include_router(router)

    @app.get("/governance", include_in_schema=False)
    @app.get("/governance/", include_in_schema=False)
    async def governance_page():
        return FileResponse(
            ASSET_DIR / "index.html",
            headers={
                "Content-Security-Policy": CSP,
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/governance/{asset}", include_in_schema=False)
    async def governance_asset(asset: str):
        if asset not in ("app.js", "style.css"):
            raise HTTPException(status_code=404)
        return FileResponse(
            ASSET_DIR / asset,
            headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-cache"},
        )

    return runtime


def create_app(runtime: Optional[GovernanceRuntime] = None):
    app = FastAPI(title="Palli Sahayak clinical knowledge governance")
    install_governance(app, runtime)
    return app
