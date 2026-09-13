"""Transactional clinical release gateway and mandatory minimized audit trail.

The store uses an independent key to authenticate the per-tenant audit chain.
It cannot protect against an administrator who controls both database and key;
production deployments should additionally export signed checkpoints off-host.
"""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, Optional
import uuid

from pydantic import ValidationError

from .models import (
    Actor,
    Condition,
    GovernanceError,
    RecommendationDraft,
    ReviewPolicy,
    Rights,
    applicability,
    timestamp,
    terms,
    unicode_offsets,
    utcnow,
)


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def require(condition: bool, code: str, status: int = 409) -> None:
    if not condition:
        raise GovernanceError(code, status)


class GovernanceStore:
    """One process-independent SQLite transaction per governed operation."""

    def __init__(self, path: str | Path, audit_key: bytes):
        require(len(audit_key) >= 32, "audit_key_too_short", 503)
        self.path = Path(path)
        self.audit_key = audit_key
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        require(not self.path.is_symlink(), "database_symlink_not_allowed", 503)
        with self._connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS objects(
                  tenant TEXT NOT NULL, id TEXT NOT NULL, revision INTEGER NOT NULL,
                  kind TEXT NOT NULL, payload TEXT NOT NULL, hash TEXT NOT NULL,
                  PRIMARY KEY(tenant,id,revision));
                CREATE TABLE IF NOT EXISTS heads(
                  tenant TEXT NOT NULL,id TEXT NOT NULL,revision INTEGER NOT NULL,
                  PRIMARY KEY(tenant,id));
                CREATE TABLE IF NOT EXISTS blobs(
                  tenant TEXT NOT NULL,id TEXT NOT NULL,content BLOB NOT NULL,
                  PRIMARY KEY(tenant,id));
                CREATE TABLE IF NOT EXISTS audit(
                  tenant TEXT NOT NULL,sequence INTEGER NOT NULL,actor TEXT NOT NULL,
                  operation TEXT NOT NULL,at TEXT NOT NULL,detail TEXT NOT NULL,
                  previous_hash TEXT NOT NULL,hash TEXT NOT NULL,
                  PRIMARY KEY(tenant,sequence));
                CREATE TABLE IF NOT EXISTS payloads(
                  tenant TEXT NOT NULL,id TEXT NOT NULL,owner TEXT NOT NULL,
                  expires_at TEXT NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(tenant,id));
                CREATE TABLE IF NOT EXISTS idempotency(
                  tenant TEXT NOT NULL,actor TEXT NOT NULL,key TEXT NOT NULL,
                  fingerprint TEXT NOT NULL,result_id TEXT NOT NULL,
                  PRIMARY KEY(tenant,actor,key));
                CREATE TRIGGER IF NOT EXISTS immutable_objects_update BEFORE UPDATE ON objects
                  BEGIN SELECT RAISE(ABORT,'immutable objects'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_objects_delete BEFORE DELETE ON objects
                  BEGIN SELECT RAISE(ABORT,'immutable objects'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_blobs_update BEFORE UPDATE ON blobs
                  BEGIN SELECT RAISE(ABORT,'immutable sources'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_blobs_delete BEFORE DELETE ON blobs
                  BEGIN SELECT RAISE(ABORT,'immutable sources'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_audit_update BEFORE UPDATE ON audit
                  BEGIN SELECT RAISE(ABORT,'append only audit'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_audit_delete BEFORE DELETE ON audit
                  BEGIN SELECT RAISE(ABORT,'append only audit'); END;
            """)
        os.chmod(self.path, 0o600)

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
            for suffix in ("", "-wal", "-shm"):
                candidate = Path(str(self.path) + suffix)
                if candidate.exists():
                    os.chmod(candidate, 0o600)

    def fingerprint(self, value: Any) -> str:
        return hmac.new(
            self.audit_key, canonical(value).encode(), hashlib.sha256
        ).hexdigest()

    def _verify(self, db, tenant: str) -> bool:
        previous, seq = "0" * 64, 0
        for row in db.execute(
            "SELECT * FROM audit WHERE tenant=? ORDER BY sequence", (tenant,)
        ):
            seq += 1
            signed = [
                tenant,
                seq,
                row["actor"],
                row["operation"],
                row["at"],
                json.loads(row["detail"]),
                previous,
            ]
            if (
                row["sequence"] != seq
                or row["previous_hash"] != previous
                or not hmac.compare_digest(row["hash"], self.fingerprint(signed))
            ):
                return False
            previous = row["hash"]
        return True

    def _audit(self, db, actor: Actor, operation: str, detail: Dict[str, Any]):
        row = db.execute(
            "SELECT sequence,hash FROM audit WHERE tenant=? ORDER BY sequence DESC LIMIT 1",
            (actor.tenant,),
        ).fetchone()
        sequence, previous = (
            (row["sequence"] + 1, row["hash"]) if row else (1, "0" * 64)
        )
        now = utcnow()
        signed = [actor.tenant, sequence, actor.id, operation, now, detail, previous]
        db.execute(
            "INSERT INTO audit VALUES(?,?,?,?,?,?,?,?)",
            (
                actor.tenant,
                sequence,
                actor.id,
                operation,
                now,
                canonical(detail),
                previous,
                self.fingerprint(signed),
            ),
        )
        return sequence

    def _put(self, db, actor, kind: str, payload: dict, id_: Optional[str] = None):
        id_ = id_ or f"{kind}_{uuid.uuid4().hex}"
        old = db.execute(
            "SELECT revision FROM heads WHERE tenant=? AND id=?", (actor.tenant, id_)
        ).fetchone()
        revision = old[0] + 1 if old else 1
        db.execute(
            "INSERT INTO objects VALUES(?,?,?,?,?,?)",
            (actor.tenant, id_, revision, kind, canonical(payload), digest(payload)),
        )
        db.execute(
            "INSERT INTO heads VALUES(?,?,?) ON CONFLICT(tenant,id) DO UPDATE SET revision=excluded.revision",
            (actor.tenant, id_, revision),
        )
        return {
            "id": id_,
            "revision": revision,
            "kind": kind,
            "hash": digest(payload),
            **payload,
        }

    def _get(
        self,
        db,
        actor,
        id_: str,
        kind: Optional[str] = None,
        revision: Optional[int] = None,
    ):
        if revision is None:
            row = db.execute(
                "SELECT o.* FROM objects o JOIN heads h USING(tenant,id,revision) WHERE o.tenant=? AND o.id=?",
                (actor.tenant, id_),
            ).fetchone()
        else:
            row = db.execute(
                "SELECT * FROM objects WHERE tenant=? AND id=? AND revision=?",
                (actor.tenant, id_, revision),
            ).fetchone()
        require(
            row is not None and (kind is None or row["kind"] == kind),
            "object_unavailable",
            404,
        )
        payload = json.loads(row["payload"])
        require(
            hmac.compare_digest(row["hash"], digest(payload)),
            "object_integrity_failure",
            503,
        )
        return {
            "id": row["id"],
            "revision": row["revision"],
            "kind": row["kind"],
            "hash": row["hash"],
            **payload,
        }

    def _list(self, db, actor, kind):
        rows = db.execute(
            "SELECT o.id FROM objects o JOIN heads h USING(tenant,id,revision) WHERE o.tenant=? AND o.kind=? ORDER BY o.rowid",
            (actor.tenant, kind),
        )
        return [self._get(db, actor, row[0], kind) for row in rows]

    def _latest_review(self, db, actor, item):
        matches = [
            review
            for review in self._list(db, actor, "review")
            if review["target_id"] == item["id"]
            and review["target_revision"] == item["revision"]
        ]
        # Database insertion order, not wall clock, determines the latest decision.
        return matches[-1] if matches else None

    def _revoked(self, db, actor, id_):
        return any(
            row["target_id"] == id_ for row in self._list(db, actor, "revocation")
        )

    def _policy(self, db, actor):
        try:
            return self._get(db, actor, "review_policy", "policy")
        except GovernanceError as exc:
            if exc.http_status != 404:
                raise
        return {"id": "review_policy", "revision": 0, **ReviewPolicy().model_dump()}

    def _source_checks(self, db, actor, anchor: dict, purpose: str):
        source = self._get(db, actor, anchor["source_id"], "source")
        reasons = []
        if self._revoked(db, actor, source["id"]):
            reasons.append("source_revoked")
        if source.get("family_id"):
            family = self._get(db, actor, source["family_id"], "source_family")
            if family["active_source_id"] != source["id"]:
                reasons.append("source_superseded")
        rights = source["rights"]
        if rights["authorized"] is not True or purpose not in rights["purposes"]:
            reasons.append("source_purpose_not_permitted")
        if timestamp(rights["valid_until"]) <= datetime.now(timezone.utc):
            reasons.append("source_permission_expired")
        text = source["normalized_text"]
        if (
            anchor["normalized_sha256"] != source["normalized_sha256"]
            or hashlib.sha256(text.encode()).hexdigest() != source["normalized_sha256"]
            or not 0 <= anchor["start"] < anchor["end"] <= len(text)
            or text[anchor["start"] : anchor["end"]] != anchor["excerpt"]
        ):
            reasons.append("evidence_mismatch")
        if anchor.get("page") is not None:
            pages = {page["page"]: page for page in source.get("pages", [])}
            page = pages.get(anchor["page"])
            if (
                page is None
                or not page["start"] <= anchor["start"] < anchor["end"] <= page["end"]
            ):
                reasons.append("page_anchor_mismatch")
        if anchor.get("block_id") is not None:
            blocks = {block["id"]: block for block in source.get("blocks", [])}
            block = blocks.get(anchor["block_id"])
            if (
                block is None
                or not block["start"] <= anchor["start"] < anchor["end"] <= block["end"]
            ):
                reasons.append("block_anchor_mismatch")
        return reasons

    def _eligibility(self, db, actor, item, purpose, approval_id=None):
        reasons = []
        current = self._get(db, actor, item["id"], "recommendation")
        if current["revision"] != item["revision"] or current["hash"] != item["hash"]:
            reasons.append("superseded_revision")
        if self._revoked(db, actor, item["id"]):
            reasons.append("recommendation_revoked")
        review = self._latest_review(db, actor, current)
        if (
            review is None
            or review["decision"] != "approved"
            or review["target_hash"] != item["hash"]
        ):
            reasons.append("current_revision_not_approved")
        elif approval_id is not None and review["id"] != approval_id:
            reasons.append("approval_changed_republish_required")
        if purpose not in item["purposes"]:
            reasons.append("recommendation_purpose_not_permitted")
        if timestamp(item["valid_until"]) <= datetime.now(timezone.utc):
            reasons.append("recommendation_expired")
        conflicts = [
            conflict
            for conflict in self._list(db, actor, "conflict")
            if item["id"] in conflict["targets"] and conflict["status"] == "open"
        ]
        if conflicts:
            reasons.append("blocking_conflict")
        for anchor in item["evidence"]:
            reasons.extend(self._source_checks(db, actor, anchor, purpose))
        return sorted(set(reasons))

    def _release(self, db, actor, id_, purpose):
        release = self._get(db, actor, id_, "release")
        require(not self._revoked(db, actor, id_), "release_revoked")
        require(purpose in release["purposes"], "release_purpose_not_permitted", 403)
        require(
            timestamp(release["valid_until"]) > datetime.now(timezone.utc),
            "release_expired",
        )
        require(
            release["manifest_hash"] == digest(release["members"]),
            "release_integrity_failure",
            503,
        )
        return release

    def _session(self, db, actor, id_):
        session = self._get(db, actor, id_, "session")
        require(
            session["owner"] == actor.id or "clinician" in actor.roles,
            "session_owner_required",
            403,
        )
        require(
            timestamp(session["expires_at"]) > datetime.now(timezone.utc),
            "session_expired",
        )
        require(not session["closed"], "session_closed")
        actor.permit(session["purpose"])
        return session

    def execute(self, actor: Actor, operation: str, data: Dict[str, Any]):
        """Never return a governed result if its audit transaction cannot commit."""
        require(bool(actor.id) and bool(actor.tenant), "principal_required", 401)
        error = None
        try:
            with self._connection() as db:
                require(self._verify(db, actor.tenant), "audit_integrity_failure", 503)
                db.execute("SAVEPOINT action")
                try:
                    result, detail = self._dispatch(db, actor, operation, data)
                except (
                    GovernanceError,
                    ValidationError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as exc:
                    db.execute("ROLLBACK TO action")
                    error = (
                        exc
                        if isinstance(exc, GovernanceError)
                        else GovernanceError("invalid_request", 422)
                    )
                    self._audit(
                        db,
                        actor,
                        operation + ".denied",
                        {"reason": error.code, "roles": sorted(actor.roles)},
                    )
                else:
                    seq = self._audit(
                        db, actor, operation, {"roles": sorted(actor.roles), **detail}
                    )
                    result = {**result, "audit_sequence": seq}
        except sqlite3.Error:
            raise GovernanceError("durable_audit_unavailable", 503) from None
        if error:
            raise error
        return result

    def _dispatch(self, db, actor, operation, data):
        if operation == "request_failure":
            return {"recorded": True}, {
                "reason": data["reason"],
                "operation": data["operation"],
            }
        if operation == "boundary_denial":
            actor.require("transport")
            return {"recorded": True}, {
                "reason": data["reason"],
                "transport": data["transport"],
            }
        if operation in ("ingestion_attempt", "ingestion_failure"):
            actor.require("editor", "clinician")
            detail = (
                {
                    "content_sha256": data["content_sha256"],
                    "media_type": data["media_type"],
                }
                if operation == "ingestion_attempt"
                else {"reason": data["reason"]}
            )
            return {"recorded": True}, detail
        if operation == "import_bundle":
            from .imports import import_bundle

            return import_bundle(self, db, actor, data)
        if operation == "source":
            actor.require("editor", "clinician")
            rights = Rights.model_validate(data["rights"])
            for purpose in rights.purposes:
                actor.permit(purpose)
            text, original = data["normalized_text"], data["original"]
            require(
                isinstance(text, str) and 0 < len(text) <= 2_000_000,
                "invalid_source_text",
                422,
            )
            require(
                isinstance(original, bytes) and 0 < len(original) <= 20 * 1024 * 1024,
                "invalid_source_bytes",
                422,
            )
            require(
                timestamp(rights.valid_until) > datetime.now(timezone.utc),
                "source_permission_expired",
            )
            title, version = data["title"], data["version"]
            require(
                isinstance(title, str)
                and 0 < len(title) <= 300
                and isinstance(version, str)
                and 0 < len(version) <= 100,
                "invalid_source_identity",
                422,
            )
            family_id = data.get("family_id")
            if family_id:
                self._get(db, actor, family_id, "source_family")
            else:
                family_id = f"family_{uuid.uuid4().hex}"
            pages, blocks = data.get("pages", []), data.get("blocks", [])
            for page in pages:
                require(
                    type(page["page"]) is int
                    and page["page"] >= 1
                    and 0 <= page["start"] <= page["end"] <= len(text),
                    "invalid_page_map",
                    422,
                )
            for block in blocks:
                require(
                    isinstance(block["id"], str)
                    and 0 <= block["start"] < block["end"] <= len(text),
                    "invalid_block_map",
                    422,
                )
            payload = {
                "title": title,
                "version": version,
                "family_id": family_id,
                "rights": rights.model_dump(),
                "normalized_text": text,
                "normalized_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "original_sha256": hashlib.sha256(original).hexdigest(),
                "media_type": data.get("media_type", "text/plain"),
                "pages": pages,
                "blocks": blocks,
                "coverage_gaps": data.get("coverage_gaps", []),
                "normalizer": data.get("normalizer", "identity-codepoints-v1"),
                "import_metadata": data.get("import_metadata", {}),
                "created_at": utcnow(),
            }
            result = self._put(db, actor, "source", payload)
            db.execute(
                "INSERT INTO blobs VALUES(?,?,?)",
                (actor.tenant, result["id"], original),
            )
            self._put(
                db,
                actor,
                "source_family",
                {"active_source_id": result["id"]},
                family_id,
            )
            return result, {
                "source_id": result["id"],
                "family_id": family_id,
                "version": version,
                "original_sha256": payload["original_sha256"],
                "normalized_sha256": payload["normalized_sha256"],
                "rights_fingerprint": self.fingerprint(payload["rights"]),
            }

        if operation in ("propose", "edit"):
            actor.require("editor", "clinician")
            draft = RecommendationDraft.model_validate(data["draft"])
            payload = draft.model_dump()
            for purpose in draft.purposes:
                actor.permit(purpose)
                for anchor in payload["evidence"]:
                    require(
                        not self._source_checks(db, actor, anchor, purpose),
                        "source_evidence_or_rights_invalid",
                    )
                    anchor["offsets"] = unicode_offsets(
                        self._get(db, actor, anchor["source_id"], "source")[
                            "normalized_text"
                        ],
                        anchor["start"],
                        anchor["end"],
                    )
            target = None
            if operation == "edit":
                target = self._get(db, actor, data["id"], "recommendation")
                require(
                    data["expected_revision"] == target["revision"], "revision_conflict"
                )
            item = self._put(
                db, actor, "recommendation", payload, target["id"] if target else None
            )
            return item, {
                "target_id": item["id"],
                "revision": item["revision"],
                "content_hash": item["hash"],
                "source_ids": sorted({e["source_id"] for e in item["evidence"]}),
            }

        if operation == "review":
            actor.require("clinician")
            require(actor.clinical_authority, "clinical_authority_not_provisioned", 403)
            item = self._get(db, actor, data["id"], "recommendation")
            require(data["expected_revision"] == item["revision"], "revision_conflict")
            require(
                data["decision"] in ("approved", "rejected", "deferred"),
                "invalid_review_decision",
                422,
            )
            rationale = data.get("rationale", "").strip()
            require(0 < len(rationale) <= 5000, "review_rationale_required", 422)
            if data["decision"] == "approved":
                require(
                    data.get("semantic_checked") is True
                    and data.get("exceptions_checked") is True
                    and data.get("coverage_acknowledged") is True,
                    "clinical_checks_required",
                )
                for purpose in item["purposes"]:
                    for anchor in item["evidence"]:
                        require(
                            not self._source_checks(db, actor, anchor, purpose),
                            "source_evidence_or_rights_invalid",
                        )
            review = self._put(
                db,
                actor,
                "review",
                {
                    "target_id": item["id"],
                    "target_revision": item["revision"],
                    "target_hash": item["hash"],
                    "decision": data["decision"],
                    "rationale": rationale,
                    "actor": actor.id,
                    "at": utcnow(),
                    "semantic_checked": data.get("semantic_checked") is True,
                    "exceptions_checked": data.get("exceptions_checked") is True,
                    "coverage_acknowledged": data.get("coverage_acknowledged") is True,
                },
            )
            return review, {
                "review_id": review["id"],
                "target_id": item["id"],
                "revision": item["revision"],
                "decision": review["decision"],
                "rationale_fingerprint": self.fingerprint(rationale),
            }

        if operation == "publish":
            actor.require("publisher")
            purposes = data["purposes"]
            require(isinstance(purposes, list) and purposes, "purposes_required", 422)
            require(
                timestamp(data["valid_until"]) > datetime.now(timezone.utc),
                "release_expired",
            )
            require(
                isinstance(data["ids"], list) and 0 < len(data["ids"]) <= 500,
                "invalid_release_members",
                422,
            )
            members = []
            for id_ in sorted(set(data["ids"])):
                item = self._get(db, actor, id_, "recommendation")
                for purpose in purposes:
                    actor.permit(purpose)
                    reasons = self._eligibility(db, actor, item, purpose)
                    require(
                        not reasons, reasons[0] if reasons else "publication_blocked"
                    )
                review = self._latest_review(db, actor, item)
                members.append(
                    {
                        "id": id_,
                        "revision": item["revision"],
                        "hash": item["hash"],
                        "approval_id": review["id"],
                    }
                )
            result = self._put(
                db,
                actor,
                "release",
                {
                    "members": members,
                    "manifest_hash": digest(members),
                    "purposes": purposes,
                    "valid_until": data["valid_until"],
                    "published_by": actor.id,
                    "at": utcnow(),
                },
            )
            return result, {
                "release_id": result["id"],
                "release_hash": result["manifest_hash"],
                "members": members,
                "purposes": purposes,
            }

        if operation == "revoke":
            actor.require("clinician", "publisher", "operator")
            target = self._get(db, actor, data["id"])
            require(
                target["kind"] in ("source", "recommendation", "release"),
                "invalid_revocation_target",
                422,
            )
            require(
                bool(data.get("reason", "").strip()), "revocation_reason_required", 422
            )
            result = self._put(
                db,
                actor,
                "revocation",
                {
                    "target_id": target["id"],
                    "reason": data["reason"][:5000],
                    "actor": actor.id,
                    "at": utcnow(),
                },
            )
            return result, {
                "revocation_id": result["id"],
                "target_id": target["id"],
                "reason_fingerprint": self.fingerprint(result["reason"]),
            }

        if operation == "conflict":
            actor.require("clinician")
            if data.get("status") == "resolved":
                require(
                    actor.clinical_authority, "clinical_authority_not_provisioned", 403
                )
            require(
                bool(data.get("reason", "").strip()), "conflict_reason_required", 422
            )
            require(
                data.get("status", "open") in ("open", "resolved"),
                "invalid_conflict_status",
                422,
            )
            if data.get("id"):
                old = self._get(db, actor, data["id"], "conflict")
                require(
                    data["expected_revision"] == old["revision"], "revision_conflict"
                )
                targets = old["targets"]
            else:
                require(
                    data.get("status", "open") == "open",
                    "new_conflict_must_be_open",
                    422,
                )
                targets = data["targets"]
                require(
                    isinstance(targets, list) and targets,
                    "conflict_targets_required",
                    422,
                )
                for id_ in targets:
                    self._get(db, actor, id_, "recommendation")
            result = self._put(
                db,
                actor,
                "conflict",
                {
                    "targets": targets,
                    "status": data.get("status", "open"),
                    "reason": data["reason"][:5000],
                    "actor": actor.id,
                    "at": utcnow(),
                },
                data.get("id"),
            )
            return result, {
                "conflict_id": result["id"],
                "targets": targets,
                "status": result["status"],
                "reason_fingerprint": self.fingerprint(result["reason"]),
            }

        if operation == "policy":
            actor.require("operator")
            current = self._policy(db, actor)
            require(
                data["expected_revision"] == current["revision"],
                "policy_revision_conflict",
            )
            policy = ReviewPolicy.model_validate(data["policy"])
            result = self._put(
                db, actor, "policy", policy.model_dump(), "review_policy"
            )
            return result, {
                "policy_revision": result["revision"],
                **policy.model_dump(),
            }

        if operation == "session":
            actor.require("requester", "clinician")
            actor.permit(data["purpose"])
            release = self._release(db, actor, data["release_id"], data["purpose"])
            policy = self._policy(db, actor)
            result = self._put(
                db,
                actor,
                "session",
                {
                    "owner": actor.id,
                    "release_id": release["id"],
                    "release_hash": release["manifest_hash"],
                    "purpose": data["purpose"],
                    "epoch": 0,
                    "input_sequence": 0,
                    "closed": False,
                    "expires_at": (
                        datetime.now(timezone.utc)
                        + timedelta(seconds=policy["session_ttl_seconds"])
                    ).isoformat(),
                },
            )
            return result, {
                "session_id": result["id"],
                "release_id": release["id"],
                "release_hash": release["manifest_hash"],
                "purpose": data["purpose"],
            }

        if operation in ("interrupt", "close_session"):
            actor.require("requester", "clinician")
            session = self._session(db, actor, data["id"])
            require(session["owner"] == actor.id, "session_owner_required", 403)
            require(session["epoch"] == data["epoch"], "stale_epoch")
            payload = {
                key: value
                for key, value in session.items()
                if key not in ("id", "revision", "kind", "hash")
            }
            payload.update(
                epoch=session["epoch"] + 1, closed=operation == "close_session"
            )
            result = self._put(db, actor, "session", payload, session["id"])
            return result, {
                "session_id": result["id"],
                "epoch": result["epoch"],
                "closed": result["closed"],
            }

        if operation == "retrieve":
            return self._retrieve(db, actor, data)
        if operation == "review_retrieval":
            actor.require("clinician")
            require(actor.clinical_authority, "clinical_authority_not_provisioned", 403)
            retrieval = self._get(db, actor, data["id"], "retrieval")
            require(retrieval["selected"], "no_evidence_to_review")
            require(
                data["decision"] in ("approved", "rejected")
                and bool(data.get("rationale", "").strip()),
                "review_decision_and_reason_required",
                422,
            )
            if data["decision"] == "approved":
                require(
                    data.get("context_reviewed") is True,
                    "request_context_review_required",
                )
            self._revalidate(db, actor, retrieval)
            result = self._put(
                db,
                actor,
                "request_review",
                {
                    "retrieval_id": retrieval["id"],
                    "retrieval_hash": retrieval["hash"],
                    "actor": actor.id,
                    "decision": data["decision"],
                    "rationale": data["rationale"][:5000],
                    "at": utcnow(),
                },
            )
            return result, {
                "request_review_id": result["id"],
                "retrieval_id": retrieval["id"],
                "decision": result["decision"],
                "rationale_fingerprint": self.fingerprint(result["rationale"]),
            }
        if operation == "answer":
            return self._answer(db, actor, data)
        if operation == "delivery":
            actor.require("requester", "clinician")
            answer = self._get(db, actor, data["id"], "answer")
            retrieval = self._get(db, actor, answer["retrieval_id"], "retrieval")
            require(retrieval["owner"] == actor.id, "request_owner_required", 403)
            require(
                data["status"] in ("displayed", "played", "interrupted", "unconfirmed"),
                "invalid_delivery_status",
                422,
            )
            # A late delivery report is an observation, not new authorization.
            # Keep it even if it reveals that a stale output was actually played.
            stale_reason = None
            try:
                self._revalidate(db, actor, retrieval)
            except GovernanceError as error:
                stale_reason = error.code
            result = self._put(
                db,
                actor,
                "delivery",
                {
                    "answer_id": answer["id"],
                    "status": data["status"],
                    "reported_by": actor.id,
                    "basis": "client_report_not_proof_of_comprehension",
                    "stale_at_receipt": stale_reason is not None,
                    "stale_reason": stale_reason,
                    "possible_unsafe_delivery": stale_reason is not None
                    and data["status"] in ("displayed", "played"),
                    "at": utcnow(),
                },
            )
            return result, {
                "delivery_id": result["id"],
                "answer_id": answer["id"],
                "status": result["status"],
                "basis": result["basis"],
                "stale_reason": stale_reason,
                "possible_unsafe_delivery": result["possible_unsafe_delivery"],
            }
        if operation == "catalog":
            actor.require(
                "requester",
                "editor",
                "clinician",
                "publisher",
                "operator",
                "auditor",
                "evaluator",
                "language_reviewer",
                "usability_reviewer",
            )
            releases = []
            for release in self._list(db, actor, "release"):
                if self._revoked(db, actor, release["id"]) or timestamp(
                    release["valid_until"]
                ) <= datetime.now(timezone.utc):
                    continue
                purposes = sorted(set(release["purposes"]).intersection(actor.purposes))
                if purposes:
                    releases.append(
                        {
                            "id": release["id"],
                            "manifest_hash": release["manifest_hash"],
                            "purposes": purposes,
                            "valid_until": release["valid_until"],
                            "member_count": len(release["members"]),
                        }
                    )
            return {
                "releases": releases,
                "actor": {
                    "id": actor.id,
                    "roles": sorted(actor.roles),
                    "clinical_authority": actor.clinical_authority,
                },
                "limits": "Release listing is not a clinical applicability check.",
            }, {"release_count": len(releases)}
        if operation == "get_retrieval":
            actor.require("requester", "clinician")
            retrieval = self._get(db, actor, data["id"], "retrieval")
            require(
                retrieval["owner"] == actor.id or "clinician" in actor.roles,
                "request_owner_required",
                403,
            )
            self._revalidate(db, actor, retrieval)
            return self._retrieval_view(db, actor, retrieval), {
                "retrieval_id": retrieval["id"]
            }
        if operation == "snapshot":
            actor.require("editor", "clinician", "publisher", "auditor", "operator")
            kinds = (
                "source",
                "recommendation",
                "review",
                "release",
                "revocation",
                "conflict",
                "request_review",
                "validation",
                "import_candidate",
                "kl4a_import",
            )
            objects = []
            for kind in kinds:
                for item in self._list(db, actor, kind):
                    if kind == "recommendation":
                        item["checks"] = self._checks(db, actor, item)
                    objects.append(item)
            if "clinician" in actor.roles:
                objects.extend(
                    self._retrieval_view(db, actor, item)
                    for item in self._list(db, actor, "retrieval")
                )
            if actor.roles.intersection({"operator", "billing"}):
                for kind in ("budget_policy", "rate", "reservation", "settlement"):
                    objects.extend(self._list(db, actor, kind))
            audit = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM audit WHERE tenant=? ORDER BY sequence DESC LIMIT 100",
                    (actor.tenant,),
                )
            ]
            return {
                "objects": objects,
                "policy": self._policy(db, actor),
                "audit": audit,
                "actor": {
                    "id": actor.id,
                    "roles": sorted(actor.roles),
                    "clinical_authority": actor.clinical_authority,
                },
            }, {"object_count": len(objects)}
        if operation == "history":
            actor.require("editor", "clinician", "publisher", "auditor")
            current = self._get(db, actor, data["id"])
            result = [
                self._get(db, actor, data["id"], revision=rev)
                for rev in range(1, current["revision"] + 1)
            ]
            return {"versions": result}, {"target_id": data["id"], "count": len(result)}
        if operation == "source_bytes":
            actor.require("editor", "clinician", "publisher")
            source = self._get(db, actor, data["id"], "source")
            blob = db.execute(
                "SELECT content FROM blobs WHERE tenant=? AND id=?",
                (actor.tenant, source["id"]),
            ).fetchone()
            require(
                blob is not None
                and hashlib.sha256(blob[0]).hexdigest() == source["original_sha256"],
                "source_integrity_failure",
                503,
            )
            return {"content": blob[0], "media_type": source["media_type"]}, {
                "source_id": source["id"],
                "original_sha256": source["original_sha256"],
            }
        if operation == "purge_payloads":
            actor.require("operator")
            count = db.execute(
                "DELETE FROM payloads WHERE tenant=? AND expires_at<=?",
                (actor.tenant, utcnow()),
            ).rowcount
            return {"deleted": count}, {"deleted": count, "reason": "retention_expired"}
        if operation == "request_payload":
            actor.require("clinician", "requester")
            retrieval = self._get(db, actor, data["id"], "retrieval")
            require(
                retrieval["owner"] == actor.id or "clinician" in actor.roles,
                "request_owner_required",
                403,
            )
            row = db.execute(
                "SELECT payload,expires_at FROM payloads WHERE tenant=? AND id=?",
                (actor.tenant, retrieval["id"]),
            ).fetchone()
            require(
                row is not None
                and timestamp(row["expires_at"]) > datetime.now(timezone.utc),
                "payload_not_retained",
                404,
            )
            return {
                "payload": json.loads(row["payload"]),
                "expires_at": row["expires_at"],
            }, {"retrieval_id": retrieval["id"]}
        if operation == "audit_verify":
            actor.require("auditor", "operator")
            return {"valid": True, "scope": "stored_chain_not_external_checkpoint"}, {
                "valid": True
            }
        # Additional ledgers/workflows share the same mandatory transaction.
        if operation.startswith("budget_"):
            from .budget import dispatch_budget

            return dispatch_budget(self, db, actor, operation, data)
        if operation.startswith("validation_"):
            from .validation import dispatch_validation

            return dispatch_validation(self, db, actor, operation, data)
        raise GovernanceError("unknown_operation", 404)

    def _needs_review(self, policy, request_id, selected, requested=False):
        if requested:
            return True
        mode = policy["mode"]
        if mode == "required":
            return True
        if mode == "sampled":
            return (
                int(self.fingerprint(request_id)[:8], 16) % 100
                < policy["sample_percent"]
            )
        return (
            mode == "policy_triggered"
            and policy["trigger_high_risk"]
            and any(item["high_risk"] for item in selected)
        )

    def _retrieve(self, db, actor, data):
        actor.require("requester", "clinician")
        session = self._session(db, actor, data["session_id"])
        require(session["owner"] == actor.id, "session_owner_required", 403)
        require(data["epoch"] == session["epoch"], "stale_epoch")
        release = self._release(db, actor, session["release_id"], session["purpose"])
        query, context = data["query"], data["context"]
        require(
            isinstance(query, str)
            and 0 < len(query.strip()) <= 4000
            and isinstance(context, dict)
            and len(context) <= 100,
            "invalid_query_context",
            422,
        )
        require(len(canonical(context)) <= 20000, "context_too_large", 422)
        request_id = data["request_id"]
        require(
            isinstance(request_id, str) and 0 < len(request_id) <= 100,
            "request_id_required",
            422,
        )
        fingerprint = self.fingerprint(data)
        previous = db.execute(
            "SELECT * FROM idempotency WHERE tenant=? AND actor=? AND key=?",
            (actor.tenant, actor.id, request_id),
        ).fetchone()
        if previous:
            require(previous["fingerprint"] == fingerprint, "idempotency_conflict")
            retrieval = self._get(db, actor, previous["result_id"], "retrieval")
            self._revalidate(db, actor, retrieval)
            return self._retrieval_view(db, actor, retrieval), {
                "retrieval_id": retrieval["id"],
                "replayed": True,
            }
        policy = self._policy(db, actor)
        session_payload = {
            key: value
            for key, value in session.items()
            if key not in ("id", "revision", "kind", "hash")
        }
        session_payload["input_sequence"] = session.get("input_sequence", 0) + 1
        session = self._put(db, actor, "session", session_payload, session["id"])
        selected, decisions = [], []
        query_terms = terms(query)
        for member in release["members"]:
            item = self._get(
                db, actor, member["id"], "recommendation", member["revision"]
            )
            require(
                item["hash"] == member["hash"], "release_member_integrity_failure", 503
            )
            reasons = self._eligibility(
                db, actor, item, session["purpose"], member["approval_id"]
            )
            match = bool(
                query_terms.intersection(
                    set().union(*(terms(keyword) for keyword in item["keywords"]))
                )
            )
            if not match:
                reasons.append("query_no_match")
            applied = applicability(
                Condition.model_validate(item["condition"]), context
            )
            if applied["status"] != "applies":
                reasons.append("applicability_" + applied["status"])
            decisions.append(
                {
                    "id": item["id"],
                    "revision": item["revision"],
                    "status": "excluded" if reasons else "included",
                    "reasons": sorted(set(reasons)),
                    "applicability": applied,
                }
            )
            if not reasons:
                selected.append({**member, "high_risk": item["high_risk"]})
        review_required = bool(selected) and self._needs_review(
            policy,
            f"{actor.tenant}:{session['id']}:{request_id}",
            selected,
            data.get("request_review") is True,
        )
        status = (
            "pending_review"
            if review_required
            else "evidence"
            if selected
            else "no_applicable_evidence"
        )
        if not selected and all(
            "query_no_match" in decision["reasons"] for decision in decisions
        ):
            status = "no_match"
        payload = {
            "session_id": session["id"],
            "epoch": session["epoch"],
            "input_sequence": session["input_sequence"],
            "owner": actor.id,
            "purpose": session["purpose"],
            "release_id": release["id"],
            "release_hash": release["manifest_hash"],
            "selected": selected,
            "decisions": decisions,
            "query_fingerprint": self.fingerprint(query),
            "context_fingerprint": self.fingerprint(context),
            "context_fields": sorted(context),
            "language": data.get("language", "en"),
            "request_fingerprint": fingerprint,
            "review_mode": policy["mode"],
            "policy_revision": policy["revision"],
            "review_required": review_required,
            "review_requested": data.get("request_review") is True,
            "status": status,
            "retriever_version": "approved-keyword-v1",
            "at": utcnow(),
        }
        retrieval = self._put(db, actor, "retrieval", payload)
        db.execute(
            "INSERT INTO idempotency VALUES(?,?,?,?,?)",
            (actor.tenant, actor.id, request_id, fingerprint, retrieval["id"]),
        )
        if data.get("consent_to_retain") is True and policy["retain_request_seconds"]:
            expiry = (
                datetime.now(timezone.utc)
                + timedelta(seconds=policy["retain_request_seconds"])
            ).isoformat()
            db.execute(
                "INSERT INTO payloads VALUES(?,?,?,?,?)",
                (
                    actor.tenant,
                    retrieval["id"],
                    actor.id,
                    expiry,
                    canonical({"query": query, "context": context}),
                ),
            )
        return self._retrieval_view(db, actor, retrieval), {
            "retrieval_id": retrieval["id"],
            **payload,
        }

    def _revalidate(self, db, actor, retrieval):
        session = self._session(db, actor, retrieval["session_id"])
        require(session["epoch"] == retrieval["epoch"], "stale_epoch")
        require(
            session.get("input_sequence", 0) == retrieval.get("input_sequence", 0),
            "stale_input_sequence",
        )
        require(
            session["release_id"] == retrieval["release_id"]
            and session["release_hash"] == retrieval["release_hash"],
            "session_release_mismatch",
        )
        release = self._release(
            db, actor, retrieval["release_id"], retrieval["purpose"]
        )
        require(
            release["manifest_hash"] == retrieval["release_hash"],
            "release_integrity_failure",
            503,
        )
        for selected in retrieval["selected"]:
            require(
                any(
                    all(
                        member[key] == selected[key]
                        for key in ("id", "revision", "hash", "approval_id")
                    )
                    for member in release["members"]
                ),
                "release_membership_mismatch",
            )
            item = self._get(
                db, actor, selected["id"], "recommendation", selected["revision"]
            )
            reasons = self._eligibility(
                db, actor, item, retrieval["purpose"], selected["approval_id"]
            )
            require(not reasons, reasons[0] if reasons else "evidence_ineligible")
        return session

    def _review_status(self, db, actor, retrieval):
        reviews = [
            review
            for review in self._list(db, actor, "request_review")
            if review["retrieval_id"] == retrieval["id"]
            and review["retrieval_hash"] == retrieval["hash"]
        ]
        return (
            reviews[-1]["decision"]
            if reviews
            else "pending"
            if retrieval["review_required"]
            else "not_required"
        )

    def _retrieval_view(self, db, actor, retrieval):
        review_status = self._review_status(db, actor, retrieval)
        visible = review_status in ("approved", "not_required")
        return {
            **retrieval,
            "selected": retrieval["selected"] if visible else [],
            "review_status": review_status,
            "status": "rejected"
            if review_status == "rejected"
            else "pending_review"
            if review_status == "pending"
            else "evidence"
            if retrieval["selected"]
            else retrieval["status"],
        }

    def _answer(self, db, actor, data):
        actor.require("requester", "clinician")
        retrieval = self._get(db, actor, data["id"], "retrieval")
        require(retrieval["owner"] == actor.id, "request_owner_required", 403)
        self._revalidate(db, actor, retrieval)
        policy = self._policy(db, actor)
        # A policy change requires new retrieval, avoiding silent weakened review.
        require(
            policy["revision"] == retrieval["policy_revision"],
            "review_policy_changed_retrieve_again",
        )
        review_status = self._review_status(db, actor, retrieval)
        if review_status in ("pending", "rejected"):
            return {
                "status": "pending_review"
                if review_status == "pending"
                else "rejected",
                "answer": "",
                "sources": [],
            }, {"retrieval_id": retrieval["id"], "review_status": review_status}
        if not retrieval["selected"]:
            return {
                "status": retrieval["status"],
                "answer": "No applicable approved information was found. Please contact your clinical team.",
                "sources": [],
            }, {"retrieval_id": retrieval["id"], "disposition": "abstained"}
        claims = []
        for selected in retrieval["selected"]:
            item = self._get(
                db, actor, selected["id"], "recommendation", selected["revision"]
            )
            wording = item["explanations"].get(retrieval["language"])
            if not wording:
                return {
                    "status": "language_not_reviewed",
                    "answer": "",
                    "sources": [],
                }, {
                    "retrieval_id": retrieval["id"],
                    "disposition": "language_not_reviewed",
                }
            claims.append(
                {
                    "recommendation_id": item["id"],
                    "revision": item["revision"],
                    "text": wording,
                    "evidence": item["evidence"],
                    "approval_id": selected["approval_id"],
                }
            )
        payload = {
            "retrieval_id": retrieval["id"],
            "claims": claims,
            "answer": "\n\n".join(claim["text"] for claim in claims),
            "generation": "reviewed-wording-v1",
            "review_status": review_status,
            "delivery": "prepared_not_confirmed",
            "clinical_action_authorized": False,
            "at": utcnow(),
        }
        answer = self._put(db, actor, "answer", payload)
        return {
            **answer,
            "status": "success",
            "sources": [anchor for claim in claims for anchor in claim["evidence"]],
        }, {
            "answer_id": answer["id"],
            "retrieval_id": retrieval["id"],
            "claim_links": [
                {
                    "recommendation_id": c["recommendation_id"],
                    "revision": c["revision"],
                    "source_ids": [e["source_id"] for e in c["evidence"]],
                }
                for c in claims
            ],
            "answer_fingerprint": self.fingerprint(payload["answer"]),
            "generation": payload["generation"],
            "review_status": review_status,
            "delivery": payload["delivery"],
            "clinical_action_authorized": False,
        }

    def _checks(self, db, actor, item):
        review = self._latest_review(db, actor, item)
        reasons = self._eligibility(db, actor, item, item["purposes"][0])
        checks = []
        for key, label, failures in (
            (
                "evidence",
                "Exact source evidence",
                {"evidence_mismatch", "page_anchor_mismatch", "block_anchor_mismatch"},
            ),
            (
                "current",
                "Current source and recommendation",
                {"superseded_revision", "source_superseded"},
            ),
            (
                "active",
                "Not revoked or expired",
                {
                    "source_revoked",
                    "recommendation_revoked",
                    "source_permission_expired",
                    "recommendation_expired",
                },
            ),
            (
                "rights",
                "Source permission recorded",
                {
                    "source_purpose_not_permitted",
                    "recommendation_purpose_not_permitted",
                },
            ),
            ("conflict", "No recorded blocking conflict", {"blocking_conflict"}),
        ):
            failed = sorted(failures.intersection(reasons))
            checks.append(
                {
                    "key": key,
                    "label": label,
                    "status": "fail" if failed else "pass",
                    "reasons": failed
                    or ["Named automated check passed; this is not clinical approval."],
                }
            )
        checks.append(
            {
                "key": "approval",
                "label": "Clinical approval for this revision",
                "status": "pass"
                if review and review["decision"] == "approved"
                else "fail"
                if review and review["decision"] == "rejected"
                else "review",
                "reasons": [
                    "Authenticated clinical decision recorded."
                    if review
                    else "Qualified clinical review is pending."
                ],
                "review_id": review["id"] if review else None,
            }
        )
        checks.append(
            {
                "key": "applicability",
                "label": "Applicability to a request",
                "status": "review",
                "reasons": [
                    "Checked at retrieval with explicit context; missing variables remain unknown."
                ],
            }
        )
        return checks
