"""Synthetic governance regression cases. No provider or real clinical data."""

from datetime import datetime, timedelta, timezone
import json
import sqlite3

import pytest

from clinical_governance import Actor, GovernanceError, GovernanceStore
from clinical_governance.models import Condition, applicability, terms, unicode_offsets


FUTURE = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()


@pytest.fixture
def actors():
    return {
        "clinical": Actor(
            "reviewer-1",
            "test-site",
            frozenset(
                {"editor", "clinician", "publisher", "operator", "auditor", "requester"}
            ),
            clinical_authority=True,
        ),
        "user": Actor("caller-1", "test-site", frozenset({"requester"})),
        "other": Actor(
            "caller-2",
            "other-site",
            frozenset({"requester", "operator", "clinician"}),
            clinical_authority=True,
        ),
    }


@pytest.fixture
def store(tmp_path):
    return GovernanceStore(
        tmp_path / "governance.sqlite3", b"synthetic-test-key-only-not-a-secret" * 2
    )


def source(store, actor, **updates):
    text = "Synthetic protocol only. Ask about the agreed plan.\nFootnote: only in a synthetic encounter."
    data = {
        "original": text.encode(),
        "normalized_text": text,
        "title": "Synthetic protocol",
        "version": "1",
        "rights": {
            "authorized": True,
            "authorization_reference": "authored synthetic fixture",
            "purposes": ["information"],
            "valid_until": FUTURE,
        },
    }
    data.update(updates)
    return store.execute(actor, "source", data)


def draft(src):
    return {
        "statement": "Ask about the synthetic agreed plan.",
        "purposes": ["information"],
        "evidence": [
            {
                "source_id": src["id"],
                "normalized_sha256": src["normalized_sha256"],
                "start": 0,
                "end": len(src["normalized_text"]),
                "excerpt": src["normalized_text"],
            }
        ],
        "condition": {"op": "eq", "field": "synthetic_encounter", "value": True},
        "keywords": ["plan", "योजना"],
        "explanations": {
            "en": "Synthetic demonstration: ask about the agreed plan.",
            "hi": "केवल परीक्षण: योजना के बारे में पूछें।",
        },
        "valid_until": FUTURE,
    }


def approve(store, actor, item, decision="approved"):
    return store.execute(
        actor,
        "review",
        {
            "id": item["id"],
            "expected_revision": item["revision"],
            "decision": decision,
            "rationale": "Synthetic fixture reviewed for software testing only.",
            "semantic_checked": True,
            "exceptions_checked": True,
            "coverage_acknowledged": True,
        },
    )


def publish(store, actor, item):
    return store.execute(
        actor,
        "publish",
        {"ids": [item["id"]], "purposes": ["information"], "valid_until": FUTURE},
    )


@pytest.fixture
def released(store, actors):
    actor = actors["clinical"]
    src = source(store, actor)
    item = store.execute(actor, "propose", {"draft": draft(src)})
    approve(store, actor, item)
    release = publish(store, actor, item)
    session = store.execute(
        actors["user"],
        "session",
        {"release_id": release["id"], "purpose": "information"},
    )
    return src, item, release, session


def retrieve(store, actors, released, **updates):
    data = {
        "session_id": released[3]["id"],
        "epoch": 0,
        "request_id": "test-request-1",
        "query": "Explain the plan",
        "context": {"synthetic_encounter": True},
        "language": "en",
    }
    data.update(updates)
    return store.execute(actors["user"], "retrieve", data)


def test_complete_release_retrieval_answer_and_audit(store, actors, released):
    result = retrieve(store, actors, released)
    assert result["status"] == "evidence"
    answer = store.execute(actors["user"], "answer", {"id": result["id"]})
    assert answer["status"] == "success"
    assert answer["clinical_action_authorized"] is False
    assert answer["delivery"] == "prepared_not_confirmed"
    assert answer["claims"][0]["evidence"][0]["source_id"] == released[0]["id"]
    assert store.execute(actors["clinical"], "audit_verify", {})["valid"]


def test_proposed_never_publishes(store, actors):
    item = store.execute(
        actors["clinical"],
        "propose",
        {"draft": draft(source(store, actors["clinical"]))},
    )
    with pytest.raises(GovernanceError, match="current_revision_not_approved"):
        publish(store, actors["clinical"], item)


@pytest.mark.parametrize("decision", ["rejected", "deferred"])
def test_subsequent_review_blocks_old_release(store, actors, released, decision):
    approve(store, actors["clinical"], released[1], decision)
    result = retrieve(store, actors, released)
    assert result["selected"] == []
    assert "current_revision_not_approved" in result["decisions"][0]["reasons"]


def test_edit_and_reapproval_do_not_revive_old_release(store, actors, released):
    new_draft = draft(released[0])
    new_draft["statement"] = "Changed synthetic statement."
    edited = store.execute(
        actors["clinical"],
        "edit",
        {"id": released[1]["id"], "expected_revision": 1, "draft": new_draft},
    )
    before = retrieve(store, actors, released)
    assert not before["selected"]
    approve(store, actors["clinical"], edited)
    after = retrieve(store, actors, released, request_id="after-reapproval")
    assert not after["selected"]
    assert "superseded_revision" in after["decisions"][0]["reasons"]
    assert publish(store, actors["clinical"], edited)["members"][0]["revision"] == 2


def test_reapproval_same_revision_requires_new_release(store, actors, released):
    approve(store, actors["clinical"], released[1], "rejected")
    approve(store, actors["clinical"], released[1])
    result = retrieve(store, actors, released)
    assert "approval_changed_republish_required" in result["decisions"][0]["reasons"]


@pytest.mark.parametrize("target_index", [0, 1, 2])
def test_all_revocation_levels_before_answer(store, actors, released, target_index):
    result = retrieve(store, actors, released)
    store.execute(
        actors["clinical"],
        "revoke",
        {"id": released[target_index]["id"], "reason": "Synthetic withdrawal"},
    )
    with pytest.raises(GovernanceError, match="revoked"):
        store.execute(actors["user"], "answer", {"id": result["id"]})


def test_new_source_version_invalidates_old_source(store, actors, released):
    source(store, actors["clinical"], family_id=released[0]["family_id"], version="2")
    result = retrieve(store, actors, released)
    assert "source_superseded" in result["decisions"][0]["reasons"]


def test_no_match_is_not_all_items_fallback(store, actors, released):
    result = retrieve(
        store,
        actors,
        released,
        query="unrelated",
        context={"synthetic_encounter": True},
    )
    assert result["status"] == "no_match"
    assert result["selected"] == []
    assert "query_no_match" in result["decisions"][0]["reasons"]


@pytest.mark.parametrize(
    "context,status",
    [({}, "unknown"), ({"synthetic_encounter": False}, "does_not_apply")],
)
def test_unknown_or_inapplicable_is_not_permission(
    store, actors, released, context, status
):
    result = retrieve(store, actors, released, context=context)
    assert result["selected"] == []
    assert result["decisions"][0]["applicability"]["status"] == status


def test_numeric_boolean_and_missing_negation():
    c = Condition(op="eq", field="flag", value=True)
    assert applicability(c, {"flag": 1})["status"] == "does_not_apply"
    negated = Condition(op="not", children=[c])
    assert applicability(negated, {})["status"] == "unknown"
    assert (
        applicability(Condition(op="gte", field="age", value=18), {"age": "21"})[
            "status"
        ]
        == "unknown"
    )


@pytest.mark.parametrize(
    "mode", ["required", "sampled", "user_requested", "policy_triggered"]
)
def test_configurable_review_modes(store, actors, released, mode):
    store.execute(
        actors["clinical"],
        "policy",
        {"expected_revision": 0, "policy": {"mode": mode, "sample_percent": 100}},
    )
    result = retrieve(
        store,
        actors,
        released,
        request_review=True
        if mode in ("user_requested", "policy_triggered")
        else False,
    )
    assert result["status"] == "pending_review"
    assert result["selected"] == []
    assert (
        store.execute(actors["user"], "answer", {"id": result["id"]})["status"]
        == "pending_review"
    )
    store.execute(
        actors["clinical"],
        "review_retrieval",
        {
            "id": result["id"],
            "decision": "approved",
            "rationale": "Synthetic request",
            "context_reviewed": True,
        },
    )
    assert (
        store.execute(actors["user"], "answer", {"id": result["id"]})["status"]
        == "success"
    )


def test_review_off_still_logs_minimized_query_context(store, actors, released):
    result = retrieve(
        store,
        actors,
        released,
        query="plan private phrase",
        context={"synthetic_encounter": True, "private_value": "DO_NOT_LOG_ME"},
    )
    snapshot = store.execute(actors["clinical"], "snapshot", {})
    raw = json.dumps(snapshot["audit"])
    assert "private phrase" not in raw and "DO_NOT_LOG_ME" not in raw
    entry = next(
        json.loads(row["detail"])
        for row in snapshot["audit"]
        if row["operation"] == "retrieve"
    )
    assert entry["review_mode"] == "off"
    assert entry["query_fingerprint"] and entry["context_fingerprint"]
    assert entry["selected"][0]["id"] == released[1]["id"]
    with pytest.raises(GovernanceError, match="payload_not_retained"):
        store.execute(actors["user"], "request_payload", {"id": result["id"]})


def test_payload_retention_requires_policy_and_consent(store, actors, released):
    store.execute(
        actors["clinical"],
        "policy",
        {"expected_revision": 0, "policy": {"retain_request_seconds": 60}},
    )
    result = retrieve(store, actors, released, consent_to_retain=True)
    assert (
        store.execute(actors["user"], "request_payload", {"id": result["id"]})[
            "payload"
        ]["query"]
        == "Explain the plan"
    )


def test_role_and_tenant_isolation(store, actors, released):
    with pytest.raises(GovernanceError):
        store.execute(
            actors["other"],
            "session",
            {"release_id": released[2]["id"], "purpose": "information"},
        )
    with pytest.raises(GovernanceError, match="role_not_authorized"):
        approve(store, actors["user"], released[1])
    unprovisioned = Actor("bad-reviewer", "test-site", frozenset({"clinician"}))
    with pytest.raises(GovernanceError, match="clinical_authority_not_provisioned"):
        approve(store, unprovisioned, released[1])


def test_evidence_mismatch_not_accepted(store, actors):
    bad = draft(source(store, actors["clinical"]))
    bad["evidence"][0]["excerpt"] = "Invented clinical recommendation"
    with pytest.raises(GovernanceError, match="source_evidence_or_rights_invalid"):
        store.execute(actors["clinical"], "propose", {"draft": bad})


def test_indic_offsets_and_tokens():
    text = "A योजना 🙂 বাংলা e\u0301"
    start, end = text.index("🙂"), text.index("🙂") + 1
    offsets = unicode_offsets(text, start, end)
    assert offsets["utf16"][1] - offsets["utf16"][0] == 2
    assert text.encode()[slice(*offsets["utf8"])].decode() == "🙂"
    assert "योजना" in terms(text) and "বাংলা" in terms(text)


def test_idempotency_revalidates_and_conflict_denied(store, actors, released):
    first = retrieve(store, actors, released)
    assert retrieve(store, actors, released)["id"] == first["id"]
    with pytest.raises(GovernanceError, match="idempotency_conflict"):
        retrieve(store, actors, released, query="other plan")
    store.execute(
        actors["clinical"],
        "revoke",
        {"id": released[1]["id"], "reason": "Synthetic withdrawal"},
    )
    with pytest.raises(GovernanceError, match="revoked"):
        retrieve(store, actors, released)


def test_interrupt_invalidates_queued_answer(store, actors, released):
    result = retrieve(store, actors, released)
    answer = store.execute(actors["user"], "answer", {"id": result["id"]})
    store.execute(actors["user"], "interrupt", {"id": released[3]["id"], "epoch": 0})
    with pytest.raises(GovernanceError, match="stale_epoch"):
        store.execute(actors["user"], "answer", {"id": result["id"]})
    late_report = store.execute(
        actors["user"], "delivery", {"id": answer["id"], "status": "played"}
    )
    assert late_report["stale_at_receipt"] and late_report["possible_unsafe_delivery"]
    assert (
        store.execute(
            actors["user"], "delivery", {"id": answer["id"], "status": "interrupted"}
        )["status"]
        == "interrupted"
    )


def test_conflict_blocks_and_resolution_is_recorded(store, actors, released):
    conflict = store.execute(
        actors["clinical"],
        "conflict",
        {"targets": [released[1]["id"]], "reason": "Synthetic contradictory sources"},
    )
    assert not retrieve(store, actors, released)["selected"]
    store.execute(
        actors["clinical"],
        "conflict",
        {
            "id": conflict["id"],
            "expected_revision": 1,
            "status": "resolved",
            "reason": "Adjudicated synthetic exception",
        },
    )
    assert retrieve(store, actors, released, request_id="resolved")["selected"]


def test_no_silent_language_fallback(store, actors, released):
    result = retrieve(store, actors, released, language="mr")
    assert (
        store.execute(actors["user"], "answer", {"id": result["id"]})["status"]
        == "language_not_reviewed"
    )


def test_review_policy_change_requires_fresh_retrieval(store, actors, released):
    result = retrieve(store, actors, released)
    store.execute(
        actors["clinical"],
        "policy",
        {"expected_revision": 0, "policy": {"mode": "required"}},
    )
    with pytest.raises(GovernanceError, match="review_policy_changed"):
        store.execute(actors["user"], "answer", {"id": result["id"]})


def test_audit_failure_rolls_back_action(store, actors, monkeypatch):
    def failed(*args, **kwargs):
        raise sqlite3.OperationalError("disk full synthetic test")

    monkeypatch.setattr(store, "_audit", failed)
    with pytest.raises(GovernanceError, match="durable_audit_unavailable"):
        source(store, actors["clinical"])
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT count(*) FROM objects").fetchone()[0] == 0


def test_tampered_audit_blocks_governed_operations(store, actors, released):
    with sqlite3.connect(store.path) as db:
        db.execute("DROP TRIGGER immutable_audit_update")
        db.execute("UPDATE audit SET detail='{}' WHERE sequence=1")
    with pytest.raises(GovernanceError, match="audit_integrity_failure"):
        retrieve(store, actors, released)


def test_source_original_is_immutable_and_download_checked(store, actors, released):
    original = store.execute(
        actors["clinical"], "source_bytes", {"id": released[0]["id"]}
    )
    assert original["content"] == released[0]["normalized_text"].encode()
    with sqlite3.connect(store.path) as db, pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE blobs SET content=X'00'")


def test_new_context_rejects_older_background_answer(store, actors, released):
    old = retrieve(store, actors, released)
    corrected = retrieve(
        store,
        actors,
        released,
        request_id="corrected-context",
        context={"synthetic_encounter": False},
    )
    assert corrected["input_sequence"] > old["input_sequence"]
    with pytest.raises(GovernanceError, match="stale_input_sequence"):
        store.execute(actors["user"], "answer", {"id": old["id"]})


def test_conflict_resolution_requires_provisioned_authority(store, actors, released):
    conflict = store.execute(
        actors["clinical"],
        "conflict",
        {"targets": [released[1]["id"]], "reason": "Synthetic conflict"},
    )
    unprovisioned = Actor("unprovisioned", "test-site", frozenset({"clinician"}))
    with pytest.raises(GovernanceError, match="clinical_authority_not_provisioned"):
        store.execute(
            unprovisioned,
            "conflict",
            {
                "id": conflict["id"],
                "expected_revision": 1,
                "status": "resolved",
                "reason": "Attempted override",
            },
        )
