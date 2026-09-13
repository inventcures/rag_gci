"""Synthetic import, reservation and validation tests with no paid API calls."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import io
import zipfile

import pytest

from clinical_governance import GovernanceError
from clinical_governance.ingestion import ingest_pdf, ingest_text, render_pdf_page
from clinical_governance.kl4a import inspect_bundle, frontmatter
from test_clinical_governance import actors, store, released, FUTURE, retrieve


def bundle_files():
    text = "Only a synthetic योजना test."
    return {
        "index.md": '---\nokf_version: "0.2"\nvendor_field: retained\n---\nSynthetic root',
        "manifest.yaml": 'okf_version: "0.2"\nprofile_version: "0.2.0"\nid: fixture\nunknown: preserve\n',
        "sources/demo.md": '---\ntype: SOP Source\ntitle: Synthetic source\nsopkb:\n  source_id: demo\n  normalized_path: sources/normalized/demo__v1.md\n  original_path: sources/originals/demo.txt\n  source_version_id: demo__v1\n---\nSource body',
        "sources/normalized/demo__v1.md": text,
        "sources/originals/demo.txt": text,
        "knowledge/item.md": '---\ntype: SOP Knowledge Piece\ntitle: Synthetic rule\ncustom:\n  retained: true\nsopkb:\n  source_id: demo\n  knowledge_item_id: item\n  review_status: approved\n  evidence: ../evidence/ev.md\n---\nUntrusted proposed meaning.',
        "evidence/ev.md": f'---\ntype: SOP Evidence\nsopkb:\n  source_id: demo\n  knowledge_item_id: item\n  span_status: exact\n  start_pos: 0\n  end_pos: {len(text)}\n---\n{text}',
        ".sopkb/items.json": '[{"statement":"Do not import cache authority","status":"approved"}]',
    }


def make_zip(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return stream.getvalue()


def test_kl4a_unknown_fields_and_originals_preserved():
    original = make_zip(bundle_files())
    result = inspect_bundle(original)
    assert result["archive_bytes"] == original
    assert result["manifest"]["unknown"] == "preserve"
    assert result["candidates"][0]["metadata"]["frontmatter"]["custom"]["retained"] is True
    assert result["candidates"][0]["clinical_approval"] is False
    assert result["candidates"][0]["local_status"] == "unreviewed_import"
    assert result["candidates"][0]["upstream_review_status"] == "approved"


def test_import_never_creates_local_approved_recommendations(store, actors):
    result = store.execute(actors["clinical"], "import_bundle", {"content": make_zip(bundle_files()),
        "rights": {"authorized": True, "authorization_reference": "synthetic author", "purposes": ["information"], "valid_until": FUTURE}})
    objects = store.execute(actors["clinical"], "snapshot", {})["objects"]
    assert result["candidate_ids"] and result["clinical_approval"] is False
    assert not [item for item in objects if item["kind"] == "recommendation"]


@pytest.mark.parametrize("path", ["../outside", "/absolute", "sources/../../escape", "sources\\escape"])
def test_zip_path_traversal_rejected(path):
    files = bundle_files()
    files[path] = "untrusted"
    with pytest.raises(GovernanceError, match="path"):
        inspect_bundle(make_zip(files))


def test_frontmatter_duplicate_keys_and_aliases_rejected():
    with pytest.raises(GovernanceError, match="ambiguous_frontmatter_key"):
        frontmatter(b"---\ntitle: one\ntitle: two\n---\n")
    with pytest.raises(GovernanceError, match="yaml_aliases"):
        frontmatter(b"---\nx: &x [1]\ny: *x\n---\n")


def test_bad_offsets_are_not_repaired_silently():
    files = bundle_files()
    files["evidence/ev.md"] = files["evidence/ev.md"].replace("start_pos: 0", "start_pos: 900")
    with pytest.raises(GovernanceError, match="codepoint_interval"):
        inspect_bundle(make_zip(files))


def test_text_preserves_combining_marks_and_original_bytes():
    original = "योजना\r\ne\u0301 and é".encode()
    parsed = ingest_text(original)
    assert parsed["original"] == original
    assert parsed["normalized_text"].encode() == original


def test_pdf_bytes_page_maps_and_review_gaps():
    import pymupdf
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Synthetic protocol: ask about the agreed plan.")
        page.insert_text((72, 140), "Footnote: synthetic encounters only.")
        content = document.tobytes()
    parsed = ingest_pdf(content)
    assert parsed["original"] == content
    assert parsed["pages"][0]["page"] == 1
    assert parsed["blocks"]
    assert "completeness_require_review" in parsed["coverage_gaps"][0]
    assert render_pdf_page(content, 1).startswith(b"\x89PNG")
    for block in parsed["blocks"]:
        assert parsed["normalized_text"][block["start"]:block["end"]].strip()


def configure_budget(store, actor, session_limit=100, monthly_limit=100):
    store.execute(actor, "budget_configure", {"expected_revision": 0, "session_limit_paise": session_limit, "monthly_limit_paise": monthly_limit})
    return store.execute(actor, "budget_rate", {"service": "synthetic-tts", "version": "test-1", "unit": "character",
        "paise_per_unit": "0.5", "valid_until": FUTURE, "billing_reference": "Synthetic rate, not provider pricing"})


def reserve(store, actors, released, rate, key="paid-job-1", units=100):
    return store.execute(actors["user"], "budget_reserve", {"session_id": released[3]["id"], "epoch": 0,
        "request_id": key, "charges": [{"rate_id": rate["id"], "max_units": units}]})


def test_budget_unconfigured_is_not_unlimited(store, actors, released):
    with pytest.raises(GovernanceError, match="inr_budget_not_configured"):
        reserve(store, actors, released, {"id": "missing"})


def test_atomic_reservations_block_concurrent_overspend(store, actors, released):
    rate = configure_budget(store, actors["clinical"], session_limit=50, monthly_limit=50)
    def attempt(key):
        try:
            return reserve(store, actors, released, rate, key)
        except GovernanceError as exc:
            return exc.code
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(attempt, ["a", "b", "c", "d"]))
    assert sum(isinstance(result, dict) for result in outcomes) == 1
    assert outcomes.count("monthly_inr_budget_exhausted") == 3


def test_reservation_replay_and_reconciliation(store, actors, released):
    rate = configure_budget(store, actors["clinical"])
    first = reserve(store, actors, released, rate)
    assert reserve(store, actors, released, rate)["id"] == first["id"]
    result = store.execute(actors["clinical"], "budget_reconcile", {"id": first["id"], "actual_paise": 20,
        "basis": "provider_usage", "reference": "synthetic-usage", "expected_revision": 0})
    assert result["actual_paise"] == 20
    usage = store.execute(actors["user"], "budget_status", {"session_id": released[3]["id"]})
    assert usage["monthly_committed_paise"] == 20
    assert usage["unreconciled_reserved_paise"] == 0


def test_uncertain_or_cancelled_jobs_remain_reserved(store, actors, released):
    rate = configure_budget(store, actors["clinical"])
    first = reserve(store, actors, released, rate)
    with pytest.raises(GovernanceError):
        store.execute(actors["clinical"], "budget_reconcile", {"id": first["id"], "actual_paise": 0,
            "basis": "timeout_assumed_free", "reference": "synthetic", "expected_revision": 0})
    usage = store.execute(actors["user"], "budget_status", {"session_id": released[3]["id"]})
    assert usage["unreconciled_reserved_paise"] == 50


def test_billing_overrun_is_recorded_not_hidden(store, actors, released):
    rate = configure_budget(store, actors["clinical"])
    first = reserve(store, actors, released, rate)
    result = store.execute(actors["clinical"], "budget_reconcile", {"id": first["id"], "actual_paise": 110,
        "basis": "provider_invoice", "reference": "synthetic_invoice", "expected_revision": 0})
    assert result["over_reserved_amount"]
    with pytest.raises(GovernanceError, match="budget_exhausted"):
        reserve(store, actors, released, rate, "another", 1)


def test_requester_cannot_fake_bill(store, actors, released):
    rate = configure_budget(store, actors["clinical"])
    first = reserve(store, actors, released, rate)
    with pytest.raises(GovernanceError, match="role_not_authorized"):
        store.execute(actors["user"], "budget_reconcile", {"id": first["id"], "actual_paise": 0,
            "basis": "confirmed_no_charge", "reference": "fake", "expected_revision": 0})


def test_engineering_test_does_not_grant_clinical_approval(store, actors, released):
    result = store.execute(actors["clinical"], "validation_record", {"release_id": released[2]["id"],
        "release_hash": released[2]["manifest_hash"], "category": "engineering", "performed": True,
        "outcome": "pass", "evidence_reference": "synthetic fixture test run", "limitations": "Software test only"})
    assert result["clinical_approval_granted"] is False
    status = store.execute(actors["clinical"], "validation_status", {"release_id": released[2]["id"]})
    assert "clinical" in status["missing_categories"] and "language" in status["missing_categories"]
    store.execute(actors["clinical"], "revoke", {"id": released[2]["id"], "reason": "Synthetic withdrawal"})
    assert not store.execute(actors["clinical"], "validation_status", {"release_id": released[2]["id"]})["current_evidence_eligible"]


def test_unperformed_validation_cannot_be_recorded_as_pass(store, actors, released):
    with pytest.raises(GovernanceError, match="validation_must_have_been_performed"):
        store.execute(actors["clinical"], "validation_record", {"release_id": released[2]["id"],
            "release_hash": released[2]["manifest_hash"], "category": "clinical", "performed": False,
            "outcome": "pass", "evidence_reference": "not real", "limitations": "not done"})
