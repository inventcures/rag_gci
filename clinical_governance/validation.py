"""Record engineering evidence separately from actual qualified human review."""

from .models import GovernanceError, utcnow
from .store import require


def dispatch_validation(store, db, actor, operation, data):
    if operation == "validation_record":
        kind = data["category"]
        require(kind in ("engineering", "clinical", "language", "usability"), "invalid_validation_category", 422)
        if kind == "engineering":
            actor.require("evaluator", "operator")
        elif kind == "clinical":
            actor.require("clinician")
            require(actor.clinical_authority, "clinical_authority_not_provisioned", 403)
        else:
            actor.require("language_reviewer" if kind == "language" else "usability_reviewer")
        require(data.get("performed") is True, "validation_must_have_been_performed", 422)
        require(data["outcome"] in ("pass", "fail", "needs_review"), "invalid_validation_outcome", 422)
        require(bool(data.get("evidence_reference")) and bool(data.get("limitations")), "validation_evidence_and_limits_required", 422)
        target = store._get(db, actor, data["release_id"], "release")
        require(data["release_hash"] == target["manifest_hash"], "validation_release_mismatch")
        # A software test cannot automatically approve the clinical release.
        result = store._put(db, actor, "validation", {
            "category": kind, "release_id": target["id"], "release_hash": target["manifest_hash"],
            "outcome": data["outcome"], "actor": actor.id, "performed": True, "at": utcnow(),
            "evidence_reference": data["evidence_reference"][:1000], "limitations": data["limitations"][:5000],
            "language": data.get("language"), "configuration": data.get("configuration", {}),
            "metrics": data.get("metrics", {}), "clinical_approval_granted": False})
        return result, {"validation_id": result["id"], "release_id": target["id"], "category": kind,
                        "outcome": result["outcome"], "evidence_fingerprint": store.fingerprint(result["evidence_reference"]),
                        "clinical_approval_granted": False}
    if operation == "validation_status":
        actor.require("clinician", "auditor", "operator", "evaluator", "language_reviewer", "usability_reviewer")
        release = store._get(db, actor, data["release_id"], "release")
        reasons = ["release_revoked"] if store._revoked(db, actor, release["id"]) else []
        for member in release["members"]:
            item = store._get(db, actor, member["id"], "recommendation", member["revision"])
            for purpose in release["purposes"]:
                reasons += store._eligibility(db, actor, item, purpose, member["approval_id"])
        records = [record for record in store._list(db, actor, "validation") if record["release_id"] == release["id"]]
        return {"records": records, "current_evidence_eligible": not reasons, "reasons": sorted(set(reasons)),
                "missing_categories": sorted({"engineering", "clinical", "language", "usability"} - {row["category"] for row in records})}, {"release_id": release["id"], "record_count": len(records)}
    raise GovernanceError("unknown_validation_operation", 404)
