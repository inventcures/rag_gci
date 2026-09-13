"""Stage KL4A candidates atomically, retaining external evidence as untrusted."""

import hashlib

from .kl4a import inspect_bundle
from .store import require


def import_bundle(store, db, actor, data):
    actor.require("editor", "clinician")
    parsed = inspect_bundle(data["content"])
    bundle_id = parsed["manifest"].get("id")
    require(
        isinstance(bundle_id, str) and 0 < len(bundle_id) <= 200,
        "bundle_id_required",
        422,
    )
    previous_imports = [
        item
        for item in store._list(db, actor, "kl4a_import")
        if item["manifest"].get("id") == bundle_id
    ]
    previous_sources = {}
    if previous_imports:
        # Untrusted upstream IDs cannot silently replace an existing local source.
        require(
            data.get("update_import_id") == previous_imports[-1]["id"],
            "bundle_already_imported_select_latest_update",
        )
        for source_id in previous_imports[-1]["source_ids"]:
            prior = store._get(db, actor, source_id, "source")
            upstream = prior["import_metadata"]["frontmatter"]["sopkb"]["source_id"]
            previous_sources[upstream] = prior
        require(
            set(previous_sources) <= set(parsed["sources"]),
            "removed_sources_require_explicit_revocation",
        )
    else:
        require(not data.get("update_import_id"), "import_update_target_mismatch")
    mapping, source_ids, candidates = {}, [], []
    for upstream_id, source in parsed["sources"].items():
        metadata = source["metadata"]["frontmatter"]
        original = source["original"]
        normalized = source["normalized_text"]
        # Missing upstream originals are explicit gaps, never labelled as a PDF.
        result, detail = store._dispatch(
            db,
            actor,
            "source",
            {
                "original": original or normalized.encode("utf-8"),
                "normalized_text": normalized,
                "title": metadata.get("title", upstream_id),
                "version": str(
                    metadata.get("sopkb", {}).get("source_version_id", "imported")
                ),
                "rights": data["rights"],
                **(
                    {"family_id": previous_sources[upstream_id]["family_id"]}
                    if upstream_id in previous_sources
                    else {}
                ),
                "normalizer": "kl4a-preserved-normalized-codepoints-v1",
                "coverage_gaps": source["coverage_gaps"],
                "media_type": "application/octet-stream"
                if original and not original.startswith(b"%PDF-")
                else "application/pdf"
                if original
                else "text/markdown",
                "import_metadata": source["metadata"],
            },
        )
        store._audit(db, actor, "import.source", detail)
        mapping[upstream_id] = result
        source_ids.append(result["id"])
    for candidate in parsed["candidates"]:
        source = mapping[candidate["source_id"]]
        anchors = [
            {
                "source_id": source["id"],
                "normalized_sha256": source["normalized_sha256"],
                "start": anchor["start"],
                "end": anchor["end"],
                "excerpt": anchor["excerpt"],
            }
            for anchor in candidate["anchors"]
        ]
        result = store._put(
            db,
            actor,
            "import_candidate",
            {
                **candidate,
                "source_id": source["id"],
                "anchors": anchors,
                "adapter_version": parsed["adapter_version"],
            },
        )
        candidates.append(result["id"])
    result = store._put(
        db,
        actor,
        "kl4a_import",
        {
            "archive_sha256": hashlib.sha256(data["content"]).hexdigest(),
            "adapter_version": parsed["adapter_version"],
            "upstream_commit": parsed["upstream_commit"],
            "source_ids": source_ids,
            "candidate_ids": candidates,
            "manifest": parsed["manifest"],
            "supersedes_import_id": data.get("update_import_id"),
            "root": parsed["root"],
            "unknown_fields_preserved": True,
            "clinical_approval": False,
            "conformance": parsed["conformance"],
        },
    )
    db.execute(
        "INSERT INTO blobs VALUES(?,?,?)", (actor.tenant, result["id"], data["content"])
    )
    return result, {
        "import_id": result["id"],
        "archive_sha256": result["archive_sha256"],
        "source_ids": source_ids,
        "candidate_ids": candidates,
        "adapter_version": result["adapter_version"],
        "clinical_approval": False,
    }
