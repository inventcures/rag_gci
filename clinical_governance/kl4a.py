"""Pinned KL4A canonical Markdown adapter, not full OKF conformance.

Original ZIP bytes and frontmatter are preserved. Upstream approval, tasks,
rules and .sopkb caches never become local clinical authorization. Imported
knowledge is a curation candidate until locally authored and reviewed.
"""

from copy import deepcopy
import io
import json
from pathlib import PurePosixPath
import stat
from typing import Any, Dict
import zipfile

import yaml

from .models import GovernanceError, unicode_offsets
from .store import require


UPSTREAM_COMMIT = "f66495923ebaea05bfe08f947caeb6510b8b1aea"
ADAPTER_VERSION = "kl4a-okf-0.2-clinical-candidates-v1"


class FrontmatterLoader(yaml.SafeLoader):
    """Reject ambiguous mappings and leave timestamp fields as strings."""


FrontmatterLoader.yaml_implicit_resolvers = deepcopy(
    yaml.SafeLoader.yaml_implicit_resolvers
)
for key, entries in FrontmatterLoader.yaml_implicit_resolvers.items():
    FrontmatterLoader.yaml_implicit_resolvers[key] = [
        pair for pair in entries if pair[0] != "tag:yaml.org,2002:timestamp"
    ]


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise GovernanceError("ambiguous_frontmatter_key", 422)
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


FrontmatterLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping
)


def load_yaml(text: str):
    require(len(text) <= 500_000, "frontmatter_size_limit", 422)
    try:
        for token in yaml.scan(text):
            require(
                not isinstance(token, (yaml.AliasToken, yaml.AnchorToken)),
                "yaml_aliases_not_supported",
                422,
            )
        result = yaml.load(text, Loader=FrontmatterLoader)
        # No Python objects, NaN, dates or other hidden executable state.
        json.dumps(result, allow_nan=False)
        return result
    except GovernanceError:
        raise
    except (yaml.YAMLError, TypeError, ValueError, RecursionError):
        raise GovernanceError("invalid_frontmatter", 422) from None


def frontmatter(content: bytes):
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise GovernanceError("utf8_bundle_required", 422) from None
    lines = text.splitlines(keepends=True)
    require(lines and lines[0].strip() == "---", "frontmatter_required", 422)
    end = next(
        (index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"),
        None,
    )
    require(end is not None, "unterminated_frontmatter", 422)
    metadata = load_yaml("".join(lines[1:end]))
    require(isinstance(metadata, dict), "frontmatter_mapping_required", 422)
    return {"raw": text, "frontmatter": metadata, "body": "".join(lines[end + 1 :])}


def _path(value: str) -> str:
    require(
        isinstance(value, str) and len(value) <= 300 and "\\" not in value,
        "invalid_bundle_path",
        422,
    )
    path = PurePosixPath(value)
    require(
        not path.is_absolute() and ".." not in path.parts and bool(path.parts),
        "unsafe_bundle_path",
        422,
    )
    return str(path)


def _reference(origin: str, reference: str) -> str:
    require(
        isinstance(reference, str)
        and not reference.startswith("/")
        and ":" not in reference
        and "\\" not in reference,
        "external_reference_not_supported",
        422,
    )
    parts = list(PurePosixPath(origin).parent.parts)
    for part in reference.split("/"):
        if part == "..":
            require(bool(parts), "unsafe_bundle_reference", 422)
            parts.pop()
        elif part not in ("", "."):
            parts.append(part)
    return _path("/".join(parts))


def inspect_bundle(content: bytes) -> Dict[str, Any]:
    """Read a bounded in-memory ZIP without filesystem extraction or links."""
    require(0 < len(content) <= 20 * 1024 * 1024, "bundle_size_limit", 422)
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries = [entry for entry in archive.infolist() if not entry.is_dir()]
            require(
                len(entries) <= 500
                and sum(entry.file_size for entry in entries) <= 30 * 1024 * 1024,
                "bundle_expansion_limit",
                422,
            )
            files = {}
            for entry in entries:
                path = _path(entry.filename)
                require(path not in files, "duplicate_bundle_path", 422)
                require(
                    not stat.S_ISLNK(entry.external_attr >> 16),
                    "bundle_symlink_not_allowed",
                    422,
                )
                require(
                    entry.file_size <= 20 * 1024 * 1024
                    and entry.file_size <= 200 * max(entry.compress_size, 1),
                    "bundle_compression_limit",
                    422,
                )
                files[path] = archive.read(entry)
    except GovernanceError:
        raise
    except (zipfile.BadZipFile, RuntimeError, OSError):
        raise GovernanceError("invalid_bundle_zip", 422) from None
    require(
        "index.md" in files and "manifest.yaml" in files,
        "bundle_root_files_required",
        422,
    )
    root = frontmatter(files["index.md"])
    manifest = load_yaml(files["manifest.yaml"].decode("utf-8"))
    require(
        root["frontmatter"].get("okf_version") == "0.2"
        and isinstance(manifest, dict)
        and manifest.get("okf_version") == "0.2"
        and manifest.get("profile_version") == "0.2.0",
        "unsupported_bundle_version",
        422,
    )
    documents = {}
    for path, value in files.items():
        if (
            path.startswith(("knowledge/", "sources/", "evidence/"))
            and path.endswith(".md")
            and not path.endswith("/index.md")
            and not path.startswith(("sources/normalized/", "sources/originals/"))
        ):
            documents[path] = frontmatter(value)
            require(
                isinstance(documents[path]["frontmatter"].get("sopkb", {}), dict),
                "sopkb_mapping_required",
                422,
            )
    sources = {}
    for path, document in documents.items():
        if document["frontmatter"].get("type") != "SOP Source":
            continue
        metadata = document["frontmatter"].get("sopkb", {})
        source_id = metadata.get("source_id")
        require(
            isinstance(source_id, str) and source_id not in sources,
            "invalid_bundle_source_id",
            422,
        )
        normalized_path = _path(metadata.get("normalized_path"))
        require(
            normalized_path.startswith("sources/normalized/")
            and normalized_path in files,
            "normalized_source_missing",
            422,
        )
        original_path = metadata.get("original_path")
        if original_path:
            original_path = _path(original_path)
            require(
                original_path.startswith("sources/originals/"),
                "original_path_invalid",
                422,
            )
        try:
            normalized = files[normalized_path].decode("utf-8")
        except UnicodeDecodeError:
            raise GovernanceError("utf8_normalized_source_required", 422) from None
        sources[source_id] = {
            "upstream_id": source_id,
            "normalized_text": normalized,
            "original": files.get(original_path) if original_path else None,
            "metadata": document,
            "normalized_path": normalized_path,
            "coverage_gaps": ["imported_clinical_semantics_require_local_review"]
            + ([] if original_path in files else ["upstream_original_missing"]),
        }
    require(bool(sources), "bundle_sources_required", 422)
    candidates = []
    for path, document in documents.items():
        if document["frontmatter"].get("type") != "SOP Knowledge Piece":
            continue
        metadata = document["frontmatter"].get("sopkb", {})
        source_id = metadata.get("source_id")
        require(source_id in sources, "candidate_source_missing", 422)
        reference = metadata.get("evidence")
        anchors, gaps = [], []
        if reference:
            evidence_path = _reference(path, reference)
            evidence_doc = documents.get(evidence_path)
            require(
                evidence_doc is not None
                and evidence_doc["frontmatter"].get("type") == "SOP Evidence",
                "candidate_evidence_missing",
                422,
            )
            evidence = evidence_doc["frontmatter"].get("sopkb", {})
            require(
                evidence.get("source_id") == source_id
                and evidence.get("knowledge_item_id")
                == metadata.get("knowledge_item_id"),
                "evidence_identity_mismatch",
                422,
            )
            start, end = evidence.get("start_pos"), evidence.get("end_pos")
            text = sources[source_id]["normalized_text"]
            offsets = unicode_offsets(text, start, end)
            anchors.append(
                {
                    "start": start,
                    "end": end,
                    "excerpt": text[start:end],
                    "offsets": offsets,
                    "upstream_span_status": evidence.get("span_status"),
                    "upstream_document": evidence_doc,
                }
            )
            if evidence.get("span_status") != "exact":
                gaps.append("upstream_span_not_exact")
        else:
            gaps.append("candidate_evidence_missing")
        candidates.append(
            {
                "upstream_id": metadata.get("knowledge_item_id"),
                "source_id": source_id,
                "metadata": document,
                "anchors": anchors,
                "coverage_gaps": gaps,
                "local_status": "unreviewed_import",
                "upstream_review_status": metadata.get("review_status"),
                "clinical_approval": False,
            }
        )
    return {
        "adapter_version": ADAPTER_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "manifest": manifest,
        "root": root,
        "sources": sources,
        "candidates": candidates,
        "archive_bytes": content,
        "unknown_fields_preserved": True,
        "conformance": "pinned_subset_not_full_okf",
        "automatically_approved": False,
    }
