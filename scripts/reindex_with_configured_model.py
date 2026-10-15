#!/usr/bin/env python3
"""
Reindex the knowledge base with the configured embedding model
==============================================================
ADR 0003 pins retrieval to bge-m3, chosen by measurement: mean hit@5 of 0.96
across the eleven supported languages, against 0.09 for the English-only index
this replaced, which scored 0.00 on all ten Indic languages.

Changing the configured model does not reindex by itself.
`get_or_create_collection` returns whatever collection is already on disk
regardless of the embedding function it was created with, and the existing
auto-rebuild path triggers on corruption rather than on a model change. This
script is the explicit migration.

Writes the embedding identity onto the collection afterwards, so a later mismatch
is diagnosable from metadata rather than inferred from dimensionality alone.

This is the one rebuild command (round-3 review ticket 20 removed the root rebuild
and cleanup scripts). It goes through reindex_service, which refuses any model but
the one the study release records, so a config.yaml that disagrees with the
release is refused before the index is opened.

Usage:
    python scripts/reindex_with_configured_model.py [--dry-run] [--yes] [--force]
    python scripts/reindex_with_configured_model.py --prune-orphans

    --dry-run        Report what would change without touching the index.
    --yes            Skip the confirmation prompt.
    --force          Rebuild even when the identity already matches.
    --prune-orphans  Delete chunks of documents no longer in the document metadata.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb  # noqa: E402

from reindex_service import (  # noqa: E402
    ReleaseModelMismatch,
    prune_orphaned_chunks,
    reindex_with_identity,
    require_release_model,
)
from retrieval_index import check_live_index, load_identity_from_config  # noqa: E402
from study_release import DEFAULT_RELEASE_PATH, StudyRelease, load_yaml  # noqa: E402

COLLECTION_NAME = "documents"
DEFAULT_CHROMA_PATH = "./data/chroma_db"
DEFAULT_CONFIG_PATH = "./config.yaml"
DEFAULT_METADATA_PATH = "./data/document_metadata.json"


def load_config(path: str) -> dict:
    import yaml

    config_path = Path(path)
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force", action="store_true",
        help="Reindex even when the identity matches. Needed to repair the "
             "collection's persisted embedding function, which the identity check "
             "cannot detect because it does not affect dimensionality.",
    )
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--release", default=str(DEFAULT_RELEASE_PATH))
    parser.add_argument(
        "--prune-orphans", action="store_true",
        help="Delete chunks of documents no longer in the document metadata, then stop.",
    )
    parser.add_argument("--metadata", default=DEFAULT_METADATA_PATH)
    args = parser.parse_args()

    if args.prune_orphans:
        return prune(args.chroma_path, args.metadata)

    config = load_config(args.config)
    identity = load_identity_from_config(config)
    print(f"Configured embedding model : {identity.model}")
    print(f"Configured dimensionality  : {identity.dimension}")

    release_path = Path(args.release)
    release = StudyRelease(
        record=load_yaml(release_path) if release_path.exists() else {}, path=release_path,
    )
    try:
        require_release_model(identity, release)
    except ReleaseModelMismatch as exc:
        print(f"\nRefused: {exc}")
        print("Fix config.yaml's embedding block, or record a new study release.")
        return 1

    client = chromadb.PersistentClient(path=args.chroma_path)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        collection = None

    if collection is None:
        print("No existing collection found. Nothing to migrate.")
        return 0

    count = collection.count()
    drift = check_live_index(collection, identity)

    print(f"Existing collection       : {COLLECTION_NAME} ({count} chunks)")
    if drift.observed:
        print(f"Existing index identity   : {drift.observed.model} "
              f"at {drift.observed.dimension}d")
    else:
        print("Existing index identity   : unobservable")

    if not drift.has_drift and not args.force:
        print("\nIndex already matches the configured model. Nothing to do.")
        return 0

    if not drift.has_drift:
        print("\nNo identity drift, rebuilding because --force was given.")
    else:
        print(f"\nDrift detected: {', '.join(drift.reasons)}")
    print(f"Remediation: {drift.remediation}")

    if count == 0:
        print("\nIndex is empty, so there is nothing to migrate.")
        return 0

    if args.dry_run:
        print(f"\nDry run. Would reindex {count} chunks with {identity.model}.")
        return 0

    if not args.yes:
        answer = input(
            f"\nThis replaces the existing {count}-chunk index. Proceed? [y/N] "
        ).strip().lower()
        if answer != "y":
            print("Aborted. The existing index is unchanged.")
            return 1

    print("\nReindexing from source documents...")
    from retrieval.embedding_runtime import configured_runtime

    result = reindex_with_identity(
        chroma_path=args.chroma_path,
        collection_name=COLLECTION_NAME,
        identity=identity,
        local_model_path=os.environ.get("EMBEDDING_MODEL_PATH") or None,
        runtime=configured_runtime(config),
        release=release,
    )
    if result.error:
        print(f"\nReindex failed: {result.error}")
        return 1

    print(f"\nChunks written: {result.chunks_written}")
    print(f"Duration      : {result.duration_seconds}s")

    verify_client = chromadb.PersistentClient(path=args.chroma_path)
    verify = check_live_index(
        verify_client.get_collection(COLLECTION_NAME), identity
    )
    if verify.has_drift:
        print(f"WARNING: still drifting after reindex: {verify.reasons}")
        return 1

    print(f"\nVerified: index now matches {identity.model} "
          f"at {identity.dimension}d.")
    print("Restart the server: it checks the index at start-up and refuses "
          "every query until the check passes.")
    return 0


def prune(chroma_path: str, metadata_path: str) -> int:
    """Delete the chunks of documents missing from the document metadata."""
    import json

    path = Path(metadata_path)
    if not path.exists():
        print(f"No document metadata at {path}; refusing to prune (it would empty the index).")
        return 1
    known = set(json.loads(path.read_text(encoding="utf-8")).keys())
    if not known:
        print(f"{path} lists no documents; refusing to prune (it would empty the index).")
        return 1
    removed = prune_orphaned_chunks(chroma_path, COLLECTION_NAME, known)
    print(f"Pruned {len(removed)} orphaned chunks ({len(known)} documents in the metadata).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
