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

Usage:
    python scripts/reindex_with_configured_model.py [--dry-run] [--yes]

    --dry-run   Report what would change without touching the index.
    --yes       Skip the confirmation prompt.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb  # noqa: E402

from retrieval_index import check_live_index, load_identity_from_config  # noqa: E402

COLLECTION_NAME = "documents"
DEFAULT_CHROMA_PATH = "./data/chroma_db"
DEFAULT_CONFIG_PATH = "./config.yaml"


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
    args = parser.parse_args()

    config = load_config(args.config)
    identity = load_identity_from_config(config)
    print(f"Configured embedding model : {identity.model}")
    print(f"Configured dimensionality  : {identity.dimension}")

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
    from reindex_service import reindex_with_identity

    result = reindex_with_identity(
        chroma_path=args.chroma_path,
        collection_name=COLLECTION_NAME,
        identity=identity,
        local_model_path=os.environ.get("EMBEDDING_MODEL_PATH") or None,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
